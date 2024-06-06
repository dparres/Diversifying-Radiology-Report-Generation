import cv2
import torch
import inspect
import functools
import torch.nn as nn
import albumentations as A
from einops import rearrange
from torchvision import transforms
from transformers.models.bert_generation import BertGenerationConfig, BertGenerationDecoder
from transformers import AutoFeatureExtractor, SwinModel, BertTokenizer

from paths import VOCAB_PATH, SWINB_IMAGENET22K_WEIGHTS
from mymodels.beam_search import prepare_inputs_for_generation, _validate_model_kwargs

DICT_DECODER_CONFIG = {
    "is_decoder": True,
    "add_cross_attention": True,
    "bos_token_id": 0,
    "eos_token_id": 2,
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "layer_norm_eps": 1.0e-05,
    "num_attention_heads": 16,
    "num_hidden_layers": 3,
    "pad_token_id": 1,
    "vocab_size": 9877,
}

class SwinBERT9k(nn.Module):
    """
    If proto is mentioned in decoder dict, loads pretrained models from proto strings.
    Otherwise, loads a BertGenerationDecoder model from decoder dict.
    """

    def __init__(self):
        super().__init__()
        
        self.processor = AutoFeatureExtractor.from_pretrained(SWINB_IMAGENET22K_WEIGHTS)
        self.encoder = SwinModel.from_pretrained(SWINB_IMAGENET22K_WEIGHTS)
        

        # Decoder
        self.tokenizer = BertTokenizer(
                                vocab_file=VOCAB_PATH, 
                                do_basic_tokenize=False, 
                                use_fast=False, 
                                max_length=128
                                )
        dec_config = BertGenerationConfig(**DICT_DECODER_CONFIG) # DICT_DECODER_CONFIG
        self.decoder = BertGenerationDecoder(dec_config)
        
        # From Encoder to Decoder
        if self.decoder.config.hidden_size != self.encoder.config.hidden_size:
            self.projection = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)
        
        # Special Tokens
        self.bos_token_id = self.decoder.config.bos_token_id
        self.eos_token_id = self.decoder.config.eos_token_id
        self.pad_token_id = self.decoder.config.pad_token_id
        
        # Data Transformations
        self.train_transform = self.transforms_train_augmented()
        
        '''self.train_transform = transforms.Compose([
                transforms.Resize(416),
                transforms.CenterCrop(384),
                transforms.RandomHorizontalFlip(),
                ])'''
        
        self.val_transform = transforms.Compose([
                transforms.Resize(416),
                transforms.CenterCrop(384),
                ])

        # Evaluation
        self.decoder.prepare_inputs_for_generation = functools.partial(prepare_inputs_for_generation, self.decoder)
        # We override _validate_model_kwargs width empty function because we add custom model kwargs that triggers
        # errors in original _validate_model_kwargs
        self.decoder._validate_model_kwargs = functools.partial(_validate_model_kwargs, self.decoder)

        # Inference
        #self.generate = self.decoder.generate
        self.config = self.decoder.config

    def transforms_train_augmented(self):
        """Transform and apply augmentation to the training set images."""
        pre_crop_size = 416 # 32px de dif
        p_train = 0.5
        p_hflip = 0.5
        shift_limit = 0.0625
        scale_limit = ((-0.2, 0.1))
        rotate_limit = 10
        scale = (0.1)
        brightness_limit = (-0.2, 0.2)
        contrast_limit = (-0.2, 0.2)
        pad_mode = cv2.BORDER_CONSTANT
        pad_val = (105/256, 105/256, 105/256)
        FINAL_SIZE = 384

        train_transform = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=shift_limit, scale_limit=scale_limit,
                rotate_limit=rotate_limit, border_mode=pad_mode, value=pad_val,
                p=p_train
            ),
            A.Perspective(
                scale=scale, pad_mode=pad_mode, pad_val=pad_val, p=p_train
            ),
            A.Resize(height=pre_crop_size, width=pre_crop_size),
            A.RandomCrop(height=FINAL_SIZE, width=FINAL_SIZE),
            #A.HorizontalFlip(p=p_hflip),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit, contrast_limit=contrast_limit,
                p=p_train
            ),
            
        ])
        return train_transform
    
    def forward(self, 
                input_ids,
                attention_mask, 
                images, 
                images_mask=None, 
                encoder_outputs=None,
                encoder_attention_mask=None, 
                **kwargs):
        
        if encoder_outputs is None:
            encoder_outputs, encoder_attention_mask = self.encode(
                images, images_mask, **kwargs)
                  
        out = self.decoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_outputs,
                    encoder_attention_mask=encoder_attention_mask,
                    labels=input_ids,
                    **kwargs
                    )
        out = vars(out)

        return {"loss": out["loss"]}
    
    # Necessary for generation
    def encode(self, images, images_mask=None):

        # Multi-image forward pass
        num_images = images.shape[1]
        images = rearrange(images, 'd0 d1 d2 d3 d4 -> (d0 d1) d2 d3 d4') # (b 3 3 224 224) -> (bx3 3 224 224)
        feature = self.encoder(images).last_hidden_state # (bx3 256 1024)

        if self.decoder.config.hidden_size != self.encoder.config.hidden_size:
            feature = self.projection(feature)

        # Masking features of empty images
        feature = feature.view(int(feature.shape[0] / num_images), num_images, feature.shape[-2], feature.shape[-1])
        feature = feature * images_mask.unsqueeze(-1).unsqueeze(-1)
        if torch.cuda.is_available():
            feature = feature.cuda()
        
        # Creating feature-wise attention mask
        feature = rearrange(feature, 'd0 d1 d2 d3 -> d0 (d1 d2) d3')
        feature_mask = (torch.sum(torch.abs(feature), dim=-1) != 0)

        return feature, feature_mask

    # TODO: use_cache True da fallo?
    def generate(self,  
                 images, images_mask=None, 
                 encoder_outputs=None, 
                 encoder_attention_mask=None, 
                 max_len = 128, num_beams = 8,
                 num_return_sequences=1, tokenizer = None, 
                 bad_words_ids=None, top_k=50, 
                 forced_eos_token_id=None,
                 output_scores=False, do_sample=False,
                 use_cache=False, 
                 return_dict_in_generate=False,
                 calc_grad=False, **kwargs):

        #put model in eval mode
        self.eval()
        batch_size = images.shape[0]
    
        #generate
        if calc_grad:
            if encoder_outputs is None:
                encoder_output, encoder_attention_mask =  self.encode(images, images_mask)

            '''
            The generate() function has a no_grad decorator that stops the 
            computational graph being returned, and this code just removes the 
            decorator and leaves the rest of the generate function unchanged.

            So we use: inspect.unwrap to calculate gradients
            '''
            hyps = inspect.unwrap(self.decoder.generate)( 
                    self=self.decoder,
                    input_ids=torch.ones((batch_size, 1), dtype=torch.long).to(encoder_output.device) * self.bos_token_id,
                    encoder_hidden_states=encoder_output,
                    encoder_attention_mask=encoder_attention_mask,
                    num_return_sequences=num_return_sequences,
                    max_length= max_len,
                    num_beams=num_beams,
                    bad_words_ids=bad_words_ids,
                    top_k=top_k,
                    output_scores=output_scores,
                    return_dict_in_generate=return_dict_in_generate,
                    do_sample=do_sample,
                    forced_eos_token_id=forced_eos_token_id,
                    bos_token_id=self.bos_token_id,
                    eos_token_id=self.eos_token_id,
                    pad_token_id=self.pad_token_id,
                    use_cache=use_cache,
            )      
        else:
            with torch.no_grad():
                if encoder_outputs is None:
                    encoder_output, encoder_attention_mask =  self.encode(images, images_mask)

                inputs_ids = torch.ones((batch_size, 1), dtype=torch.long).to(encoder_output.device) * self.bos_token_id
                hyps = self.decoder.generate(
                        input_ids=inputs_ids,
                        encoder_hidden_states=encoder_output,
                        encoder_attention_mask=encoder_attention_mask,
                        num_return_sequences=num_return_sequences,
                        max_length= max_len,
                        num_beams=num_beams,
                        bad_words_ids=bad_words_ids,
                        top_k=top_k,
                        output_scores=output_scores,
                        return_dict_in_generate=return_dict_in_generate,
                        do_sample=do_sample,
                        forced_eos_token_id=forced_eos_token_id,
                        bos_token_id=self.bos_token_id,
                        eos_token_id=self.eos_token_id,
                        pad_token_id=self.pad_token_id,
                        use_cache=use_cache,
                )    

            #lenght penalty falta
        self.train()
        outs = hyps
        #print("hyps: ", hyps.sequences) 
        if tokenizer is not None:
            hyps = [self.tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False) for h in hyps.sequences]
        #print("hyps: ", hyps)
        return hyps, outs
