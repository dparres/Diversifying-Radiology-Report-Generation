import torch
import pathlib
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate as pytorch_default_collate
import random

from paths import IMAGES_MIMIC_PATH, DICT_CSV_MIMIC_PATH
from mydatasets.mydatasets_utils import ifcc_clean_report, vilmedic_collate

class mimic_Dataset(Dataset):

    def __init__(self, 
                 transform, 
                 tokenizer,
                 processor,
                 partition = "train",
                 text_preprocessing="ifcc_clean_report",
                 multi_image=3):

        self.transform = transform
        self.tokenizer = tokenizer
        self.processor = processor
        self.partition = partition
        self.text_preprocessing = text_preprocessing if text_preprocessing is None else eval(text_preprocessing)
        self.multi_image = multi_image

        self.random_padding = False
        if self.partition == "train":
            self.random_padding = True

        # Load CSV partition
        self.csv_path = DICT_CSV_MIMIC_PATH[self.partition]
        self.dataset_df = pd.read_csv(self.csv_path)
        # Remove empty text from self.dataset_df
        self.remove_empty_text()

        # Set images path
        self.img_root_dir = pathlib.Path(IMAGES_MIMIC_PATH) if IMAGES_MIMIC_PATH is not None else pathlib.Path.cwd()

    def __len__(self):
        return len(self.dataset_df)
    
    def __getitem__(self, idx):

        img_list_from_idx = []
        num_images = len(self.dataset_df.iloc[idx].images.split(","))

        # Process all images from patient idx
        for i in range(num_images):

            img_name = self.img_root_dir / self.dataset_df.iloc[idx].images.split(",")[i]
            image = Image.open(img_name).convert('RGB')
            #image.save('rad.png')

            if isinstance(self.transform, transforms.Compose):
                # If torchvision transformation
                image = self.transform(image)
            elif isinstance(self.transform, A.core.composition.Compose):
                # If Albumentations transformation
                image = self.transform(image=np.asarray(image))['image']
            else:
                raise ValueError("Unknown transformation type. Supported types: torchvision.transforms.Compose, albumentations.core.composition.Compose")
            
            # Image Processor
            image = self.processor(image, random_padding=self.random_padding, return_tensors="pt").pixel_values
            image = image.squeeze()
                
            img_list_from_idx.append(image)

        # Report
        text = self.dataset_df.iloc[idx].text
        text = self.text_preprocessing(text)
        if self.partition == "train":
            text = self.change_sentences_order(text)

        # Calculate images_mask
        im_and_immask = vilmedic_collate([img_list_from_idx], self.multi_image)
        images = im_and_immask["images"]
        images_mask = im_and_immask["images_mask"]
              
        return {'idx': idx, 'text': text, 'image': images, "images_mask": images_mask}
    
    def remove_empty_text(self):

        def num_phrases(text):
            text = text.replace("\n", "")
            ls_text = text.split(".")
            #remove all '' elememts from ls_text
            ls_text = list(filter(lambda a: a != '', ls_text))
            return len(ls_text)
        self.dataset_df["num_phrases"] = self.dataset_df.text.apply(num_phrases)
        print("Len before removing empty text", len(self.dataset_df))
        self.dataset_df = self.dataset_df[self.dataset_df.num_phrases > 0].copy()
        print("Len after removing empty text", len(self.dataset_df))

    '''
    Text Augmentation proposed in the paper
    '''
    def change_sentences_order(self, input_string):

        # Split the string into sentences
        sentences = input_string.split('.')

        # Remove empty strings from the list (resulting from the split)
        sentences = [sentence.strip() for sentence in sentences if sentence]

        # Shuffle the order of the sentences
        random.shuffle(sentences)

        # Join the sentences back into a string
        output_string = ' . '.join(sentences)

        # Add a period at the end of the string if it's missing
        if not output_string.endswith('.'):
            output_string += '.'

        return output_string

    def get_collate_fn(self):
        def collate_fn(batch):

            images =  pytorch_default_collate([s['image'] for s in batch])
            images_mask = pytorch_default_collate([s['images_mask'] for s in batch])
            idx = pytorch_default_collate([s['idx'] for s in batch])
            phrases = [s['text'] for s in batch]

            # Report Tokenizer
            seq = self.tokenizer(
                                phrases, max_length=128,
                                padding='max_length', 
                                truncation=True, 
                                return_tensors='pt'
                                )

            collated = {
                        "idx": idx,
                        "images": images, 
                        "input_ids": seq.input_ids, 
                        "images_mask": images_mask,
                        "attention_mask": seq.attention_mask,
                        "text": phrases
                        }
            
            return collated
        return collate_fn
