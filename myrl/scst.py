import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf.listconfig import ListConfig

from myscorers.bertscore.bertscore import BertScorer
from myscorers.myradgraph.myradgraph import myRadGraph

# scst loss intuition:
# https://ai.stackexchange.com/questions/2405/how-do-i-handle-negative-rewards-in-policy-gradients-with-the-cross-entropy-loss

REWARD_COMPLIANT = {
    "F1RadGraph": [myRadGraph, 1],
    "BertScorer": [BertScorer, 1],
}

def scst_loss(input,
              seq,
              reward_sampling,
              reward_greedy,
              scores_weights,
              pad_token_id):
    # HuggingFace TopKLogitsWarper (if top_k > 0) puts -float('inf') for non top_k (or if we use bad_words_ids)
    # Padding can then have logits -float('inf') though we have to pad because hyp generation is finished
    # masking -float('inf') * 0 results in NaN s, therefore need to do:
    input[input == -float("Inf")] = 0.

    # Masked logits
    mask = (seq > pad_token_id).float()
    input = input.squeeze(-1)
    input = input * mask
    input = input / torch.sum(mask)

    # SCST Loss
    delta_rewards = [torch.tensor(rs).cuda() - torch.tensor(rg).cuda()  for rs, rg in
                     zip(reward_sampling, reward_greedy)]
        
    loss = [scores_weights[i] * (-input * r.unsqueeze(-1).expand_as(input)) for i, r in enumerate(delta_rewards)]
    
    # Compute mean loss
    #  Double sum: sum words, then sum sentences. Division is done beforehand 'input = input / torch.sum(mask)'
    loss = sum([torch.sum(l) for l in loss])

    # Stats
    delta_reward = torch.mean(torch.stack(delta_rewards))
    delta_reward_per_metric = torch.mean(torch.stack(delta_rewards), dim=-1)

    return loss, delta_reward, delta_reward_per_metric


class SCST(nn.Module):
    def __init__(self, 
                 bos_token_id, 
                 eos_token_id, 
                 pad_token_id, 
                 scores, 
                 scores_args=None, 
                 scores_weights=None, 
                 top_k=0, 
                 use_nll=False):
        super().__init__()
        
        self.max_length = 128
        self.top_k = top_k
        self.use_nll = use_nll
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.scores = scores
        self.scores_args = scores_args
        self.scores_weights = scores_weights

        assert self.scores is not None

        if not isinstance(scores, (list, ListConfig)):
            scores = [scores]

        scores = list((map(lambda x: x, scores)))
        assert all([score in REWARD_COMPLIANT for score in scores]), "{} not in {}".format(scores,
                                                                                           REWARD_COMPLIANT.keys())
        # Scores weights
        if len(scores) > 1 or use_nll:
            assert scores_weights is not None, "You need to mention scores_weights"
            assert isinstance(scores_weights, (list, ListConfig)), "scores_weights must be a list"
            if self.use_nll:
                assert len(scores_weights) == len(scores) + 1, "Mention nll_weight + as much scores_weights as scores"
            else:
                assert len(scores_weights) == len(scores), "Mention as much scores_weights as scores"

        else:
            self.scores_weights = [1.0]

        # Scores args
        if scores_args is not None:
            if not isinstance(scores_args, (list, ListConfig)):
                scores_args = [scores_args]
            assert len(scores_args) == len(scores), \
                "You need to mention as much scores_args as scores (i.e. [{arg1:value}, {}, {arg1:value}])"
        else:
            self.scores_args = [None] * len(scores)

        self.scorers = []
        self.scorers_index = []
        for index, score in enumerate(scores):
            scorer, scorer_index = REWARD_COMPLIANT[score]
            if self.scores_args[index] is not None:                
                scorer = scorer(**self.scores_args[index])
            else:
                scorer = scorer()
            self.scorers.append(scorer)
            self.scorers_index.append(scorer_index)

    def forward_greedy(self, input_ids, images, model, images_mask=None):
        
        assert not torch.is_grad_enabled(), "Please add torch.no_grad() decorator."

        # Forward Greedy
        _, out = model.generate(
            images, 
            images_mask=images_mask,
            max_len=128,
            num_beams=1,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True,
            forced_eos_token_id=True,
            use_cache=False, # dpm: esto daba error!
            calc_grad=False,
        )

        # Calculate Greedy Rewards
        greedy_input_ids = out.sequences
        reward_greedy, generated_reports_list, reference_reports_list = self.get_reward(greedy_input_ids.detach().data, input_ids, model.tokenizer)

        return {
            "reward_greedy": reward_greedy, 
            "hyp_list": generated_reports_list, 
            "ref_list": reference_reports_list
            }

    def forward_sampling(self, input_ids, attention_mask, reward_greedy, images, model, images_mask=None):
        
        assert torch.is_grad_enabled()

        #batch_size = input_ids.shape[0]
        # Calculate NLL
        if self.use_nll:
            decoder_out = model(input_ids, attention_mask, 
                                        images, images_mask=images_mask)
            nll_loss = decoder_out['loss']

        # Forward Sampling
        out, _ = model.generate(
            images, images_mask=images_mask,
            max_len=128,
            num_beams=1,
            num_return_sequences=1,
            bad_words_ids=[[self.pad_token_id], [self.bos_token_id]],
            top_k=self.top_k,
            forced_eos_token_id=True,
            output_scores=True,
            do_sample=True,
            use_cache=False, #True, # dpm: esto daba error!
            return_dict_in_generate=True,
            calc_grad=True,
        )

        # Calculate Report Probabilities from Forward Sampling
        sampled_ids = out.sequences[:, 1:].contiguous()
        logits = torch.stack(out.scores, dim=1)
        logits = F.log_softmax(logits, dim=-1)
        sampled_logits = logits.gather(2, sampled_ids.unsqueeze(-1))

        # Calculate Sampling Rewards  
        reward_sampling, generated_reports_list, _ = self.get_reward(sampled_ids.data, input_ids, model.tokenizer)

        # Calculate SCST Loss to do Backward using Greedy and Sampling Rewards
        loss, delta_reward, delta_reward_per_metric = scst_loss(sampled_logits,
                                                                sampled_ids.data,
                                                                reward_sampling,
                                                                reward_greedy,
                                                                # avoid nll_weight if present
                                                                self.scores_weights[-len(self.scores):],
                                                                self.pad_token_id
                                                                )
        # Add NLL Loss to SCST Loss
        if self.use_nll:
            loss += self.scores_weights[0] * nll_loss

        return {
            "loss": loss, 
            "delta_reward": delta_reward, 
            "delta_reward_per_metric": delta_reward_per_metric, 
            "reward_sampling": reward_sampling, 
            "hyp_list": generated_reports_list, 
            "nll_loss": nll_loss
            }

    def get_reward(self, generated_reports_input_ids, reference_reports_input_ids, tokenizer):

        generated_reports_list = []
        reference_reports_list = []
        for gen_rep, ref_rep in zip(generated_reports_input_ids, reference_reports_input_ids):
            
            generated_reports_list.append(tokenizer.decode(gen_rep, skip_special_tokens=True, clean_up_tokenization_spaces=False))

            reference_reports_list.append(tokenizer.decode(ref_rep, skip_special_tokens=True, clean_up_tokenization_spaces=False))        
            
        reward_list = []
        for scorer, scorer_index in zip(self.scorers, self.scorers_index):

            try:
                score_out = scorer(generated_reports_list, reference_reports_list)
            except:
                score_out = torch.tensor(0.)
                print("Except del reward")

            reward_list.append(score_out)
                    
        return reward_list, generated_reports_list, reference_reports_list
