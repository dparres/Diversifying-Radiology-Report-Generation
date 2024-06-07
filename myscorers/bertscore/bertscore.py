import os
import torch
import bert_score
import torch.nn as nn

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BertScorer(nn.Module):
    def __init__(self):
        
        super().__init__()
         
        with torch.no_grad():
            self.mybert_scorer = bert_score.BERTScorer(
                                        model_type='distilbert-base-uncased',
                                        num_layers=5,
                                        batch_size=64,
                                        #nthreads=4,
                                        all_layers=False,
                                        idf=False,
                                        #idf_sents=None,
                                        device='cuda',
                                        lang='en',
                                        rescale_with_baseline=True,
                                        baseline_path=None
                                        )
    
    def forward(self, hyps, refs):
        p, r, f = self.mybert_scorer.score(hyps, refs)
        
        return torch.mean(f).item()