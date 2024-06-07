import torch.nn as nn
from rouge_score import rouge_scorer
from six.moves import zip_longest
import numpy as np
import torch

class Rouge(nn.Module):
    def __init__(self, rouges, **kwargs):
        super().__init__()
        rouges = [r.replace('rougel', 'rougeL') for r in rouges]
        self.scorer = rouge_scorer.RougeScorer(rouges, use_stemmer=True)
        self.rouges = rouges

    def forward(self, refs, hyps):
        scores = []
        for target_rec, prediction_rec in zip_longest(refs, hyps):
            if target_rec is None or prediction_rec is None:
                raise ValueError("Must have equal number of lines across target and "
                                 "prediction.")
            scores.append(self.scorer.score(target_rec, prediction_rec))

        # aggregator = scoring.BootstrapAggregator()
        # for score in scores:
        #     aggregator.add_scores(score)
        # print(aggregator.aggregate())
        f1_rouge = [s[self.rouges[0]].fmeasure for s in scores]
        return np.mean(f1_rouge), f1_rouge


class Rouge1(Rouge):
    def __init__(self, **kwargs):
        super(Rouge1, self).__init__(rouges=['rouge1'])


class Rouge2(Rouge):
    def __init__(self, **kwargs):
        super(Rouge2, self).__init__(rouges=['rouge2'])


class RougeL(Rouge):
    def __init__(self, **kwargs):
        super(RougeL, self).__init__(rouges=['rougeL'])


from torchmetrics import Metric
from typing import Any, List, Optional, Union
from torch import Tensor

class TMRougeL(Metric):

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    scores: List[Tensor]
    

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
       
        self.model = Rouge(rouges=['rougeL'], **kwargs)

        self.add_state("scores", [], dist_reduce_fx="cat")       

    def update(self, preds: List[str], target: List[str]) -> None:
        """Store predictions/references for computing BERT scores.

        It is necessary to store sentences in a tokenized form to ensure the DDP mode working.
        """
        res = self.model(refs=target, hyps=preds)
        self.scores.extend(res[1])

    def compute(self) -> float:
        """Calculate Rouge score."""        
        return torch.mean(torch.tensor(self.scores, dtype=torch.float))
    
    
class TMRouge1(Metric):

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    scores: List[Tensor]
    

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
       
        self.model = Rouge(rouges=['rouge1'], **kwargs)

        self.add_state("scores", [], dist_reduce_fx="cat")
       

    def update(self, preds: List[str], target: List[str]) -> None:
        """Store predictions/references for computing BERT scores.

        It is necessary to store sentences in a tokenized form to ensure the DDP mode working.
        """
     
        res = self.model(refs=target, hyps=preds)
        self.scores.extend(res[1])
  

    def compute(self) -> float:
        """Calculate Rouge score."""

        return np.mean(self.scores)
    



class TMRouge2(Metric):

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    scores: List[Tensor]
        
    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
       
        self.model = Rouge(rouges=['rouge'], **kwargs)

        self.add_state("scores", [], dist_reduce_fx="cat")
       
    def update(self, preds: List[str], target: List[str]) -> None:
        """Store predictions/references for computing BERT scores.

        It is necessary to store sentences in a tokenized form to ensure the DDP mode working.
        """
     
        res = self.model(refs=target, hyps=preds)
        self.scores.extend(res[1])
  
    def compute(self) -> float:
        """Calculate Rouge score."""

        return np.mean(self.scores)