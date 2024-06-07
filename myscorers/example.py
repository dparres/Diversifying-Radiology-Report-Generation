
import os
import sys
import torch
import torch.nn as nn
from radgraph import F1RadGraph

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir))) # "/home/user/RRG/rrg"


scorer_rg = F1RadGraph(reward_level="all")

r = ["There is no new lung lesion. Unchanged position of right chest large for pigtail catheter and 2 pigtail catheters within the liver. Low lung volumes with bibasilar atelectasis. Small right pneumothorax is slightly decreased from prior. Probable small left pleural effusion is demonstrated with decreased left base atelectasis and decrease conspicuity of perihilar consolidation, possibly due to leftward rotation."]
h = ["The right chest large for pigtail catheter remains in an unchanged position, and there are two pigtail catheters within the liver. Low lung volumes are noted, accompanied by bibasilar atelectasis. The small right pneumothorax has slightly decreased compared to the previous assessment. A probable small left pleural effusion is evident, along with decreased left base atelectasis and reduced conspicuity of perihilar consolidation, possibly attributable to leftward rotation."]
# RG:  (0.8, 0.7058823529411765, 0.7200000000000001)


rg_v, _, ref, hyp = scorer_rg(h, r)

entities_h = hyp[0]['entities']
entities_r = ref[0]['entities']

my_ent = []
for pos, (k_v_h, k_v_r) in enumerate(zip(entities_h.items(), entities_r.items())):       
    print(pos, k_v_r[1]['tokens'], k_v_h[1]['tokens'])

print("RG: ", rg_v)

# RG:  (0.7213114754098361, 0.5846153846153846, 0.5633802816901409)
# añadiendo mas info que no aparece en ref si que puede bajar el RG
# RG:  (0.6875, 0.5507246376811594, 0.5135135135135135)
