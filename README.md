# Diversifying-Radiology-Report-Generation

Official implementation of [*'Improving Radiology Report Generation Quality and Diversity through Reinforcement Learning and Text Augmentation'*](https://www.mdpi.com/2306-5354/11/4/351).

  ![training_workflow](https://github.com/dparres/Diversifying-Radiology-Report-Generation/assets/114649578/dafd3871-971c-430c-a1ad-3fcf99653d02)
  ![training_workflow](https://github.com/dparres/Diversifying-Radiology-Report-Generation/assets/114649578/8f6d2642-f57b-4381-a3ee-01cf81ac1f94)

## Installation

```
conda create -n rrg_env python=3.9
conda activate rrg_env

pip install vilmedic==1.3.2
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install albumentations
```
If there is any error related to the CV2 package:
```
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless 
pip install opencv-python==4.7.0.72
```
 [Download vocab.tgt](https://storage.googleapis.com/vilmedic_dataset/checkpoints/RRG/emnlp22_rl_findings_bertscore_128.zip) and update the file paths.py

 
## Training workflow
Navigate to the `train` directory and execute the following commands for each stage.

Stage 1 with Negative Log-Likelihood
```
python mytrain_nll.py \
    --exp_name exp1_stage1 \
    --model_arch SwinBERT9k \
    --hnm True
```

Stage 2 with Reinforcement Learning
```
python mytrain_rl.py \
    --exp_name exp1_stage2 \
    --model_arch SwinBERT9k \
    --load_weights exp1_stage1/last_model.pt \
    --hnm True \
    --scores_weights 0.01,0.495,0.495 \
    --scores BertScorer,F1RadGraph \
    --scores_args {},{\"reward_level\":\"partial\"} \
    --use_nll True \
    --top_k 0
```

## BibTeX
```
@Article{dparres2024RRG,
AUTHOR = {Parres, Daniel and Albiol, Alberto and Paredes, Roberto},
TITLE = {Improving Radiology Report Generation Quality and Diversity through Reinforcement Learning and Text Augmentation},
JOURNAL = {Bioengineering},
VOLUME = {11},
YEAR = {2024},
DOI = {10.3390/bioengineering11040351}
}
```

