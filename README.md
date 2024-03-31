# Diversifying-Radiology-Report-Generation

Official implementation of *'Improving Radiology Report Generation Quality and Diversity through Reinforcement Learning and Text Augmentation'*.

  ![training_workflow](https://github.com/dparres/Diversifying-Radiology-Report-Generation/assets/114649578/dafd3871-971c-430c-a1ad-3fcf99653d02)
  ![training_workflow](https://github.com/dparres/Diversifying-Radiology-Report-Generation/assets/114649578/8f6d2642-f57b-4381-a3ee-01cf81ac1f94)

## Installation

```
conda create env -n rrg_env python=3.8
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

 
