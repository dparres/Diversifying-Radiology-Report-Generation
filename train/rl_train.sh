# conda activate hroc

python mytrain_rl.py \
    --exp_name exp1_stage2 \
    --model_arch SwinBERT9k \
    --load_weights exp1_stage1/last_model.pt \
    --scores_weights 0.01,0.495,0.495 \
    --scores BertScorer,F1RadGraph \
    --scores_args {},{\"reward_level\":\"partial\"} \
    --use_nll True \
    --top_k 0

