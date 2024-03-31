import os
import sys
import json
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
import multiprocessing
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir)))

from myrl.scst import SCST
from myscorers.bleu.bleu import Bleu
from myscorers.rouge.rouge import Rouge
from mymodels.swinbert9k import SwinBERT9k
from mydatasets.mimic_dataset import mimic_Dataset
from myscorers.chexbert.chexbert import myF1ChexBert
from myscorers.bertscore.bertscore import BertScorer
from myscorers.myradgraph.myradgraph import myRadGraph
from train.train_utils import multiassign, Hard_Negative_Mining

torch.set_float32_matmul_precision('medium')

####################################################################
# Load Arguments
####################################################################

parser = argparse.ArgumentParser(description='Train NLL for RRG.')

parser.add_argument('--exp_name', type=str, help='Experiment name.')
parser.add_argument('--model_arch', type=str, help='Architecture to train')
parser.add_argument('--load_weights', type=str, default=None, help='Load weights.')
parser.add_argument('--hnm', type=bool, default=False, help='Use Hard Negative Mining.')

# RL args
parser.add_argument('--scores', type=str, default=None, help='Load weights.')
parser.add_argument('--scores_args', type=str, default=None, help='Load weights.')
parser.add_argument('--scores_weights', type=str, default=None, help='Load weights.')
parser.add_argument('--use_nll', type=bool, default=True, help='Use NLL in SCST.')
parser.add_argument('--top_k', type=int, default=0, help='top_k value.')

args = parser.parse_args()

# Prepare RL args
scores = args.scores.split(",")

scores_weights = args.scores_weights.split(',')
for i in range(len(scores_weights)):
    scores_weights[i] = float(scores_weights[i])

scores_args = args.scores_args.split(",")
for i in range(len(scores_args)):
    scores_args[i] = json.loads(scores_args[i])

print(35*'*')
print('exp_name:', args.exp_name)
print('model_arch:', args.model_arch)
if args.load_weights != None:
    print("load_weights: ", args.load_weights)
    args.load_weights = "../EXPERIMENTS/" + args.load_weights
print('hnm:', args.hnm)
print('scores:', scores)
print('scores_args:', scores_args)
print('scores_weights:', scores_weights)
print('use_nll:', args.use_nll)
print('top_k:', args.top_k)
print(35*'*')

EXP_DIR_PATH = "../EXPERIMENTS/" + args.exp_name
if not os.path.exists(EXP_DIR_PATH):
    os.makedirs(EXP_DIR_PATH)

####################################################################
# Load Scorers
####################################################################

bleu_scorer = Bleu(n=4)
rougel_scorer = Rouge(rouges=['rougeL'])
f1cxb_scorer = myF1ChexBert()
radgraph_scorer = myRadGraph(reward_level='partial')
bert_scorer = BertScorer()

####################################################################
# Load Model
####################################################################

DICT_MODELS = {
    "SwinBERT9k": SwinBERT9k(),
}
device = 'cuda:0'
model = DICT_MODELS[args.model_arch]

if args.load_weights != None:
    model.load_state_dict(torch.load(args.load_weights))
    print("Model initialized with weights: ", args.load_weights, "!")

# RL
scst = SCST(
    bos_token_id=model.bos_token_id, 
    eos_token_id=model.eos_token_id,
    pad_token_id=model.pad_token_id,
    scores=scores,
    scores_args=scores_args,
    scores_weights=scores_weights,
    use_nll=args.use_nll,
    top_k=args.top_k)

####################################################################
# Dataset Class
####################################################################

val_dataset = mimic_Dataset(
                transform=model.val_transform, 
                tokenizer=model.tokenizer,
                processor=model.processor,
                partition = "validation"
                )

train_dataset = mimic_Dataset(
                transform=model.train_transform, 
                tokenizer=model.tokenizer,
                processor=model.processor,
                partition = "train"
                )

####################################################################
# DataLoader Class
####################################################################

batch_size = 2
accumulate_grad_batches = 12
num_workers = multiprocessing.cpu_count()-1
print("Num workers", num_workers)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size, 
    shuffle=True, 
    num_workers=num_workers,
    collate_fn=train_dataset.get_collate_fn())

val_dataloader = DataLoader(
    val_dataset, 
    1, 
    shuffle=False, 
    num_workers=num_workers,
    collate_fn=val_dataset.get_collate_fn())

####################################################################
# Training settings
####################################################################

# Training hyperparameters
epochs=50
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8) 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Params: ", count_parameters(model))

####################################################################
# Training
####################################################################

# Load model in GPU
model.to(device)

best_f1cxb = -9999999.9
best_rg = -9999999.9
epoch_best_f1cxb = 0
epoch_best_rg = 0
print("\n---- Start Training ----")
for epoch in range(epochs):
    
    #epoch = 1
    # Train
    train_loss = 0
    test_loss = 0
    if args.hnm:
        train_hnm_loss = 0
        dict_loss = {}
    
    if epoch != 0: # Do Test first
        model.train()
        optimizer.zero_grad()
        with tqdm(iter(train_dataloader), desc="Epoch " + str(epoch), unit="batch") as tepoch:
            for steps, batch in enumerate(tepoch):
                
                pixel_values = batch['images'].to(device)
                inputs_id = batch['input_ids'].to(device)
                attention_mask  = batch['attention_mask'].to(device)
                images_mask  = batch['images_mask'].to(device)
                ids = batch["idx"].to('cpu').numpy()

                # 1 Greedy
                with torch.no_grad():
                    model.eval()
                    out = scst.forward_greedy(
                            input_ids=inputs_id,
                            images=pixel_values, #.cuda(),
                            model=model, 
                            images_mask=images_mask
                    )
                    # out contains: 
                    # reward_greedy, greedy_hyp_list, ref_list
                    
                # 2. Sampling
                model = model.train()
                out = scst.forward_sampling(
                        input_ids=inputs_id,
                        attention_mask=attention_mask,
                        reward_greedy=out["reward_greedy"],
                        images=pixel_values, #.cuda(),
                        model=model, 
                        images_mask=images_mask
                )
                # out contains: 
                # loss, delta_reward, delta_reward_per_metric, reward_sampling, sampling_hyp_list, nll_loss 
                
                loss = out["loss"] # Reward Loss
                nll_loss = out["nll_loss"] # NLL Loss

                # Calculate gradients
                loss.backward()

                if steps % accumulate_grad_batches == 0 and steps != 0:

                    # Update parameters
                    optimizer.step()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                if args.hnm:
                    multiassign(dict_loss, ids, 
                                    [nll_loss.to('cpu').detach().numpy()])

                train_loss += nll_loss.item()
                tepoch.set_description(f'Train Epoch [{epoch}/{epochs-1}] Loss: {nll_loss.item():.4f}')
            
            optimizer.zero_grad()

        # HNM
        if args.hnm:
            HNM_trainloader = Hard_Negative_Mining(
                dict_loss, train_dataset, batch_size, num_workers=num_workers)
            
            with tqdm(iter(HNM_trainloader), desc="Epoch " + str(epoch), unit="batch") as tepoch:
                for steps, batch in enumerate(tepoch):
                    
                    pixel_values = batch['images'].to(device)
                    inputs_id = batch['input_ids'].to(device)
                    attention_mask  = batch['attention_mask'].to(device)
                    images_mask  = batch['images_mask'].to(device)

                    # 1 Greedy
                    with torch.no_grad():
                        model.eval()
                        out = scst.forward_greedy(
                            input_ids=inputs_id,
                            images=pixel_values, #.cuda(),
                            model=model, 
                            images_mask=images_mask
                        )
                        # out contains: 
                        # reward_greedy, greedy_hyp_list, ref_list
                    
                    # 2. Sampling
                    model = model.train()
                    out = scst.forward_sampling(
                            input_ids=inputs_id,
                            attention_mask=attention_mask,
                            reward_greedy=out["reward_greedy"],
                            images=pixel_values, #.cuda(),
                            model=model, 
                            images_mask=images_mask
                    )
                    # out contains: 
                    # loss, delta_reward, delta_reward_per_metric, reward_sampling, sampling_hyp_list, nll_loss 
                    
                    loss = out["loss"]
                    nll_loss = out["nll_loss"]

                    # Calculate gradients
                    loss.backward()

                    if steps % accumulate_grad_batches == 0 and steps != 0:

                        # Update parameters
                        optimizer.step()

                        # zero the parameter gradients
                        optimizer.zero_grad()

                    train_hnm_loss += nll_loss.item()
                    tepoch.set_description(f'HNM Epoch [{epoch}/{epochs-1}] Loss: {nll_loss.item():.4f}')

                optimizer.zero_grad()
                
        lr_scheduler.step()

    # Test
    l_refs = []
    l_hyps = []
    model.eval()
    with torch.no_grad():
        with tqdm(iter(val_dataloader), desc="Epoch " + str(epoch), unit="batch") as tepoch:
            for steps, batch in enumerate(tepoch):
                
                pixel_values = batch['images'].to(device)
                inputs_id = batch['input_ids'].to(device)
                attention_mask  = batch['attention_mask'].to(device)
                images_mask  = batch['images_mask'].to(device)

                decoder_out = model(inputs_id, attention_mask, pixel_values, images_mask=images_mask)          
                loss = decoder_out['loss']

                generated_reports, _ = model.generate(
                    pixel_values, images_mask=images_mask,
                    tokenizer=model.tokenizer,
                    num_beams=2,
                    max_len=128,
                    return_dict_in_generate=True,
                    output_scores=True)

                reference_reports = batch['text']

                for r, h in zip(reference_reports, generated_reports):
                    l_refs.append(r)
                    l_hyps.append(h)

                test_loss += loss.item()
                tepoch.set_description(f'Test Epoch [{epoch}/{epochs-1}] Loss: {loss.item():.4f}')
    
    # Calculate metrics
    calculated_bleu = bleu_scorer(l_refs, l_hyps)[0]
    calculated_rougel = rougel_scorer(refs=l_refs, hyps=l_hyps)[0]
    calculated_f1cxb = f1cxb_scorer.calculate(l_refs, l_hyps)
    calculated_rg = radgraph_scorer(l_refs, l_hyps)
    calculated_bertscore = bert_scorer(l_hyps, l_refs)

    if best_f1cxb < calculated_f1cxb:
        try:
            os.remove(EXP_DIR_PATH + "/best_cxb_" + str(epoch_best_f1cxb) + "_model.pt")
        except OSError as e:
            print("Nothing is deleted")
        best_f1cxb = calculated_f1cxb
        epoch_best_f1cxb = epoch
        torch.save(model.state_dict(), EXP_DIR_PATH + "/best_cxb_" + str(epoch) + "_model.pt")
    if best_rg < calculated_rg:
        try:
            os.remove(EXP_DIR_PATH + "/best_rg_" + str(epoch_best_rg) + "_model.pt")
        except OSError as e:
            print("Nothing is deleted")
        best_rg = calculated_rg
        epoch_best_rg = epoch
        torch.save(model.state_dict(), EXP_DIR_PATH + "/best_rg_" + str(epoch) + "_model.pt")
    
    train_loss /= (len(train_dataloader.dataset) / batch_size)
    test_loss /= len(val_dataloader.dataset)
    print("\tTrain Loss: ", train_loss)
    print("\tTest Loss: ", test_loss)

    # Open the file in write mode ('w')
    with open(EXP_DIR_PATH + "/log.txt", 'a') as file:
        # Write the string to the file
        file.write("EPOCH: " + str(epoch) + "\n")
        file.write("\tBLEU4: \t\t" + str(calculated_bleu) + "\n")
        file.write("\tRougeL: \t\t" + str(calculated_rougel) + "\n")
        file.write("\tF1cXb: \t\t" + str(calculated_f1cxb) + "\n")
        file.write("\tRG: \t\t" + str(calculated_rg) + "\n")
        file.write("\tBERTscore: \t\t" + str(calculated_bertscore) + "\n")
        file.write("\tTrLoss: \t\t" + str(train_loss) + "\n")
        file.write("\tTsLoss: \t\t" + str(test_loss) + "\n")
        if args.hnm and epoch != 0:
            train_hnm_loss /= len(HNM_trainloader.dataset)
            file.write("\tHNMLoss: \t" + str(train_hnm_loss) + "\n")
            print("t\HNM Loss: ", train_hnm_loss)
        file.write("------------------------------\n")

# Save Final weights
torch.save(model.state_dict(), EXP_DIR_PATH + "/last_model.pt")
