IMAGES_MIMIC_PATH = "/home/Data/NEW/mimic-cxr/2.0.0/files_jpg_512/files"

DICT_CSV_MIMIC_PATH = {
    "train": "/home/Data/NEW/mimic-cxr/2.0.0/RRG/mimic-cxr/findings/train.metadata.csv",
    "validation": "/home/Data/NEW/mimic-cxr/2.0.0/RRG/mimic-cxr/findings/validate.metadata.csv",
    "test": "/home/Data/NEW/mimic-cxr/2.0.0/RRG/mimic-cxr/findings/test.metadata.csv",
    "train_llama3": "/home/Data/llama3_data/patients/"
}

VOCAB_PATH = "/home/Data/vocab/emnlp22_rl_findings_bertscore_128/vocab.tgt"

SWINB_IMAGENET22K_WEIGHTS = "microsoft/swin-base-patch4-window12-384-in22k"
SWINS_IMAGENET22K_WEIGHTS = "microsoft/swin-small-patch4-window7-224"

PATH_IDS_NO_RG_TRAIN = "/home/dparres/myRRG/lightRRG/mydatasets/TRAIN_SAMPLES_NO_rg.txt"
PATH_IDS_NO_RG_TEST = "/home/dparres/myRRG/lightRRG/mydatasets/TEST_SAMPLES_NO_rg.txt"


PATH_OPENI_REPORTS = "/home/Data/NEW/mimic-cxr/openi/ecgen-radiology/"
PATH_OPENI_IMGS = '/home/Data/NEW/mimic-cxr/openi/imgs/'

CHEXPERT_TRAIN_CSV = "/home/dparres/mychexpert/train.csv"
CHEXPERT_VALID_CSV = "/home/dparres/mychexpert/valid.csv"
CHEXPERT_IMAGES = "/home/dparres/mychexpert/"


DICT_MIMIC_OBS_TO_INT = {
    "enlarged cardiomediastinum": 0,      
    "cardiomegaly": 1,  
    "lung opacity": 2,       
    "lung lesion": 3,     
    "edema": 4,           
    "consolidation": 5,          
    "pneumonia": 6,          
    "atelectasis": 7,        
    "pneumothorax": 8,      
    "pleural effusion": 9,       
    "pleural other": 10,   
    "fracture": 11,      
    "support devices": 12,          
    "no finding": 13
}

DICT_MIMIC_INT_TO_OBS = {
    0: "enlarged cardiomediastinum",      
    1: "cardiomegaly",  
    2: "lung opacity",       
    3: "lung lesion",     
    4: "edema",           
    5: "consolidation",          
    6: "pneumonia",          
    7: "atelectasis",        
    8: "pneumothorax",      
    9: "pleural effusion",       
    10: "pleural other",   
    11: "fracture",      
    12: "support devices",          
    13: "no finding"
}

DICT_CHEXPERT_INT_TO_OBS = {
    0: "no finding",
    1: "enlarged cardiomediastinum",
    2: "cardiomegaly",
    3: "lung opacity",
    4: "lung lesion",
    5: "edema",
    6: "consolidation",
    7: "pneumonia",
    8: "atelectasis",
    9: "pneumothorax",
    10: "pleural effusion",
    11: "pleural other",
    12: "fracture",
    13: "support devices"
}