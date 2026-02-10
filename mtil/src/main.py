import copy
import os
import numpy as np
import clip_
import torch
import csv
import pandas as pd
from . import utils
from .args import parse_arguments
from .models import evaluate

from .models import apply_lora_all,load_lora_all,merge_lora_back_to_original_all
from .models import alora
import torchvision.models as models
import torch.nn as nn
import math
def main(args):
    utils.seed_all(args.seed)

    assert args.train_mode in ["whole", "text", "image", "adapter","lora"]

    if args.eval_only:  
        model, train_preprocess, val_preprocess = clip_.load(args.model, jit=False)

        if args.lora_paths != '':
            lora_paths_list = args.lora_paths.split(',')
            index=0
            max_len=len(lora_paths_list)
            for lora_path in lora_paths_list:
                list_lora_layers = apply_lora_all(args, model)
                load_lora_all(args, list_lora_layers, lora_path)
                model = merge_lora_back_to_original_all(args, model, list_lora_layers,index,max_len)
                index+=1
        model = model.cuda()
        

        evaluate(model, args, val_preprocess)

    else:
        
        print('----------------------finetune model----------------------')
        model=alora(args)
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
