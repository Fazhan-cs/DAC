import os
import clip_.clip as clip
import clip_
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from .. import datasets, templates
from .lrbmutil import clip_loss,compute_orthogonal_loss,load_lora_weights
from .loralib import mark_only_lora_as_trainable,get_lora_parameters
from .loralib import apply_lora_all,save_lora_all,load_lora_all,merge_lora_back_to_original_all
import torch
from torch.utils.data import DataLoader, Dataset
try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False


import torch
def alora(args):
    model, train_preprocess, val_preprocess = clip_.load(args.model, jit=False)

    loading=None
    if args.load:
        lora_paths_list = args.lora_paths.split(',')
        loading=load_lora_weights(lora_paths_list)

        index=0
        max_len=len(lora_paths_list)
        for lora_path in lora_paths_list:
            list_lora_layers = apply_lora_all(args, model)
            load_lora_all(args, list_lora_layers, lora_path)
            model = merge_lora_back_to_original_all(args, model, list_lora_layers,index,max_len)
            index += 1
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
    )
    if args.template is not None:
        template = getattr(templates, args.template)[0]
    else:
        template2 = dataset.template
    if args.few_shot > 0:
        print('=====few-shot======')
        few_shot_data = {} 

        for images, labels in dataset.train_loader:
            for image, label in zip(images, labels):
                label = label.item()
                if label not in few_shot_data:
                    few_shot_data[label] = []
                if len(few_shot_data[label]) < args.few_shot:
                    few_shot_data[label].append(image)
        few_shot_images = []
        few_shot_labels = []

        for label, images in few_shot_data.items():
            few_shot_images.extend(images)
            few_shot_labels.extend([label] * len(images))

        few_shot_images = torch.stack(few_shot_images)
        few_shot_labels = torch.tensor(few_shot_labels)
        print(f"Few-shot images shape: {few_shot_images.shape}, type: {few_shot_images.dtype}")
        print(f"Few-shot labels shape: {few_shot_labels.shape}, type: {few_shot_labels.dtype}")

        few_shot_dataset = torch.utils.data.TensorDataset(few_shot_images, few_shot_labels)
        few_shot_data_loader = DataLoader(few_shot_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)


    if args.few_shot > 0:
        num_batches = len(few_shot_data_loader)
    else:  # full_shot
        num_batches = len(dataset.train_loader)
    if args.epochs is not None: # # False
        total_iterations = args.epochs * num_batches
    else:
        total_iterations = args.iterations  # 1000
    print("Iterations per epoch:", num_batches)
    print("Total iterations:", total_iterations)

    print("[Training mode] lora")
    list_lora_layers = apply_lora_all(args, model)
        
    model = model.cuda()
    mark_only_lora_as_trainable(model)
    total_params_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)

    print('The number of Total Trainable Parameters------------------:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"Total Trainable Parameters Memory Size: {total_params_size / 1024 / 1024:.2f} MB")

    lora_params = get_lora_parameters(model, bias='none')
    gate_params = []
    for name, param in model.named_parameters():
        if 'lora_w' in name:
            gate_params.append(param)
    params = [
        {'params': lora_params, 'lr': args.lr, 'weight_decay': 1e-2, 'betas': (0.9, 0.999)},
    ]
    optimizer = torch.optim.AdamW(params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iterations, eta_min=1e-6)

    logit_scale = model.logit_scale
    devices = list(range(torch.cuda.device_count()))
    print("Using devices", devices)
    model = torch.nn.DataParallel(model, device_ids=devices) 
    save_path = os.path.join(args.save, 'rank')
    os.makedirs(save_path, exist_ok=True)
    text_cls = [template2(x) for x in dataset.classnames]
    for iteration in tqdm(range(total_iterations + 1)):
        if iteration % num_batches == 0:
            if args.few_shot>0:  # default is -1
                data_iter = iter(few_shot_data_loader)
            else:
                data_iter = iter(dataset.train_loader)

        model.train()

        try:
            images, labels = next(data_iter)
        except:
            data_iter = iter(dataset.train_loader)
            images, labels = next(data_iter)
        images, labels = images.cuda(), labels.cuda()
        classnames = dataset.classnames





        texts = [template2(classnames[label]) for label in labels]
        texts = clip.tokenize(texts).cuda()
        text_embeds  = model(None, texts)
        image_embeds  = model(images, None)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale.exp()


        loss = clip_loss(logits_per_text,labels)


        
        o_loss = 0
        if loading:
            o_loss=compute_orthogonal_loss(model,loading)
            
        loss=loss+o_loss*0.1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if iteration % 20 == 0:
                print("Loss:", loss.item()) 
                print("oLoss:", o_loss*0.1) 

  
    # Saving model
    if args.save is not None:
        save_lora_all(args, list_lora_layers)
        
