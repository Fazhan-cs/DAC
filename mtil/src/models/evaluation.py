import clip_.clip as clip
import torch
from tqdm import tqdm

from .. import datasets
from ..datasets.common import get_dataloader, maybe_dictionarize
import torch.nn.functional as F
def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


@torch.no_grad()
def zeroshot_classifier(classnames, templates, model, args):
    if not isinstance(templates, list):
        templates = [templates]
    with torch.no_grad():

        zeroshot_weights_i = []
        for classname in classnames:
            texts = [template(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            if args.non_text == True:
                class_embeddings = model.encode_text(texts, -1)  # embed with text encoder
            else:
                class_embeddings = model(None,texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights_i.append(class_embedding)
        zeroshot_weights_i = torch.stack(zeroshot_weights_i, dim=1).cuda()
    return zeroshot_weights_i


@torch.no_grad()
def zeroshot_eval(model, loader, zeroshot_weights, args):
    top1, top5, n = 0.0, 0.0, 0.0
    for i, data in enumerate(tqdm(loader)):

        data = maybe_dictionarize(data)
        images = data["images"].cuda()
        target = data["labels"].cuda()

    
        image_features = model(images,None)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ zeroshot_weights
       
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return top1, top5


def eval_single_dataset(image_classifier, dataset, args):
    model = image_classifier
    input_key = "images"
    image_enc = None

    model.eval()
    #--------------------------------------------------------------------------------------------------------------------------------------
    zeroshot_weights = zeroshot_classifier(
        dataset.classnames, dataset.templates, model,args
    )

    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc
    )
    top1, top5 = zeroshot_eval(model, dataloader, zeroshot_weights, args)
    top1_rounded = round(top1, 2)
    print(f"Top-1 accuracy: {top1:.1f}")
    return top1_rounded

import pandas as pd
def evaluate(image_classifier, args, val_preprocess):
    if args.eval_datasets is None:
        return
    top1_list = [] 
    for i, dataset_name in enumerate(args.eval_datasets):
        print("Evaluating on", dataset_name)  
        print(args.batch_size)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )
        top1=eval_single_dataset(image_classifier, dataset, args)
        top1_list.append(top1)
    # dataset_name='ImageNetSUB'
    # print("Evaluating on", dataset_name)  # Caltech101
    # print(args.batch_size) 
    # dataset_class = getattr(datasets, dataset_name)
    # dataset = dataset_class(
    #     val_preprocess,
    #     location=args.data_location,
    #     batch_size=args.batch_size,
    #     batch_size_eval=args.batch_size_eval,
    # )
    # top1=eval_single_dataset(image_classifier,feature_extractor,Autoencoder_list, dataset, args)
    # top1_list.append(top1)
    df = pd.DataFrame([top1_list])
    df.to_csv(args.resfile, mode='a', index=False, header=False)
    print("Top1 results saved to top1_results.csv")
