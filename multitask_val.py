#!/usr/bin/env python

import argparse
import pathlib
import time, yacs

import numpy as np
import torch
from typing import Tuple, Union
from pytorch_image_classification import create_transform
from pytorch_image_classification import create_collator
import torch.nn.functional as F
import tqdm
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from fvcore.common.checkpoint import Checkpointer
from torch.utils.data import Dataset, DataLoader

from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_loss,
    create_model,
    get_default_config,
    update_config,
)
from pytorch_image_classification.utils import (
    AverageMeter,
    create_logger,
    get_rank,
)
def getLabelmap(label_list):
    label_map={}
    for i in label_list:
        if i not in label_map.keys():
            label_map[i]=len(label_map)
    print(label_map)
    return label_map

class LabelData(Dataset):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame,configs,istrain=False, transforms=None):
        if istrain:
          self.files = [configs.dataset.dataset_dir +"/"+ file for file in test_df["filename"].values]
        else:
          self.files = [configs.dataset.dataset_dir +"/"+ file for file in train_df["filename"].values]
        self.y1 = train_df["artist"].values.tolist()
        self.label_map1=getLabelmap(self.y1)
        self.y2 = train_df["style"].values.tolist()
        self.label_map2=getLabelmap(self.y2)
        self.y3 = train_df["genre"].values.tolist()
        self.label_map3=getLabelmap(self.y3)
        if not istrain:
          self.y1 = test_df["artist"].values.tolist()
          self.y2 = test_df["style"].values.tolist()
          self.y3 = test_df["genre"].values.tolist()
        self.transforms = transforms
        
    def __len__(self):
        return len(self.y1)
    
    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert('RGB')
        label1 = self.label_map1[self.y1[i]]
        label2 = self.label_map2[self.y2[i]]
        label3 = self.label_map3[self.y3[i]]
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, [label1,label2,label3]

def create_dataloader(config: yacs.config.CfgNode,is_train: bool) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    if is_train:
        df = pd.read_csv(config.dataset.cvsfile_train)
        if config.dataset.subname=="K100":
          train_df, valid_df = train_test_split(df, stratify=df["artist"].values)
        else:
          train_df, valid_df = train_test_split(df, stratify=df["label"].values)
        train_dataset = LabelData(train_df,config,create_transform(config, is_train=True))
        val_dataset = LabelData(valid_df,config,create_transform(config, is_train=False))

        if dist.is_available() and dist.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset)
        else:
            train_sampler = torch.utils.data.sampler.RandomSampler(
                train_dataset, replacement=False)
            val_sampler = torch.utils.data.sampler.SequentialSampler(
                val_dataset)

        train_collator = create_collator(config)

        # train_batch_sampler = torch.utils.data.sampler.BatchSampler(
        #     train_sampler,
        #     batch_size=config.train.batch_size,
        #     drop_last=config.train.dataloader.drop_last)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            # batch_sampler=train_batch_sampler,
            num_workers=config.train.dataloader.num_workers,
            collate_fn=train_collator,
            pin_memory=config.train.dataloader.pin_memory,
            worker_init_fn=worker_init_fn)

        # val_batch_sampler = torch.utils.data.sampler.BatchSampler(
        #     val_sampler,
        #     batch_size=config.validation.batch_size,
        #     drop_last=config.validation.dataloader.drop_last)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            # batch_sampler=val_batch_sampler,
            num_workers=config.validation.dataloader.num_workers,
            pin_memory=config.validation.dataloader.pin_memory,
            worker_init_fn=worker_init_fn)
        return train_loader, val_loader
    else:
        train_df = pd.read_csv(config.dataset.cvsfile_train)
        test_df = pd.read_csv(config.dataset.cvsfile_test)
        dataset = LabelData(train_df,test_df,config,False,create_transform(config, is_train=False))
        test_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.test.batch_size,
                num_workers=config.test.dataloader.num_workers,
                # sampler=sampler,
                shuffle=False,
                drop_last=False,
                pin_memory=config.test.dataloader.pin_memory)
                
        return test_loader

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    update_config(config)
    config.freeze()
    return config


def evaluate(config, model, test_loader, loss_func, logger):
    device = torch.device(config.device)

    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()

    pred_raw_all = []
    pred_prob_all = []
    pred_label_all = []
    with torch.no_grad():
        for data, targets in tqdm.tqdm(test_loader):
            data = data.to(device)
            targets = [ tar.to(device) for tar in targets ]

            outputs = model(data)
            loss = loss_func(outputs, targets)

            pred_raw_all.append(outputs[0].cpu().numpy())
            pred_prob_all.append(F.softmax(outputs[0], dim=1).cpu().numpy())

            _, preds = torch.max(outputs[0], dim=1)
            pred_label_all.append(preds.cpu().numpy())

            loss_ = loss.item()
            correct_ = preds.eq(targets[0]).sum().item()
            num = data.size(0)

            loss_meter.update(loss_, num)
            correct_meter.update(correct_, 1)

        accuracy = correct_meter.sum / len(test_loader.dataset)

        elapsed = time.time() - start
        logger.info(f'Elapsed {elapsed:.2f}')
        logger.info(f'Loss {loss_meter.avg:.4f} Accuracy {accuracy:.4f}')

    preds = np.concatenate(pred_raw_all)
    probs = np.concatenate(pred_prob_all)
    labels = np.concatenate(pred_label_all)
    return preds, probs, labels, loss_meter.avg, accuracy


def main():
    config = load_config()

    if config.test.output_dir is None:
        output_dir = pathlib.Path(config.test.checkpoint).parent
    else:
        output_dir = pathlib.Path(config.test.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    logger = create_logger(name=__name__, distributed_rank=get_rank())

    model = create_model(config)
    model = apply_data_parallel_wrapper(config, model)
    checkpointer = Checkpointer(model)
    checkpointer.load(config.test.checkpoint)

    test_loader = create_dataloader(config, is_train=False)
    _, test_loss = create_loss(config)

    preds, probs, labels, loss, acc = evaluate(config, model, test_loader,
                                               test_loss, logger)

    output_path = output_dir / f'predictions.npz'
    np.savez(output_path,
             preds=preds,
             probs=probs,
             labels=labels,
             loss=loss,
             acc=acc)


if __name__ == '__main__':
    main()
