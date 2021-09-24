from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_loss,
    create_model,create_transform,
    get_files,
    create_optimizer,
    create_scheduler,
    get_default_config,
    update_config,
    worker_init_fn,
)
from pytorch_image_classification.config.config_node import ConfigNode
from pytorch_image_classification.utils import (
    AverageMeter,
    DummyWriter,
    compute_accuracy,
    count_op,
    create_logger,
    create_tensorboard_writer,
    find_config_diff,
    get_env_info,
    get_rank,
    save_config,
    set_seed,
    setup_cudnn,
)
import apex
from pytorch_image_classification import create_transform
from pytorch_image_classification import create_collator
from PIL import Image
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from fvcore.common.checkpoint import Checkpointer
import torch.distributed as dist
import pandas as pd

import argparse
import pathlib
import time
import torch

from typing import Tuple, Union
import torch, yacs
import numpy as np

global_step = 0

def getLabelmap(label_list):
    label_map={}
    for i in label_list:
        if i not in label_map.keys():
            label_map[i]=len(label_map)
    print(label_map)
    return label_map

class LabelData(Dataset):
    def __init__(self, df: pd.DataFrame,configs, transforms=None):
        self.files = [configs.dataset.dataset_dir +"/"+ file for file in df["file"].values]
        self.y1 = df["artist"].values.tolist()
        self.label_map1=getLabelmap(self.y1)
        self.y2 = df["style"].values.tolist()
        self.label_map2=getLabelmap(self.y2)
        self.y3 = df["genre"].values.tolist()
        self.label_map3=getLabelmap(self.y3)
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
          train_df, valid_df = train_test_split(df, stratify=df["artist"].values)
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

        train_batch_sampler = torch.utils.data.sampler.BatchSampler(
            train_sampler,
            batch_size=config.train.batch_size,
            drop_last=config.train.dataloader.drop_last)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=config.train.dataloader.num_workers,
            collate_fn=train_collator,
            pin_memory=config.train.dataloader.pin_memory,
            worker_init_fn=worker_init_fn)

        val_batch_sampler = torch.utils.data.sampler.BatchSampler(
            val_sampler,
            batch_size=config.validation.batch_size,
            drop_last=config.validation.dataloader.drop_last)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=config.validation.dataloader.num_workers,
            pin_memory=config.validation.dataloader.pin_memory,
            worker_init_fn=worker_init_fn)
        return train_loader, val_loader
    else:
        test_df = pd.read_csv(config.dataset.cvsfile_test)
        dataset = LabelData(test_df,config,create_transform(config, is_train=False))
        test_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.test.batch_size,
                num_workers=config.test.dataloader.num_workers,
                # sampler=sampler,
                shuffle=False,
                drop_last=False,
                pin_memory=config.test.dataloader.pin_memory)
                
        return test_loader

def compute_multi_accuracy(outputs, targets):
    batch_size = targets[0].size(0)
    with torch.no_grad():
        acc = []
        for out,target in zip(outputs,targets):
            _, preds = torch.max(out, dim=1)
            correct_ = preds.eq(target).sum().item()
            acc.append(correct_ *(1/batch_size*1.0))
        return acc
    

def train(epoch, config, model, optimizer, scheduler, loss_func, train_loader,logger,tensorboard_writer,tensorboard_writer2):

    global global_step
    logger.info(f'Train {epoch}/{global_step}')#
    device = torch.device(config.device)

    model.train()

    loss_meter = AverageMeter()
    correct_meter1 = AverageMeter()
    correct_meter2 = AverageMeter()
    correct_meter3 = AverageMeter()
    start = time.time()
    losses = []
    for step, (data, targets) in enumerate(train_loader):
            step += 1
            global_step += 1

            if get_rank() == 0 and step == 1:
                if config.tensorboard.train_images:
                    image = torchvision.utils.make_grid(data,
                                                        normalize=True,
                                                        scale_each=True)
                    tensorboard_writer.add_image('Train/Image', image, epoch)
            
            data = data.to(device)
            targets = [ tar.to(device) for tar in targets ]

            optimizer.zero_grad()

            outputs = model(data)
            loss_ = loss_func(outputs, targets)

            
            correct_ = compute_multi_accuracy(outputs,targets)
            acc1,acc2,acc3 = correct_;acc1,acc2,acc3 = torch.Tensor([acc1]),torch.Tensor([acc2]),torch.Tensor([acc3])
            if config.train.distributed:
                loss_all_reduce = dist.all_reduce(loss_,
                                                op=dist.ReduceOp.SUM,
                                                async_op=True)
                acc1_all_reduce = dist.all_reduce(acc1,
                                                op=dist.ReduceOp.SUM,
                                                async_op=True)
                acc2_all_reduce = dist.all_reduce(acc2,
                                                op=dist.ReduceOp.SUM,
                                                async_op=True)
                acc3_all_reduce = dist.all_reduce(acc3,
                                                op=dist.ReduceOp.SUM,
                                                async_op=True)
                loss_all_reduce.wait()
                acc1_all_reduce.wait()
                acc2_all_reduce.wait()
                acc3_all_reduce.wait()
                loss_.div_(dist.get_world_size())
                acc1.div_(dist.get_world_size())
                acc2.div_(dist.get_world_size())
                acc3.div_(dist.get_world_size())

            correct_ = torch.cat(acc1,acc2,acc3)# correct_ = [acc1,acc2,acc3]
            num = data.size(0)
            # loss = sum(loss_)
            loss_.backward()
            optimizer.step()
            # print("loss： ",loss)
            # print("loss： ",loss.item())
            loss_meter.update(loss_.item(), num)
            correct_meter1.update(correct_[0], num)
            correct_meter2.update(correct_[1], num)
            correct_meter3.update(correct_[2], num)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            if get_rank() == 0:
                if step % config.train.log_period == 0 or step == len(
                        train_loader):
                    logger.info(f'Epoch {epoch} '
                                f'Step {step}/{len(train_loader)} '
                                f'lr {scheduler.get_last_lr()[0]:.6f} '
                                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                                f'acc1@ {correct_meter1.val:.4f} ({correct_meter1.avg:.4f}) '
                                f'acc2@ {correct_meter2.val:.4f} ({correct_meter2.avg:.4f}) '
                                f'acc3@ {correct_meter3.val:.4f} ({correct_meter3.avg:.4f}) ')
                tensorboard_writer2.add_scalar('Train/RunningLoss',
                                               loss_meter.avg, global_step)
                tensorboard_writer2.add_scalar('Train/RunningAcc1',
                                               correct_meter1.avg, global_step)
                tensorboard_writer2.add_scalar('Train/RunningAcc2',
                                               correct_meter2.avg, global_step)
                tensorboard_writer2.add_scalar('Train/RunningAcc3',
                                               correct_meter3.avg, global_step)
                tensorboard_writer2.add_scalar('Train/RunningLearningRate',
                                               scheduler.get_last_lr()[0],
                                               global_step)

            scheduler.step()

    # accuracy = correct_meter1.sum / len(train_loader.dataset)
    # accuracy2 = correct_meter1.sum / len(train_loader.dataset)
    # accuracy3 = correct_meter1.sum / len(train_loader.dataset)
    if get_rank() == 0:
        elapsed = time.time() - start
        logger.info(f'Elapsed {elapsed:.2f}')
        tensorboard_writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        tensorboard_writer.add_scalar('Train/Acc1', correct_meter1.avg, epoch)
        tensorboard_writer.add_scalar('Train/Acc2', correct_meter2.avg, epoch)
        tensorboard_writer.add_scalar('Train/Acc3', correct_meter3.avg, epoch)
        tensorboard_writer.add_scalar('Train/Time', elapsed, epoch)
        tensorboard_writer.add_scalar('Train/LearningRate',
                                        scheduler.get_last_lr()[0], epoch)

def validate(epoch, config, model, loss_func, val_loader, logger,
             tensorboard_writer):
    logger.info(f'Val {epoch}')

    device = torch.device(config.device)

    model.eval()    

    loss_meter = AverageMeter()
    correct_meter1 = AverageMeter()
    correct_meter2 = AverageMeter()
    correct_meter3 = AverageMeter()

    start = time.time()
    with torch.no_grad():
        for step, (data, targets) in enumerate(val_loader):
            if get_rank() == 0:
                if config.tensorboard.val_images:
                    if epoch == 0 and step == 0:
                        image = torchvision.utils.make_grid(data,
                                                            normalize=True,
                                                            scale_each=True)
                        tensorboard_writer.add_image('Val/Image', image, epoch)

            data = data.to(
                    device, non_blocking=config.validation.dataloader.non_blocking)
            targets = [ tar.to(device) for tar in targets ]

            outputs = model(data)
            loss = loss_func(outputs, targets)

            acc1 = compute_multi_accuracy(outputs,targets)
            # acc,acc2,acc3 = acc1;acc,acc2,acc3 = torch.Tensor([acc1]),torch.Tensor([acc2]),torch.Tensor([acc3])

            if config.train.distributed:
                    loss_all_reduce = dist.all_reduce(loss,
                                                    op=dist.ReduceOp.SUM,
                                                    async_op=True)
                    acc1_all_reduce = dist.all_reduce(acc,
                                                    op=dist.ReduceOp.SUM,
                                                    async_op=True)
                    acc2_all_reduce = dist.all_reduce(acc2,
                                                    op=dist.ReduceOp.SUM,
                                                    async_op=True)
                    acc3_all_reduce = dist.all_reduce(acc3,
                                                    op=dist.ReduceOp.SUM,
                                                    async_op=True)
                    loss_all_reduce.wait()
                    acc1_all_reduce.wait()
                    acc2_all_reduce.wait()
                    acc3_all_reduce.wait()

                    loss.div_(dist.get_world_size())
                    acc1.div_(dist.get_world_size())
                    acc2.div_(dist.get_world_size())
                    acc3.div_(dist.get_world_size())

            loss = loss.item()
            # acc1=torch.cat((acc,acc2,acc3),0)#acc1 = [acc,acc2,acc3]

            num = data.size(0)
            loss_meter.update(loss, num)
            correct_meter1.update(acc1[0], num)
            correct_meter2.update(acc1[1], num)
            correct_meter3.update(acc1[2], num)

            if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
    logger.info(f'Epoch {epoch} '
                                    f'loss {loss_meter.avg:.4f} '
                                    f'acc1@ {correct_meter1.avg:.4f} '
                                    f'acc2@ {correct_meter2.avg:.4f}'
                                    f'acc3@ {correct_meter3.avg:.4f}')

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')
        

    if get_rank() == 0:
        if epoch > 0:
            tensorboard_writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/Acc1', correct_meter1.avg, epoch)
        tensorboard_writer.add_scalar('Val/Acc2', correct_meter2.avg, epoch)
        tensorboard_writer.add_scalar('Val/Acc3', correct_meter3.avg, epoch)
        tensorboard_writer.add_scalar('Val/Time', elapsed, epoch)
        if config.tensorboard.model_params:
            for name, param in model.named_parameters():
                tensorboard_writer.add_histogram(name, param, epoch)
    return correct_meter1.avg
    # return      
def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.dataloader.pin_memory = False
    if not config.train.use_kfold:
        config.train.fold_num = 1
    if args.resume != '':
        config_path = pathlib.Path(args.resume) / 'config.yaml'
        config.merge_from_file(config_path.as_posix())
        config.merge_from_list(['train.resume', True])
    config.merge_from_list(['train.dist.local_rank', args.local_rank])

    config.model.multitask = True
    df = pd.read_csv(config.dataset.cvsfile_train)
    datasets=[df['artist'].nunique(),df['style'].nunique(),df['genre'].nunique()]
    config.dataset.multi_task = datasets
    config = update_config(config)
    config.freeze()
    return config

def main():
    global global_step
    config = load_config()

    set_seed(config)
    setup_cudnn(config)

    epoch_seeds = np.random.randint(np.iinfo(np.int32).max // 2,
                                    size=config.scheduler.epochs)
    
    if config.train.distributed:
        dist.init_process_group(backend=config.train.dist.backend,
                                init_method=config.train.dist.init_method,
                                rank=config.train.dist.node_rank,
                                world_size=config.train.dist.world_size)
        torch.cuda.set_device(config.train.dist.local_rank)
    
    output_dir = pathlib.Path(config.train.output_dir)
    print(output_dir)
    if get_rank() == 0:
        if not config.train.resume and output_dir.exists():
            raise RuntimeError(
                f'Output directory `{output_dir.as_posix()}` already exists')
        output_dir.mkdir(exist_ok=True, parents=True)
        if not config.train.resume:
            save_config(config, output_dir / 'config.yaml')
            save_config(get_env_info(config), output_dir / 'env.yaml')
            diff = find_config_diff(config)
            if diff is not None:
                save_config(diff, output_dir / 'config_min.yaml')
    
    logger = create_logger(name=__name__,
                           distributed_rank=get_rank(),
                           output_dir=output_dir,
                           filename='log.txt')
    logger.info(config)
    logger.info(get_env_info(config))

    model = create_model(config)
    
    optimizer = create_optimizer(config, model)
    if config.device != 'cpu' and config.train.use_apex:
        model, optimizer = apex.amp.initialize(
            model, optimizer, opt_level=config.train.precision)

    
    model = apply_data_parallel_wrapper(config, model)
    train_loader, val_loader = create_dataloader(config, is_train=True)

    scheduler = create_scheduler(config,
                                  optimizer,
                                  steps_per_epoch=len(train_loader))
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                save_to_disk=get_rank() == 0)

    
    if config.train.resume:
        checkpoint_config = checkpointer.resume_or_load('', resume=True)
        global_step = checkpoint_config['global_step']
        start_epoch = checkpoint_config['epoch']
        config.defrost()
        config.merge_from_other_cfg(ConfigNode(checkpoint_config['config']))
        config.freeze()

    if get_rank() == 0 and config.train.use_tensorboard:
        tensorboard_writer = create_tensorboard_writer(
            config, output_dir, purge_step=config.train.start_epoch + 1)
        tensorboard_writer2 = create_tensorboard_writer(
            config, output_dir / 'running', purge_step=global_step + 1)
    else:
        tensorboard_writer = DummyWriter()
        tensorboard_writer2 = DummyWriter()
    
    train_loss, val_loss = create_loss(config)
    fold=0
    best_acc=0
    for epoch in range(config.scheduler.epochs):
        train(epoch, config, model, optimizer, scheduler, train_loss, train_loader,logger
            , tensorboard_writer, tensorboard_writer2)

        if config.train.val_period > 0 and (epoch % config.train.val_period== 0):
                acc1=validate(epoch, config, model, val_loss, val_loader, logger,
                        tensorboard_writer)
        tensorboard_writer.flush()
        tensorboard_writer2.flush()
        if ((((epoch % config.train.checkpoint_period
                    == 0) or (epoch == config.scheduler.epochs))and acc1>best_acc) or acc1>best_acc):
                if config.train.use_kfold:
                    checkpoint_config = {
                        'epoch': epoch,
                        'fold':fold,
                        'global_step': global_step,
                        'config': config.as_dict(),
                    }
                else:
                    checkpoint_config = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'config': config.as_dict(),
                    }
                if get_rank() == 0:
                    print("improve {} from {} save checkpoint!".format(best_acc,acc1))
                    best_acc = acc1
                    checkpointer.save(f'checkpoint_bestacc', **checkpoint_config)

    tensorboard_writer.close()
    tensorboard_writer2.close()

if __name__ == '__main__':
    main()