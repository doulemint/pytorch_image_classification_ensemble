import argparse
import pathlib
import time,os

try:
    import apex
except ImportError:
    pass
import pandas as pd
import numpy as np

from train import train,validate,load_config
from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_loss,
    create_model,
    get_files,
    create_optimizer,
    create_scheduler,discriminative_lr_params,
    prepare_dataloader,
    get_default_config,
    update_config,
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
from pytorch_image_classification import create_transform
from pytorch_image_classification.models import get_model
from pytorch_image_classification.datasets import MyDataset, pesudoMyDataset

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from fvcore.common.checkpoint import Checkpointer

from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit

def train_model(epoch,model, train_loader, val_loader, criterion, optimizer, device, max_epoch, summary_writer=None):
        # model.to(device)
    # for  in range(1, max_epoch + 1):  # loop over the dataset multiple times

        running_loss = 0.0
        running_accuracy = 0.0
        count = 0
        for step, (image,label) in enumerate(train_loader, 1):
            # get the inputs; data is a list of [inputs, labels]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(image)
            loss = criterion(outputs, label)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            logits = torch.max(outputs, dim=1)[1]
            if label.dim() != 1:
                label = torch.max(label, dim=1)[1]
            count += outputs.size(0)
            running_accuracy += (logits == label).sum().item()

        # print statistics
        print('TRAIN - Epoch {} loss: {:.4f} accuracy: {:.4f}'.format(epoch,
                                                                      running_loss / step, running_accuracy / count))
        if summary_writer is not None:
            summary_writer.add_scalar('train', running_accuracy / count, epoch)

        running_accuracy = 0.0
        count = 0
        model.eval()
        for step, (image,label) in enumerate(val_loader, 1):
            
            with torch.no_grad():
                outputs = model(image)

                logits = torch.max(outputs, dim=1)[1]
                count += outputs.size(0)
                running_accuracy += (logits == label).sum().item()

        print('VAL - Epoch {}  accuracy: {:.4f}'.format(epoch,
                                                        running_accuracy / count))

        if summary_writer is not None:
            summary_writer.add_scalar('val', running_accuracy / count, epoch)

def main():
    config = load_config()

    set_seed(config)
    setup_cudnn(config)

    # epoch_seeds = np.random.randint(np.iinfo(np.int32).max // 2,
    #                                 size=config.scheduler.epochs)

    if config.train.distributed:
        dist.init_process_group(backend=config.train.dist.backend,
                                init_method=config.train.dist.init_method,
                                rank=config.train.dist.node_rank,
                                world_size=config.train.dist.world_size)
        torch.cuda.set_device(config.train.dist.local_rank)

    output_dir = pathlib.Path(config.train.output_dir)
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
    
    data_root = config.dataset.dataset_dir+'train/'
    batch_size=config.train.batch_size

    if config.dataset.type=='dir':
        train_clean = get_files(data_root,'train',output_dir/'label_map.pkl')
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.7, random_state=0)
        for trn_idx, val_idx in sss.split(train_clean['filename'], train_clean['label']):
            train_frame = train_clean.loc[trn_idx]
            test_frame  = train_clean.loc[val_idx]
        test_clean=get_files(config.dataset.dataset_dir+'val/','train',output_dir/'label_map.pkl')
    elif config.dataset.type=='df':
        train_clean =  pd.read_csv(config.dataset.cvsfile_train)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.7, random_state=0)
        for trn_idx, val_idx in sss.split(train_clean['image'], train_clean['label']):
            train_frame = train_clean.loc[trn_idx]
            test_frame  = train_clean.loc[val_idx]
        test_clean =  pd.read_csv(config.dataset.cvsfile_test)


    baseline=0
    device=config.device

    models_opt=["resnet50","resnet101","efficientnet-b5"]#
    models = []
    config.defrost()
    for opt in models_opt:
        config.model.name=opt
        models.append(get_model(config,pretrained=False))
    config.freeze()
    
    if device=='cpu':
        num_workers=1
    else:
        num_workers=2

    soft=True
    
    labeled_dataset = MyDataset(train_frame, data_root, transforms=create_transform(config, is_train=False), output_label=True,soft=soft,
                        n_class=config.dataset.n_classes,label_smooth=config.augmentation.use_label_smoothing,
                        epsilon=config.augmentation.label_smoothing.epsilon,is_df=config.dataset.type=='df')
    labeled_dataloader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataset = MyDataset(test_clean,config.dataset.dataset_dir+'val/',
                        transforms=create_transform(config, is_train=False),is_df=config.dataset.type=='df')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    
    
    best_acc=0

    if baseline==1:
        base_output_dir = pathlib.Path(config.train.output_dir+"/baseline")
        base_output_dir.mkdir(exist_ok=True, parents=True)
        #just run three model for 10 epoches -- baseline
        for i in range(len(models)):
            best_acc=0
            models[i].to(device)
            macs, n_params = count_op(config, models[i])
            logger.info(f'name   : {models_opt[i]}')
            logger.info(f'MACs   : {macs}')
            logger.info(f'#params: {n_params}')
            
            optimizer = create_optimizer(config, models[i])
            if config.device != 'cpu' and config.train.use_apex:
                model, optimizer = apex.amp.initialize(
                    models[i], optimizer, opt_level=config.train.precision)
            model = apply_data_parallel_wrapper(config, models[i])
            scheduler =create_scheduler(config,
                                 optimizer,
                                 steps_per_epoch=len(train_frame))
            checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=base_output_dir,
                                save_to_disk=get_rank() == 0)
            if get_rank() == 0 and config.train.use_tensorboard:
                tensorboard_writer = create_tensorboard_writer(
                    config, output_dir, purge_step=config.train.start_epoch + 1)
                tensorboard_writer2 = create_tensorboard_writer(
                    config, output_dir / 'running', purge_step= 1)
            else:
                tensorboard_writer = DummyWriter()
                tensorboard_writer2 = DummyWriter()
            train_loss, val_loss = create_loss(config)

            for j in range(config.scheduler.epochs):
                train(j,config,models[i],optimizer,scheduler,train_loss,labeled_dataloader,logger,tensorboard_writer,tensorboard_writer2) #global_step problem
                acc=validate(j,config,models[i],val_loss,test_dataloader,logger,tensorboard_writer)
                if get_rank() == 0:
                    if best_acc<acc:
                        print("improve {} from {} save checkpoint!".format(acc,best_acc))
                        best_acc=acc
                        checkpoint_config = {
                                    'models_opt':models_opt[i],
                                    'epoch': j,
                                    'config': config.as_dict(),
                        }
                        checkpointer.save(f'checkpoint_{models_opt[i]}', **checkpoint_config)
            del model, optimizer, scheduler,tensorboard_writer,tensorboard_writer2

    else:
        for i in range(len(models_opt)):
            best_acc=0
            ckp_pth= config.test.checkpoint+f'/checkpoint_{models_opt[i]}.pth'
            # print(ckp_pth)
            if os.path.exists(ckp_pth):
                checkpoint = torch.load(ckp_pth, map_location='cpu')
                if isinstance(models[i],
                      (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                    models[i].module.load_state_dict(checkpoint['model'])
                    print(f"load model from {str(ckp_pth)}")
                else:
                    models[i].load_state_dict(checkpoint['model'])
                    print(f"load model from {str(ckp_pth)}")
            models[i].to(device)
            macs, n_params = count_op(config, models[i])
            logger.info(f'name   : {models_opt[i]}')
            logger.info(f'MACs   : {macs}')
            logger.info(f'#params: {n_params}')
            optimizer = create_optimizer(config, models[i])
            if config.device != 'cpu' and config.train.use_apex:
                model, optimizer = apex.amp.initialize(
                    models[i], optimizer, opt_level=config.train.precision)
                model = apply_data_parallel_wrapper(config, model)
            else:
                model = apply_data_parallel_wrapper(config, models[i])

            scheduler =create_scheduler(config,
                                 optimizer,
                                 steps_per_epoch=len(train_frame))
            checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                save_to_disk=get_rank() == 0)
            if get_rank() == 0 and config.train.use_tensorboard:
                tensorboard_writer = create_tensorboard_writer(
                    config, output_dir, purge_step=config.train.start_epoch + 1)
                tensorboard_writer2 = create_tensorboard_writer(
                    config, output_dir / 'running', purge_step= 1)
            else:
                tensorboard_writer = DummyWriter()
                tensorboard_writer2 = DummyWriter()
            train_loss, val_loss = create_loss(config)
            # model.to(device)
            models[i]=model
            if i != 0:
                pseudo_labeled_dataloader = DataLoader(pesudoMyDataset(train_frame, test_frame,data_root, models[i - 1], device, transforms=create_transform(config, is_train=True), 
                        soft=soft,n_class=config.dataset.n_classes,label_smooth=config.augmentation.use_label_smoothing,
                        epsilon=config.augmentation.label_smoothing.epsilon,is_df=config.dataset.type=='df'), 
                        batch_size=batch_size, shuffle=True, num_workers=num_workers)

            for j in range(config.scheduler.epochs):
                if i == 0:
                    # if j!=0:
                    #     break
                    continue
                    train(j,config,model,optimizer,scheduler,train_loss,labeled_dataloader, logger, tensorboard_writer,tensorboard_writer2)
                else:       
                    train(j,config,models[i],optimizer,scheduler,train_loss,pseudo_labeled_dataloader, logger, tensorboard_writer,tensorboard_writer2)
                acc=validate(j,config,models[i],val_loss,test_dataloader,logger,tensorboard_writer)
                if get_rank() == 0:
                    if best_acc<acc:
                        print("improve {} from {} save checkpoint!".format(acc,best_acc))
                        best_acc=acc
                        checkpoint_config = {
                                    'models_opt':models_opt[i],
                                    'epoch': j,
                                    'config': config.as_dict(),
                        }
                        checkpointer.save(f'checkpoint_{models_opt[i]}', **checkpoint_config)
                    
            models[i]=model
            tensorboard_writer.close()
            tensorboard_writer2.close()
            del model, optimizer, scheduler,tensorboard_writer,tensorboard_writer2


if __name__ == '__main__':
    main()