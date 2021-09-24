#todo list
#random forest
#hard voting 
#soft voting(wright)
#linear regression?
#(2 layer percepton)
import argparse
import pathlib
import time


try:
    import apex
except ImportError:
    pass
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision,os
import pandas as pd
import tqdm
from pytorch_image_classification import create_transform
from pytorch_image_classification.models import get_model
from pytorch_image_classification.losses import TaylorCrossEntropyLoss
from pytorch_image_classification.datasets import MyDataset, pesudoMyDataset
from train import validate,send_targets_to_device

import torch,torchvision
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader,TensorDataset
from fvcore.common.checkpoint import Checkpointer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit

from fvcore.common.checkpoint import Checkpointer

from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_dataloader,create_dataset,
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
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_image_classification import create_transform
from pytorch_image_classification.models import get_model


class ShallowNetwork(nn.Module):
    def __init__(self,n_classes):
        super(ShallowNetwork,self).__init__()
        self.feature_size=n_classes*3
        self.fc1 = nn.Linear(self.feature_size, n_classes*2)
        self.fc2 = nn.Linear(n_classes*2, n_classes)
    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x), inplace=True),training=self.training)
        x = self.fc2(x)
        return x
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            # print(param)
            param.requires_grad = False
def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--resume', type=str, default='')#multitask
    parser.add_argument('--multitask', type=bool, default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if args.multitask != None:
         config.merge_from_list(['config.model.multitask', args.local_rank])
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
    config = update_config(config)
    config.freeze()
    return config

def evaluate(NN, test_dataloader,test_loss,epoch,logger,tensorboard_writer,config,device):        
        NN.eval()
        

        import time; start = time.time()
        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()

        for data,targets in test_dataloader:
            # pred_raw_all = []
            # for model in models: 
            #     with torch.no_grad():
            #         data = data.to(device)
            #         targets = targets.to(device)
            #         outputs = model(data)
            #         pred_raw_all.append(outputs)

            # pred_raw_all = torch.cat(pred_raw_all,1)
            data = data.to(device)#;print(data.size());print(targets.size())
            
            targets = torch.flatten(targets).to(device)
            test_loss = nn.CrossEntropyLoss(reduction='mean')

            with torch.no_grad():       
                probs = NN(data)
                # print(probs,targets)
                loss = test_loss(probs, targets.long())

            acc1, acc5 = compute_accuracy(config,
                                          probs,
                                          targets,
                                          augmentation=False,
                                          topk=(1, 5))
            loss = loss.item()
            acc1 = acc1.item()
            acc5 = acc5.item()

            num = data.size(0)
            loss_meter.update(loss, num)
            acc1_meter.update(acc1, num)
            acc5_meter.update(acc5, num)

        if get_rank() == 0:
            elapsed = time.time() - start
            logger.info(f'Elapsed {elapsed:.2f}')
            logger.info(
                    f'val/Epoch {epoch} '
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                    # f'sk_acc@1 {sk_acc1_meter.val:.4f} ({sk_acc1_meter.avg:.4f}) '
                    f'acc@1 {acc1_meter.val:.4f} ({acc1_meter.avg:.4f}) '
                    f'acc@5 {acc5_meter.val:.4f} ({acc5_meter.avg:.4f})')

            tensorboard_writer.add_scalar('val/Loss', loss_meter.avg, epoch)
            tensorboard_writer.add_scalar('Val/Acc1', acc1_meter.avg, epoch)
            tensorboard_writer.add_scalar('Val/Acc5', acc5_meter.avg, epoch)
            tensorboard_writer.add_scalar('Val/Time', elapsed, epoch)
        return acc1_meter.val
        

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
    
    data_root = config.dataset.dataset_dir
    batch_size=config.train.batch_size

    device=config.device
    if device=='cpu':
        num_workers=1
    else:
        num_workers=2
    soft=False

    # if config.augmentation.use_albumentations:
    #     if config.dataset.type=='dir':
    #         train_clean = get_files(data_root,'train',output_dir/'label_map.pkl')
    #         # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.7, random_state=0)
    #         # for trn_idx, val_idx in sss.split(train_clean['filename'], train_clean['label']):
    #         #     train_frame = train_clean.loc[trn_idx]
    #         #     test_frame  = train_clean.loc[val_idx]
    #         test_clean=get_files(config.dataset.dataset_dir+'val/','train',output_dir/'label_map.pkl')
    #         labeled_dataset = MyDataset(train_clean, data_root, transforms=create_transform(config, is_train=False), output_label=True,soft=soft,
    #                     n_class=config.dataset.n_classes,label_smooth=config.augmentation.use_label_smoothing,
    #                     epsilon=config.augmentation.label_smoothing.epsilon,is_df=config.dataset.type=='df')
    #         labeled_dataloader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #         test_dataset = MyDataset(test_clean,config.dataset.dataset_dir+'/val',
    #                         transforms=create_transform(config, is_train=False),is_df=config.dataset.type=='df')
    #         test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    #     elif config.dataset.type=='df':
    #         train_dataset, val_dataset = create_dataset(config, True)
    #         # train_clean =  pd.read_csv(config.dataset.cvsfile_train)
    #         # test_clean =  pd.read_csv(config.dataset.cvsfile_test)
    #         labeled_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    #         # test_dataset = MyDataset(test_clean, data_root, transforms=create_transform(config, is_train=False),data_type=config.dataset.subname)
    #         test_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # else:
    #     if config.train.use_test_as_val:
    #         labeled_dataloader,test_dataloader = create_dataloader(config, is_train=True)


    baseline=0
    

    # models_opt=["efficientnet-b5",
    # # "efficientnet-b5","efficientnet-b5"
    # ]#
    # models_checkpoints=['/root/artwork/pytorch_image_classification/experiments/wiki22/efficientnet-b5/exp01_CM_AUg/checkpoint_bstacc.pth',
    #         # '/root/artwork/pytorch_image_classification/experiments/wiki22/efficientnet-b5/exp01_RC_AUg/checkpoint_bstacc.pth',
    #         # '/root/artwork/pytorch_image_classification/experiments/wiki22/efficientnet-b5/exp01_SC_Aug/checkpoint_bstacc.pth'
    #         ]
    # models = []
    # config.defrost()
    # for opt,ckp_pth in zip(models_opt,models_checkpoints):
    #     config.model.name=opt
    #     models.append(get_model(config,pretrained=False))
    #     if os.path.exists(ckp_pth):
    #             checkpoint = torch.load(ckp_pth, map_location='cpu')
    #             if isinstance(models[-1],
    #                   (nn.DataParallel, nn.parallel.DistributedDataParallel)):
    #                 models[-1].module.load_state_dict(checkpoint['model'])
    #                 print(f"load model from {str(ckp_pth)}")
    #             else:
    #                 models[-1].load_state_dict(checkpoint['model'])
    #                 print(f"load model from {str(ckp_pth)}")
    #     set_parameter_requires_grad(models[-1],False)
    #     models[-1].to(device)
    # config.freeze()
    npz_files = [
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp010_sc/predictions_train.npz',
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp09_MU/predictions_train.npz',
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp09_rc/predictions_train.npz',
    # '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp02_cm/predictions_train.npz'
               ]
    X=[]
    for f in npz_files:
        # print(f)
        X.append(np.load(f)['probs'])
    gt=np.load(f)['gt']
    X=np.concatenate(X,axis=1)
    torch.set_printoptions(precision=6)

    NN = ShallowNetwork(config.dataset.n_classes);NN.to(device)
    optimizer = create_optimizer(config, NN)
    model = load_model('simpleModel.h5')
    weights=model.get_weights()
    
    NN, optimizer = apex.amp.initialize(NN, optimizer, opt_level=config.train.precision)
    
    NN.fc1.weight.data=torch.from_numpy(np.transpose(weights[0])).to(device)
    print(NN.fc1.weight.data,np.transpose(weights[0]))
    NN.fc1.bias.data=torch.from_numpy(weights[1]).to(device)
    NN.fc2.weight.data=torch.from_numpy(np.transpose(weights[2])).to(device)
    NN.fc2.bias.data=torch.from_numpy(weights[3]).to(device)
    scheduler =create_scheduler(config,
                                 optimizer,
                                 steps_per_epoch=len(gt))
    checkpointer = Checkpointer(NN,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                save_to_disk=get_rank() == 0)
    if get_rank() == 0 and config.train.use_tensorboard:
        tensorboard_writer = create_tensorboard_writer(
                    config, output_dir, purge_step=config.train.start_epoch + 1)
    else:
        tensorboard_writer = DummyWriter()
        
    
    npz_test_files = [
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp010_sc/predictions_test.npz',
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp09_MU/predictions_test.npz',
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp09_rc/predictions_test.npz',
    # '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp02_cm/predictions_test.npz'
               ]

    train_loss, test_loss = create_loss(config)
    X_test=[]
    for f in npz_test_files:
        # print(np.load(f)['preds'].shape)
        X_test.append(np.load(f)['probs'])
    gt_test=np.load(f)['gt']    
    X_test=np.concatenate(X_test,axis=1)

     # transform to torch tensor
    tensor_x_test = torch.Tensor(X_test)
    # tensor_y = torch.Tensor(my_y)
    
    # gt_test=[]
    # for _, targets in tqdm.tqdm(test_dataloader):
    #   # print(targets)
    #   gt_test.extend(targets)
    
    # gt_test = np.array(gt_test)
    print(tensor_x_test.size())
    gt_test = torch.Tensor(gt_test).view(-1,1)
    print(gt_test.size()) 
    my_dataset = TensorDataset(tensor_x_test,gt_test) # create your datset
    test_dataloader = DataLoader(my_dataset,batch_size=128, num_workers=num_workers) 
    # acc=evaluate(NN,test_dataloader,test_loss,0,logger,tensorboard_writer,config,device)

    
    tensor_x = torch.Tensor(X)
    # gt=[]
    # for _, targets in tqdm.tqdm(labeled_dataloader):
    #   gt.extend(targets)
    # # print(len(gt),len(gt[0]))
    gt = torch.Tensor(gt).view(-1,1)
    print(gt.size())
    
    best_acc=0
    my_dataset = TensorDataset(tensor_x,gt) # create your datset
    labeled_dataloader = DataLoader(my_dataset,batch_size=128, num_workers=num_workers) 

    
    for epoch in range(config.scheduler.epochs):
        
        # pred_prob_all = []
        # pred_label_all = []
        import time;start = time.time()
        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()
        
        for data,targets in labeled_dataloader:
            # pred_raw_all = []
            # for model in models: 
            #     with torch.no_grad():
            #         data = data.to(device)
            #         targets = targets.to(device)
            #         outputs = model(data)
            #         pred_raw_all.append(outputs)
            data = data.to(device)
            targets = torch.flatten(targets).to(device)
            optimizer.zero_grad()
            # pred_raw_all = torch.cat(pred_raw_all,1)#;print(pred_raw_all,'\n',targets);
            NN.train()
            probs = NN(data)
            loss = train_loss(probs, targets.long())

            acc1, acc5 = compute_accuracy(config,
                                          probs,
                                          targets,
                                          augmentation=False,
                                          topk=(1, 5))
            loss.backward()
            optimizer.step()

            loss = loss.item()
            acc1 = acc1.item()
            acc5 = acc5.item()

            num = data.size(0)
            loss_meter.update(loss, num)
            acc1_meter.update(acc1, num)
            acc5_meter.update(acc5, num)

        if get_rank() == 0:
            elapsed = time.time() - start
            logger.info(f'Elapsed {elapsed:.2f}')
            logger.info(
                    f'train/Epoch {epoch} '
                    f'lr {scheduler.get_last_lr()[0]:.6f} '
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                    # f'sk_acc@1 {sk_acc1_meter.val:.4f} ({sk_acc1_meter.avg:.4f}) '
                    f'acc@1 {acc1_meter.val:.4f} ({acc1_meter.avg:.4f}) '
                    f'acc@5 {acc5_meter.val:.4f} ({acc5_meter.avg:.4f})')

            tensorboard_writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
            tensorboard_writer.add_scalar('Train/Acc1', acc1_meter.avg, epoch)
            tensorboard_writer.add_scalar('Train/Acc5', acc5_meter.avg, epoch)
            tensorboard_writer.add_scalar('Train/Time', elapsed, epoch)
            tensorboard_writer.add_scalar('Train/LearningRate',scheduler.get_last_lr()[0], epoch)
            tensorboard_writer.flush()
        scheduler.step()
        #evaluate(NN, test_dataloader,test_loss,epoch,logger,tensorboard_writer,config,device)
        acc=evaluate(NN,test_dataloader,test_loss,epoch,logger,tensorboard_writer,config,device)
        if best_acc<acc and get_rank()==0:
                checkpoint_config = {
                            'epoch': epoch,
                            'best_acc' : best_acc,
                            'config': config.as_dict(),
                        }
                if get_rank() == 0:
                        logger.info(f"improve {acc} from {best_acc} save checkpoint!")
                        best_acc = acc
                        checkpointer.save(f'checkpoint_bstacc', **checkpoint_config)
        tensorboard_writer.flush()
    tensorboard_writer.close()

def stack_keras():
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
    
    data_root = config.dataset.dataset_dir
    batch_size=config.train.batch_size

    device=config.device
    if device=='cpu':
        num_workers=1
    else:
        num_workers=2
    soft=False

    # if config.augmentation.use_albumentations:
    #     if config.dataset.type=='dir':
    #         train_clean = get_files(data_root,'train',output_dir/'label_map.pkl')
    #         # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.7, random_state=0)
    #         # for trn_idx, val_idx in sss.split(train_clean['filename'], train_clean['label']):
    #         #     train_frame = train_clean.loc[trn_idx]
    #         #     test_frame  = train_clean.loc[val_idx]
    #         test_clean=get_files(config.dataset.dataset_dir+'val/','train',output_dir/'label_map.pkl')
    #         labeled_dataset = MyDataset(train_clean, data_root, transforms=create_transform(config, is_train=False), output_label=True,soft=soft,
    #                     n_class=config.dataset.n_classes,label_smooth=config.augmentation.use_label_smoothing,
    #                     epsilon=config.augmentation.label_smoothing.epsilon,is_df=config.dataset.type=='df')
    #         labeled_dataloader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #         test_dataset = MyDataset(test_clean,config.dataset.dataset_dir+'/val',
    #                         transforms=create_transform(config, is_train=False),is_df=config.dataset.type=='df')
    #         test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    #     elif config.dataset.type=='df':
    #         train_dataset, val_dataset = create_dataset(config, True)
    #         # train_clean =  pd.read_csv(config.dataset.cvsfile_train)
    #         # test_clean =  pd.read_csv(config.dataset.cvsfile_test)
    #         labeled_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    #         # test_dataset = MyDataset(test_clean, data_root, transforms=create_transform(config, is_train=False),data_type=config.dataset.subname)
    #         test_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # else:
    #     if config.train.use_test_as_val:
    #         labeled_dataloader,test_dataloader = create_dataloader(config, is_train=True)


    baseline=0
    

    # models_opt=["efficientnet-b5",
    # # "efficientnet-b5","efficientnet-b5"
    # ]#
    # models_checkpoints=['/root/artwork/pytorch_image_classification/experiments/wiki22/efficientnet-b5/exp01_CM_AUg/checkpoint_bstacc.pth',
    #         # '/root/artwork/pytorch_image_classification/experiments/wiki22/efficientnet-b5/exp01_RC_AUg/checkpoint_bstacc.pth',
    #         # '/root/artwork/pytorch_image_classification/experiments/wiki22/efficientnet-b5/exp01_SC_Aug/checkpoint_bstacc.pth'
    #         ]
    # models = []
    # config.defrost()
    # for opt,ckp_pth in zip(models_opt,models_checkpoints):
    #     config.model.name=opt
    #     models.append(get_model(config,pretrained=False))
    #     if os.path.exists(ckp_pth):
    #             checkpoint = torch.load(ckp_pth, map_location='cpu')
    #             if isinstance(models[-1],
    #                   (nn.DataParallel, nn.parallel.DistributedDataParallel)):
    #                 models[-1].module.load_state_dict(checkpoint['model'])
    #                 print(f"load model from {str(ckp_pth)}")
    #             else:
    #                 models[-1].load_state_dict(checkpoint['model'])
    #                 print(f"load model from {str(ckp_pth)}")
    #     set_parameter_requires_grad(models[-1],False)
    #     models[-1].to(device)
    # config.freeze()
    npz_files = [
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp010_sc/predictions_train.npz',
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp09_MU/predictions_train.npz',
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp09_rc/predictions_train.npz',
    # '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp02_cm/predictions_train.npz'
               ]
    X=[]
    for f in npz_files:
        # print(f)
        X.append(np.load(f)['probs'])
    gt=np.load(f)['gt']
    X=np.concatenate(X,axis=1)
    # print(X[0],X.shape);

    torch.set_printoptions(precision=6)
    NN = ShallowNetwork(config.dataset.n_classes);NN.to(device)
    if os.path.exists():
        from tensorflow.keras.models import load_model 
        model = load_model('simpleModel.h5')
        weights=model.get_weights()
        NN.fc1.weight.data=torch.from_numpy(np.transpose(weights[0])).to(device)
        NN.fc1.bias.data=torch.from_numpy(weights[1]).to(device)
        NN.fc2.weight.data=torch.from_numpy(np.transpose(weights[2])).to(device)
        NN.fc2.bias.data=torch.from_numpy(weights[3]).to(device)

    optimizer = create_optimizer(config, NN)
    scheduler =create_scheduler(config,
                                 optimizer,
                                 steps_per_epoch=len(gt))
    checkpointer = Checkpointer(NN,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                save_to_disk=get_rank() == 0)
    if get_rank() == 0 and config.train.use_tensorboard:
        tensorboard_writer = create_tensorboard_writer(
                    config, output_dir, purge_step=config.train.start_epoch + 1)
    else:
        tensorboard_writer = DummyWriter()
        
    
    npz_test_files = [
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp010_sc/predictions_test.npz',
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp09_MU/predictions_test.npz',
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp09_rc/predictions_test.npz',
    # '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp02_cm/predictions_test.npz'
               ]

    train_loss, test_loss = create_loss(config)
    X_test=[]
    for f in npz_test_files:
        # print(np.load(f)['preds'].shape)
        X_test.append(np.load(f)['probs'])
    gt_test=np.load(f)['gt']    
    X_test=np.concatenate(X_test,axis=1)

     # transform to torch tensor
    tensor_x_test = torch.Tensor(X_test)
    # tensor_y = torch.Tensor(my_y)
    
    # gt_test=[]
    # for _, targets in tqdm.tqdm(test_dataloader):
    #   # print(targets)
    #   gt_test.extend(targets)
    
    # gt_test = np.array(gt_test)
    print(tensor_x_test.size())
    gt_test = torch.Tensor(gt_test).view(-1,1)
    print(gt_test.size()) 
    my_dataset = TensorDataset(tensor_x_test,gt_test) # create your datset
    test_dataloader = DataLoader(my_dataset,batch_size=128, num_workers=num_workers) 
    # acc=evaluate(NN,test_dataloader,test_loss,0,logger,tensorboard_writer,config,device)

    
    tensor_x = torch.Tensor(X)
    # gt=[]
    # for _, targets in tqdm.tqdm(labeled_dataloader):
    #   gt.extend(targets)
    # # print(len(gt),len(gt[0]))
    gt = torch.Tensor(gt).view(-1,1)
    print(gt.size())
    
    best_acc=0
    my_dataset = TensorDataset(tensor_x,gt) # create your datset
    labeled_dataloader = DataLoader(my_dataset,batch_size=128, num_workers=num_workers) 

    
    for epoch in range(config.scheduler.epochs):
        
        # pred_prob_all = []
        # pred_label_all = []
        import time;start = time.time()
        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()
        
        for data,targets in labeled_dataloader:
            # pred_raw_all = []
            # for model in models: 
            #     with torch.no_grad():
            #         data = data.to(device)
            #         targets = targets.to(device)
            #         outputs = model(data)
            #         pred_raw_all.append(outputs)
            data = data.to(device)
            targets = torch.flatten(targets).to(device)
            optimizer.zero_grad()
            # pred_raw_all = torch.cat(pred_raw_all,1)#;print(pred_raw_all,'\n',targets);
            NN.train()
            probs = NN(data)
            loss = train_loss(probs, targets.long())

            acc1, acc5 = compute_accuracy(config,
                                          probs,
                                          targets,
                                          augmentation=False,
                                          topk=(1, 5))
            loss.backward()
            optimizer.step()

            loss = loss.item()
            acc1 = acc1.item()
            acc5 = acc5.item()

            num = data.size(0)
            loss_meter.update(loss, num)
            acc1_meter.update(acc1, num)
            acc5_meter.update(acc5, num)

        if get_rank() == 0:
            elapsed = time.time() - start
            logger.info(f'Elapsed {elapsed:.2f}')
            logger.info(
                    f'train/Epoch {epoch} '
                    f'lr {scheduler.get_last_lr()[0]:.6f} '
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                    # f'sk_acc@1 {sk_acc1_meter.val:.4f} ({sk_acc1_meter.avg:.4f}) '
                    f'acc@1 {acc1_meter.val:.4f} ({acc1_meter.avg:.4f}) '
                    f'acc@5 {acc5_meter.val:.4f} ({acc5_meter.avg:.4f})')

            tensorboard_writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
            tensorboard_writer.add_scalar('Train/Acc1', acc1_meter.avg, epoch)
            tensorboard_writer.add_scalar('Train/Acc5', acc5_meter.avg, epoch)
            tensorboard_writer.add_scalar('Train/Time', elapsed, epoch)
            tensorboard_writer.add_scalar('Train/LearningRate',scheduler.get_last_lr()[0], epoch)
            tensorboard_writer.flush()
        scheduler.step()
        #evaluate(NN, test_dataloader,test_loss,epoch,logger,tensorboard_writer,config,device)
        acc=evaluate(NN,test_dataloader,test_loss,epoch,logger,tensorboard_writer,config,device)
        if best_acc<acc and get_rank()==0:
                checkpoint_config = {
                            'epoch': epoch,
                            'best_acc' : best_acc,
                            'config': config.as_dict(),
                        }
                if get_rank() == 0:
                        logger.info(f"improve {acc} from {best_acc} save checkpoint!")
                        best_acc = acc
                        checkpointer.save(f'checkpoint_bstacc', **checkpoint_config)
        tensorboard_writer.flush()
    tensorboard_writer.close()


if __name__ == '__main__':
    main()

            