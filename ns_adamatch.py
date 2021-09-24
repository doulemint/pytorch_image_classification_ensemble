import argparse
import pathlib
import time,os

try:
    import apex
except ImportError:
    pass
import pandas as pd
import numpy as np
from evaluate import evaluate
# import tensorflow as tf

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
import torch.nn.functional as F
from pytorch_image_classification import create_transform
from pytorch_image_classification.models import get_model
from pytorch_image_classification.losses import TaylorCrossEntropyLoss
from pytorch_image_classification.datasets import MyDataset, pesudoMyDataset
from train import validate,send_targets_to_device

import torch,torchvision
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from fvcore.common.checkpoint import Checkpointer

from sklearn.model_selection import StratifiedShuffleSplit

def  generate_pseudo_labels(weak_images_train, weak_images_test, teacher_models, confidence_thres):
       
    for model in teacher_models:   
       model.eval()

    with torch.no_grad():
        # pass train images into models
        preds_1 = teacher_models[0](weak_images_train)
        preds_2 = teacher_models[1](weak_images_train)
        final_predictions_train = torch.stack((preds_1, preds_2), dim=0).mean(0)

        # pass test images into models
        preds_1 = teacher_models[0](weak_images_test)
        preds_2 = teacher_models[1](weak_images_test)
        # print("preds_1",preds_1.size())
        # print("final_predictions_test: ",torch.stack((preds_1, preds_2), dim=0).size())

        final_predictions_test = torch.stack((preds_1, preds_2), dim=0).mean(0)
        # print("final_predictions_train: ",torch.nn.Softmax(dim=1)(final_predictions_train).max(1),final_predictions_train.size())
        # final_predictions_test_, _ = torch.nn.Softmax()(
        #     torch.tensor(final_predictions_test)).max(1)
        final_predictions_test_ = torch.nn.Softmax(dim=1)(final_predictions_test)
        final_predictions_test_,_ = torch.max(final_predictions_test_,dim=1)
        # print("final_predictions_test_: ",final_predictions_test_,final_predictions_test_.size())

        # print("1: ",torch.sum(final_predictions_test_ > confidence_thres))

        # compute thresholding mask
        test_mask = final_predictions_test_ > confidence_thres
        # print("test_mask: ",test_mask,test_mask.size())

        # concatenate all predictions
        all_predictions = torch.cat(
            (final_predictions_train, final_predictions_test), dim=0
        )

    return all_predictions, test_mask

def cross_entropy_loss(data: torch.Tensor, target: torch.Tensor,
                       reduction: str) -> torch.Tensor:
    target = torch.nn.Softmax(dim=1)(target)
    logp = F.log_softmax(data, dim=1)
    loss = torch.sum(-logp * target, dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')

def compute_loss_target(predictions, pseudo_labels,gt, alpha):
    # print("predictions: ",predictions,predictions.size())
    # print("pseudo_labels: ",pseudo_labels,pseudo_labels.size())
    # pseudo_labels = pseudo_labels.to(dtype=torch.long)

    if gt is not None:
        # print("gt: ",gt.size())
        # loss_func1 = TaylorCrossEntropyLoss(reduction='mean')
        # loss_func = cross_entropy_with_soft_target(reduction='mean')
        # loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        loss_func1=nn.CrossEntropyLoss(reduction="mean")
        # loss_function = nn.CrossEntropyLoss()
        with torch.no_grad():
            target_loss  = cross_entropy_loss(predictions,pseudo_labels,'mean')
            # student_loss = cross_entropy_loss(predictions,gt,'mean')
            # _, fk_targets = pseudo_labels.max(dim=1)
            #target_loss = loss_func1(predictions,pseudo_labels)
            # target_loss = torch.tensor(loss_func(predictions.cpu().numpy(),pseudo_labels.cpu().numpy()).numpy())#loss_func(predictions,pseudo_labels)
            student_loss = loss_func1(predictions,gt)
            # print("target_loss: ",target_loss," student_loss: ",student_loss)
            # print("total loss: ",((1 - alpha) * target_loss) + (alpha * student_loss))
            return ((1 - alpha) * target_loss) + (alpha * student_loss)
    else:
        # _, fk_targets = pseudo_labels.max(dim=1)
        # loss_func = nn.CrossEntropyLoss(reduction='none')#cross_entropy_with_soft_target(reduction='none')
        # student_loss = loss_func(predictions,fk_targets)
        student_loss = cross_entropy_loss(predictions,pseudo_labels)
        # print("student_loss: ",student_loss)
        return student_loss

def get_alpha(epoch, total_epochs):
    initial_alpha = 0.1
    final_alpha = 0.5
    modified_alpha = (
        final_alpha - initial_alpha
    ) / total_epochs * epoch + initial_alpha
    return modified_alpha

def main():
    config = load_config()
    global_step = 0


    set_seed(config)
    setup_cudnn(config)

    best_acc=0

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
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        for trn_idx, val_idx in sss.split(train_clean['filename'], train_clean['label']):
            train_frame = train_clean.loc[trn_idx]
            val_frame  = train_clean.loc[val_idx]
        test_clean=get_files(config.dataset.dataset_dir+'val/','train',output_dir/'label_map.pkl')
    elif config.dataset.type=='df':
        train_clean =  pd.read_csv(config.dataset.cvsfile_train)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        for trn_idx, val_idx in sss.split(train_clean['image'], train_clean['label']):
            train_frame = train_clean.loc[trn_idx]
            val_frame  = train_clean.loc[val_idx]
        test_clean =  pd.read_csv(config.dataset.cvsfile_test)
    
    soft = False
    
    weak_labeled_dataset = MyDataset(train_frame, data_root, transforms=create_transform(config, is_train=False), output_label=True,is_df=config.dataset.type=='df')
    strong_labeled_dataset = MyDataset(train_frame, data_root, transforms=create_transform(config, is_train=True), output_label=True,is_df=config.dataset.type=='df')

    weak_unlabeled_dataset = MyDataset(val_frame, data_root, transforms=create_transform(config, is_train=False),is_df=config.dataset.type=='df')
    strong_unlabeled_dataset = MyDataset(val_frame, data_root, transforms=create_transform(config, is_train=True),is_df=config.dataset.type=='df')

    num_workers=config.train.dataloader.num_workers
    weak_labeled_dataloader = DataLoader(weak_labeled_dataset, batch_size=batch_size, num_workers=num_workers)
    strong_labeled_dataloader = DataLoader(strong_labeled_dataset, batch_size=batch_size, num_workers=num_workers)
    weak_unlabeled_dataloader = DataLoader(weak_unlabeled_dataset, batch_size=batch_size//4, num_workers=num_workers)
    strong_unlabeled_dataloader = DataLoader(strong_unlabeled_dataset, batch_size=batch_size//4, num_workers=num_workers)
    
    test_dataset = MyDataset(test_clean,config.dataset.dataset_dir+'val/',
                        transforms=create_transform(config, is_train=False),is_df=config.dataset.type=='df')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    
    student_model_opt = "resnet50"
    teacher_model_opt = ["resnet50","efficientnet-b5"]

    TEMPERATURE = 10
    log_softmax = torch.nn.LogSoftmax(dim=1)
    kl_divergence = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    device = config.device
    num_epochs = config.scheduler.epochs
    teacher_model = []
    config.defrost()
    for opt in teacher_model_opt:
        config.model.name=opt
        teacher_model.append(get_model(config,pretrained=False))
        ckp_pth= config.test.checkpoint+f'/checkpoint_{opt}.pth'
        # print(ckp_pth)
        if os.path.exists(ckp_pth):
                checkpoint = torch.load(ckp_pth, map_location='cpu')
                if isinstance(teacher_model[-1],
                      (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                    teacher_model[-1].module.load_state_dict(checkpoint['model'])
                    print(f"load model from {str(ckp_pth)}")
                else:
                    teacher_model[-1].load_state_dict(checkpoint['model'])
                    print(f"load model from {str(ckp_pth)}")
        teacher_model[-1].to(device)
        macs, n_params = count_op(config, teacher_model[-1])
        logger.info(f'name   : {opt}')
        logger.info(f'MACs   : {macs}')
        logger.info(f'#params: {n_params}')
    config.model.name=student_model_opt
    student_model=get_model(config,pretrained=False)
    ckp_pth= config.test.checkpoint+f'/checkpoint_{student_model_opt}.pth'
    if config.train.checkpoint != '':
        checkpoint = torch.load(config.train.checkpoint, map_location='cpu')
        if isinstance(student_model,
            (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            student_model.module.load_state_dict(checkpoint['model'])
            print(f"load model from {str(config.train.checkpoint)}")
        else:
            student_model.load_state_dict(checkpoint['model'])
            print(f"load model from {str(config.train.checkpoint)}")

    elif os.path.exists(ckp_pth):
      checkpoint = torch.load(ckp_pth, map_location='cpu')
      if isinstance(student_model,
          (nn.DataParallel, nn.parallel.DistributedDataParallel)):
          student_model.module.load_state_dict(checkpoint['model'])
          print(f"load model from {str(ckp_pth)}")
      else:
          student_model.load_state_dict(checkpoint['model'])
          print(f"load model from {str(ckp_pth)}")
    student_model.to(device)
    macs, n_params = count_op(config, student_model)
    logger.info(f'name   : {student_model_opt}')
    logger.info(f'MACs   : {macs}')
    logger.info(f'#params: {n_params}')
    config.freeze()

    optimizer = create_optimizer(config, student_model)
    if config.device != 'cpu' and config.train.use_apex:
        student_model, optimizer = apex.amp.initialize(
                    student_model, optimizer, opt_level=config.train.precision)
    student_model = apply_data_parallel_wrapper(config, student_model)
    scheduler =create_scheduler(config,
                                 optimizer,
                                 steps_per_epoch=len(train_frame))

    checkpointer = Checkpointer(student_model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                save_to_disk=get_rank() == 0)
    # checkpointer.load(config.test.checkpoint)
    if get_rank() == 0 and config.train.use_tensorboard:
        tensorboard_writer = create_tensorboard_writer(
                    config, output_dir, purge_step=config.train.start_epoch + 1)
        tensorboard_writer2 = create_tensorboard_writer(
                    config, output_dir / 'running', purge_step= 1)
    else:
        tensorboard_writer = DummyWriter()
        tensorboard_writer2 = DummyWriter()
    
    _, val_loss = create_loss(config)

    # preds, probs, labels, loss, acc = evaluate(config, student_model, test_dataloader,
    #                                            val_loss, logger)
    for j in range(config.scheduler.epochs):
        logger.info(f'Train {j} {global_step}')
        start = time.time()

        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()
        student_model.train()
        # step=0
        for step,(weak_batch_train,weak_batch_test,strong_batch_train,strong_batch_test,) in enumerate(zip(
            weak_labeled_dataloader, weak_unlabeled_dataloader, strong_labeled_dataloader, strong_unlabeled_dataloader)): 
            step += 1
            global_step += 1

            if get_rank() == 0 and step == 1:
                if config.tensorboard.train_images:
                    image = torchvision.utils.make_grid(weak_batch_train,
                                                        normalize=True,
                                                        scale_each=True)
                    tensorboard_writer.add_image('Train/Image', image, j)

            optimizer.zero_grad()

            weak_image_train = weak_batch_train[0].to(device,non_blocking=config.train.dataloader.non_blocking)
            targets = weak_batch_train[1].to(device,non_blocking=config.train.dataloader.non_blocking)
            weak_image_test  = weak_batch_test[0].to(device,non_blocking=config.train.dataloader.non_blocking)
            strong_image_train = strong_batch_train[0].to(device,non_blocking=config.train.dataloader.non_blocking)
            strong_image_test = strong_batch_test[0].to(device,non_blocking=config.train.dataloader.non_blocking)

            targets = send_targets_to_device(config, targets, device)

            num_train = strong_image_train.size(0)
            # print("num_train: ",num_train)
            # print("strong_image_test: ",strong_image_test.size())
            # print("weak_image_test: ",weak_image_test.size()

            # strong_image=torch.cat((strong_image_train, strong_image_test), dim=0)
            student_prediction=student_model(torch.cat((strong_image_train, strong_image_test), dim=0))
            student_prediction_train=student_prediction[:num_train]
            student_prediction_test=student_prediction[num_train:]
            # print("student_prediction_test: ",student_prediction_test.size())

            
            #calcutate c_tau
            # print("student_prediction_train",student_prediction_train)
            row_wise_max = F.softmax(student_prediction_train, dim=1)#torch.nn.Softmax(dim=1)(student_prediction_train)
            row_wise_max,_ = torch.max(row_wise_max,dim=1)
            # print("row_wise_max: ",row_wise_max,row_wise_max.size())
            final_sum=torch.mean(row_wise_max)
            # final_sum = row_wise_max.mean(0)
            # print("final_sum: ",final_sum)
            c_tau = 0.8 * final_sum

            pseudo_labels,test_mask=generate_pseudo_labels(
                weak_image_train,weak_image_test,teacher_model,c_tau
            )
            ## allign target label distribtion to student_prediction_train distribution
            predicts=torch.cat((student_prediction_train, student_prediction_test), dim=0)
            expectation_ratio = torch.mean(predicts) / torch.mean(pseudo_labels)
            # print("expectation_ratio: ",expectation_ratio)
            pseudo_labels = F.normalize((pseudo_labels*expectation_ratio), p=2, dim=1) # L2 normalization

            # pseudo_labels = pseudo_labels.to(dtype=torch.long)
            _, pseudo_labels = pseudo_labels.max(dim=1)
            # print("loss: ",val_loss(student_prediction_train,pseudo_labels[:num_train]))
            # print("loss1: ",kl_divergence(log_softmax(student_prediction_train / TEMPERATURE),
            # log_softmax(pseudo_labels[:num_train] / TEMPERATURE)))
            # print("pseudo_labels: ",pseudo_labels.size())

            alpha = get_alpha(j, num_epochs)
            train_loss = compute_loss_target(student_prediction_train,pseudo_labels[:num_train],targets,alpha)
            # print("loss2: ",train_loss)
            test_loss = compute_loss_target(student_prediction_test,pseudo_labels[num_train:],None,alpha)

            loss = train_loss + (test_loss[test_mask]).mean()
            if config.device != 'cpu' and config.train.use_apex:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            if config.train.gradient_clip > 0:
                if config.device != 'cpu' and config.train.use_apex:
                    torch.nn.utils.clip_grad_norm_(
                        apex.amp.master_params(optimizer),
                        config.train.gradient_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(),
                                                config.train.gradient_clip)
            if config.train.subdivision > 1:
                for param in student_model.parameters():
                    param.grad.data.div_(config.train.subdivision)
            optimizer.step()

            acc1, acc5 = compute_accuracy(config,
                                      student_prediction_train,
                                      targets,
                                      augmentation=True,
                                      topk=(1, 5))

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
            loss_meter.update(loss.cpu().item(),(num_train+test_loss.size(0)))
            acc1_meter.update(acc1.cpu().item(), num_train)
            acc5_meter.update(acc5.cpu().item(), num_train)
            
            if get_rank() == 0:
                if step % config.train.log_period == 0 or step == len(
                        weak_labeled_dataloader):
                    logger.info(
                        f'Epoch {j} '
                        f'Step {step}/{len(weak_labeled_dataloader)} '
                        f'lr {scheduler.get_last_lr()[0]:.6f} '
                        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        # f'sk_acc@1 {sk_acc1_meter.val:.4f} ({sk_acc1_meter.avg:.4f}) '
                        f'acc@1 {acc1_meter.val:.4f} ({acc1_meter.avg:.4f}) '
                        f'acc@5 {acc5_meter.val:.4f} ({acc5_meter.avg:.4f})')

                    tensorboard_writer2.add_scalar('Train/RunningLoss',
                                                loss_meter.avg, global_step)
                    tensorboard_writer2.add_scalar('Train/RunningAcc1',
                                                acc1_meter.avg, global_step)
                    tensorboard_writer2.add_scalar('Train/RunningAcc5',
                                                acc5_meter.avg, global_step)
                    tensorboard_writer2.add_scalar('Train/RunningLearningRate',
                                                scheduler.get_last_lr()[0],
                                                global_step)
            scheduler.step()
        # print("step: ",step)

        logger.info(f'Epoch {j} '
                    f'loss {loss_meter.avg:.4f} '
                    f'acc@1 {acc1_meter.avg:.4f} '
                    f'acc@5 {acc5_meter.avg:.4f}')
        if get_rank() == 0:
            elapsed = time.time() - start
            logger.info(f'Elapsed {elapsed:.2f}')

            tensorboard_writer.add_scalar('Train/Loss', loss_meter.avg, j)
            tensorboard_writer.add_scalar('Train/Acc1', acc1_meter.avg, j)
            tensorboard_writer.add_scalar('Train/Acc5', acc5_meter.avg, j)
            tensorboard_writer.add_scalar('Train/Time', elapsed, j)
            tensorboard_writer.add_scalar('Train/LearningRate',
                                        scheduler.get_last_lr()[0], j)
    
        acc=validate(j, config, student_model, val_loss, test_dataloader, logger,tensorboard_writer)

        tensorboard_writer.flush()
        tensorboard_writer2.flush()

        if ((((j % config.train.checkpoint_period
                    == 0) or (j == config.scheduler.epochs))and acc>best_acc) or acc>best_acc):
                    checkpoint_config = {
                            'epoch': j,
                            'global_step': global_step,
                            'config': config.as_dict(),
                        }
                    if get_rank() == 0:
                        logger.info(f"improve {acc} from {best_acc} save checkpoint!")
                        best_acc = acc
                        checkpointer.save(f'checkpoint_bstacc', **checkpoint_config)

    tensorboard_writer.close()
    tensorboard_writer2.close()

if __name__ == '__main__':
    main()

        
