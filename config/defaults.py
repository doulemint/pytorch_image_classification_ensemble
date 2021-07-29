from yacs.config import CfgNode as CN
import os

_C = CN()

_C.seed = 42            # set random seed
_C.workers = 12           # set number of data loading workers (default: 4)
_C.beta1 = 0.9           # adam parameters beta1
_C.beta2 = 0.999         # adam parameters beta2
_C.mom = 0.9             # momentum parameters
_C.wd = 1e-4             # weight-decay   # clw note: origin is 1e-4, but kaggle top solution use 1e-6 TODO
    # wd = 1e-6
_C.resume = None         # path to latest checkpoint (default: none),should endswith ".pth" or ".tar" if used
_C.start_epoch = 0

_C.dataset = "/data/nextcloud/dbc2017/files/images"
_C.dataset_merge_csv = "/data/nextcloud/dbc2017/files/images/train"
# _C.num_classes = len(os.listdir(os.path.join(_C.dataset, 'train')))
_C.submit_example =  "./submit_example.csv"
_C.checkpoints = "./checkpoints/"        # path to save checkpoints
_C.log_dir = "./logs/"                   # path to save log files
_C.submits = "./submits/"                # path to save submission files
_C.evaluate = True

_C.model_name = "efficientnet-b3"  # "resnet18", "resnet34", "resnet50"、"se_resnext50_32x4d"、"resnext50_32x4d"、
                                    # "shufflenet_v2_x1_0"、"shufflenetv2_x0.5"、"efficientnet-b3"、“efficientnet-b4”、
                                    # “efficientnet-b5”、 vit_base_patch16_384  vit_large_patch16_384
                                    # tf_efficientnet_l2_ns_475
_C.sampler = "RandomSampler"   # "RandomSampler"、"WeightedSampler"、"imbalancedSampler"（和WeightedSampler基本一样）


_C.optim = "adam"  # "adam","radam","novograd",sgd","ranger","ralamb","over9000","lookahead","lamb"
    #step_milestones = [9, 12, 14]
    #step_milestones = [12, 16, 19]
    #step_milestones = [10, 21, 27]
    #step_milestones = [7, 12, 17]
_C.step_milestones = [7, 10, 12]  # 第几个epoch开始下降,比如设置5,则前5个epoch保持lr 0.01,第6个开始下降为0.001
_C.epochs = 13
_C.step_gamma = 0.1

_C.bs = 32         # clw note: bs=128, 配合input_size=784, workers = 12，容易超出共享内存大小  报错：ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
_C.input_size = (512, 512) if "vit" not in _C.model_name else (384, 384)   # clw note：注意是 w, h   512、384、784、(800, 600)

_C.freeze_bn_epochs = 0
_C.accum_iter = 1
_C.drop_out_rate = 0.2 if "efficientnet" in _C.model_name else 0.0
    #drop_out_rate = 0.0
_C.loss_func = "TaylorCrossEntropyLoss" #  "LabelSmoothingLoss"、 "LabelSmoothingLoss_clw", "CELoss"、"BCELoss"、"FocalLoss"、“FocalLoss_clw”、 "TaylorCrossEntropyLoss",
                                     # "SymmetricCrossEntropy", "BiTemperedLogisticLoss"
_C.label_smooth_epsilon = 0.3
_C.gpu_id = "0"           # default gpu id
_C.fp16 = True          # use float16 to train the model
_C.opt_level = "O1"      # if use fp16, "O0" means fp32，"O1" means mixed，"O2" means except BN，"O3" means only fp16

_C.do_mixup_in_dataset = 0
_C.do_cutmix_in_dataset = 0   # in __get_items__()  clw note:因为有可能是同一个类的, 相当于标签上看起来没有做cutmix一样...
_C.do_cutmix_in_batch = 0.5
_C.lr_scheduler = "cosine_change_per_epoch"  # lr scheduler method: "step", "cosine_change_per_epoch", "cosine_change_per_batch", "adjust","on_loss","on_acc",    adjust不需要配置这里的epoch和lr
        #lr = 1e-4  # adam: 1e-4, 3e-4, 5e-4
_C.lr = 1e-4
_C.epochs = 10

_C.use_kfold = False
                
def get_default_config():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()