import timm
import tensorflow as tf
import torch.nn as nn

from timm.models.layers.classifier import ClassifierHead
from torchvision import models
from pretrainedmodels import models as pm
import pretrainedmodels

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False

def get_model(configs,feature_extract=False):
    if configs.model.name.startswith("resnext50_32x4d"):
        # model = torchvision.models.resnext50_32x4d(pretrained=True)
        # model.avgpool = nn.AdaptiveAvgPool2d(1)
        # model.fc = nn.Linear(2048, configs.num_classes)
        ####
        model = timm.create_model('resnext50_32x4d', pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        n_features = model.fc.in_features
        # model.fc = nn.Linear(n_features, configs.dataset.n_classes)

    elif configs.model.name.startswith("se_resnext50_32x4d"):
        #model = se_resnext50_32x4d_clw(configs.num_classes)    # 自定义se_resnext50_32x4d, result not good

        # model = pretrainedmodels.se_resnext50_32x4d(pretrained="imagenet")
        # set_parameter_requires_grad(model, feature_extract)
        # n_features =2048
        # model.last_linear=nn.Linear(2048, configs.num_cdataset.n_classeslasses)
        # model.avg_pool = nn.AdaptiveAvgPool2d(1)  # clw note: 在senet.py中，默认是self.avg_pool = nn.AvgPool2d(7, stride=1)，这里的7是根据imagenet输入224来的，所以要改一下，否则输出就不是 32,2048,1,1了

        model = timm.create_model('seresnext50_32x4d', pretrained=True)   # not good
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, configs.num_classes)

    elif configs.model.name.startswith("se_resnext101_32x4d"):  # TODO: pretrainedmodels.se_resnext50_32x4d()
        model = pretrainedmodels.se_resnext101_32x4d(pretrained="imagenet")
        set_parameter_requires_grad(model, feature_extract)
        n_features =2048
        # model.last_linear = nn.Linear(2048, configs.dataset.n_classes)
        # model.avg_pool = nn.AdaptiveAvgPool2d(1)

    elif configs.model.name.startswith("pnasnet"):  # TODO: pretrainedmodels.se_resnext50_32x4d()
        model = pretrainedmodels.pnasnet5large(pretrained="imagenet")
        set_parameter_requires_grad(model, feature_extract)
        n_features = 4320
        model.last_linear=nn.Linear(4320, configs.dataset.n_classes)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)  # clw note: 在senet.py中，默认是self.avg_pool = nn.AvgPool2d(7, stride=1)，这里的7是根据imagenet输入224来的，所以要改一下，否则输出就不是 32,2048,1,1了

    elif configs.model.name.startswith("resnet18"):
        model = models.resnet18(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        n_features = 512
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, configs.dataset.n_classes)

    elif configs.model.name.startswith("resnet34"):
        if configs.pretrain_pth is not None:
            model = models.resnet34(pretrained=False)
            model.load_state_dict(torch.load(configs.pretrain_pth))
        else:
            model = models.resnet34(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        n_features = 512
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, configs.dataset.n_classes)

    elif configs.model.name.startswith("resnet50"):
        model = models.resnet50(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(2048, configs.dataset.n_classes)
        # print(model.relu)

        model = timm.create_model("resnet50", pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, configs.num_classes)

    elif configs.model.name.startswith("efficientnet-b0"):
        model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        model.classifier = nn.Linear(model.classifier.in_features, configs.dataset.n_classes)
    elif configs.model.name.startswith("efficientnet-b2"):
        model = timm.create_model('tf_efficientnet_b2_ns', pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        model.classifier = nn.Linear(model.classifier.in_features, configs.dataset.n_classes)
    elif configs.model.name.startswith("efficientnet-b3"):
        #model = timm.create_model('tf_efficientnet_b3_ns', pretrained=True, num_classes=configs.num_classes, drop_path_rate=configs.drop_out_rate, drop_rate=configs.drop_out_rate)
        model = timm.create_model('tf_efficientnet_b3_ns', pretrained=True, num_classes=configs.dataset.n_classes, drop_path_rate=configs.drop_out_rate, drop_rate=configs.drop_out_rate)
        set_parameter_requires_grad(model, feature_extract)
        model.classifier = nn.Linear(model.classifier.in_features, configs.dataset.n_classes)
    elif configs.model.name.startswith("efficientnet-b4"):
        #model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=configs.num_classes, drop_path_rate=configs.drop_out_rate)  # drop_path_rate=0.2~0.5
        model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=configs.dataset.n_classes)
        set_parameter_requires_grad(model, feature_extract)
        model.classifier = nn.Linear(model.classifier.in_features, configs.dataset.n_classes)
    elif configs.model.name.startswith("efficientnet-b5"):
        model = timm.create_model('tf_efficientnet_b5_ns', pretrained=True, num_classes=configs.dataset.n_classes, drop_path_rate=0.2)
        set_parameter_requires_grad(model, feature_extract)
        n_features = model.classifier.in_features
        # model.classifier = nn.Linear(model.classifier.in_features, configs.dataset.n_classes)
    elif configs.model.name.startswith("vit_base_patch16_384"):
        model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=configs.dataset.n_classes)  # , drop_rate=0.1)
        set_parameter_requires_grad(model, feature_extract)
        model.head = nn.Linear(model.head.in_features, configs.dataset.n_classes)
    elif configs.model.name.startswith("vit_large_patch16_384"):
        model = timm.create_model('vit_large_patch16_384', pretrained=True, num_classes=configs.dataset.n_classes)
        set_parameter_requires_grad(model, feature_extract)
        model.head = nn.Linear(model.head.in_features, configs.dataset.n_classes)
    elif configs.model.name.startswith('vit_base_resnet50_384'):
        model = timm.create_model('vit_base_resnet50_384', pretrained=True, num_classes=configs.dataset.n_classes)
        set_parameter_requires_grad(model, feature_extract)
        model.head = nn.Linear(model.head.in_features, configs.dataset.n_classes)

    ######AugReg series
    elif configs.model.name.startswith('vit_large_r50_s32_384'):
        model = timm.create_model('vit_large_r50_s32_384', num_classes=configs.dataset.n_classes)
        if configs.model.vit.load_checkpoint != None:
            filename = configs.model.vit.load_checkpoint
        # Non-default checkpoints need to be loaded from local files.
            if not tf.io.gfile.exists(f'{filename}.npz'):
                tf.io.gfile.copy(f'gs://vit_models/augreg/{filename}.npz', f'{filename}.npz')
            timm.models.load_checkpoint(model, f'{filename}.npz')
            set_parameter_requires_grad(model, feature_extract)
            model.head = nn.Linear(model.head.in_features, configs.dataset.n_classes)

    elif configs.model.name == 'shufflenetv2_x0_5':  # clw modify
        model = models.shufflenet_v2_x0_5(pretrained=True)  # clw modify
        set_parameter_requires_grad(model, feature_extract)
        model.fc = nn.Linear(model.fc.in_features, configs.dataset.n_classes)
        ####
        mean=0
        std=0.01
        bias=0
        nn.init.normal_(model.fc.weight, mean, std)
        if hasattr(model.fc, 'bias') and model.fc.bias is not None:
            nn.init.constant_(model.fc.bias, bias)
        ####

    elif configs.model.name == 'shufflenet_v2_x1_0':  # clw modify
        model = models.shufflenet_v2_x1_0(pretrained=False)  # clw modify
        #model = models.shufflenet_v2_x1_0(pretrained=True)  # clw modify
        set_parameter_requires_grad(model, feature_extract)
        model.fc = nn.Linear(model.fc.in_features, configs.dataset.n_classes)
        ####
        mean=0
        std=0.01
        bias=0
        nn.init.normal_(model.fc.weight, mean, std)
        if hasattr(model.fc, 'bias') and model.fc.bias is not None:
            nn.init.constant_(model.fc.bias, bias)
        ####

    elif configs.model.name == 'shufflenetv2_x1_5':  # clw modify
        model = models.shufflenet_v2_x1_5(pretrained=False)  # clw modify
        set_parameter_requires_grad(model, feature_extract)
        model.fc = nn.Linear(model.fc.in_features, configs.dataset.n_classes)
        ####
        mean=0
        std=0.01
        bias=0
        nn.init.normal_(model.fc.weight, mean, std)
        if hasattr(model.fc, 'bias') and model.fc.bias is not None:
            nn.init.constant_(model.fc.bias, bias)
        ####

    return model , n_features

#I didn't use it
class MultiTaskModel(nn.Module):
    """
    Creates a MTL model with the encoder from "arch" and with dropout multiplier ps.
    """
    def __init__(self, config,ps=0.5):
        super(MultiTaskModel,self).__init__()
        self.model,n_features = get_model(config)
                #fastai function that creates an encoder given an architecture
        self.multitask = config.model.multitask
        self.n_classes = config.dataset.multi_task
        if  config.model.multitask:
            for j,i in enumerate(config.dataset.multi_task):
                setattr(self,'fc_'+str(j), ClassifierHead(n_features,i)) 
        else:
            self.fc = ClassifierHead(n_features,config.dataset.n_classes)

    def forward(self,x):
        x = self.model.forward_features(x)
        if self.multitask:
            return [ getattr(self, 'fc_'+str(i))(x) for i in range(len(self.n_classes))]
        else:
            x = self.fc(x)
            return x
