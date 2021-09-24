from typing import Tuple, Union

import pathlib
from PIL import Image,ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torchvision
import yacs.config
import pandas as pd
import json,cv2,pickle

from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from pytorch_image_classification import create_transform


import os
from glob import glob
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
from itertools import chain

def get_files(root,mode,label_map_dir):
    if mode == "test":
        files = []
        for img in os.listdir(root):
            files.append(root + img)
        files = pd.DataFrame({"filename":files})
        return files
    else:
        all_data_path, labels = [], []
        image_folders = list(map(lambda x: root + x, os.listdir(root)))
        # print("image_folders",image_folders)
        all_images = list(chain.from_iterable(list(map(lambda x: glob(x + "/*"), image_folders))))
        # print("all_images",all_images)
        if mode == "val":
            print("loading val dataset")
        elif mode == "train":
            print("loading train dataset")
        else:
            raise Exception("Only have mode train/val/test, please check !!!")
        if os.path.exists(label_map_dir):
            with open(label_map_dir, 'rb')  as f:
                label_dict=pickle.load(f)
            for file in tqdm(all_images):
                all_data_path.append(file)
                name=file.split(os.sep)[-2]
                labels.append(label_dict[name])
        else:
            label_dict={}
            for file in tqdm(all_images):
                all_data_path.append(file)
                name=file.split(os.sep)[-2] #['', 'data', 'nextcloud', 'dbc2017', 'files', 'images', 'train', 'Diego_Rivera', 'Diego_Rivera_21.jpg']
                # print(name)
                if name not in label_dict:
                    label_dict[name]=len(label_dict)
                labels.append(label_dict[name])
            pickle.dump(label_dict,open(label_map_dir, 'wb'))
            # labels.append(int(file.split(os.sep)[-2]))
        print(label_dict)
        all_files = pd.DataFrame({"filename": all_data_path, "label": labels})
        return all_files

class SubsetDataset(Dataset):
    def __init__(self, subset_dataset, transform=None):
        self.subset_dataset = subset_dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset_dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset_dataset)

def getLabelmap(label_list):
    label_map={}
    for i in label_list:
        if i not in label_map.keys():
            label_map[i]=len(label_map)
    print(label_map)
    return label_map

def get_img(imgsrc):
    im_bgr = cv2.imread(imgsrc)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    return im_rgb
def get_img2(imgsrc):
    img = np.asarray(Image.open(imgsrc).convert('RGB'))
    return img
    
class LabelData(Dataset):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame,configs,istrain=False, transforms=None):
        self.files = [configs.dataset.dataset_dir +"/"+ file for file in train_df["filename"].values]
        self.y1 = train_df["artist"].values.tolist()
        self.label_map1=getLabelmap(self.y1)
        self.y2 = train_df["style"].values.tolist()
        self.label_map2=getLabelmap(self.y2)
        self.y3 = train_df["genre"].values.tolist()
        self.label_map3=getLabelmap(self.y3)
        if istrain:
          self.y1 = train_df["artist"].values.tolist()
          self.y2 = train_df["style"].values.tolist()
          self.y3 = train_df["genre"].values.tolist()
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

class MyDataset(Dataset):
    def __init__(self, df, data_root, transforms=None, output_label=True,data_type=None,soft=False,n_class=50,label_smooth=False,epsilon=0.2,is_df=False):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.data_type = data_type
        self.is_df=is_df
        
        self.output_label = output_label
        
        if output_label == True:
            self.labels = self.df['label'].values
        if soft==True:
            self.labels = np.identity(n_class)[self.labels].astype(np.float32)
            if label_smooth:
                self.labels = self.labels*(1 - epsilon)+ np.ones_like(self.labels) * epsilon / n_class

            
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        # get labels
        if self.output_label:
            target = self.labels[index]
          
        # img  = get_img("{}/{}".format(self.data_root, self.df.loc[index]['filename']))
        if self.is_df:       
            img  = get_img2("{}".format(self.df.loc[index]['file']))
        else:
          if self.data_type=='wiki22':
                img = get_img2("{}/{}".format(self.data_root, self.df.loc[index]['image']))
          else:
            img  = get_img2("{}".format(self.df.loc[index]['filename']))

        if self.transforms:
            img = self.transforms(image=img)['image']
        
        if self.output_label == True:
            return img, target
        else:
            return img

#append unlabel data and pesudo labels
#todo: acce this part
class pesudoMyDataset(MyDataset):
    def __init__(self, df,unlabel_df, data_root,model, device,soft=False,transforms=None, output_label=True,n_class=50,label_smooth=False,epsilon=0.2,is_df=False):
        
        super(pesudoMyDataset, self).__init__(
            df=df, data_root=data_root,transforms=transforms,soft=soft,n_class=n_class,label_smooth=label_smooth,epsilon=epsilon,is_df=is_df)
        self.df=pd.concat([self.df,unlabel_df]).reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        
        self.output_label = output_label
        self.labels=list(self.labels)
        names='filename'
        if is_df:
            names='file'
        
        images=[]
        if output_label == True:
            for i,image in enumerate(unlabel_df[names].values):
                image=get_img2("{}".format(image))
                image=self.transforms(image=image)['image'].numpy()
                # image = np.array((image.transpose(2,0,1)-127.5)/127.5,dtype=np.float32)
                images.append(image)
                if (i%64==0 and i!=0) or i==len(unlabel_df[names].values)-1: 
                    with torch.no_grad():
                        output = model(torch.from_numpy(np.array(images)).to(device))
                    if soft is True:
                        logit = F.softmax(output,dim=1).squeeze(0).cpu().numpy()
                    else:
                        logit = torch.max(output,dim=1)[1].item()
                    images=[]
                    self.labels.extend(logit)
                #normalize
                       
        # print(self.labels[-10:])


class Data(Dataset):
    def __init__(self, df: pd.DataFrame,label_map,configs, transforms=None):
        if configs.dataset.subname == 'K100':
          self.files = [configs.dataset.dataset_dir +"/"+ file for file in df["filename"].values]
        else:
          self.files = [configs.dataset.dataset_dir +"/"+ file for file in df["image"].values]
        if configs.dataset.subname == 'K100':
          self.y = df["artist"].values.tolist()
        else:
          self.y = df["label"].values.tolist()
        self.label_map=label_map
        self.transforms = transforms
        self.albuAug=configs.augmentation.use_albumentations
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        
        label = self.label_map[self.y[i]]
        if self.transforms is not None:
            if self.albuAug:
                img  = get_img2(self.files[i])
                img = self.transforms(image=img)['image']
            else:
                img = Image.open(self.files[i]).convert('RGB')
                img = self.transforms(img)
            
        return img, label

def create_dataset(config: yacs.config.CfgNode,
                   is_train: bool) -> Union[Tuple[Dataset, Dataset], Dataset]:
    if config.dataset.name in [
            'CIFAR10',
            'CIFAR100',
            'MNIST',
            'FashionMNIST',
            'KMNIST',
    ]:
        module = getattr(torchvision.datasets, config.dataset.name)
        if is_train:
            if config.train.use_test_as_val:
                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = module(config.dataset.dataset_dir,
                                       train=is_train,
                                       transform=train_transform,
                                       download=True)
                test_dataset = module(config.dataset.dataset_dir,
                                      train=False,
                                      transform=val_transform,
                                      download=True)
                return train_dataset, test_dataset
            else:
                dataset = module(config.dataset.dataset_dir,
                                 train=is_train,
                                 transform=None,
                                 download=True)
                val_ratio = config.train.val_ratio
                assert val_ratio < 1
                val_num = int(len(dataset) * val_ratio)
                train_num = len(dataset) - val_num
                lengths = [train_num, val_num]
                train_subset, val_subset = torch.utils.data.dataset.random_split(
                    dataset, lengths)

                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = SubsetDataset(train_subset, train_transform)
                val_dataset = SubsetDataset(val_subset, val_transform)
                return train_dataset, val_dataset
        else:
            transform = create_transform(config, is_train=False)
            dataset = module(config.dataset.dataset_dir,
                             train=is_train,
                             transform=transform,
                             download=True)
            return dataset
    elif config.dataset.name == 'ImageNet':
        if is_train:

            if config.dataset.type == 'dir':
                dataset_dir = pathlib.Path(config.dataset.dataset_dir).expanduser()
                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = torchvision.datasets.ImageFolder(
                    dataset_dir / 'train', transform=train_transform)
                val_dataset = torchvision.datasets.ImageFolder(dataset_dir / 'val',
                                                            transform=val_transform)
                return train_dataset, val_dataset
            elif config.dataset.type == 'df':
                if config.model.multitask:
                  df = pd.read_csv(config.dataset.cvsfile_train)
                  if config.dataset.subname=="K100":
                    train_df, valid_df = train_test_split(df, stratify=df["artist"].values)
                  else:
                    train_df, valid_df = train_test_split(df, stratify=df["label"].values)
                  train_dataset = LabelData(train_df,valid_df,config,create_transform(config, is_train=True))
                  val_dataset = LabelData(train_df,valid_df,config,create_transform(config, is_train=False))
                  return train_dataset, val_dataset

                df = pd.read_csv(config.dataset.cvsfile_train)
                label_map={}
                if config.dataset.jsonfile:
                    with open(config.dataset.jsonfile, "r") as f:
                        label_map = json.load(f)
                else:
                    if config.dataset.subname == 'K100':
                      label_map = getLabelmap(df['artist'])
                    else:
                      label_map = getLabelmap(df['label'])
                    # label_map = {int(v): k for k, v in label_map.items()}
                if config.train.use_test_as_val:
                    train_df = pd.read_csv(config.dataset.cvsfile_train)
                    valid_df = pd.read_csv(config.dataset.cvsfile_test)
                else:
                    if config.dataset.subname == 'K100':
                        train_df, valid_df = train_test_split(df, stratify=df["artist"].values)
                    else:
                        train_df, valid_df = train_test_split(df, stratify=df["label"].values)
                train_transform = create_transform(config, is_train=False)    
                # label_map = {int(k): v for k, v in label_map.items()}
                val_transform = create_transform(config, is_train=False)
                train_ds = Data(train_df,label_map,config,train_transform)
                valid_ds = Data(valid_df,label_map,config,val_transform)
                return train_ds, valid_ds
            else:
                raise ValueError() 
        else:
            if config.dataset.type == 'df':
              
                df = pd.read_csv(config.dataset.cvsfile_train)
                if config.dataset.subname == 'K100':
                    label_map = getLabelmap(df['artist'])
                else:
                    label_map = getLabelmap(df['label'])
                df = pd.read_csv(config.dataset.cvsfile_test)
                val_transform = create_transform(config, is_train=False)
                valid_ds = Data(df,label_map,config,val_transform)
                
                return valid_ds
            if config.dataset.type=='dir'and config.augmentation.use_albumentations:
                test_clean=get_files(config.dataset.dataset_dir+'val/','train',config.train.output_dir+'/label_map.pkl')
                test_dataset = MyDataset(test_clean,config.dataset.dataset_dir+'val/',
                        transforms=create_transform(config, is_train=False),is_df=config.dataset.type=='df')
                return test_dataset
              
            dataset_dir = pathlib.Path(config.dataset.dataset_dir).expanduser()
            val_transform = create_transform(config, is_train=False)
            val_dataset = torchvision.datasets.ImageFolder(dataset_dir / 'val',
                                                            transform=val_transform)
            return val_dataset
    else:
        raise ValueError()



