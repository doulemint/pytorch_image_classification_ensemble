from typing import Tuple, Union

import numpy as np
import PIL.Image, cv2
import torch
import torchvision
import yacs.config


class CenterCrop:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.CenterCrop(
            config.dataset.image_size)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class Normalize:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image: PIL.Image.Image) -> np.ndarray:
        image = np.asarray(image).astype(np.float32) / 255.
        image = (image - self.mean) / self.std
        return image

class toNumpy:
    def __init__(self):
        pass

    def __call__(self, image: PIL.Image.Image) -> np.ndarray:
        return np.asarray(image)

class RandomCrop:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.RandomCrop(
            config.dataset.image_size,
            padding=config.augmentation.random_crop.padding,
            fill=config.augmentation.random_crop.fill,
            padding_mode=config.augmentation.random_crop.padding_mode)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class RandomResizeCrop:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.RandomResizedCrop(
            config.dataset.image_size)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class RandomHorizontalFlip:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.RandomHorizontalFlip(
            config.augmentation.random_horizontal_flip.prob)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)


class Resize:
    def __init__(self, config: yacs.config.CfgNode):
        self.transform = torchvision.transforms.Resize(config.tta.resize)

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.transform(data)

class StepCrop:
    def __init__(self):
        pass

    def __call__(self, data: PIL.Image.Image) -> PIL.Image.Image:
        return self.DataAugmentation3(data)

    def DataAugmentation3(self,image):
        n = 8
        im_list = []
        iv_list = []
        patch_initial = np.array([0,0])
        patch_scale = 1/n #find 5 patch on diagonal
        smaller_dim = np.min(image.shape[0:2])
        #print(smaller_dim)
        image = cv2.resize(image,((smaller_dim,smaller_dim)))
        patch_size = int(patch_scale * smaller_dim)
        #print(patch_size)
        for i in range(n):
            patch_x = patch_initial[0];
            patch_y = patch_initial[1];
            patch_image = image[patch_x:patch_x+patch_size,patch_y:patch_y+patch_size]
            #print(patch_image.shape)
            #patch_image = zoomin(patch_image,3)
            #print(patch_image.shape)
            x2 = smaller_dim - patch_x
            patch_image2 = image[x2-patch_size:x2,patch_y:patch_y+patch_size]
            #patch_image2 = zoomin(patch_image2,3)
            patch_initial = np.array([patch_x+patch_size,patch_y+patch_size])
            iv_list.append(patch_image)
            im_list.append(patch_image2)
        im_list = im_list[1:n]
        im_h = cv2.vconcat(iv_list);
        #print(im_h.shape)
        width = patch_size*(n-1)
        #print(width)
        image = cv2.resize(image,(width,width))
        im_v=cv2.hconcat(im_list)
        #print(im_v.shape)
        im_v = cv2.vconcat([image,im_v])
        #print(im_v.shape)
        img = cv2.hconcat([im_v,im_h])
        return img
class ToTensor:
    def __call__(
        self, data: Union[np.ndarray, Tuple[np.ndarray, ...]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if isinstance(data, tuple):
            return tuple([self._to_tensor(image) for image in data])
        else:
            return self._to_tensor(data)

    @staticmethod
    def _to_tensor(data: np.ndarray) -> torch.Tensor:
        if len(data.shape) == 3:
            return torch.from_numpy(data.transpose(2, 0, 1).astype(np.float32))
        else:
            return torch.from_numpy(data[None, :, :].astype(np.float32))
