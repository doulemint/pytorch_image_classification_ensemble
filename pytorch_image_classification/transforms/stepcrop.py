# import albumentations.augmentations.functional as F
import numpy as np
import random
import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform


class StepcropAlbu(ImageOnlyTransform):
    """Rotate the input by 90 degrees zero or more times.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(self,n=8,pos=0,*args, **kwargs):
        super(StepcropAlbu, self).__init__(*args, **kwargs)
        self.n = n
        self.pos=pos

    def apply(self, img,**params):
        return self.DataAugmentation3(img,self.n,self.pos)

    def DataAugmentation3(self,image,n=8,pos=0):
        n = random.randrange(2,16)
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
            patch_x = patch_initial[0]
            patch_y = patch_initial[1]
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
        im_h = cv2.vconcat(iv_list)
        #print(im_h.shape)
        width = patch_size*(n-1)
        #print(width)
        image = cv2.resize(image,(width,width))
        im_v=cv2.hconcat(im_list)
        #print(im_v.shape)
        if pos==0:
            im_v = cv2.vconcat([image,im_v])
            #print(im_v.shape)
            img = cv2.hconcat([im_v,im_h])
        elif pos==1:
            im_v = cv2.vconcat([im_v,image])
            #print(im_v.shape)
            img = cv2.hconcat([im_v,im_h])
        elif pos==2:
            im_v = cv2.vconcat([im_v,image])
            #print(im_v.shape)
            img = cv2.hconcat([im_h,im_v])
        elif pos==3:
            im_v = cv2.vconcat([image,im_v])
            #print(im_v.shape)
            img = cv2.hconcat([im_h,im_v])
        return img

class CornerCrop(ImageOnlyTransform):
    
    def __init__(self, *args, **kwargs):
        super(CornerCrop, self).__init__(*args, **kwargs)

    def apply(self, img, **params):
        return self.DataAugmentation3(img)

    def DataAugmentation3(self,image):
        img = None
        try:
            n = 5
            patch_scale = 0.125 #find 8 patches on corner
            smaller_dim = np.min(image.shape[0:2])
            image = cv2.resize(image,(smaller_dim,smaller_dim))
            im_list = []
            im_list2 = []
            w = image.shape[0]
            h = image.shape[1]
            dw = int(w/5)
            dh = int(h/5)
            patch_initial = np.array([0,0])
            patch_size = int(patch_scale * smaller_dim)
            for i in range(n):
                for j in range(n):
                    if (i%2)==0 and (j%2)==0:
                        patch_x = i * dw
                        patch_y = j * dh
                        patch_image = image[patch_x:patch_x+patch_size,patch_y:patch_y+patch_size]
                        # patch_image = zoomin(patch_image,3)
                        #print(patch_image.shape)
                        im_list.append(patch_image)
                    elif not i%2==0 :
                        patch_x = i * dw
                        patch_y = j * dh
                        patch_image = image[patch_x:patch_x+patch_size,patch_y:patch_y+patch_size]
                        # patch_image = zoomin(patch_image,3)
                        #print(patch_image.shape)
                        im_list2.append(patch_image)
            im_l = im_list2[0:7]
            im_h = cv2.vconcat(im_list[1:])
            width = patch_size*7
            image = cv2.resize(image,(width,width))
            im_v = cv2.vconcat([image,cv2.hconcat(im_l)])
            img = cv2.hconcat([im_v,im_h])
        except Exception as e:
            print(str(e))
        return img



class RandomCropStitch(ImageOnlyTransform):
    
    def __init__(self, *args, **kwargs):
        super(RandomCropStitch, self).__init__(*args, **kwargs)

    def apply(self, img, **params):
        return self.DataAugmentation3(img)

    def DataAugmentation3(self,image):
        n = 8
        num = 16
        im_list = []
        patch_scale = 1/n 
        smaller_dim = np.min(image.shape[0:2])
        patch_size = int(patch_scale * smaller_dim)
        #print(patch_size)
        for i in range(num):
            rang = smaller_dim - patch_size
            patch_x = random.randrange(0,rang);
            patch_y = random.randrange(0,rang);
            patch_image = image[patch_x:patch_x+patch_size,patch_y:patch_y+patch_size]
            # patch_image = zoomin(patch_image,3)
            im_list.append(patch_image)
        im_l = im_list[0:7]
        #print(len(im_l))
        im_h = cv2.vconcat(im_list[7:-1]);
        width = patch_size*7
        image = cv2.resize(image,(width,width))
        im_v = cv2.vconcat([image,cv2.hconcat(im_l)])
        img = cv2.hconcat([im_v,im_h])
        return img