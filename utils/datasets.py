import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from data.default_path import DatasetCatalog
import os.path as osp
from pycocotools.coco import COCO
from collections import defaultdict
import cv2
import io

from utils.mc_reader import MemcachedReader

from utils.augmentations import *



def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)

    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path,img_size=416,augment=False,use_mix=True, multiscale=True, normalized_labels=True,data_type="coco_train"):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        self.property = DatasetCatalog.get(data_type)
        self.root = self.property['root']
        json_root  = self.property['json_file']
        self.annotations = COCO(json_root)
        self.coco_name2id()

        self.img_files = self.update_file()


        self.cat2label = {item:idx for idx,item in enumerate(self.annotations.getCatIds())}
        #update the img_files

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.data_type = data_type
        self.use_mix= use_mix
        self.mix_iter = 0

        self.reader = MemcachedReader()

    def coco_name2id(self):
        keys = self.annotations.imgs.keys()
        self.label = defaultdict(int)
        for key in keys:
            id_info = self.annotations.imgs[key]
            self.label[id_info['file_name']] = id_info['id']

    def annotation_build(self,img_key):
        img_id = self.label[img_key]
        an_id = self.annotations.getAnnIds(img_id)

        label_matrix = []
        for id in an_id:
            an = self.annotations.anns[id]
            class_type = an['category_id']
            bbox = an['bbox']
            label = self.cat2label[class_type]
            label_matrix.append([label,*bbox])
        label_matrix = np.asarray(label_matrix).astype(np.float)
        return label_matrix

    def update_file(self):
        keys = self.label.keys()
        self.new_keys = []
        for img_key in keys:
            key = self.label[img_key]
            an_id  = self.annotations.getAnnIds(key)
            if len(an_id) !=0:
                self.new_keys.append(img_key)
        return self.new_keys

    def open(self,img):
        try:
            filebytes = self.reader(img)
            buff = io.BytesIO(filebytes)
            image = Image.open(buff).convert('RGB')
        except:
            img = img
            image = Image.open(img).convert('RGB')
        return image

    def get_mixup_img(self, read_cnt):
        mixup_idx = np.random.randint(1, len(self.img_files))
        mixup_img = self.img_files[(read_cnt + mixup_idx) % len(self.img_files)]
        return mixup_img
    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        label_matrix = self.annotation_build(img_path)


        img_path = osp.join(self.root,img_path)



        #already rgb
        img = np.asarray(self.open(img_path))

        img_label = label_matrix[:,0]
        img_scores = np.ones_like(img_label)
        img_target = label_matrix[:,1:]


        if self.use_mix:
            mix_id = self.get_mixup_img(index)
            mix_img_path = osp.join(self.root, mix_id)
            mix_label_matrix = self.annotation_build(mix_id)

            mix_label = mix_label_matrix[:,0]
            mix_scores = np.ones_like(mix_label)
            mix_target = mix_label_matrix[:,1:]
            mix_img = np.asarray(self.open(mix_img_path))
            img,gtboxes, gtlabels, gtscores,gt_mix_index = image_mixup(img,img_target,img_label,img_scores,mix_img,mix_target,mix_label,mix_scores)
            gtscores = torch.from_numpy(gtscores).float()
            gt_mix_index = torch.from_numpy(gt_mix_index).float()
        else:
            gtlabels = img_label
            gtboxes = img_target
            gtscores = img_scores
            gt_mix_index = torch.zeros(img_target.shape[0])

            h,w,c = img.shape

            gtboxes[:,0]/=w
            gtboxes[:,1]/=h
            gtboxes[:,2]/=w
            gtboxes[:,3]/=h
            gtscores = torch.from_numpy(gtscores).float()
        if self.augment:
            img,gtboxes, gtlabels, gtscores  = image_augment(img, gtboxes, gtlabels,
                          gtscores, self.img_size,pixel_means)
            # img = img.astype(np.uint8)
            # h,w,c  = img.shape
            # gtboxes[:, 0] = gtboxes[:, 0] * w
            # gtboxes[:, 1] = gtboxes[:, 1] * h
            # gtboxes[:, 2] = gtboxes[:, 2] * w
            # gtboxes[:, 3] = gtboxes[:, 3] * h
            #
            # gtboxes[:,2] +=gtboxes[:,0]
            # gtboxes[:,3] +=gtboxes[:,1]
            # for boxes in gtboxes.astype(np.int):
            #     img = cv2.rectangle(img,tuple(boxes[:2]),tuple(boxes[2:]),(0,0,255),3)
            # cv2.imshow("fuck",img[:,:,::-1])
            # cv2.waitKey()
        img  = ToTensor(img)
        _, h, w = img.shape


        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)


        # Pad to square resolution
        # using
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        targets = None

        label_matrix = np.concatenate([gtlabels[:,None],gtboxes],axis=1)

        if (label_matrix.shape[0] != 0):
            boxes = torch.from_numpy(label_matrix)
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * boxes[:, 1]
            y1 = h_factor * boxes[:, 2]
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3])
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4])
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h


            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes
        # Apply augmentations
        if self.use_mix:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
        return img_path, img, targets,gtscores,gt_mix_index




        #pad_img

        # return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets,gt_scores,gt_mix_index = list(zip(*batch))

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        if len(targets) !=0:
            targets = torch.cat(targets, 0)
            gt_scores = torch.cat(gt_scores,0)
            gt_mix_index = torch.cat(gt_mix_index,0)
        else:
            targets = None
            gt_scores = None
            gt_mix_index  =None

            # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets,gt_scores,gt_mix_index


    def __len__(self):
        return len(self.img_files)



class InferenceDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=False,data_type="coco_train"):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        self.property = DatasetCatalog.get(data_type)
        self.root = self.property['root']

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()


        img_path = osp.join(self.root,img_path)

        # img = cv2.imread(img_path)
        # label_matrix = label_matrix.astype(np.int)
        # label_matrix[:,3] +=label_matrix[:,1]
        # label_matrix[:,4] +=label_matrix[:,2]
        # for label in label_matrix:
        #     img = cv2.rectangle(img,tuple(label[1:3]),tuple(label[3:]),(0,0,255),3)
        #     cv2.imshow("fuck", img)
        #     cv2.waitKey()
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))


        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape

        # Pad to square resolution
        # using
        img, pad = pad_to_square(img, 0)

        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        return img_path, img,pad

    def collate_fn(self, batch):
        paths, imgs, pad = list(zip(*batch))
        # Remove empty placeholder targets
            # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, pad


    def __len__(self):
        return len(self.img_files)