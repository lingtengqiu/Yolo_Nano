from __future__ import division

from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from models.darknet import *
from models.yolo_nano_helper import YoloNano
from torch.nn.parallel import DataParallel

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import cv2
import os.path as osp

VIDEO_SIZE={
    "vid_65132":(1920,1080),
    "vid_64708": (1920, 1080),
    "multi_person":(1920,1080)
}
def str_id(cnt):
    cnt = str(cnt)
    pre=""
    for _ in range(8-len(cnt)):
        pre+='0'
    return pre+cnt

@torch.no_grad()
def inference(model, path, conf_thres, nms_thres, img_size, batch_size,data_type,video_id):
    model.eval()
    # Get dataloader
    dataset = InferenceDataset(path, img_size=img_size, augment=False, multiscale=False,data_type =data_type)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor



    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vider_write = cv2.VideoWriter(osp.join("./result", "{}.mp4".format(video_id)), fourcc, 30.0,
                                  VIDEO_SIZE[video_id])
    for batch_i, (img_id, imgs, pads) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        outputs = model(imgs)

        outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        for id,output,pad in zip(img_id,outputs,pads):
            img =cv2.imread(id)
            h,w,c = img.shape
            square_edge = max(h,w)
            ratio = square_edge/imgs.shape[-1]
            if output is None:
                vider_write.write(img)
                continue
            output = output.detach().cpu().numpy()[:]
            output[:,:4]*=ratio
            output[:,0]-= pad[0]
            output[:,1]-= pad[2]
            output[:,2]-= pad[1]
            output[:,3]-= pad[3]

            for out in output:
                category = int(out[-1])
                if category>0:
                    continue
                out = out[:4].astype(np.int).tolist()
                img = cv2.rectangle(img,tuple(out[:2]),tuple(out[2:4]),(0,0,255),3)
            vider_write.write(img)



    # Concatenate sample statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.7, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--data_type", type=str, default="coco_test", help="Dataset type")
    parser.add_argument("--video_id", type=str, default="vid_65132", help=" video id info")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)


    valid_path = data_config["valid"]


    class_names = load_classes(data_config["names"])

    # Initiate model
    if "yolov3" in opt.model_def:
        model = Darknet(opt.model_def).to(device)
        model.apply(weights_init_normal)
    else:
        kargs = get_nano_info(opt.model_def)
        model = YoloNano(**kargs).to(device)
        model.apply(weights_init_normal)
        model = DataParallel(model)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    inference(
        model,
        path=valid_path,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=1,
        data_type = opt.data_type,
        video_id = opt.video_id
    )

