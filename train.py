from __future__ import division

from models.darknet import *
from models.yolo_nano_helper import YoloNano
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from utils.dataloader_utils import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from utils.board import Visualizer
from utils.optim import get_lr_at_epoch,set_lr
from torch.nn.parallel import DataParallel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=1, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--visual", type=bool, default=True, help=" video id info")
    parser.add_argument("--lr_policy", type=str, default="cosine", help="lr_decay methods")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="base_lr")
    parser.add_argument("--max_epoch", type=float, default=120,help="base_lr")
    parser.add_argument("--warm_up_epoch", type=float, default=1, help="base_lr")
    parser.add_argument("--warm_up_lr", type=float, default=0.0, help="base_lr")
    parser.add_argument("--none_mix_epoch", type=float, default= 100, help="base_lr")
    parser.add_argument("--augment", type=bool, default= False, help="Use augment? paper said use augmentor may be get trouble result")
    parser.add_argument("--mix_up", type=bool, default= True, help="Use mix_up? paper said use mixup may be get btter result")
    opt = parser.parse_args()
    print(opt)
    if opt.visual:
        logger = Visualizer("./result")


    # logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    if "yolov3" in opt.model_def:
        model = Darknet(opt.model_def).to(device)
        model.apply(weights_init_normal)
        # If specified we start from checkpoint
        if opt.pretrained_weights:
            if opt.pretrained_weights.endswith(".pth"):
                model.load_state_dict(torch.load(opt.pretrained_weights))
            else:
                model.load_darknet_weights(opt.pretrained_weights)
    else:
        kargs = get_nano_info(opt.model_def)
        model = YoloNano(**kargs).to(device)
        model.apply(weights_init_normal)

    # Get dataloader
    dataset = ListDataset(train_path, augment=opt.augment,use_mix=opt.mix_up, multiscale=opt.multiscale_training,normalized_labels=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_init_fn_seed
    )
    opt.max_epoch*=len(dataloader)
    opt.max_epoch+=2000
    opt.warm_up_epoch *= len(dataloader)

    optimizer = torch.optim.Adam(model.parameters(),lr = opt.lr)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):
        if epoch > opt.none_mix_epoch:
            dataloader.dataset.use_mix = False
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets,img_scores,gt_mix_index) in enumerate(dataloader):


            batches_done = len(dataloader) * epoch + batch_i

            cur_lr = get_lr_at_epoch(opt,batches_done)
            set_lr(optimizer,cur_lr)

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            img_scores = Variable(img_scores.to(device),requires_grad = False)

            loss, outputs = model(imgs,targets=targets,img_scores=img_scores,gt_mix_index = gt_mix_index)
            loss.backward()

            if batches_done % opt.gradient_accumulations == 0:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()




            # ---------------   -
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
            tensorboard_log += [("loss", loss.item())]
            #logger.write_group(tensorboard_log,batches_done)
            logger.line("train/loss",loss.item(),batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"
            log_str +=f"\nBatch_LR {cur_lr}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)
            model.seen += imgs.size(0)
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                conf_thres=0.01,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=1,
                data_type="coco_test"
            )
            precision = precision[0]
            recall = recall[0]
            AP = AP[0]
            f1 = f1[0]
            ap_class = ap_class[0]
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.write_group(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
