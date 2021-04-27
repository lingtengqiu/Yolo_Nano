#!/usr/bin/python3
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : lingteng qiu
# * Email         : 
# * Create time   : 2018-11-06 09:52
# * Last modified : 2018-11-06 09:52
# * Filename      : board.py
# * Description   : before we use the visualizer tool like tensorboardX and visdom but Know we change the method we only use the tensorboardX 
# **********************************************************
from datetime import datetime
import os 
import glob
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import os.path as osp
import socket
class Visualizer(object):
    def __init__(self,save_dir_root = None):
        if(save_dir_root == None):
            save_dir_root = osp.join(osp.dirname(osp.abspath(__file__)))
        runs = sorted(glob.glob(osp.join(save_dir_root,'logger','logger_*')))
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
        save_dir = osp.join(save_dir_root,'logger','logger_{}'.format(run_id+1))
        log_dir = osp.join(save_dir,'models',datetime.now().strftime("%b%d_%H_%M_%S")+"_"+socket.gethostname())
        self.log_dir = log_dir
        self.writer = SummaryWriter(logdir = log_dir)
    def line(self,tag,scalar_value,ite):
        '''
        tag : axis of y
        scalar_value : value
        ite : which step 
        '''
        self.writer.add_scalar(tag,scalar_value,ite)
    def lines(self,tag,ite,**kwargs):
        '''
        tag,and ite the same as line
        but for kwargs,example as followed:
        kwargs---> is a dictionary
        kwargs = {
        "xsinx" : val,
        "xcosx" : val,
        "xtanx" : val......
        }
        '''
        self.writer.add_scalars(tag,kwargs,ite)
    def images(self,tag,tensor,n_row,ite,normal = False,xmin = None,xmax =None):
        '''
        @tag:the name of graph_window
        @tensor:(B,N,H,W)
        @n_row:each row to show the image,
        '''
        assert len(tensor.shape) == 4,"your input size must be 4 dim"
        if xmin!=None and xmax!=None:
            grid_img = make_grid(tensor[:n_row].clone().cpu().data,n_row,
                    normalize = False,range =(xmin,xmax))
        grid_img = make_grid(tensor[:n_row].clone().cpu().data,n_row,
            normalize = normal)
        self.writer.add_image(tag,grid_img,ite)
    def write_group(self,list,step):
        for e in list:
            self.line("train/{}".format(e[0]),float(e[1]),step)
    def close(self):
        self.writer.close()
if __name__ =="__main__":
    vis = Visualizer()

