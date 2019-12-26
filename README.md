# PyTorch-YOLO_Nano
A minimal PyTorch implementation of YOLO_Nano
- [Yolo_Nano](https://arxiv.org/abs/1910.01271)
- [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/abs/1902.04103v3)  
##### trick In here I have done  
[Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/abs/1902.04103v3) tell us that fixup in object detection can increase the mAP, So I realize it and test in result. 
- [x] Data Augmentation  
- [x] Fixup  
- [x] Cosine lr decay  
- [x] Warm up
##### Download COCO
    $ cd data/
    $ bash get_coco_dataset.sh
## Module Pipeline
![Pipeline](assets/structure.png)
## training
```bash
bash train.sh
```
better Para:  
   --epochs 120  
   --batch_size 8  
   --model_def ./config/yolo-nano_person.cfg  
   --warm_up_lr 2  
   --lr 2.5e-4  
   --fix_up True  
## Result
In this engineer we only train our model using coco-train person class  
we compare with yolov-3，yolo-tiny. We got competitive results.  

Methods |mAP@50|mAP|weights| Model 
:--------------:|:--:|:--:|:--: |:--:  
 yolov3(paper)      | 74.4 |40.3 | 204.8M|  [Google Disk](https://pjreddie.com/media/files/yolov3.weights)
 yolov3-tiny(paper) | 38.8 |15.6 | 35.4M |  [Google Disk](https://pjreddie.com/media/files/yolov3-tiny.weights)
 yolo-nano          | 55.6 |27.7 | 22.0M |  [Baidu WebDisk](https://pan.baidu.com/s/1Rp0is2LqA91XwjRc41mGaw)  
 
Baidu WebDisk Key: p2j3
## Ablation Result
 Augmentation| fixup | mAP 
:--------------:|:--:|:--:
No|No|54.3
Yes|No|53.9
No|YES|55.6
YES|YES|54.8  

## Inference Result
![Pipeline](assets/show.jpg)
