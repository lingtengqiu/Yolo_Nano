import os
class DatasetCatalog(object):
    DATASETS = {
        "coco_test": {
            "img_dir": "../val2017",
            "annotations":"../annotations/person_keypoints_val2017.json"
        },
        "coco_train": {
            "img_dir": "../train2017",
            "annotations":"../annotations/person_keypoints_train2017.json"
        },
        "vid65132":{
            "img_dir":"../frame/VID20191016165132.mp4/",
            "annotations":"None"
        },
        "vid64708": {
            "img_dir": "../frame/VID20191016164708.mp4/",
            "annotations": "None"
        },
        "multi_person": {
            "img_dir": "../frame/multi_person",
            "annotations": "None"
        },
        "MOT17-07": {
            "img_dir": "../frame/MOT17-07-DPM/img1",
            "annotations": "None"
        }
    }

    @staticmethod
    def get(name):
        if "coco" in name:
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root = attrs["img_dir"],
                json_file = attrs['annotations']
            )
        elif "vid" in name:
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root = attrs["img_dir"],
                json_file = attrs['annotations']
            )
        elif "multi" in name or "MOT" in name:
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root = attrs["img_dir"],
                json_file = attrs['annotations']
            )
        else:
            raise RuntimeError("Dataset not available: {}".format(name))
        return args