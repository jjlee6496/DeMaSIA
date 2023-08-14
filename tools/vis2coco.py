# Reference: https://github.com/caodoanh2001/Convert-Visdrone-to-COCO/blob/main/vis2coco.py
#            https://github.com/fcakyon/small-object-detection-benchmark/blob/main/visdrone/visdrone_to_coco.py
#            https://github.com/open-mmlab/mmtracking/blob/master/tools/convert_datasets/mot/mot2coco.py
#
# This script converts VisDrone-MOT labels into COCO style.
# Official website of the VisDrone-MOT dataset: https://github.com/VisDrone/VisDrone-Dataset
#
# Label format of VisDrone-MOT dataset:
#   annotations:
#       <frame_index> # The frame index of the video frame. starts from 1 but COCO style starts from 0,
#       <target_id>, <x1>, <y1>, <w>, <h>,
#       <score> # whether to be ignored, 
#       <class_id>,
#       <truncation> # 잘린 정도, (i.e., no truncation = 0 (truncation ratio 0%), and partial truncation = 1 (truncation ratio 1% ~ 50%))
#       <occlusion> # 가려짐 정도, (i.e., no occlusion = 0 (occlusion ratio 0%), partial occlusion = 1 (occlusion ratio 1% ~ 50%), 
#		              and heavy occlusion = 2 (occlusion ratio 50% ~ 100%))
#
# Classes in VisDrone:
#   0: 'ignored regions'
#   1: 'pedestrian'
#   2: 'people'
#   3: 'bicycle'
#   4: 'car'
#   5: 'van'
#   6: 'truck'
#   7: 'tricycle'
#   8: 'awning-tricycle'
#   9: 'bus'
#   10: 'motor',
#   11: 'others'
#
#   USELESS classes are not included into the json file.

import argparse
import os
import mmcv

import os.path as osp
from collections import defaultdict

from PIL import Image
from tqdm import tqdm

USELESS = [-1, 0, 2, 3, 7, 8, 10, 11] # pedestrian: 1,  vehicles: 4, 5, 6, 9

def parse_args():
    parser = argparse.ArgumentParser('Convert VisDrone-MOT labels to COCO-VID format')
    parser.add_argument('-i', '--input', help='path of VisDrone data')
    parser.add_argument(
        '-o', '--output', help='path to save coco formatted label file')

    return parser.parse_args()


def parse_gts(gts):
    outputs = defaultdict(list)
    for gt in gts:
        gt = gt.strip().split(',')
        frame_id, ins_id = map(int, gt[:2])
        category_id = int(gt[7])
        bbox = list(map(float, gt[2:6]))
        
        area=bbox[2] * bbox[3]
        conf = float(gt[6])
        if gt[8] == 0 and gt[9] == 1:
            visibility = float(0.75)
        elif gt[8] == 1 and gt[9] == 0:
            visibility = float(0.75)
        elif gt[8] == 1 and gt[9] == 1:
            visibility = float(0.5) #0.5~0.9
        else:
            visibility = float(0.25) #0~0.5

        if category_id in USELESS:
            continue
        
        if conf == 0:
            continue
                
        anns = dict(
            category_id=category_id,
            bbox=bbox,
            area=area,
            iscrowd=False,
            visibility=visibility,
            mot_instance_id=ins_id,
            mot_conf=conf,
            mot_class_id=category_id)
        outputs[frame_id].append(anns)
    return outputs


# instance id 의심됨
def main():
    args = parse_args()
    if not osp.isdir(args.output):
        os.makedirs(args.output)
    
    train_dir = 'VisDrone2019-MOT-train'
    val_dir = 'VisDrone2019-MOT-val'
    test_dir = 'VisDrone2019-MOT-test-dev'
    sets = [train_dir, val_dir, test_dir]
    vid_id, img_id, ann_id = 1, 1, 1
    
    for subset in sets:
        ins_id = 0
        print(f'Converting {subset} set to COCO format')
        in_folder = osp.join(args.input, subset)
        out_file = osp.join(args.output, f'{subset}_cocoformat.json')
        outputs = defaultdict(list)
        outputs['categories'] = [
            dict(id=1, name='pedestrian'),
            dict(id=4, name='car'),
            dict(id=5, name='van'),
            dict(id=6, name='truck'),
            dict(id=9, name='bus')
            ]

        video_names = os.listdir(osp.join(in_folder, 'sequences'))
        for video_name in tqdm(video_names):
            
            # basic params
            # parse_gt = 'test' not in subset
            ins_maps = dict()
            
            video_folder = osp.join(in_folder, 'sequences' , video_name)
            img_names = sorted(os.listdir(video_folder))
            temp_img = Image.open(osp.join(video_folder, img_names[0]))
            width, height = temp_img.size
            
            video = dict(
                id = vid_id,
                name=video_name,
                #fps=??,
                width=width,
                height=height)
            
            gts = mmcv.list_from_file(f'{in_folder}/annotations/{video_name}.txt')
            img2gts = parse_gts(gts)
            
            for frame_id, name in enumerate(img_names):
                img_name = f'{subset}/sequences/{video_name}/{name}'
                mot_frame_id = int(name.split('.')[0])
                image = dict(
                    id=img_id,
                    video_id=vid_id,
                    file_name=img_name,
                    height=height,
                    width=width,
                    frame_id=frame_id,
                    mot_frame_id=mot_frame_id
                )
                gts = img2gts[mot_frame_id]
                for gt in gts:
                    gt.update(id=ann_id, image_id=img_id)
                    mot_ins_id = gt['mot_instance_id']
                    
                    if mot_ins_id in ins_maps:
                        gt['instance_id'] = ins_maps[mot_ins_id]
                    else:
                        gt['instance_id'] = ins_id
                        ins_maps[mot_ins_id] = ins_id
                        ins_id += 1
                    outputs['annotations'].append(gt)
                    ann_id += 1
                outputs['images'].append(image)
                img_id += 1
            outputs['videos'].append(video)
            vid_id += 1
            outputs['num_instances'] = ins_id
            print(f'{subset} has {ins_id} instances.')
            mmcv.dump(outputs, out_file)
            print(f'Done! Saved as {out_file}')
  

if __name__ == '__main__':
    main()