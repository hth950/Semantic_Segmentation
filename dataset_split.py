
import shutil
import random
from glob import glob
import json
import ade_custom
import os

min = 0
max = 0

classes = {
            'background' : 0, 'vehicle': 0, 'bus': 0, 'truck': 0, 'policeCar': 0, 'ambulance': 0, 'schoolBus': 0, 'otherCar': 0, 
            'motorcycle': 0, 'bicycle': 0, 'twoWheeler': 0, 'pedestrian': 0, 'rider': 0, 'freespace': 0,
            'curb': 0, 'sidewalk': 0, 'crossWalk': 0, 'safetyZone': 0, 'speedBump': 0, 'roadMark': 0, 'whiteLane': 0,
            'yellowLane': 0, 'blueLane': 0, 'redLane': 0, 'stopLane': 0, 'constructionGuide': 0, 'trafficDrum': 0,
            'rubberCone': 0, 'trafficSign': 0, 'trafficLight': 0, 'warningTriangle': 0, 'fence': 0
    }
file_list = {
    'background' : [], 'vehicle': [], 'bus': [], 'truck': [], 'policeCar': [], 'ambulance': [], 'schoolBus': [], 'otherCar': [], 
        'motorcycle': [], 'bicycle': [], 'twoWheeler': [], 'pedestrian': [], 'rider': [], 'freespace': [],
        'curb': [], 'sidewalk': [], 'crossWalk': [], 'safetyZone': [], 'speedBump': [], 'roadMark': [], 'whiteLane': [],
        'yellowLane': [], 'blueLane': [], 'redLane': [], 'stopLane': [], 'constructionGuide': [], 'trafficDrum': [],
        'rubberCone': [], 'trafficSign': [], 'trafficLight': [], 'warningTriangle': [], 'fence': []
    }

root = '/data/36-3/'
img_path = root + 'img_dir/'
ann_path = root + 'ann_dir/'
img_paths = glob(img_path+'*')
ann_paths = glob(ann_path+'*')
img_move_path = root + 'img_val/'
ann_move_path = root + 'ann_val/'
img_move_path_t = root + 'img_test/'
ann_move_path_t = root + 'ann_test/'
label_path = '/data/3_label/'
label_folder = glob(label_path+'*')

def make_label_list(label_folder=[]):
    for i in range(len(label_folder)):
        label_path = label_folder[i]+'/*.json'
        label_path = glob(label_path)
        
        for j in range(len(label_path)):
            name = os.path.basename(label_path[j]).split('.')[0]
            with open(label_path[j], 'r') as f:
                data = json.load(f)
            for k in range(len(data['annotations'])):
                l_class = data['annotations'][k]['class'].lower()
                if l_class == 'background':
                    continue
                    
                if l_class in classes:
                    classes[l_class] = classes[l_class]+1
                    file_list[l_class].append(name)
    return classes, file_list

def s_label_split(img_path='', ann_path='', img_move_path='', ann_move_path='', img_move_path_t='', ann_move_path_t='', classes={}, file_list={},min=10, max=2000):
    res = {}
    keys = []
    vals = []

    for key,val in classes.items():
        vals.append(val)
        
    classes = dict(sorted(classes.items(), key=lambda x:x[1]))
    for key,val in classes.items():
        if int(val) > min and int(val) < max:
            keys.append(key)
            res[key] = val

    for key, val in res.items():
        if key in file_list:
            files = file_list[key]
            random.shuffle(files)
            val_length = int(len(files) * (8/10))
            test_length = int(len(files) * (9/10))
            
            for file1 in files[val_length:test_length]:
                img_o = img_path + file1 + '.jpg'
                ann_o = ann_path + file1 + '.png'
                if not os.path.exists(img_o) or not os.path.exists(ann_o):
                    continue
                shutil.move(img_o, img_move_path)
                shutil.move(ann_o, ann_move_path)
            
            for file2 in files[test_length:]:
                img_o = img_path + file2 + '.jpg'
                ann_o = ann_path + file2 + '.png'
                if not os.path.exists(img_o) or not os.path.exists(ann_o):
                    continue
                shutil.move(img_o, img_move_path_t)
                shutil.move(ann_o, ann_move_path_t)
    print('s_label_split finish!')

def label_split(img_path='', ann_path='', img_move_path='', ann_move_path='', img_move_path_t='', ann_move_path_t=''):
    img_paths = glob(img_path +'*')
    ann_paths = glob(ann_path +'*')
    ann_paths_name = []
    for path in ann_paths:
        ann_paths_name.append(os.path.basename(path).split('.')[0])

    random.shuffle(img_paths)

    val_length = int(len(img_paths) * (8/10))
    test_length = int(len(img_paths) * (9/10))

    for path in img_paths[val_length:test_length]:
        ann_path1 = ann_path
        if not os.path.exists(path):
            continue
        if not os.path.basename(path).split('.')[0] in ann_paths_name:
            continue
        else:
            ann_path1 = ann_path1 + os.path.basename(path).split('.')[0] + '.png'
        shutil.move(path, img_move_path)
        shutil.move(ann_path1, ann_move_path)

    for path in img_paths[test_length:]:
        ann_path2 = ann_path
        if not os.path.exists(path):
            continue
        if not os.path.basename(path).split('.')[0] in ann_paths_name:
            continue
        else:
            ann_path2 = ann_path2 + os.path.basename(path).split('.')[0] + '.png'
        shutil.move(path, img_move_path_t)
        shutil.move(ann_path2, ann_move_path_t)
    print('label_split finish!')

def restore_split(img_path='', ann_path='', img_move_path='', ann_move_path='', img_move_path_t='', ann_move_path_t=''):
    path1 = glob(img_move_path+'*')
    path2 = glob(ann_move_path+'*')
    path3 = glob(img_move_path_t+'*')
    path4 = glob(ann_move_path_t+'*')

    img_path = img_path
    ann_path = ann_path

    for i in path1:
        shutil.move(i, img_path)
    for j in path2:
        shutil.move(j, ann_path)
    for k in path3:
        shutil.move(k, img_path)
    for l in path4:
        shutil.move(l, ann_path)
    
    print('restore Done!')

def dict_key_lower(data):
    if isinstance(data, dict):
        return {k.lower():dict_key_lower(v) for k,v in data.items()}
    elif isinstance(data, list):
        return [dict_key_lower(v) for v in data]
    else:
        return data

