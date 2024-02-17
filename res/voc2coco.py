import os
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re

id_images = 0

def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, 'r') as f:
        labels_str = f.read().splitlines()
    labels_ids = list(range(1, len(labels_str)+1))
    print(dict(zip(labels_str, labels_ids)))
    return dict(zip(labels_str, labels_ids))

def strdateformat(str): #getting YYYY-MM-DD from filename
    cus_lens = [4, 2, 2]
    dt=re.findall("([0-9]+)", str)
    dt0=dt[0]
    res = []
    strt = 0
    for size in cus_lens:
        res.append(dt0[strt : strt + size])
        strt += size
    dat=res[0]+"\\"+res[1]+"\\"+res[2]
    return dat

def get_annpaths(ann_dir_path: str = None,
                 ann_ids_path: str = None,
                 ext: str = '',
                 annpaths_list_path: str = None) -> List[str]:
    # If use annotation paths list
    if annpaths_list_path is not None:
        with open(annpaths_list_path, 'r') as f:
            ann_paths = f.read().split()
        return ann_paths

    # If use annotaion ids list
    ext_with_dot = '.' + ext if ext != '' else ''
    with open(ann_ids_path, 'r') as f:
        ann_ids = f.read().split()
    ann_paths = [os.path.join(ann_dir_path, aid+ext_with_dot) for aid in ann_ids]
    return ann_paths


def get_image_info(annotation_root, extract_num_from_imgid=True):
    global id_images

    path = annotation_root.findtext('path')
    if path is None:
        filename = annotation_root.findtext('filename')
    else:
        filename = os.path.basename(path)

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'id': id_images,
        'file_name': filename,
        'height': height,
        'width': width,
        'date_captured': strdateformat(filename),
        'license':1,
        'coco_url': "",
        'flickr_url': ""
    }

    id_images = id_images+1

    return image_info


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(float(bndbox.findtext('xmin'))) - 1 if int(float(bndbox.findtext('xmin'))) > 0 else 0
    ymin = int(float(bndbox.findtext('ymin'))) - 1 if int(float(bndbox.findtext('ymin'))) > 0 else 0
    xmax = int(float(bndbox.findtext('xmax')))
    ymax = int(float(bndbox.findtext('ymax')))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'id': None,
        'image_id': id_images,
        'category_id': category_id,
        'iscrowd': 0,
        'area': o_width * o_height,
        'bbox': [xmin, ymin, o_width, o_height],
        'segmentation': [xmin,ymin,xmin,(ymin + o_height), (xmin + o_width), (ymin + o_height), (xmin + o_width), ymin]  # This script is not for segmentation
    }
    return ann


def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True):
    output_json_dict = {
        "info": {
                    "description": "",
                    "url": "",
                    "version": "",
                    "year": 2021,
                    "contributor": "",
                    "data_created": "2021-05-11"
                },
        "licenses": [{
                    "id": 1,
                    "name": None,
                    "url": None
                }],
        "categories": [],
        "images": [],
        "annotations": []
    }
    bnd_id = 0  # START_BOUNDING_BOX_ID, TODO input as args ?
    print('Start converting !')
    for a_path in tqdm(annotation_paths):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(annotation_root=ann_root,
                                  extract_num_from_imgid=extract_num_from_imgid)
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in label2id.items():
        category_info = {'id': label_id, 'name': label, 'supercategory': 'none'}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)#.replace('\\\\', '\\')
        f.write(output_json)


def main():
    parser = argparse.ArgumentParser(
        description='This script support converting voc format xmls to coco format json')
    parser.add_argument('--ann_dir', type=str, default=None,
                        help='path to annotation files directory. It is not need when use --ann_paths_list')
    parser.add_argument('--ann_ids', type=str, default=None,
                        help='path to annotation files ids list. It is not need when use --ann_paths_list')
    parser.add_argument('--ann_paths_list', type=str, default=None,
                        help='path of annotation paths list. It is not need when use --ann_dir and --ann_ids')
    parser.add_argument('--labels', type=str, default=None,
                        help='path to label list.')
    parser.add_argument('--output', type=str, default='output.json', help='path to output json file')
    parser.add_argument('--ext', type=str, default='', help='additional extension of annotation file')
    parser.add_argument('--extract_num_from_imgid', action="store_true",
                        help='Extract image number from the image filename')
    args = parser.parse_args()
    label2id = get_label2id(labels_path=args.labels)
    ann_paths = get_annpaths(
        ann_dir_path=args.ann_dir,
        ann_ids_path=args.ann_ids,
        ext=args.ext,
        annpaths_list_path=args.ann_paths_list
    )
    convert_xmls_to_cocojson(
        annotation_paths=ann_paths,
        label2id=label2id,
        output_jsonpath=args.output,
        extract_num_from_imgid=args.extract_num_from_imgid
    )


if __name__ == '__main__':
    main()