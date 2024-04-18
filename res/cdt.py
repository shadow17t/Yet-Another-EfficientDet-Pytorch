import os
import json
import csv
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split

json_dt=r'datasets\\vehicles_det\\annotations\\coco_annotations.json'
source1 = r"C:/Users/62852/Documents/GitHub/Object-Tagging-PascalVOC-export/JPEGImages"
source2 = r"C:/Users/62852/Documents/GitHub/Object-Tagging-PascalVOC-export/Annotations"

def getlistfromjson(jsonfile):
    with open(jsonfile) as jf:
        data = json.load(jf)
        names=[]
        images=data['images']

        for image in images:
            filename = image.get('file_name')
            names.append(filename)
        return names

def getlistfromdir(source, ext=".txt"):
    list=[]
    for file in os.listdir(source):
        if file.endswith(ext):
            list.append(file)
    return list

def checkdups(filelist):
    tmp  = set()
    dups = set(x for x in filelist if (x in tmp or tmp.add(x)))
    list(dups)
    if len(dups) != 0:
        print("duplicates: ", dups)
    else: print("No duplicates")

def compareList(l1,l2):
   l1.sort()
   l2.sort()
   if(l1==l2):
      return "Equal"
   else:
      return "Non equal"

def listfromjson(jsonfile):
    with open(jsonfile) as jf:
        data = json.load(jf)
        anno=[[],[],[]]
        anndata=data['annotations']

        for ann in anndata:
            id = ann.get('id')
            anno[0].append(id)
            image_id = ann.get('image_id')
            anno[1].append(image_id)
            category_id = ann.get('category_id')
            anno[2].append(category_id)
    return anno


def writecsv(csvfile, datalist):
# a = [[1,2,3,4],[5,6,7,8]]

    with open(csvfile,"w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=' ')
        csvWriter.writerows(datalist)


if __name__ == '__main__':
    # jpglist=getlistfromdir(source=source1, ext=".jpg")
    # checkdups(jpglist)
    # print(jpglist[:5])

    # xmllist=getlistfromdir(source=source2, ext=".xml")
    # checkdups(xmllist)
    # print(xmllist[:5])

    # fl=getlistfromjson(json_dt)
    # checkdups(fl)
    # print(fl[:5])

    # print("Comparing jpg and json list: ", compareList(jpglist,fl))

    # jl=listfromjson(json_dt)
    # print(jl)

    # writecsv("newcsv.csv", jl)

    # memuat file json
    with open(json_dt) as f:
        data = json.load(f)

    # memakai pd.json_normalize untuk konversi JSON ke DataFrame
    df = pd.json_normalize(data['annotations'])

    # membuat kolom tambahan untuk list data
    df[['bbox0', 'bbox1','bbox2', 'bbox3']] = df["bbox"].tolist()
    df[['seg0', 'seg1','seg2', 'seg3', 'seg4','seg5', 'seg6', 'seg7']] = df["segmentation"].tolist()
    df = df.drop(columns=['bbox', 'segmentation'])

    # simpan ke csv sebelum pembentukan file train & test
    df.to_csv(r'datasets\\vehicles_det\\annotations\\before_split.csv', index = None)

    # split data dengan rasio 8:2
    y=df['category_id']
    x=df.drop(columns=['category_id'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y)
    
    # menempelkan y ke x
    train = pd.concat([x_train, y_train], axis=1, join='inner')
    test = pd.concat([x_test, y_test], axis=1, join='inner')

    # susun ulang kolom & urut data berdasarkan id anotasi
    cols = ['id', 'image_id', 'category_id', 'iscrowd', 'area', 'bbox0', 'bbox1', 'bbox2', 'bbox3', 'seg0', 'seg1', 'seg2', 'seg3', 'seg4', 'seg5', 'seg6', 'seg7']
    train=train[cols].sort_values(by=['id'], ascending=True)
    test=test[cols].sort_values(by=['id'], ascending=True)

    print(train)
    print(test)

    # simpan masing2 data ke csv
    train.to_csv(r'datasets\\vehicles_det\\annotations\\train.csv', index = None)
    test.to_csv(r'datasets\\vehicles_det\\annotations\\test.csv', index = None)