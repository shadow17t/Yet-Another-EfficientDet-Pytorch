import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

import re
id = 0
id2 = 0
category = ["car", "motorcycle", "other vehicles", "person"]

def strdateformat(str):
    # dt=str[0:8]
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

def imageslist(path):
    global id
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = int(float(bbx.find('xmin').text))
            ymin = int(float(bbx.find('ymin').text))
            xmax = int(float(bbx.find('xmax').text))
            ymax = int(float(bbx.find('ymax').text))
            label = member.find('name').text

            value = (id,
                     root.find('filename').text,
                     int(float(root.find('size')[0].text)),
                     int(float(root.find('size')[1].text)),
                     strdateformat(root.find('filename').text),
                    #  root.find('filename').text,
                     1,
                     "",
                     ""#,
                    #  label,
                    #  xmin,
                    #  ymin,
                    #  xmax,
                    #  ymax
                     )
            xml_list.append(value)
            id=id+1
    column_name = ['id','file_name', 'width', 'height', 'date_captured', 'license', 'coco_url', 'flickr_url']#,
                #    'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name) #
    djson=xml_df.to_json('temp.json', orient = "records")
    return djson #xml_df

def annotationslist(path):
    global id2
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = int(float(bbx.find('xmin').text))
            ymin = int(float(bbx.find('ymin').text))
            xmax = int(float(bbx.find('xmax').text))
            ymax = int(float(bbx.find('ymax').text))
            label = member.find('name').text
            pjg=xmax-xmin
            lbr=ymax-ymin
            luas=pjg*lbr

            value = (id2, # id
                     id, #image_id
                     category.index(label),
                     0,
                     luas,
                     [
                         
                     ],
                     "",
                     ""#,
                     )
            xml_list.append(value)
            id=id+1
    column_name = ['id','image_id', 'category_id', 'iscrowd', 'area', 'bbox', 'segmentation']
    xml_df = pd.DataFrame(xml_list, columns=column_name) #
    djson=xml_df.to_json('temp.json', orient = "records")
    return djson #xml_df


def main():
    # datasets = ['train', 'test']
    # for ds in datasets:
    #     image_path = os.path.join(os.getcwd(), ds, 'Annotation')
    #     xml_df = xml_to_csv(image_path)
    #     xml_df.to_csv('labels_{}.csv'.format(ds), index=None)
    #     print('Successfully converted xml to csv.')
    # image_path = os.path.join(os.getcwd(), 'tagging/object_detection/Object-Tagging-PascalVOC-export/Annotations') #nama folder
    xml_files = os.path.join(r"C:\\Users\\62852\\Documents\\GitHub\\Object-Tagging-PascalVOC-export\\Annotations") #nama folder
    img_list = imageslist(xml_files) 
    # xml_df.to_csv('labelfiles.csv', index=None) #nama output csv
    print('Successfully converted xml to json.')
    # print(xml_df)

main()

