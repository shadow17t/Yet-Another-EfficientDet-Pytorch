import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
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

            value = (root.find('filename').text,
                     int(float(root.find('size')[0].text)),
                     int(float(root.find('size')[1].text)),
                     label,
                     xmin,
                     ymin,
                     xmax,
                     ymax
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    # datasets = ['train', 'test']
    # for ds in datasets:
    #     image_path = os.path.join(os.getcwd(), ds, 'Annotation')
    #     xml_df = xml_to_csv(image_path)
    #     xml_df.to_csv('labels_{}.csv'.format(ds), index=None)
    #     print('Successfully converted xml to csv.')
    # image_path = os.path.join(os.getcwd(), 'tagging/object_detection/Object-Tagging-PascalVOC-export/Annotations') #nama folder
    xml_files = os.path.join(r"C:\\Users\\62852\\Documents\\GitHub\\Object-Tagging-PascalVOC-export\\Annotations") #nama folder
    xml_df = xml_to_csv(xml_files) 
    xml_df.to_csv('labelfiles.csv', index=None) #nama output csv
    print('Successfully converted xml to csv.')

main()

