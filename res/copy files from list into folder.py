import os
import shutil

source = r"C:/Users/62852/Documents/GitHub/Object-Tagging-PascalVOC-export/JPEGImages"
instances_train=r'datasets/vehicles_det/annotations/instances_train.txt'
instances_val=r'datasets/vehicles_det/annotations/instances_val.txt'
dest_train=r"C:/Users/62852/Documents/GitHub/Yet-Another-EfficientDet-Pytorch/datasets/vehicles_det/train"
dest_val=r"C:/Users/62852/Documents/GitHub/Yet-Another-EfficientDet-Pytorch/datasets/vehicles_det/val"

def copyfiles(source, instances, destination): # instances adalah txt berupa daftar file
    files_list = os.listdir(source)
    with open(instances, 'r') as fp:
        namelist = fp.read().splitlines()
    
    for file_name in namelist:
        if file_name in files_list:
            dest_path = os.path.join(destination,file_name)
            shutil.copy(os.path.join(source,file_name),dest_path)

if __name__ == '__main__':
    copyfiles(source, instances_train, dest_train)
    copyfiles(source, instances_val, dest_val)