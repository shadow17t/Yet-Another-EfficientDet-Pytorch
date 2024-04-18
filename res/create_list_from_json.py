import os, json

json_train=r'datasets\\vehicles_det\\annotations\\instances_train.json'
json_val=r'datasets\\vehicles_det\\annotations\\instances_val.json'

instances_train= r"C:\\Users\\62852\\Documents\\GitHub\\Yet-Another-EfficientDet-Pytorch\\datasets\vehicles_det\\annotations\\instances_train.txt"
instances_val= r"C:\\Users\\62852\\Documents\\GitHub\\Yet-Another-EfficientDet-Pytorch\\datasets\vehicles_det\\annotations\\instances_val.txt"

def listfromjson(jsonfile, instances):
    with open(jsonfile) as jf:
        data = json.load(jf)
        train_names=[]
        images=data['images']

        for image in images:
            filename = image.get('file_name')
            train_names.append(filename)

    # open file in write mode
    with open(instances, 'w') as fp:
        for item in train_names:
        # write each item on a new line
            fp.write("%s\n" % item)

if __name__ == '__main__':
    listfromjson(json_train, instances_train)
    listfromjson(json_val, instances_val)