import os, json

instances_train= r"C:\\Users\\62852\\Documents\\GitHub\\Yet-Another-EfficientDet-Pytorch\\datasets\vehicles_det\\annotations\\instances_train.txt"
instances_val= r"C:\\Users\\62852\\Documents\\GitHub\\Yet-Another-EfficientDet-Pytorch\\datasets\vehicles_det\\annotations\\instances_val.txt"

with open(r'datasets\\vehicles_det\\annotations\\instances_train.json') as jsonfile:
    data = json.load(jsonfile)
    train_names=[]
    images=data['images']

    for image in images:
#         for name in images:
            filename = image.get('file_name')
            train_names.append(filename)

with open(r'datasets\\vehicles_det\\annotations\\instances_val.json') as jsonfile2:
    data2 = json.load(jsonfile2)
    val_names=[]
    images2=data2['images']

    for image2 in images2:
#         for name in images:
            filename2 = image2.get('file_name')
            val_names.append(filename2)


# open file in write mode
with open(instances_train, 'w') as fp:
    for item in train_names:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('instances_train Done')

with open(instances_val, 'w') as fq:
    for item2 in val_names:
        # write each item on a new line
        fq.write("%s\n" % item2)
    print('instances_val Done')