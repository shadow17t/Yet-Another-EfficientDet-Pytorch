import json

aDict = {
    "info": {
        "description": "",
        "url": "",
        "version": "",
        "year": 2021,
        "contributor": "",
        "data_created": "2021-05-10"
    },
    "licenses": [
        {
            "id": 1,
            "name": None,
            "url": None
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "car",
            "supercategory": "None"
        },
        {
            "id": 2,
            "name": "motorcyle",
            "supercategory": "None"
        },
        {
            "id": 3,
            "name": "other vehicles",
            "supercategory": "None"
        },
        {
            "id": 4,
            "name": "person",
            "supercategory": "None"
        }
    ],
    "images": [
        {
            "id": 0,
            "file_name": "1111.jpg",
            "width": 512,
            "height": 512,
            "date_captured": "2020/12/9",
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        },
    ]
}
# {"a":54, "b":87}

jsonString = json.dumps(aDict)
jsonFile = open("data.json", "w")
jsonFile.write(jsonString)
jsonFile.close()