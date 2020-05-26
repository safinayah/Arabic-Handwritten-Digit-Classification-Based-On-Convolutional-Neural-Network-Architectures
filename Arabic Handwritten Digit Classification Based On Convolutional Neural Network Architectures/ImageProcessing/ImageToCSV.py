import re

from PIL import Image
import numpy as np
import os
import csv
import PIL.ImageOps

# from PIL import Image,ImageEnhance
fileList = []


def createFileList(myDir, format='.bmp'):
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList


# """load original image """

myFileList = createFileList('/image/folder/path')

for file in fileList:
    print(file)
    img_file = Image.open(file)
    img_file = img_file.resize((28, 28))
    width, height = img_file.size
    # add RegEx code to split files the following for ADBase and MADBase datasets
    file = re.split("writer(\d+).*digit(\d+)\.bmp", file)
    # file[2] is label
    print(file[2])
    # saves the labels in file called Labels.csv in CSV form
    # note to change the path
    with open("/labels/folder/path/Labels.csv", 'a', newline='') as txt_file:
        txt_file.write(" ".join(file[2]) + "\n")

    # uncomment if you need to display image after reshape
    # not recommended for folders with huge numbers of photos
    # img_file.show()
    format = img_file.format
    mode = img_file.mode

    value = np.asarray(img_file.getdata(), dtype=np.int)
    print(np.int)
    # uncomment if data is not flattened
    # value = value.flatten()
    # writes dataset on file called Data.csv in CSV form
    # note to change the path

    with open("/data/folder/path/Data.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(value)
