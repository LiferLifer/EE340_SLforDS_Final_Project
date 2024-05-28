import os

import numpy as np
from . import ImageProcessor as ip

def read(sourcepath, label):
    # 遍历读取文件夹里的所有文件
    sourcepath = sourcepath+f"{label}/"
    features = []
    labels = []
    for root, dirs, files in os.walk(sourcepath):
        for file in files:
            file_path = os.path.join(root, file)
            feature = np.load(file_path)
            features.append(feature)
            labels.append(label)
            # print(file_path)
    return np.array(features), np.array(labels)

def read_whole(sourcepath):
    features = []
    labels = []
    for i in range(1, 4):
        print(f"Reading label {i}")
        fs, ls = read(sourcepath, i)
        features.append(fs)
        labels.append(ls)
    return np.concatenate(features), np.concatenate(labels)

def read_portion(sourcepath, rg):
    features = []
    labels = []
    for i in rg:
        print(f"Reading label {i}")
        fs, ls = read(sourcepath, i)
        features.append(fs)
        labels.append(ls)
    return np.concatenate(features), np.concatenate(labels)

def read_single(sourcepath):
    return np.load(sourcepath)

def read_single_img(path, height=80):
    # print(f"Reading image {path}")
    img = ip.read_image(path)
    img = ip.shrink(img, height)
    img = ip.adjust_contrast(img, 2.0, -20)
    img = ip.gradient_graph(img)
    return np.resize(img, (1,height*height))

def read_imgs(sourcepath, label, height=80):
    # 遍历读取文件夹里的所有文件
    sourcepath = sourcepath + f"{label}/"
    features = []
    labels = []
    for root, dirs, files in os.walk(sourcepath):
        for file in files:
            file_path = os.path.join(root, file)
            feature = read_single_img(file_path, height)
            features.append(feature)
            labels.append(label)
            # print(file_path)
    return np.array(features), np.array(labels)

def read_imgs_whole(sourcepath, height=80):
    labels = []
    features = []
    for i in range(1,4):
        feature, label = read_imgs(sourcepath, i, height)
        features.append(feature)
        labels.append(label)
    return np.concatenate(features), np.concatenate(labels)



if __name__ == "__main__":
    sourcepath = "../statics/FundusDomainTest_features/"
    label = 1
    fs, ls = read(sourcepath,1)
    print(fs.shape)
    print(ls.shape)

    fs, ls = read_whole(sourcepath)
    print(fs.shape)
    print(ls.shape)