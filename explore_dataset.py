"""
A script to explore and visualize the data base
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from util import wider_util

FLAG_VIS = True
TRAIN_ANNO_PATH = "Data/WIDER_FACE/wider_face_split/wider_face_train_bbx_gt.txt"

txt_path = os.path.join(os.path.expanduser("~"), TRAIN_ANNO_PATH)
faces = wider_util.anno_test_2_dict(txt_path)

img_names = list(faces.keys())
nb_train_img = len(img_names)
print("There are {} training images.".format(nb_train_img))

nb_face_dist = []
for img_name in img_names:
    nb_face_dist.append(faces[img_name].shape[0])
print("Average number of faces in one image is {}".format(np.mean(nb_face_dist)))
plt.figure()
plt.hist(nb_face_dist, bins=20, range=[-1, 20])
plt.title("Distribution of number of faces in one image")
plt.show()

if FLAG_VIS:
    import random
    import cv2

    img_dir = os.path.join(os.path.expanduser("~"), "Data/WIDER_FACE/WIDER_train/images")
    random.shuffle(img_names)
    for img_name in img_names:
        img = cv2.imread(os.path.join(img_dir, img_name))
        tmp = faces[img_name]
        if tmp.shape[0] == 0:
            print("No face annotated in image {}".format(img_name))
            continue
        else:
            loc_tl = tmp.loc[:, ["x1", "y1"]].values
            loc_br = loc_tl + tmp.loc[:, ["w", "h"]].values
            loc = np.hstack([loc_tl, loc_br])
            wider_util.vis_face(img, loc)
