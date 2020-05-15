"""
A script to explore and visualize the data base
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
import tensorflow as tf

from util import wider_util

FLAG_VIS = True

TRAIN_ANNO_PATH = "Data/WIDER_FACE/wider_face_split/wider_face_train_bbx_gt.txt"

txt_path = os.path.join(os.path.expanduser("~"), TRAIN_ANNO_PATH)
# faces = wider_util.anno_test_2_dict(txt_path)
# nb_img = len(faces)
# face_arr = np.empty((nb_img, 2), dtype=object)
#
# for idx, img_path in tqdm.tqdm(enumerate(faces)):
#     face_arr[idx, 0] = img_path
#     face_arr[idx, 1] = faces[img_path].values.astype(float).tostring()


# np.save("data/wider_array_anno", face_arr)
col_name = ["x1", "y1", "w", "h", "blur", "expression", "illumination",
            "invalid", "occlusion", "pose"]


faces = {}
face_arr = np.load("data/wider_array_anno.npy", allow_pickle=True)

for img_path, faces_ctt in tqdm.tqdm(face_arr):
    faces[img_path] = pd.DataFrame(
        np.frombuffer(faces_ctt).reshape(-1, 10).astype(int),
        columns=col_name,
    )

img_names = list(faces.keys())
nb_train_img = len(img_names)
print("There are {} training images.".format(nb_train_img))

nb_face_dist = []
for img_name in img_names:
    nb_face_dist.append(faces[img_name].shape[0])
print("Average number of faces in one image is {}".
      format(np.mean(nb_face_dist)))
plt.figure()
plt.hist(nb_face_dist, bins=20, range=[-1, 20])
plt.title("Distribution of number of faces in one image")
plt.show()

if FLAG_VIS:
    import random
    import cv2

    img_dir = os.path.join(os.path.expanduser("~"),
                           "Data/WIDER_FACE/WIDER_train/images")
    random.shuffle(img_names)
    for img_name in img_names:
        img = cv2.imread(os.path.join(img_dir, img_name))
        tmp = faces[img_name]
        if tmp.shape[0] > 0:
            loc_tl = tmp.loc[:, ["x1", "y1"]].values
            loc_br = loc_tl + tmp.loc[:, ["w", "h"]].values
            loc = np.hstack([loc_tl, loc_br])
            wider_util.vis_face(img, loc)
        else:
            print("No face annotated in image {}".format(img_name))


