"""
wider_util contains some functions to process WIDER FACE database
"""
import json
import pandas as pd
import numpy as np
import cv2
import random
import os
import tqdm


def anno_test_2_dict(txt_path):
    """
    interpret face information stored in txt to dict
    For example:

    ------------------------
    0--Parade/0_Parade_marchingband_1_849.jpg
    1
    449 330 122 149 0 0 0 0 0 0
    0--Parade/0_Parade_Parade_0_904.jpg
    1
    361 98 263 339 0 0 0 0 0 0
    0--Parade/0_Parade_marchingband_1_799.jpg
    21
    78 221 7 8 2 0 0 0 0 0
    78 238 14 17 2 0 0 0 0 0
    .....
    ------------------------

    The format of txt ground truth.
    File name
    Number of bounding box
    x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose

    All the detail information is stored in a array

    Args:
        txt_path:

    Returns:

    """
    col_name = ["x1", "y1", "w", "h", "blur", "expression", "illumination",
                "invalid", "occlusion", "pose"]
    face_info = {}
    cnt = 0
    with open(txt_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                print("End of the file")
                break
            if "/" in line:
                cnt += 1
                img_name = line.rstrip('\n')
                nb_box = int(f.readline())
                if nb_box == 0:
                    face_info[img_name] = pd.DataFrame(
                        np.asarray(boxes, dtype=int), columns=col_name
                    )
                else:
                    boxes = []
                    for idx in range(nb_box):
                        line = f.readline().rstrip('\n').strip()
                        boxes.append([int(x) for x in line.split(" ")])
                    face_info[img_name] = pd.DataFrame(
                        np.asarray(boxes, dtype=int), columns=col_name
                    )
            else:
                continue
            if cnt % 1000 == 0:
                print("{} images are processed".format(cnt))
    print("Find {} images in train dataset.".format(cnt))

    return face_info


def plot_rect(img, pnt_tl, pnt_br, color=(0, 0, 255), thickness=1):
    cv2.rectangle(img, pnt_tl, pnt_br, color=color, thickness=thickness)


def vis_face(img, loc, color=(0, 0, 255), thickness=2):
    nb_face = loc.shape[0]
    for idx in range(nb_face):
        plot_rect(img, (loc[idx, 0], loc[idx, 1]), (loc[idx, 2], loc[idx, 3]),
                  color=color, thickness=thickness)

    cv2.imshow("Face", img)
    cv2.waitKey(0)
