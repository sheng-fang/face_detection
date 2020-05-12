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
    col_name = ["x1", "y1", "w", "h", "blur", "expression", "illumination", "invalid", "occlusion", "pose"]
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
                    face_info[img_name] = pd.DataFrame(np.asarray(boxes, dtype=int), columns=col_name)
                else:
                    boxes = []
                    for idx in range(nb_box):
                        line = f.readline().rstrip('\n').strip()
                        boxes.append([int(x) for x in line.split(" ")])
                    face_info[img_name] = pd.DataFrame(np.asarray(boxes, dtype=int), columns=col_name)
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
        plot_rect(img, (loc[idx, 0], loc[idx, 1]), (loc[idx, 2], loc[idx, 3]), color=color, thickness=thickness)

    cv2.imshow("Face", img)
    cv2.waitKey(0)


def dict_2_json(name, src_dict, save_dir=""):
    """
    TODO
    Args:
        name:
        src_dict:
        save_dir:

    Returns:

    """
    class JSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'to_json'):
                return obj.to_json(orient='records')
            return json.JSONEncoder.default(self, obj)
    img_names = list(src_dict.keys())
    with open(os.path.join(save_dir, name), 'w') as fp:
        for img_name in tqdm.tqdm(img_names[0:2]):
            json.dump({img_name: src_dict[img_name]}, fp, cls=JSONEncoder)


# if __name__ == '__main__':
#     txt_path = os.path.join(os.path.expanduser("~"), "data/WIDER_FACE/wider_face_split/wider_face_train_bbx_gt.txt")
#     faces = anno_test_2_dict(txt_path)
#     img_dir = os.path.join(os.path.expanduser("~"), "data/WIDER_FACE/WIDER_train/images")
#     img_names = list(faces.keys())
#     random.shuffle(img_names)
#     for img_name in img_names:
#         img = cv2.imread(os.path.join(img_dir, img_name))
#         tmp = faces[img_name]
#         if tmp is None:
#             print("No face annotated in image {}".format(img_name))
#             continue
#         else:
#             loc_tl = tmp.loc[:, ["x1", "y1"]].values
#             loc_br = loc_tl + tmp.loc[:, ["w", "h"]].values
#             loc = np.hstack([loc_tl, loc_br])
#             vis_face(img, loc)

    print("Debug line.")
