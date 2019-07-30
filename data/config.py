classes_coco = ('person', 'nose', 'LeftEye', 'RightEye', 'LeftEar', 'RightEar', 'LeftShoulder', 'RightShoulder',
                'LeftElbow', 'RightElbow', 'LeftWrist', 'RightWrist', 'LeftHip', 'RightHip', 'LeftKnee', 'RightKnee',
                'LeftAnkle', 'RightAnkle')

coco_kpts_mirror_map = {
    2: 3,    # 'left_eye' <--> 'right_eye'
    4: 5,    # 'left_ear' <--> 'right_ear'
    6: 7,    # 'left_shoulder' <--> 'right_shoulder'
    8: 9,    # 'left_elbow' <--> 'right_elbow'
    10: 11,  # 'left_wrist' <--> 'right_wrist'
    12: 13,  # 'left_hip' <--> 'right_hip'
    14: 15,  # 'left_knee' <--> 'right_knee'
    16: 17   # 'left_ankle' <--> 'right_ankle'
}

cfg_coco = {
    'name': 'coco',
    'classes': classes_coco,
    'kpts_mirror_map': coco_kpts_mirror_map,
    'inp_dim': 300,
    'means': (107, 114, 118),
}
