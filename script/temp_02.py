#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

def count_files(dir_path):
    """返回 dir_path 下所有文件的数量（不包含子目录）"""
    return sum(
        1 for fn in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, fn))
    )

root_dir = '/home/xu_23/Project/nvs_solver/dataset/bs-ergb/1_TEST_split'

for sub in sorted(os.listdir(root_dir)):
    sub_path = os.path.join(root_dir, sub)
    if not os.path.isdir(sub_path):
        continue

    img_dir = os.path.join(sub_path, "images")
    evt_dir = os.path.join(sub_path, "events")

    if not os.path.isdir(img_dir) or not os.path.isdir(evt_dir):
        print(f"[跳过] {sub}：缺少 'images' 或 'events' 文件夹")
        continue

    n_imgs = count_files(img_dir)
    n_evts = count_files(evt_dir)

    if n_imgs != n_evts:
        print(f"[不匹配] {sub}: images({n_imgs}) vs events({n_evts})")