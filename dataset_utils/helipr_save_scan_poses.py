# Based on submap-extraction tools from: https://sites.google.com/view/mulran-pr/tool

import glob
import os
import numpy as np
import csv
import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
basedir = 'data/Town1/'


def findNnPoseUsingTime(target_time, all_times, data_poses):
    time_diff = np.abs(all_times - target_time)
    nn_idx = np.argmin(time_diff)
    return data_poses[nn_idx]


sequences = ['Aeva','Avia','Ouster','Velodyne']

for sequence in sequences:
    sequence_path = basedir +"/LiDAR/" +sequence
    scan_names = sorted(glob.glob(os.path.join(sequence_path, '*.bin')))

    with open(basedir + "/LiDAR_GT" + f'/{sequence}_gt.txt', 'r') as f:
        lines = f.readlines()
        data_poses = []
        for line in lines:
            data_poses.append(line.split())
        
    data_poses_ts = np.asarray([int(t) for t in np.asarray(data_poses)[:, 0]])

    for scan_name in scan_names:
        scan_time = int(scan_name.split('/')[-1].split('.')[0])
        scan_pose = findNnPoseUsingTime(scan_time, data_poses_ts, data_poses)
        with open(basedir +"/LiDAR_GT/"+ f'/{sequence}_sync_poses.csv', 'a', newline='') as csvfile:
            posewriter = csv.writer(
                csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            posewriter.writerow(scan_pose)