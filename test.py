import numpy as np
from pypcd import pypcd

pcd_path = "/home/ma/git/incloud_comment/data/hetero_data_processed/Aeva/Town1/lidar/1689670334765774526.pcd"

pc = pypcd.PointCloud.from_path(pcd_path)

## Get data from pcd (x, y, z, intensity, ring, time)
np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)

print(pc)
