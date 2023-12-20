import os 
import csv 
import argparse 
import numpy as np 
import pandas as pd 
import open3d as o3d 
from tqdm import tqdm 
from glob import glob 
import multiprocessing as mp 
import sys

# pip3 install --upgrade git+https://github.com/klintan/pypcd.git
from pypcd import pypcd

np.random.seed(0)

def remove_ground_plane(xyzr):
    # Make open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzr[:,:3])

    # Remove ground plane 
    _, inliers = pcd.segment_plane(0.2, 3, 250)
    not_ground_mask = np.ones(len(xyzr), bool)
    not_ground_mask[inliers] = 0
    xyzr = xyzr[not_ground_mask]
    return xyzr 

def downsample_point_cloud(xyzr, voxel_size=0.05):
    # Make xyz pointcloud 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzr[:,:3])

    # Downsample point cloud using open3d functions
    pcd_ds, ds_trace, ds_ids = pcd.voxel_down_sample_and_trace(voxel_size, pcd.get_min_bound(), pcd.get_max_bound(), False)
    inv_ids = [ids[0] for ids in ds_ids]
    ds_reflectances = np.asarray(xyzr[:, 3])[inv_ids]
    return np.hstack((pcd_ds.points, ds_reflectances.reshape(-1,1)))

# vox_sz = 0.3
def pnv_preprocessing(xyzr, num_points = 4096, vox_sz = 0.1, dist_thresh = 25):

    # Cut off points past dist_thresh
    dist = np.linalg.norm(xyzr[:,:3], axis=1)
    xyzr = xyzr[dist <= dist_thresh]

    # Slowly increase voxel size until we have less than num_points
    while len(xyzr) > num_points:
      xyzr = downsample_point_cloud(xyzr, vox_sz)
      vox_sz += 0.01
    # Re-sample some points to bring to num_points if under num_points 
    ind = np.arange(xyzr.shape[0])
        
    if num_points - len(ind) > 0:
        # print(f"ds num : {len(ind)}, extra num : {num_points - len(ind)}")
        if(len(ind) < (num_points - len(ind))):
            np.save("tgt_pcd.npy", xyzr)
        extra_points_ind = np.random.choice(xyzr.shape[0], num_points - len(ind), replace = False)
        ind = np.concatenate([ind, extra_points_ind])
    xyzr = xyzr[ind,:]
    assert len(xyzr) == num_points

    # Regularize xyz to be between -1, 1 in x,y planes 
    xyzr[:,0] = xyzr[:,0] - xyzr[:,0].mean()
    xyzr[:,1] = xyzr[:,1] - xyzr[:,1].mean()

    scale = np.max(abs(xyzr[:,:2]))
    xyzr[:,:3] = xyzr[:,:3] / 1000
    return xyzr 

def get_pointcloud_tensor(xyzr):

    # Filter our points near the origin 
    r = np.linalg.norm(xyzr[:,:3], axis = 1)
    r_filter = np.logical_and(r > 0.1, r < 80)
    xyzr = xyzr[r_filter]

    # Cap out reflectance
    ref = xyzr[:,3]
    ref[ref > 1000] = 1000 
    xyzr[:,3] = ref 

    # Remove ground plane and pre-process  
    xyzr = remove_ground_plane(xyzr)
    xyzr = pnv_preprocessing(xyzr,dist_thresh=80)

    return xyzr

os_data_format = {
    'Aeva' : [('x', np.float32), ('y', np.float32), ('z', np.float32), ('reflectivity', np.float32), ('velocity', np.float32), ('time_offset_ns', np.int32), ('line_index', np.uint8)],
    'Avia' : [('x', np.float32), ('y', np.float32), ('z', np.float32), ('reflectivity', np.uint8), ('tag', np.uint8), ('line', np.uint8), ('offset_time', np.uint32)],
    'Ouster' : [('x', np.float32), ('y', np.float32), ('z', np.float32),('intensity', np.float32), ('time', np.uint32), ('reflectivity', np.uint16), ('ring', np.uint16), ('ambient', np.uint16)],
    'Velodyne' : [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32), ('ring', np.uint16), ('time', np.float32)]}

def process_pointcloud(ARGS):
    pc_path, source_dir, save_dir,env = ARGS 
    
    # pcd file
    if env in ['Aeva','Avia']:
        scan = pypcd.PointCloud.from_path(pc_path)        

        np_x = (np.array(scan.pc_data['x'], dtype=np.float32)).astype(np.float32)
        np_y = (np.array(scan.pc_data['y'], dtype=np.float32)).astype(np.float32)
        np_z = (np.array(scan.pc_data['z'], dtype=np.float32)).astype(np.float32)
        np_r = (np.array(scan.pc_data['reflectivity'], dtype=np.float32)).astype(np.float32)
        xyzr = np.transpose(np.vstack((np_x, np_y, np_z, np_r)))
        
    # bin file
    else: 
        scan = np.fromfile(pc_path, dtype=os_data_format[env])
        xyzr = np.stack((scan['x'], scan['y'], scan['z'], scan['intensity']), axis = -1)
        
    # xyzr = np.fromfile(pc_path, dtype = np.float32).reshape(-1,4)
    if len(xyzr) == 0:
        return None  
    xyzr = get_pointcloud_tensor(xyzr)
    save_path = pc_path.replace(source_dir, save_dir).split('.')[0]
    np.save(save_path, xyzr)

def multiprocessing_preprocessing(run, source_dir, save_dir,env):
    # Prepare inputs 
    clouds_raw = sorted(glob(os.path.join(run, 'lidar', '*')))
    ARGS = [[c, source_dir, save_dir,env] for c in clouds_raw]
        
    # Multiprocessing the pre-processing 
    with mp.Pool(32) as p:
        _ = list(tqdm(p.imap(process_pointcloud, ARGS), total = len(ARGS)))

def global_csv_to_northing_easting(csv_path, source_dir, save_dir):
    df = pd.DataFrame(columns = ['timestamp', 'northing', 'easting'])
    scan_timestamps = sorted([x.split('.')[0] for x in os.listdir(os.path.join(os.path.dirname(csv_path), 'lidar'))])
        
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter = ',')
        for idx, row in tqdm(enumerate(reader)):
            new_row = [scan_timestamps[idx], row[1], row[2]]
            df.loc[idx] = new_row 

    # Save new dataframe 
    df = df[df.timestamp != '1567496784952532897'] # Exclude edge case faulty scan 
    save_path = os.path.join(os.path.dirname(csv_path), 'pd_northing_easting.csv').replace(source_dir, save_dir)
    df.to_csv(save_path)


def process_MulRan(root, save_dir):
    # environments = ['Aeva', 'Avia','Ouster','Velodyne'] # KAIST and Sejong not used, but can edit this to pre-process them as well 
    environments = ['Aeva','Avia'] # KAIST and Sejong not used, but can edit this to pre-process them as well 
    # environments = ['Ouster'] # KAIST and Sejong not used, but can edit this to pre-process them as well 

    # environments = ['DCC_01', 'RiverSide_01'] # KAIST and Sejong not used, but can edit this to pre-process them as well 
    for env in environments:
        for run in os.listdir(os.path.join(root, env)): # ENV01, ENV02, ENV03
            print(os.path.join(save_dir, env, run, 'lidar'))
            if not os.path.exists(os.path.join(save_dir, env, run, 'lidar')):
                os.makedirs(os.path.join(save_dir, env, run, 'lidar'))
            global_csv_to_northing_easting(os.path.join(root, env, run, 'scan_poses.csv'), root, save_dir)
            multiprocessing_preprocessing(os.path.join(root, env, run), root, save_dir,env)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type = str, required = True, help = 'Root for MulRan Dataset')
    parser.add_argument('--save_dir', type = str, required = True, help = 'Directory to save pre-processed data to')
    args = parser.parse_args()

    process_MulRan(args.root, args.save_dir)