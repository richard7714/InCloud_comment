import open3d as o3d
import numpy as np

# 예제로 사용할 NumPy 행렬 (n x 4)
# 각 열은 x, y, z, intensity 값을 나타냄
# point_cloud_np = np.array([
#     [1.0, 2.0, 3.0, 0.8],
#     [4.0, 5.0, 6.0, 0.6],
#     [7.0, 8.0, 9.0, 0.9]
# ])

point_cloud_np = np.load("tgt_pcd.npy")

# NumPy 배열을 Open3D의 PointCloud 데이터로 변환
point_cloud_o3d = o3d.geometry.PointCloud()
point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_np[:, :3])  # x, y, z 열
# point_cloud_o3d.colors = o3d.utility.Vector3dVector(point_cloud_np[:, 3:])  # intensity 값은 colors로 설정

# 시각화 창 생성
vis = o3d.visualization.Visualizer()
vis.create_window()

# point cloud 추가
vis.add_geometry(point_cloud_o3d)

# 시각화 업데이트
vis.update_geometry()
vis.poll_events()
vis.update_renderer()

# 이미지로 저장할 파일 경로
image_file_path = "point_cloud_visualization.png"

# 이미지로 저장
vis.capture_screen_image(image_file_path)

# 창 닫기
vis.destroy_window()
