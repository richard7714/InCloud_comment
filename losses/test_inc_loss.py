import torch
import torch.nn.functional as F

t = torch.tensor(
    [
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12]
    ])

out = t.unsqueeze(0)-t.unsqueeze(1)

print(out)

# # 예시 데이터 생성
# B = 3  # 배치 크기
# D = 4  # 차원 크기
# old_rep = torch.randn(B, D)  # B x D 크기의 예시 행렬 생성

# # 행렬 브로드캐스팅을 이용한 두 행렬 간의 차이 계산
# old_vec = old_rep.unsqueeze(0) - old_rep.unsqueeze(1)  # B x D x D (broadcasting)

# # 정규화
# norm_old_vec = F.normalize(old_vec, p=2, dim=2)

# # 각 벡터 간의 각도 계산
# old_angles = torch.bmm(norm_old_vec, norm_old_vec.transpose(1, 2)).view(-1)

# print("Old Representation:")
# print(old_rep)
# print("\nBroadcasted Vector Difference:")
# print(old_vec)
# print("\nNormalized Vector Difference:")
# print(norm_old_vec)
# print("\nAngles between Normalized Vectors:")
# print(old_angles)
