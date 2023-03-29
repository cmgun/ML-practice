import numpy as np
from scipy.interpolate import RegularGridInterpolator

# 定义数据点的坐标和值
# 期限
x = np.array([1, 3, 6, 12, 24, 60])
# K/S_0
y = np.array([0.90, 0.95, 1.00, 1.05, 1.10])
z = np.array([[14.2, 13.0, 12.0, 13.1, 14.5],
              [14.0, 13.0, 12.0, 13.1, 14.2],
              [14.1, 13.3, 12.5, 13.4, 14.3],
              [14.7, 14.0, 13.5, 14.0, 14.8],
              [15.0, 14.4, 14.0, 14.5, 15.1],
              [14.8, 14.6, 14.4, 14.7, 15.0]])

# 创建插值函数
f = RegularGridInterpolator((x, y), z, method='linear')

# 定义要进行插值的点的坐标
points = np.array([1.5, 1.04])

# 进行插值
z_new = f(points)

print(z_new)