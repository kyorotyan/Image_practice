import numpy as np
import matplotlib.pyplot as plt

def gauss(x, mu, sigma):
    """ガウス関数の定義"""
    return np.exp(-(x-mu)**2 / (2*sigma**2))

# 描画範囲と解像度を設定
filament_length = 80
num_molecules = 15
molecule_size = filament_length / (num_molecules - 1)
x = np.linspace(0, filament_length, 250)
y = np.linspace(-10, 10, 250)
X, Y = np.meshgrid(x, y)

# アクチン繊維のシミュレーション (二重螺旋構造)
y_actin = 10 * np.sin(2 * np.pi * x / molecule_size)
y_actin_neg = -y_actin

# Zにアクチン繊維のシミュレーションの結果を設定
Z = np.zeros((250, 250))
for i, y_value in enumerate(y_actin):
    Z[:, i] = y_value
Z_neg = np.zeros((250, 250))
for i, y_value in enumerate(y_actin_neg):
    Z_neg[:, i] = y_value

# ガウス関数をフィティングして蛍光効果をシミュレート
sigma = 1.5
Z_gauss = np.zeros_like(Z)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z_gauss += Z[i, j] * gauss(np.sqrt((X-X[0, j])**2 + (Y-Y[i, 0])**2), 0, sigma)
        Z_gauss += Z_neg[i, j] * gauss(np.sqrt((X-X[0, j])**2 + (Y-Y[i, 0])**2), 0, sigma)

# 画像を正規化
max_value = Z_gauss.max()
if max_value != 0:
    Z_gauss = Z_gauss / max_value

# 描画
plt.imshow(Z_gauss, cmap='gray', extent=(0, filament_length, -10, 10))
plt.axis('off')
plt.savefig("actin_simulation.png", bbox_inches='tight', pad_inches=0)
plt.show()
