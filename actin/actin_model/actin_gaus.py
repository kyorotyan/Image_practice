import numpy as np
import matplotlib.pyplot as plt

def gauss(x, mu, sigma):
    """ガウス関数の定義"""
    return np.exp(-(x-mu)**2 / (2*sigma**2))

# 描画範囲と解像度を設定
x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x, y)

# アクチン繊維のシミュレーション (ここでは単純な直線を使用)

Z = np.exp(-X**2 / (2*0.5**2))

# ガウス関数をフィティングして蛍光効果をシミュレート
sigma = 1.5
Z_gauss = np.zeros_like(Z)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        if Z[i, j] > 0:
            Z_gauss += Z[i, j] * gauss(np.sqrt((X-X[0, j])**2 + (Y-Y[i, 0])**2), 0, sigma)

# 画像を正規化
Z_gauss = Z_gauss / Z_gauss.max()

# 描画
plt.imshow(Z_gauss, cmap='gray')
plt.axis('off')
plt.savefig("actin_simulation.png", bbox_inches='tight', pad_inches=0)
plt.show()
