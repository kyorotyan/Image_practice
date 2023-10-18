import numpy as np
import matplotlib.pyplot as plt

def gauss(x, y, mu_x, mu_y, sigma):
    """2D ガウス関数の定義"""
    return np.exp(-((x-mu_x)**2 + (y-mu_y)**2) / (2*sigma**2))

# アクチン繊維のパラメータ
filament_length = 80
num_molecules = 15
molecule_size = filament_length / (num_molecules - 1)

# 描画範囲と解像度を設定
x = np.linspace(0, filament_length, 1000)
y = np.linspace(-10, 10, 250)
X, Y = np.meshgrid(x, y)

# アクチン繊維のシミュレーション (二重螺旋構造)
y_actin = 1.5 * np.sin(2 * np.pi * x / filament_length)
y_actin_neg = -y_actin

x_molecules = np.linspace(0, filament_length, num_molecules)
y_molecules = 1.5 * np.sin(2 * np.pi * x_molecules / filament_length)
y_molecules_neg = -y_molecules

# ガウス関数をフィティングして蛍光効果をシミュレート
sigma = 0.5
Z_gauss = np.zeros_like(X)

for xi, ym, ym_neg in zip(x_molecules, y_molecules, y_molecules_neg):
    Z_gauss += gauss(X, Y, xi, ym, sigma)
    Z_gauss += gauss(X, Y, xi, ym_neg, sigma)

# 描画
plt.imshow(Z_gauss, cmap='gray', extent=(0, filament_length, -10, 10), aspect='auto', origin='lower')
plt.axis('off')
plt.savefig("fluorescent_actin_simulation.png", bbox_inches='tight', pad_inches=0)
plt.show()
