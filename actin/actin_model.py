import numpy as np
import matplotlib.pyplot as plt

def plot_sin_with_circles():
    # x軸の範囲を定義
    x = np.linspace(-0.5, 8, 100)

    # sin関数の値を計算
    y = np.sin(x)

    # 円の半径と色を定義
    radius = 15
    circle_color = 'red'

    # sin関数のプロット
    plt.plot(x, y, color='blue', linewidth=2)

    # 円の描画
    plt.scatter(x, y, s=radius*13, c=circle_color, edgecolors='black', linewidths=1)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Sin Function with Circles')
    plt.grid(True)
    plt.show()

# sin関数に円を付けて表示
plot_sin_with_circles()
