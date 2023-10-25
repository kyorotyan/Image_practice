import cv2
import numpy as np

def get_gaussian(mx, my, sig, r_mesh=20, xy_h=0.2):
    x_min,x_max = mx-r_mesh, mx+r_mesh
    y_min,y_max = my-r_mesh, my+r_mesh
    Nx = Ny = 2400
    x_mesh = np.linspace(x_min, x_max, Nx)
    y_mesh = np.linspace(y_min, y_max, Ny)
    x_gauss, y_gauss = np.meshgrid(x_mesh, y_mesh)
    x_mu    = mx
    y_mu    = my
    gauss_mu    = np.array([x_mu, y_mu])
    xx_sigma    = sig
    yy_sigma    = sig
    xy_sigma    = 0.0
    yx_sigma    = 0.0
    gauss_sigma = np.array([[xx_sigma, xy_sigma],[yx_sigma, yy_sigma]])
    gauss_sigma_det = np.linalg.det(gauss_sigma)
    gauss_sigma_inv = np.linalg.inv(gauss_sigma)
    #ガウス分布の強度
    I_max   = 100
    def func_gauss(x, y):
        mtx = np.array([x, y]) - gauss_mu
        return I_max*np.exp(- mtx.dot(gauss_sigma_inv).dot(mtx[np.newaxis, :].T) / 2.0) / (2*np.pi*np.sqrt(gauss_sigma_det))
    intensity = np.vectorize(func_gauss)(x_gauss,y_gauss)
    return intensity

#アクチンフィラメントの座標計算
molecule_size    = 5.5
num_molecules   = 13
d_filament  = 9.0 - molecule_size
filament_length       = num_molecules * molecule_size
x_min   = -5
x_max   = 80
N_x     = 85
N_repeat = 3
x   = np.linspace(x_min, x_max, N_x)
f1 = (d_filament/2)*np.sin((2*np.pi/filament_length)*x)
f2 = -(d_filament/2)*np.sin((2*np.pi/filament_length)*x-2*np.pi*((molecule_size/2)/filament_length))
x1  = np.arange(0, N_repeat*filament_length+molecule_size, molecule_size)
x2  = np.add(x1, molecule_size/2)
y1 = (d_filament/2)*np.sin((2*np.pi/filament_length)*x1)
y2 = -(d_filament/2)*np.sin((2*np.pi/filament_length)*x2-2*np.pi*((molecule_size/2)/filament_length))
n_round = 1
x1 = np.round(x1, decimals=n_round)
y1 = np.round(y1, decimals=n_round)
x2 = np.round(x2, decimals=n_round)
y2 = np.round(y2, decimals=n_round)
r_mesh = 120
xy_h    = 65.0
x_min,x_max = np.amin(x1)-r_mesh, np.amax(x2)+r_mesh
y_min,y_max = np.amin(y1)-r_mesh, np.amax(y2)+r_mesh
x_mesh = np.arange(x_min, x_max, xy_h)
y_mesh = np.arange(y_min, y_max, xy_h)
x_mesh = np.round(x_mesh, decimals=n_round)
y_mesh = np.round(y_mesh, decimals=n_round)
x_gauss, y_gauss = np.meshgrid(x_mesh, y_mesh)
intensity_gaussian = np.zeros((len(y_mesh), len(x_mesh)), np.float64)
scale = 132.9
rg = int(r_mesh/xy_h)
sig = scale * 2
for i in range(len(x1)):
    for coord_set in [(x1[i], y1[i]), (x2[i], y2[i])]:
        px, py = coord_set
        g_intensity = get_gaussian(px, py, sig, r_mesh=r_mesh, xy_h=xy_h)
        
        idx_x = int((px-x_min)/xy_h)
        idx_y = int((py-y_min)/xy_h)

        start_x = max(idx_x-rg, 0)
        end_x = min(idx_x+rg, intensity_gaussian.shape[1])

        start_y = max(idx_y-rg, 0)
        end_y = min(idx_y+rg, intensity_gaussian.shape[0])

        g_intensity_cropped = g_intensity[0:end_y-start_y, 0:end_x-start_x]

        intensity_gaussian[start_y:end_y, start_x:end_x] += g_intensity_cropped



# 末端アクチン分子の重心を計算
centroid_x = np.mean([x1[-1], x2[-1]])
centroid_y = np.mean([y1[-1], y2[-1]])

#末端アクチン分子の重心が中心になるように
new_img_size = int(2 * (r_mesh / xy_h + 1))

# 新しいサイズの空の強度マップの作成
new_intensity_map = np.zeros((new_img_size, new_img_size), np.float64)

#新しい画像の中心に重心を配置するための計算
offset_x = new_img_size // 2 - int((centroid_x - x_min) * np.power(10, n_round))
offset_y = new_img_size // 2 - int((centroid_y - y_min) * np.power(10, n_round))

for i in range(intensity_gaussian.shape[0]):
    for j in range(intensity_gaussian.shape[1]):
        new_i = i + offset_y
        new_j = j + offset_x
        if 0 <= new_i < new_img_size and 0 <= new_j < new_img_size:
            new_intensity_map[new_i, new_j] = intensity_gaussian[i, j]

img_max = 255

output_size = (int(new_img_size * np.sqrt(scale)), int(new_img_size * np.sqrt(scale)))
rescaled_intensity_map = cv2.resize(new_intensity_map, output_size, interpolation=cv2.INTER_CUBIC)

# 画像を保存
rescaled_intensity_max = np.amax(rescaled_intensity_map)
rescaled_intensity_map = np.divide(np.multiply(rescaled_intensity_map.copy(), img_max), rescaled_intensity_max)
rescaled_img = rescaled_intensity_map.astype(np.uint8)
cv2.imwrite('rescaled_gaussian.jpg', rescaled_img)

'rescaled_gaussian.jpg'
