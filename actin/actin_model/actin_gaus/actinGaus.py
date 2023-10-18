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
    I_max   = 100
    def func_gauss(x, y):
        mtx = np.array([x, y]) - gauss_mu
        return I_max*np.exp(- mtx.dot(gauss_sigma_inv).dot(mtx[np.newaxis, :].T) / 2.0) / (2*np.pi*np.sqrt(gauss_sigma_det))
    intensity = np.vectorize(func_gauss)(x_gauss,y_gauss)
    return intensity

d_gactin    = 5.5
N_subunit   = 13
d_filament  = 9.0 - d_gactin
pitch       = N_subunit * d_gactin
x_min   = -5
x_max   = 80
N_x     = 85
N_repeat = 3
x   = np.linspace(x_min, x_max, N_x)
f1 = (d_filament/2)*np.sin((2*np.pi/pitch)*x)
f2 = -(d_filament/2)*np.sin((2*np.pi/pitch)*x-2*np.pi*((d_gactin/2)/pitch))
x1  = np.arange(0, N_repeat*pitch+d_gactin, d_gactin)
x2  = np.add(x1, d_gactin/2)
y1 = (d_filament/2)*np.sin((2*np.pi/pitch)*x1)
y2 = -(d_filament/2)*np.sin((2*np.pi/pitch)*x2-2*np.pi*((d_gactin/2)/pitch))
n_round = 1
x1 = np.round(x1, decimals=n_round)
y1 = np.round(y1, decimals=n_round)
x2 = np.round(x2, decimals=n_round)
y2 = np.round(y2, decimals=n_round)
r_mesh = 120
xy_h    = 1/np.power(10,n_round)
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
    px,py = x1[i],y1[i]
    idx_x = int((px-x_min)*np.power(10,n_round))
    idx_y = int((py-y_min)*np.power(10,n_round))
    g_intensity = get_gaussian(px,py,sig,r_mesh=r_mesh, xy_h=xy_h)
    intensity_gaussian[(idx_y-rg):(idx_y+rg),(idx_x-rg):(idx_x+rg)] += g_intensity
    px,py = x2[i],y2[i]
    idx_x = int(np.round((px-x_min)*np.power(10,n_round), decimals=1))
    idx_y = int(np.round((py-y_min)*np.power(10,n_round), decimals=1))
    g_intensity = get_gaussian(px,py,sig,r_mesh=r_mesh, xy_h=xy_h)
    intensity_gaussian[(idx_y-rg):(idx_y+rg),(idx_x-rg):(idx_x+rg)] += g_intensity

# Calculate the centroid of the terminal actin molecule
centroid_x = np.mean([x1[-1], x2[-1]])
centroid_y = np.mean([y1[-1], y2[-1]])

# Define the size of the new image, ensuring that the centroid of the terminal actin molecule is at the center
new_img_size = int(2 * (r_mesh / xy_h + 1))

# Create a blank intensity map of the new size
new_intensity_map = np.zeros((new_img_size, new_img_size), np.float64)

# Calculate the offset to place the centroid at the center of the new image
offset_x = new_img_size // 2 - int((centroid_x - x_min) * np.power(10, n_round))
offset_y = new_img_size // 2 - int((centroid_y - y_min) * np.power(10, n_round))

# Transfer the intensities from the old map to the new map with the offset
for i in range(intensity_gaussian.shape[0]):
    for j in range(intensity_gaussian.shape[1]):
        new_i = i + offset_y
        new_j = j + offset_x
        if 0 <= new_i < new_img_size and 0 <= new_j < new_img_size:
            new_intensity_map[new_i, new_j] = intensity_gaussian[i, j]

img_max = 255
# ...
# Rescale the image such that 1 pixel = 1 nm
output_size = (int(new_img_size * np.sqrt(scale)), int(new_img_size * np.sqrt(scale)))
rescaled_intensity_map = cv2.resize(new_intensity_map, output_size, interpolation=cv2.INTER_CUBIC)

# Save the rescaled image
rescaled_intensity_max = np.amax(rescaled_intensity_map)
rescaled_intensity_map = np.divide(np.multiply(rescaled_intensity_map.copy(), img_max), rescaled_intensity_max)
rescaled_img = rescaled_intensity_map.astype(np.uint8)
cv2.imwrite('rescaled_gaussian.jpg', rescaled_img)

'rescaled_gaussian.jpg'
