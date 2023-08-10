import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d


def get_gaussian(mx, my, sig, r_mesh=20, xy_h=0.2):
    x_min,x_max = mx-r_mesh, mx+r_mesh
    y_min,y_max = my-r_mesh, my+r_mesh
    #x_mesh = np.arange(x_min, x_max, xy_h)
    #y_mesh = np.arange(y_min, y_max, xy_h)
    #Nx = np.round((x_max-x_min)/xy_h)
    #Ny = np.round((y_max-y_min)/xy_h)
    Nx = Ny = 2400
    # 80 -> 1600
    # 120 -> 2400
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
#d_filament  = 5.9 - d_gactin
d_filament  = 9.0 - d_gactin
pitch       = N_subunit * d_gactin


## structure of actin filament
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

#'''
rg = int(r_mesh/xy_h)
sig = scale * 2
for i in range(len(x1)):

    print(i)

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


img_max = 255
intensity_max = np.amax(intensity_gaussian)
intensity_gaussian = np.divide(np.multiply(intensity_gaussian.copy(), img_max), intensity_max)
print(intensity_max)

img_gaussian = intensity_gaussian.copy()
img_gaussian = img_gaussian.astype(np.uint8)
cv2.imwrite('gaussian.jpg', img_gaussian)


fig,ax=plt.subplots()
ax.set_aspect('equal')
ax.contourf(x_gauss, y_gauss, intensity_gaussian, 10, cmap=plt.cm.bone)
plt.show()




'''


r = int(6)
img = np.zeros((int(r*2+1), int(r*2+1)), np.float64)

sx = 0
sy = 0
scale_half = 0.5*scale
idx_rx = np.add(r, (np.divmod(np.add(np.subtract(x_mesh, sx), scale_half), scale))[0])
idx_ry = np.add(r, (np.divmod(np.add(np.subtract(y_mesh, sy), scale_half), scale))[0])


for j in idx_ry:
    if 0<=j<2*r+1:
        for i in idx_rx:
            if 0<=i<2*r+1:
                img[j,i] += intensity_gaussian[j,i]

cv2.imwrite('img_gaussian.jpg', img.astype(np.uint8))




img_max = 100
intensity_gaussian = np.divide(np.multiply(intensity_gaussian.copy(), img_max), intensity_max)
intensity_max = np.amax(intensity_gaussian)
print(intensity_max)

img_gaussian = intensity_gaussian.copy()
img_gaussian = img_gaussian.astype(np.uint8)
cv2.imwrite('gaussian.jpg', img_gaussian)
'''






# 1 pixel = 132.9 nm

'''
#fig,ax = plt.figure()
fig,ax=plt.subplots()
ax.set_aspect('equal')
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(x_gauss, y_gauss, intensity, rstride=1, cstride=1, cmap=cm.coolwarm)
ax.contourf(x_gauss, y_gauss, intensity_gaussian, 10, cmap=plt.cm.bone)
plt.show()

'''