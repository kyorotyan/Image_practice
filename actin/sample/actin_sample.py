import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


d_gactin    = 5.5
N_subunit   = 13
#d_filament  = 5.9 - d_gactin
d_filament  = 9.0 - d_gactin
pitch       = N_subunit * d_gactin


x_min   = -5
x_max   = 80
N_x     = 85
x   = np.linspace(x_min, x_max, N_x)
f1 = (d_filament/2)*np.sin((2*np.pi/pitch)*x)
f2 = -(d_filament/2)*np.sin((2*np.pi/pitch)*x-2*np.pi*((d_gactin/2)/pitch))

x1  = np.arange(0, pitch+d_gactin, d_gactin)
x2  = np.add(x1, d_gactin/2)
print(x2)
y1 = (d_filament/2)*np.sin((2*np.pi/pitch)*x1)
y2 = -(d_filament/2)*np.sin((2*np.pi/pitch)*x2-2*np.pi*((d_gactin/2)/pitch))

print(x1)
print(y1)


n_round = 2
x1 = np.round(x1, decimals=n_round)
y1 = np.round(y1, decimals=n_round)
x2 = np.round(x2, decimals=n_round)
y2 = np.round(y2, decimals=n_round)

print(x1)
print(y1)


fig,ax=plt.subplots(figsize=(8,1))
fig.subplots_adjust(bottom=0.3,left=0.2)

ax.grid()

for i in range(len(x1)):
    px,py = x1[i],y1[i]
    draw_circle = patches.Circle(xy=(px, py), radius=d_gactin/2, color = (0, 0, 1, 0.5))
    ax.add_patch(draw_circle)

    px,py = x2[i],y2[i]
    draw_circle = patches.Circle(xy=(px, py), radius=d_gactin/2, color = (1, 0, 0, 0.5))
    ax.add_patch(draw_circle)

ax.plot(x, f1, color = (0, 0, 1), linewidth = 1.0)
ax.plot(x, f2, color = (1, 0, 0), linewidth = 1.0)
ax.scatter(x1,y1, s=5, marker='o')
ax.scatter(x2,y2, s=5, marker='o')

plt.show()
