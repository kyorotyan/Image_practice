import numpy as numpy
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 6))

ax = fig.add_subplot(111)
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)

C = plt.Circle(xy = (0, 0), radius = 2.75, color = "red")

ax.add_patch(C)
plt.xlabel('Length(nm)')
plt.ylabel('Length(nm)')
plt.title('Actin Molecules')
plt.show()