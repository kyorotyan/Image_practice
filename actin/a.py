import numpy as np
import matplotlib.pyplot as plt

def draw_actin_filament():
    filament_length = 80  
    molecule_size = 5 
    num_molecules = 13  

    t = np.linspace(0, filament_length, 1000)  

    y = np.sin(2 * np.pi *  t / filament_length)

    plt.plot(t, y, color='blue', linewidth=2)

    x_molecules = np.linspace(0, filament_length, num_molecules)
    y_molecules = np.sin(2 * np.pi *  x_molecules / filament_length)

    for x, y in zip(x_molecules, y_molecules):
        circle = plt.Circle((x, y), radius=molecule_size/2 , color='red')
        plt.gca().add_patch(circle)

    plt.xlabel('Length')
    plt.ylabel('Amplitude')
    plt.title('Actin Filament with Molecules')
    plt.grid(True)
    plt.show()

draw_actin_filament()
