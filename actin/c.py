import numpy as np
import matplotlib.pyplot as plt

def draw_actin_filament():
    filament_length = 80  
    num_molecules = 14

    molecule_size = filament_length / (num_molecules - 1)  # Adjust the molecule size

    t = np.linspace(0, filament_length, 1000)  

    y = 2.5 * np.sin(2 * np.pi *  t / filament_length)
    y_neg = -1 * y  # Negate the y values

    plt.plot(t, y, color='red', linewidth=2)
    plt.plot(t, y_neg, color='blue', linewidth=2)  # Plot the negative y values with a red line

    x_molecules = np.linspace(0, filament_length, num_molecules)
    y_molecules = 2.5 * np.sin(2 * np.pi *  x_molecules / filament_length)
    y_molecules_neg = -1 * y_molecules  # Negate the y_molecule values

    # Shift x_molecules for blue circles to be half way between red circles
    x_molecules_blue = np.roll(x_molecules, -1) - molecule_size / 2
    y_molecules_blue = 2.5 * np.sin(2 * np.pi * x_molecules_blue / filament_length)

    for x, y, y_neg in zip(x_molecules, y_molecules, y_molecules_neg):
        circle = plt.Circle((x, y), radius=molecule_size/2 , color='red', alpha=0.5)  # Add transparency
        plt.gca().add_patch(circle)

    for x_blue, y_blue in zip(x_molecules_blue, y_molecules_blue):
        circle_blue = plt.Circle((x_blue, -y_blue), radius=molecule_size/2 , color='blue', alpha=0.5)  # Plot negative y values with blue circles
        plt.gca().add_patch(circle_blue)

    plt.gca().set_aspect('equal')

    plt.xlabel('Length')
    plt.ylabel('Amplitude')
    plt.title('Actin Filament with Molecules')
    plt.grid(True)
    plt.show()

draw_actin_filament()
