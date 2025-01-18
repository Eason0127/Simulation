import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon, disk

# Initialize image size
image_size = 1024
image = np.zeros((image_size, image_size), dtype=np.float32)

# Define tumor boundary as an irregular polygon
tumor_boundary_x = [300, 350, 400, 450, 480, 460, 420, 360, 310, 280]
tumor_boundary_y = [400, 350, 340, 360, 410, 460, 500, 520, 490, 440]
polygon_rr, polygon_cc = polygon(tumor_boundary_y, tumor_boundary_x, shape=image.shape)

# Assign intensity for the tumor region
image[polygon_rr, polygon_cc] = 0.6  # Tumor body intensity

# Add smaller irregular shapes to represent cells inside the tumor
num_cells = 50  # Number of cells
cell_min_radius = 5
cell_max_radius = 15
for _ in range(num_cells):
    # Randomly place cells inside the tumor boundary
    x_cell = np.random.randint(min(tumor_boundary_x), max(tumor_boundary_x))
    y_cell = np.random.randint(min(tumor_boundary_y), max(tumor_boundary_y))

    # Ensure the cell center is inside the tumor
    if image[y_cell, x_cell] == 0.6:
        # Randomize cell size and intensity
        cell_radius = np.random.randint(cell_min_radius, cell_max_radius)
        cell_intensity = np.random.uniform(0.7, 0.9)

        # Draw the cell
        cell_rr, cell_cc = disk((y_cell, x_cell), cell_radius, shape=image.shape)
        image[cell_rr, cell_cc] = cell_intensity

# Display and save the image
plt.imshow(image, cmap="gray")
plt.axis('off')
plt.tight_layout()
plt.savefig("tumor_sample.png", bbox_inches='tight', pad_inches=0)
plt.show()