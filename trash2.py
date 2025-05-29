import numpy as np
from PIL import Image

# Image parameters
img_size = 1024
px_size_um = 0.2
spacing_um = 14.0

# Convert physical spacing to pixel period
period_px = int(spacing_um / px_size_um)
stripe_width = period_px // 2

# Create a blank (black) image
img = np.zeros((img_size, img_size), dtype=np.uint8)

# Define the central square region (512x512)
region_size = img_size // 2
start = (img_size - region_size) // 2
end = start + region_size

# Draw vertical stripes in the top half
for x in range(start, end):
    if ((x - start) // stripe_width) % 2 == 0:
        img[start:start + region_size // 2, x] = 255

# Draw horizontal stripes in the bottom half
for y in range(start + region_size // 2, end):
    if ((y - (start + region_size // 2)) // stripe_width) % 2 == 0:
        img[y, start:end] = 255

# Save the image
output_path = 'Rayleigh criterion/14_test.png'
Image.fromarray(img).save(output_path)
print(f"Image saved to {output_path}")

