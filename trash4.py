import numpy as np


wavelength = 532e-9
bandwidth = 2e-9
co_length = (2 * np.log(2) / np.pi) * (wavelength ** 2 / bandwidth)
