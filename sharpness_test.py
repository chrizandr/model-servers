import cv2
import numpy as np
import pdb

# Load the image in grayscale
image = cv2.imread('test2.jpeg', cv2.IMREAD_GRAYSCALE)

# # Laplacian of Gaussian
# blurred = cv2.GaussianBlur(image, (3, 3), 0)
# laplacian = cv2.Laplacian(image, cv2.CV_64F)
# mean_laplacian = laplacian.mean()
# stddev_laplacian = laplacian.std()
# min_measure = np.min(laplacian)
# max_measure = np.max(laplacian)
# sharpness_score_mean = (mean_laplacian - min_measure) / (max_measure - min_measure)
# sharpness_score_stddev = (stddev_laplacian - min_measure) / (max_measure - min_measure)
# pdb.set_trace()

# # Laplacian of Lumination
# img_HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
# L = img_HLS[:, :, 1]
# u = np.mean(L)
# LP = cv2.Laplacian(L, cv2.CV_16S, ksize = 3)
# s = np.sum(LP/u)

# # Gradient of Laplcian of Lumination
# gy, gx = np.gradient(L)
# gnorm = np.sqrt(gx**2 + gy**2)
# sharpness = np.average(gnorm)
# pdb.set_trace()
# print("sharpness 1", sharpness)

# # Gradient magnitude
# ksize = -1
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
# gY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
# # the gradient magnitude images are now of the floating point data
# # type, so we need to take care to convert them back a to unsigned
# # 8-bit integer representation so other OpenCV functions can operate
# # on them and visualize them
# # gX = cv2.convertScaleAbs(gX)
# # gY = cv2.convertScaleAbs(gY)
# # combine the gradient representations into a single image
# combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

# Load the image in grayscale
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Calculate gradient magnitude using Sobel operator
# gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
# gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
# gradient_magnitude = cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0)

# # Calculate local average intensity using a 3x3 kernel
# local_average = cv2.filter2D(image, cv2.CV_64F, np.ones((3, 3)) / 9)

# # Calculate local contrast by dividing gradient magnitude by local average
# local_contrast = gradient_magnitude / (local_average + 1e-6)

# # Calculate acutance by summing local contrast and normalizing
# acutance = np.sum(local_contrast) / (image.shape[0] * image.shape[1])

import cv2
import numpy as np

# Load the image in grayscale
# Calculate the FFT
fft_image = np.fft.fftshift(np.fft.fft2(image))

high_pass_kernel = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]])

# Apply the high-pass filter in the frequency domain
filtered_fft = fft_image * high_pass_kernel

# Calculate the power spectrum
power_spectrum = np.abs(fft_image) ** 2

# Calculate the sharpness score based on high-frequency content
high_frequency_energy = np.sum(power_spectrum)
sharpness_score = high_frequency_energy

print("Sharpness Score (Frequency Domain):", sharpness_score)
pdb.set_trace()