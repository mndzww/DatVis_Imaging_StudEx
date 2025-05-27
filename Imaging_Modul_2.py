import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

img = cv2.imread("Tresholding.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)

canvas = img.copy()
cv2.drawContours(canvas, [cnt], -1, (0, 255, 0), 2)
cv2.imwrite("Segmented-Freeform-Object.jpg", canvas)

rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)

# Dapatkan rotasi matrix
center = rect[0]
angle = rect[2]
(h, w) = img.shape[:2]
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

cv2.imwrite("Rotated-Freeform.jpg", rotated)


pts = cnt[:, 0, :]
x, y = pts[:, 0], pts[:, 1]

tck, u = splprep([x, y], s=3)
u_new = np.linspace(0, 1, 200)
x_new, y_new = splev(u_new, tck)

# Plot hasil spline
plt.figure()
plt.plot(x_new, y_new, 'b-')
plt.scatter(x, y, color='red', s=5, label="Original")
plt.gca().invert_yaxis()
plt.title("Interpolasi Kontur Bentuk Tak Beraturan")
plt.savefig("Interpolasi-Kontur-Freeform.jpg")

y_arr = np.array(y_new)
x_arr = np.array(x_new)

ymax_idx = np.argmax(y_arr)
ymin_idx = np.argmin(y_arr)

dy = y_arr[ymax_idx] - y_arr[ymin_idx]
dx = x_arr[ymax_idx] - x_arr[ymin_idx]
length_px = np.sqrt(dx**2 + dy**2)

print(f"Panjang objek tak beraturan (dalam pixel): {length_px:.2f}")

pixel_per_cm = 37.79527563816727  # didapat dari objek referensi
length_cm = length_px / pixel_per_cm
print(f"Panjang objek: {length_cm:.2f} cm")
