import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import os

# Buat folder output jika belum ada
os.makedirs("output", exist_ok=True)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    cv2.imwrite("Captured-Image.jpg", frame)
cap.release()
image = cv2.imread("output/Captured-Image.jpg")
b, g, r = cv2.split(image)
cv2.imwrite("output/Blue-Channel.jpg", b)
cv2.imwrite("output/Green-Channel.jpg", g)
cv2.imwrite("output/Red-Channel.jpg", r)
median = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5)
cv2.imwrite("output/Median-Filter.jpg", median)
_, thresh = cv2.threshold(median, 127, 255, cv2.THRESH_BINARY)
_, thresh_inv = cv2.threshold(median, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite("output/Tresholding-Image.jpg", thresh)
cv2.imwrite("output/Inverse-Tresholding-Image.jpg", thresh_inv)
# Titik sudut objek (contoh dummy)
pts_src = np.array([[50, 200], [400, 200], [50, 500], [400, 500]], dtype=np.float32)
width, height = 500, 300
pts_dst = np.array([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]], dtype=np.float32)

M = cv2.getPerspectiveTransform(pts_src, pts_dst)
unwarped = cv2.warpPerspective(thresh, M, (width, height))
unwarped_inv = cv2.warpPerspective(thresh_inv, M, (width, height))
cv2.imwrite("output/Unwarped-Image.jpg", unwarped)
cv2.imwrite("output/Unwarped-Inverse-Image.jpg", unwarped_inv)
un_img = cv2.imread("output/Unwarped-Image.jpg")
lines_img = un_img.copy()
cv2.line(lines_img, (50, 0), (50, height), (0, 0, 255), 2)  # contoh vertikal
cv2.line(lines_img, (0, 150), (width, 150), (255, 0, 0), 2) # contoh horizontal
cv2.imwrite("output/Perpendicular-Lines.jpg", lines_img)
contours, _ = cv2.findContours(unwarped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = max(contours, key=cv2.contourArea)
cnt = cnt.squeeze()

x, y = cnt[:, 0], cnt[:, 1]
tck, u = splprep([x, y], s=3)
unew = np.linspace(0, 1, 100)
out = splev(unew, tck)

# Simpan titik interpolasi
np.savetxt("output/Interpolated-Contour-Points.txt", np.column_stack(out))

# Plot
plt.figure()
plt.plot(out[0], out[1], 'r-', label='Spline')
plt.scatter(x, y, s=1, alpha=0.5, label='Original')
plt.legend()
plt.axis('equal')
plt.title("Interpolasi Spline Parametrik")
plt.savefig("output/Interpolasi-Spline-Parametrik.jpg")

# Scatter Plot
plt.figure()
plt.scatter(out[0], out[1], s=10)
plt.title("Scatter of Interpolated Points")
plt.axis('equal')
plt.savefig("output/Scatter-Plot-of-Interpolated-Contour-Points.jpg")
y_arr = np.array(out[1])
x_arr = np.array(out[0])
ymax_idx = np.argmax(y_arr)
ymin_idx = np.argmin(y_arr)

pt_max = (int(x_arr[ymax_idx]), int(y_arr[ymax_idx]))
pt_min = (int(x_arr[ymin_idx]), int(y_arr[ymin_idx]))

extrema_img = cv2.cvtColor(unwarped, cv2.COLOR_GRAY2BGR)
cv2.circle(extrema_img, pt_max, 5, (0, 0, 255), -1)
cv2.circle(extrema_img, pt_min, 5, (255, 0, 0), -1)
cv2.imwrite("output/Global-Extrema-Points.jpg", extrema_img)

# Hitung panjang dalam pixel
dy = y_arr[ymax_idx] - y_arr[ymin_idx]
dx = x_arr[ymax_idx] - x_arr[ymin_idx]
length_px = np.sqrt(dx**2 + dy**2)
print("Panjang dalam pixel:", length_px)
# Muat data kalibrasi
data = np.load("calibration_data.npz")
px_per_cm = data['px_per_cm'].item()  # pastikan format dict

length_cm = length_px / px_per_cm
print("Panjang dalam cm:", length_cm)
manual_cm = float(input("Masukkan panjang hasil pengukuran manual (cm): "))
KSR = abs((length_cm - manual_cm) / manual_cm) * 100
print(f"KSR (Koefisien Selisih Relatif): {KSR:.2f}%")
