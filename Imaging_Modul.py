import cv2
import numpy as np
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk

# --- Fungsi Proses ---

def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("Captured-Image1.jpg", frame)
        messagebox.showinfo("Info", "Gambar berhasil disimpan sebagai Captured-Image1.jpg")
    else:
        messagebox.showerror("Error", "Gagal menangkap gambar")
    cap.release()

def extract_rgb():
    image = cv2.imread("Captured-Image1.jpg")
    if image is None:
        messagebox.showerror("Error", "Captured-Image1.jpg tidak ditemukan")
        return
    b, g, r = cv2.split(image)
    cv2.imwrite("Blue-Channel.jpg", b)
    cv2.imwrite("Green-Channel.jpg", g)
    cv2.imwrite("Red-Channel.jpg", r)
    messagebox.showinfo("Info", "RGB Channel disimpan")

def apply_median_filter():
    image = cv2.imread("Captured-Image1.jpg")
    if image is None:
        messagebox.showerror("Error", "Captured-Image1.jpg tidak ditemukan")
        return
    median = cv2.medianBlur(image, 5)
    cv2.imwrite("Median-Filter.jpg", median)
    messagebox.showinfo("Info", "Median filter diterapkan")

def apply_thresholding():
    image = cv2.imread("Captured-Image1.jpg")
    if image is None:
        messagebox.showerror("Error", "Captured-Image1.jpg tidak ditemukan")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, inv_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("Tresholding.jpg", thresh)
    cv2.imwrite("Inverse-Tresholding-Image.jpg", inv_thresh)
    messagebox.showinfo("Info", "Thresholding dan Inverse disimpan")

# --- GUI Setup ---

root = Tk()
root.title("Modul Praktikum Image Processing - Percobaan 1")
root.geometry("400x350")

Label(root, text="Percobaan 1 - Segmentasi Dasar", font=("Arial", 14)).pack(pady=10)

Button(root, text="1. Capture Image", width=30, command=capture_image).pack(pady=5)
Button(root, text="2. Extract RGB Channels", width=30, command=extract_rgb).pack(pady=5)
Button(root, text="3. Apply Median Filter", width=30, command=apply_median_filter).pack(pady=5)
Button(root, text="4. Apply Thresholding", width=30, command=apply_thresholding).pack(pady=5)

Label(root, text="File output akan disimpan di folder kerja", font=("Arial", 10)).pack(pady=20)

root.mainloop()
