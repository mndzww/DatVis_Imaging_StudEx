# import cv2
# import numpy as np
# import torch
# import torchvision
# from torchvision import transforms

# def capture_frame(camera_index=0):
#     cap = cv2.VideoCapture(camera_index)
#     if not cap.isOpened():
#         print("Error: Camera not found!")
#         return None
#     return cap

# def detect_human(frame, model, transform, device):
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = transform(img).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         preds = model(img)[0]
    
#     human_boxes = []
#     for i, label in enumerate(preds['labels']):
#         if label == 1 and preds['scores'][i] > 0.6:  # Label 1 untuk manusia
#             box = preds['boxes'][i].cpu().numpy().astype(int)
#             human_boxes.append(box)
    
#     if not human_boxes:
#         return None
    
#     return max(human_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))  # Pilih objek terbesar

# def process_image(frame, model, transform, device, reference_height_cm=170, reference_pixels=500):
#     human_box = detect_human(frame, model, transform, device)
#     if human_box is None:
#         print("No human detected!")
#         return frame, None
    
#     x1, y1, x2, y2 = human_box
#     height_cm = ((y2 - y1) / reference_pixels) * reference_height_cm
    
#     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.putText(frame, f"Height: {height_cm:.2f} cm", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
#     return frame, height_cm

# def main():
#     cap = capture_frame(0)
#     if cap is None:
#         return
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
#     model.eval()
    
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((480, 640)),
#         transforms.ToTensor()
#     ])
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture frame!")
#             break
        
#         processed_frame, height_cm = process_image(frame, model, transform, device)
        
#         if height_cm:
#             print(f"Detected height: {height_cm:.2f} cm")
        
#         cv2.imshow("Height Measurement", processed_frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


import cv2
import numpy as np

def capture_frame(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Camera not found!")
        return None
    return cap

def detect_human(frame, hog):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    humans, _ = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
    
    if len(humans) == 0:
        return None
    
    max_human = max(humans, key=lambda rect: rect[2] * rect[3])  # Ambil objek terbesar
    return max_human

def process_image(frame, hog, reference_height_cm=170, reference_pixels=500):
    human_box = detect_human(frame, hog)
    if human_box is None:
        print("No human detected!")
        return frame, None
    
    x, y, w, h = human_box
    height_cm = (h / reference_pixels) * reference_height_cm
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, f"Height: {height_cm:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame, height_cm

def main():
    cap = capture_frame(0)
    if cap is None:
        return
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame!")
            break
        
        processed_frame, height_cm = process_image(frame, hog)
        
        if height_cm:
            print(f"Detected height: {height_cm:.2f} cm")
        
        cv2.imshow("Height Measurement", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
