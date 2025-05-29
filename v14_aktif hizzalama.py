# -*- coding: utf-8 -*-
import gxipy as gx
import numpy as np
import cv2
import json
from log.logger import operation_logger

# --- Ayarlar ---
CAMERA_SN = "FDC24100405"
json_file_path = "api_get_20250514_094755.json"
real_width_cm = 107
real_height_cm = 30
camera_width_px = 1920
camera_height_px = 1200
camera_real_width_cm = 139
camera_real_height_cm = 86
SCALE_PERCENT = 0
DISTANCE_TOLERANCE_PX = 30  # Ort. mesafe toleransı

# --- Ölçekleme Hesabı ---
scale_px_per_cm_x = camera_width_px / camera_real_width_cm
scale_px_per_cm_y = camera_height_px / camera_real_height_cm

# --- JSON Referans Noktalarını Yükle ---
with open(json_file_path, "r") as f:
    data = json.load(f)
shape_points = data[0]['points']
x_coords, y_coords = zip(*shape_points)
min_x, max_x = min(x_coords), max(x_coords)
min_y, max_y = min(y_coords), max(y_coords)
raw_width = max_x - min_x
raw_height = max_y - min_y
shape_scale_x_cm = real_width_cm / raw_width
shape_scale_y_cm = real_height_cm / raw_height
final_scale_x = shape_scale_x_cm * scale_px_per_cm_x
final_scale_y = shape_scale_y_cm * scale_px_per_cm_y
shape_center_x = (min_x + max_x) / 2
shape_center_y = (min_y + max_y) / 2

# --- Yardımcı Fonksiyonlar ---
def scale_and_mirror_points(points, scale_percent=0):
    mirrored_scaled = []
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    scale_ratio = 1 + scale_percent / 100.0
    for x_cm, y_cm in points:
        mirrored_y_cm = (max_y + min_y) - y_cm
        new_x = center_x + (x_cm - center_x) * scale_ratio
        new_y = center_y + (mirrored_y_cm - center_y) * scale_ratio
        mirrored_scaled.append((new_x, new_y))
    return mirrored_scaled

def is_contour_fully_inside(reference_cnt, test_cnt):
    for point in test_cnt[:, 0]:
        test_pt = (float(point[0]), float(point[1]))
        if cv2.pointPolygonTest(reference_cnt, test_pt, False) < 0:
            return False
    return True

def mean_distance_between_contours(test_cnt, reference_cnt):
    distances = [abs(cv2.pointPolygonTest(reference_cnt, (float(pt[0]), float(pt[1])), True))
                 for pt in test_cnt[:, 0]]
    return np.mean(distances) if distances else float('inf')

def get_contour_center(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

# --- Kontur Algılama Sınıfları ---
class ContourStorage:
    def __init__(self):
        self.contours = []
    def save_contours(self, contours):
        self.contours = contours
    def get_contours(self):
        return self.contours

class ContourProcessor:
    def __init__(self, contour_storage):
        self.contour_storage = contour_storage

    def process_frame(self, image: np.ndarray):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            if mean_intensity > 227:
                thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY_INV, 31, 12)
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=3)
                edges = cv2.Canny(closed, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > 3000]
            else:
                thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY_INV, 21, 5)
                morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=3)
                contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                h, w = gray.shape
                filtered = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 5000:
                        continue
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if not (w * 0.15 < cx < w * 0.85 and h * 0.15 < cy < h * 0.85):
                        continue
                    filtered.append(cnt)
            if not filtered:
                return None, "Yeterli kontur bulunamadı."
            largest = max(filtered, key=cv2.contourArea)
            self.contour_storage.save_contours(filtered)
            return largest, None
        except Exception as e:
            return None, f"Contour processing error: {e}"

# --- Kamera Gösterimi ---
def show_daheng_realtime(camera_sn):
    storage = ContourStorage()
    processor = ContourProcessor(storage)
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num == 0:
        print("Hiçbir kamera bulunamadı.")
        return

    cam = device_manager.open_device_by_sn(camera_sn)
    cam.stream_on()
    print(f"Kamera {camera_sn} ile bağlantı kuruldu. Görüntü başlatılıyor...")

    scaled_reference = [(int(x * final_scale_x), int(y * final_scale_y))
                        for x, y in scale_and_mirror_points(shape_points, scale_percent=SCALE_PERCENT)]
    ref_cnt = np.array(scaled_reference, dtype=np.int32).reshape((-1, 1, 2))
    ref_offset = None
    hizalama_aktif = False
    hizalama_onayi_alindi = False
    hizalanmis_live_center = None

    try:
        while True:
            raw_image = cam.data_stream[0].get_image()
            if raw_image is None:
                continue
            rgb_image = raw_image.convert("RGB")
            frame = rgb_image.get_numpy_array()
            if frame is None:
                continue

            frame_for_processing = frame.copy()
            display_frame = frame.copy()

            image_center_x = frame.shape[1] // 2
            image_center_y = frame.shape[0] // 2
            offset_x = image_center_x - int(shape_center_x * final_scale_x)
            offset_y = image_center_y - int(shape_center_y * final_scale_y)

            if ref_offset is None:
                ref_offset = ref_cnt + np.array([[[offset_x, offset_y]]], dtype=np.int32)

            live_cnt, error = processor.process_frame(frame_for_processing)

            if live_cnt is not None:
                fully_inside = is_contour_fully_inside(ref_offset, live_cnt)
                if fully_inside:
                    result = "OK"
                    reason = "Tamamen icerde"
                else:
                    avg_dist = mean_distance_between_contours(live_cnt, ref_offset)
                    if avg_dist < DISTANCE_TOLERANCE_PX:
                        result = "OK"
                        reason = f"Konturler hizali (mesafe: {avg_dist:.1f}px)"
                    else:
                        result = "NOT OK"
                        reason = f"Disarida ve uzak (mesafe: {avg_dist:.1f}px)"

                color = (0, 255, 0) if result == "OK" else (0, 0, 255)
                cv2.drawContours(display_frame, [live_cnt], -1, color, 2)
                cv2.putText(display_frame, result, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
                cv2.putText(display_frame, reason, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                mask_red = np.zeros_like(frame[:, :, 0])
                cv2.drawContours(mask_red, [live_cnt], -1, 255, -1)
                mask_blue = np.zeros_like(frame[:, :, 0])
                cv2.drawContours(mask_blue, [ref_offset], -1, 255, -1)
                outside_mask = cv2.subtract(mask_red, cv2.bitwise_and(mask_red, mask_blue))
                inside_mask = cv2.subtract(mask_blue, cv2.bitwise_and(mask_red, mask_blue))
                overlay = display_frame.copy()
                overlay[outside_mask > 0] = [0, 0, 255]
                overlay[inside_mask > 0] = [0, 255, 0]
                cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)

                if result == "OK" and not hizalama_onayi_alindi:
                    hizalanmis_live_center = get_contour_center(live_cnt)
                    hizalama_aktif = True
                    hizalama_onayi_alindi = True
                    print("Otomatik hizalama yapıldı ve merkez referanslandı.")

                if hizalama_aktif:
                    current_center = get_contour_center(live_cnt)
                    if current_center:
                        dx = current_center[0] - hizalanmis_live_center[0]
                        dy = current_center[1] - hizalanmis_live_center[1]
                        ref_offset = ref_offset + np.array([[[dx, dy]]], dtype=np.int32)
                        hizalanmis_live_center = current_center
                        cv2.putText(display_frame, "Hizalama aktif", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            else:
                cv2.putText(display_frame, error or "Contour Error", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)

            cv2.namedWindow("Daheng Realtime OK/NOT OK", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Daheng Realtime OK/NOT OK", 1000, 800)
            cv2.imshow("Daheng Realtime OK/NOT OK", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stream_off()
        cam.close_device()
        cv2.destroyAllWindows()
        print("Kamera kapatıldı ve pencere kapatıldı.")

if __name__ == "__main__":
    show_daheng_realtime(CAMERA_SN)