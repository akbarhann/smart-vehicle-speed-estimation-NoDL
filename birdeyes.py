import cv2
import numpy as np
from centroid import CentroidTracker
import csv

VIDEO_PATH = 'cars2.mp4'
csv_output_path = 'vehicle_data.csv'
# === LOAD ROI YANG TELAH DISIMPAN ===
initial_roi = np.load("saved_roi1.npy").astype(np.float32).reshape(-1, 1, 2)

# === INISIALISASI TRACKING ROI ===
orb = cv2.ORB_create(1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = 29.97
print(f"[INFO] Detected FPS: {FPS}")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r'd:\\VehicleDetection\\result.mp4', fourcc, FPS, (frame_width, frame_height))
bird_out = cv2.VideoWriter(r'd:\\VehicleDetection\\bird_view_output.mp4', fourcc, FPS, (400, 600))
ret, initial_frame = cap.read()
ref_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
ref_kp, ref_des = orb.detectAndCompute(ref_gray, None)

# === INISIALISASI PELACAKAN & PERHITUNGAN ===
tracker = CentroidTracker()
bgsub = cv2.createBackgroundSubtractorMOG2(history=550, varThreshold=100, detectShadows=False,)
bgsub.setBackgroundRatio(0.93)
bgsub.setNMixtures(3)
bgsub.setVarInit(60)
bgsub.setVarMin(12)
bgsub.setVarMax(100)

counted_ids = set()
track_history = {}
crossing_times = {}
estimated_speeds = {}
frame_number = 0
count = 0
SPEED_LIMIT_KMPH = 100  

def classify_speed(speed):
    return "Overspeed" if speed > SPEED_LIMIT_KMPH else "Normal"

def write_csv_header(path):
    with open(path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'ID', 'Speed (km/h)', 'Classification'
        ])
        writer.writeheader()

def append_csv(path, data):
    with open(path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'ID', 'Speed (km/h)', 'Classification'
        ])
        writer.writerow(data)
def ccw(A, B, C):
    """Check if three points are listed in a counterclockwise order."""
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A, B, C, D):
    """Return True if line segment AB intersects CD"""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


write_csv_header(csv_output_path)   
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1
    if frame_number == 100:
        cv2.imwrite("frame_100.jpg", frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)

    if des is None or ref_des is None or len(kp) < 10:
        continue

    matches = bf.match(ref_des, des)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50]

    ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    frame_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(ref_pts, frame_pts, cv2.RANSAC, 5.0)
    if H is None:
        continue

    roi_transformed = cv2.perspectiveTransform(initial_roi, H)
    
    roi_transformed_int = np.int32(roi_transformed)

    cv2.polylines(frame, [roi_transformed_int], isClosed=True, color=(0, 255, 0), thickness=2)

    # Garis entry (atas) dan exit (bawah) dari ROI dinamis
    # === DETEKSI GARIS ENTRY DAN EXIT OTOMATIS ===
    # Ambil 4 titik ROI
    pts = [p[0] for p in roi_transformed]
    # Urutkan berdasarkan nilai Y (atas ke bawah)
    sorted_pts = sorted(pts, key=lambda p: p[1])
    # Dua titik dengan Y terkecil adalah ENTRY, dua lainnya EXIT
    entry_pt1, entry_pt2 = sorted_pts[0], sorted_pts[1]
    exit_pt1, exit_pt2 = sorted_pts[2], sorted_pts[3]
    
    # Pastikan titik-titik dalam urutan: top-left, top-right, bottom-left, bottom-right
    sorted_pts_by_x = sorted(sorted_pts[:2], key=lambda p: p[0])  # Entry line (top-left to top-right)
    top_left, top_right = sorted_pts_by_x[0], sorted_pts_by_x[1]

    sorted_pts_by_x = sorted(sorted_pts[2:], key=lambda p: p[0])  # Exit line (bottom-left to bottom-right)
    bottom_left, bottom_right = sorted_pts_by_x[0], sorted_pts_by_x[1]

    # Gabungkan dalam urutan untuk homografi
    ordered_pts = np.float32([top_left, top_right, bottom_left, bottom_right])

    # Tujuan bird's-eye view (misalnya 400x200)
    bird_width, bird_height = 400, 600
    dst_pts = np.float32([
        [0, 0],
        [bird_width - 1, 0],
        [0, bird_height - 1],
        [bird_width - 1, bird_height - 1]
    ])

    # Hitung homografi dari ROI ke tampilan top-down
    H_bird = cv2.getPerspectiveTransform(ordered_pts, dst_pts)

    # Terapkan ke frame
    bird_view = cv2.warpPerspective(frame, H_bird, (bird_width, bird_height))

    # Tampilkan jendela bird's eye view
    # Salin untuk menggambar di bird_view
    bird_overlay = bird_view.copy()

    # Gambar entry-exit path untuk setiap kendaraan yang sudah melewati ROI
    for objectID, times in crossing_times.items():
        if 'entry_point' in times:
            entry_pt = tuple(np.int32(times['entry_point']))
            cv2.circle(bird_overlay, entry_pt, 5, (255, 0, 0), -1)  # Biru

        if 'exit_point' in times:
            exit_pt = tuple(np.int32(times['exit_point']))
            cv2.circle(bird_overlay, exit_pt, 5, (0, 0, 255), -1)  # Merah

            # Gambar garis dari entry ke exit
            if 'entry_point' in times:
                cv2.line(bird_overlay, entry_pt, exit_pt, (255, 255, 255), 1)

            # Tampilkan ID juga
            cv2.putText(bird_overlay, f"ID {objectID}", (entry_pt[0] + 5, entry_pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Tampilkan hasil visualisasi jalur di bird-view
    cv2.imshow("Bird's Eye View", bird_overlay)
    bird_out.write(bird_overlay)


    #menggunakan bird virw
    entry_bird = ((0 + bird_width) / 2, 0 )
    exit_bird = ((0 + bird_width) / 2, bird_height )


    # Deteksi objek
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    fgmask = bgsub.apply(blur)
    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 800:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 1500:
                boxes.append((x, y, w, h))

    # Gabungkan box terdekat
    merged = []
    for box in boxes:
        x, y, w, h = box
        merged_flag = False
        for i in range(len(merged)):
            mx, my, mw, mh = merged[i]
            if abs(x - mx) < 50 and abs(y - my) < 50:
                nx = min(x, mx)
                ny = min(y, my)
                nw = max(x + w, mx + mw) - nx
                nh = max(y + h, my + mh) - ny
                merged[i] = (nx, ny, nw, nh)
                merged_flag = True
                break
        if not merged_flag:
            merged.append(box)

    centroids = []
    for x, y, w, h in merged:
        cx = x + w // 2
        cy = y + h // 2
        centroids.append((cx, cy))

    objects = tracker.update(centroids)

    def point_line_distance(p, a, b):
        px, py = p
        x1, y1 = a
        x2, y2 = b
        return abs((y2 - y1)*px - (x2 - x1)*py + x2*y1 - y2*x1) / (((y2 - y1)**2 + (x2 - x1)**2) ** 0.5)

    for objectID, (cx, cy) in objects.items():
        current_pos = (cx, cy)
        previous_pos = track_history.get(objectID, None)
        track_history[objectID] = current_pos  # Simpan posisi sekarang

        pt = np.array([[[cx, cy]]], dtype=np.float32)
        pt_bird = cv2.perspectiveTransform(pt, H_bird)[0][0]

        if objectID not in crossing_times:
            crossing_times[objectID] = {}

        # Lanjut jika belum ada posisi sebelumnya
        if previous_pos is None:
            continue

        # Deteksi interseksi garis entry
        if 'entry' not in crossing_times[objectID]:
            if intersect(previous_pos, current_pos, entry_pt1, entry_pt2):
                crossing_times[objectID]['entry'] = cap.get(cv2.CAP_PROP_POS_MSEC)
                crossing_times[objectID]['entry_point'] = pt_bird
                counted_ids.add(objectID)
                count += 1
                print(f"[ENTRY] Object {objectID} at frame {frame_number}")

        # Deteksi interseksi garis exit
        if 'entry' in crossing_times[objectID] and 'exit' not in crossing_times[objectID]:
            if intersect(previous_pos, current_pos, exit_pt1, exit_pt2):
                crossing_times[objectID]['exit'] = cap.get(cv2.CAP_PROP_POS_MSEC)
                crossing_times[objectID]['exit_point'] = pt_bird

                entry_time_ms = crossing_times[objectID]['entry']
                exit_time_ms = crossing_times[objectID]['exit']
                time_seconds = (exit_time_ms - entry_time_ms) / 1000.0
                print(FPS)

                if time_seconds > 0:
                    pt1 = crossing_times[objectID]['entry_point']
                    pt2 = crossing_times[objectID]['exit_point']
                    pixel_distance_bird = np.linalg.norm(pt2 - pt1)

                    dis = 19.0
                    pixel_to_meter_bird = dis / np.linalg.norm(np.array(exit_bird) - np.array(entry_bird))

                    speed_mps = pixel_distance_bird * pixel_to_meter_bird / time_seconds
                    speed_kmph = speed_mps * 3.6
                    estimated_speeds[objectID] = speed_kmph
                    classification = classify_speed(speed_kmph)

                    print(f"[SPEED] ID {objectID}: {speed_kmph:.2f} km/h | dist: {pixel_distance_bird * pixel_to_meter_bird:.2f} m | time: {time_seconds:.2f}s")
                    cv2.putText(frame, f"{speed_kmph:.1f} km/h", (cx - 20, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    append_csv(csv_output_path, {
                        'ID': objectID,
                        'Speed (km/h)': f"{speed_kmph:.2f}",
                        'Classification': classification,
                    })    
    y_offset = 50
    x_offset = frame.shape[1] - 300  # pojok kanan atas
    line_height = 25
    cv2.putText(frame, "ID  | Speed (km/h) | Class", (x_offset, y_offset - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    for idx, objectID in enumerate(sorted(estimated_speeds.keys())):
        speed = estimated_speeds[objectID]
        classification = classify_speed(speed)
        text = f"{objectID: <3} | {speed:6.2f}       | {classification}"
        cv2.putText(frame, text, (x_offset, y_offset + idx * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


    # Gambar kotak dan garis ROI
    for x, y, w, h in merged:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.line(frame, tuple(entry_pt1.astype(int)), tuple(entry_pt2.astype(int)), (255, 0, 255), 2)
    cv2.line(frame, tuple(exit_pt1.astype(int)), tuple(exit_pt2.astype(int)), (0, 255, 255), 2)
    cv2.putText(frame, f"Count: {count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    out.write(frame)
    cv2.imshow("ROI + Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
bird_out.release()

cv2.destroyAllWindows()
