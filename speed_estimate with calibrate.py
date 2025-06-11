import cv2
import numpy as np

# === KONFIGURASI VIDEO ===
VIDEO_PATH = 'test.mp4'  # 🔧 GANTI DENGAN FILE VIDEO KAMU
# =========================

# Variabel global untuk menyimpan titik ROI
roi_points = []

def draw_points_and_lines(img, points):
    """Menggambar ulang semua titik dan garis ROI di frame."""
    for i, pt in enumerate(points):
        cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)
        if i > 0:
            cv2.line(img, tuple(points[i-1]), tuple(pt), (255, 0, 0), 2)
    if len(points) == 4:
        cv2.line(img, tuple(points[3]), tuple(points[0]), (255, 0, 0), 2)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(roi_points) < 4:
        roi_points.append([x, y])
        img_copy = param.copy()
        draw_points_and_lines(img_copy, roi_points)
        cv2.imshow("Klik 4 Titik ROI", img_copy)

def select_initial_roi(video_path):
    global roi_points
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca video.")
        cap.release()
        return None, None
    clone = frame.copy()

    print("Klik 4 titik membentuk ROI (searah jarum jam atau berlawanan).")
    print("Tekan 'u' untuk undo, 's' untuk simpan jika sudah 4 titik, atau ESC untuk keluar.")
    cv2.imshow("Klik 4 Titik ROI", clone)
    cv2.setMouseCallback("Klik 4 Titik ROI", click_event, clone)

    while True:
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            print("Dibatalkan.")
            roi_points = []
            break
        elif key == ord('u'):  # Undo
            if roi_points:
                roi_points.pop()
                img_copy = clone.copy()
                draw_points_and_lines(img_copy, roi_points)
                cv2.imshow("Klik 4 Titik ROI", img_copy)
        elif key == ord('s'):  # Save
            if len(roi_points) == 4:
                np.save("test.npy", roi_points)
                print("Titik ROI disimpan ke saved_roi.npy")
                break
            else:
                print("Harus klik tepat 4 titik sebelum menyimpan.")

    cv2.destroyWindow("Klik 4 Titik ROI")

    if len(roi_points) != 4:
        return None, None

    return frame, np.array(roi_points, dtype=np.float32).reshape(-1, 1, 2)

# ==== Proses ROI Tracking ====
def track_roi(video_path, initial_frame, initial_roi):
    cap = cv2.VideoCapture(video_path)
    orb = cv2.ORB_create(1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ref_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
    ref_kp, ref_des = orb.detectAndCompute(ref_gray, None)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(frame_gray, None)

        if des is None or ref_des is None or len(kp) < 10:
            continue

        matches = bf.match(ref_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]

        ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        frame_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(ref_pts, frame_pts, cv2.RANSAC, 5.0)

        if H is not None:
            roi_transformed = cv2.perspectiveTransform(initial_roi, H)
            cv2.polylines(frame, [np.int32(roi_transformed)], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow("Tracking ROI Dinamis", frame)
        if cv2.waitKey(3) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# === MAIN PROGRAM ===
initial_frame, initial_roi = select_initial_roi(VIDEO_PATH)
print("Titik ROI akhir:", roi_points)

if initial_roi is not None:
    track_roi(VIDEO_PATH, initial_frame, initial_roi)
