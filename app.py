from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import safe_join
from flask_cors import CORS
from dotenv import load_dotenv
import cv2
import numpy as np
import os
import threading
import requests
import mediapipe as mp
import base64
from collections import defaultdict
# Gunakan modul solutions standar (menghindari import ke mediapipe.python.* yang bisa bermasalah di beberapa build)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join("static", "uploads")
PROCESSED_FOLDER = os.path.join("static", "processed")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

# Pastikan folder ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# === Utility functions ===
def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def euclid(p1, p2): 
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def get_kpt_xy(k, i): 
    return (float(k[i, 0]), float(k[i, 1]))

def is_visible(k, i, thr=0.20): 
    # Lower visibility threshold for noisier frames so reps still count
    return float(k[i, 2]) >= thr

class RepetitionCounter:
    def __init__(self):
        self.pushup_cnt = 0
        self.pushup_state = "up"
        self.squat_cnt = 0
        self.squat_state = "top"
        self.jj_cnt = 0
        self.jj_state = "closed"

    def update_pushup(self, k):
        # Allow counting with only one side visible (left or right). Hip/ankle optional for plank.
        left_core = all(is_visible(k, i) for i in [11, 13, 15])
        right_core = all(is_visible(k, i) for i in [12, 14, 16])
        if not (left_core or right_core):
            return

        # If posture looks standing (head far above ankles), skip push-up counting to avoid JJ cross-detect.
        if all(is_visible(k, i) for i in [0, 27, 28]):
            head_y = get_kpt_xy(k, 0)[1]
            ankle_y = (get_kpt_xy(k, 27)[1] + get_kpt_xy(k, 28)[1]) / 2
            if (ankle_y - head_y) > 0.5 * ankle_y:  # tall posture, likely standing
                return

        elbows = []
        hips = []
        if left_core:
            elbows.append(angle(get_kpt_xy(k, 11), get_kpt_xy(k, 13), get_kpt_xy(k, 15)))
            if all(is_visible(k, i) for i in [23, 27]):
                hips.append(angle(get_kpt_xy(k, 11), get_kpt_xy(k, 23), get_kpt_xy(k, 27)))
        if right_core:
            elbows.append(angle(get_kpt_xy(k, 12), get_kpt_xy(k, 14), get_kpt_xy(k, 16)))
            if all(is_visible(k, i) for i in [24, 28]):
                hips.append(angle(get_kpt_xy(k, 12), get_kpt_xy(k, 24), get_kpt_xy(k, 28)))

        elbow = float(np.mean(elbows))
        hip_avg = float(np.mean(hips)) if hips else None
        plank_ok = (hip_avg is None) or hip_avg > 135  # skip plank check if hips not visible

        if plank_ok:
            if self.pushup_state == "up" and elbow < 110:  # looser bottom threshold
                self.pushup_state = "down"
            elif self.pushup_state == "down" and elbow > 140:  # looser top threshold
                self.pushup_cnt += 1
                self.pushup_state = "up"

    def update_squat(self, k):
        need = [23, 24, 25, 26, 27, 28]
        if not all(is_visible(k, i) for i in need):
            return
        L = angle(get_kpt_xy(k, 23), get_kpt_xy(k, 25), get_kpt_xy(k, 27))
        R = angle(get_kpt_xy(k, 24), get_kpt_xy(k, 26), get_kpt_xy(k, 28))
        knee = (L + R) / 2
        hip_y = (get_kpt_xy(k, 23)[1] + get_kpt_xy(k, 24)[1]) / 2
        knee_y = (get_kpt_xy(k, 25)[1] + get_kpt_xy(k, 26)[1]) / 2
        depth_ok = hip_y > knee_y - 5
        if self.squat_state == "top" and knee < 90 and depth_ok:
            self.squat_state = "bottom"
        elif self.squat_state == "bottom" and knee > 160:
            self.squat_cnt += 1
            self.squat_state = "top"

    def update_jj(self, k, frame_w):
        need = [15, 16, 27, 28, 0]
        if not all(is_visible(k, i) for i in need):
            return
        wristY = (get_kpt_xy(k, 15)[1] + get_kpt_xy(k, 16)[1]) / 2
        headY = get_kpt_xy(k, 0)[1]
        hands_up = wristY < headY
        ankles = euclid(get_kpt_xy(k, 27), get_kpt_xy(k, 28))
        feet_apart = ankles > 0.25 * frame_w
        feet_closed = ankles < 0.18 * frame_w

        # Hitung ketika transisi dari posisi awal (kaki rapat, tangan bawah) ke posisi open (kaki buka, tangan di atas).
        if self.jj_state == "closed" and feet_apart and hands_up:
            self.jj_cnt += 1
            self.jj_state = "open"
        elif self.jj_state == "open" and feet_closed and not hands_up:
            self.jj_state = "closed"

# Singleton pose for realtime frames and per-stream counters
POSE_SINGLETON = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
STREAM_STATE = defaultdict(lambda: RepetitionCounter())

def draw_text(img, text, x, y, scale=0.9, thickness=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def process_realtime_frame(stream_id, frame):
    """
    Process a single frame for realtime detection using shared pose model and per-stream counters.
    Returns annotated frame and current counts.
    """
    height, width = frame.shape[:2]
    results = POSE_SINGLETON.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    counter = STREAM_STATE[stream_id]

    if results.pose_landmarks:
        kpts = np.zeros((33, 3), dtype=np.float32)
        for i, lm in enumerate(results.pose_landmarks.landmark):
            kpts[i] = [lm.x * width, lm.y * height, lm.visibility]

        counter.update_pushup(kpts)
        counter.update_squat(kpts)
        counter.update_jj(kpts, width)

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=2, circle_radius=2),
        )

    draw_text(frame, f"Push-up: {counter.pushup_cnt}", 20, 40)
    draw_text(frame, f"Squat: {counter.squat_cnt}", 20, 80)
    draw_text(frame, f"Jumping Jack: {counter.jj_cnt}", 20, 120)

    return frame, counter

def process_video_mediapipe(filepath):
    counter = RepetitionCounter()
    cap = cv2.VideoCapture(filepath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    base_filename = os.path.basename(filepath)
    processed_filename = f"processed_{base_filename}"
    out_path = os.path.join(app.config["PROCESSED_FOLDER"], processed_filename)

    # Coba codec h264 (avc1) untuk kompatibilitas web, fallback ke mp4v
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Inisialisasi Pose menggunakan variabel mp_pose yang diimport langsung
    with mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose_detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(rgb)

            if results.pose_landmarks:
                kpts = np.zeros((33, 3), dtype=np.float32)
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    kpts[i] = [lm.x * width, lm.y * height, lm.visibility]

                counter.update_pushup(kpts)
                counter.update_squat(kpts)
                counter.update_jj(kpts, width)

                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=2, circle_radius=2),
                )

            draw_text(frame, f"Push-up: {counter.pushup_cnt}", 20, 40)
            draw_text(frame, f"Squat: {counter.squat_cnt}", 20, 80)
            draw_text(frame, f"Jumping Jack: {counter.jj_cnt}", 20, 120)
            writer.write(frame)

    cap.release()
    writer.release()
    return processed_filename, counter

def process_and_notify(filepath, video_id, webhook_url):
    webhook_url = webhook_url or os.getenv("LARAVEL_WEBHOOK_URL")
    try:
        processed_filename, counter = process_video_mediapipe(filepath)
        payload = {
            "video_id": video_id,
            "status": "completed",
            "processed_filename": processed_filename,
            "details": {
                "pushup": counter.pushup_cnt,
                "squat": counter.squat_cnt,
                "jj": counter.jj_cnt
            }
        }
        print(f"Sending webhook for video {video_id} to {webhook_url}")
        resp = requests.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()

    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        payload = {
            "video_id": video_id,
            "status": "failed",
            "details": {"error": str(e)}
        }
        if webhook_url:
            requests.post(webhook_url, json=payload, timeout=10)

# === API ROUTES ===
@app.route("/api/process", methods=["POST"])
@app.route("/process_video", methods=["POST"])  # backward compatibility for Laravel client
def process_video_route():
    if "video" not in request.files:
        return jsonify({"error": "Tidak ada file video."}), 400
    
    video_id = request.form.get('video_id')
    webhook_url = request.form.get('webhook_url')

    if not all([video_id, webhook_url]):
        return jsonify({"error": "video_id and webhook_url are required."}), 400

    file = request.files["video"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    thread = threading.Thread(
        target=process_and_notify,
        args=(filepath, video_id, webhook_url)
    )
    thread.start()
    return jsonify({"message": "Video accepted for processing."}), 202

@app.route("/api/download/<path:filename>", methods=["GET"])
@app.route("/download/<path:filename>", methods=["GET"])  # backward compatibility
def download_named_file(filename):
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename, as_attachment=True)

@app.route("/api/realtime_frame", methods=["POST"])
def realtime_frame():
    data = request.get_json(silent=True) or {}
    stream_id = data.get("stream_id")
    frame_b64 = data.get("frame")
    if not stream_id or not frame_b64:
        return jsonify({"error": "stream_id and frame are required"}), 400

    try:
        if frame_b64.startswith("data:"):
            frame_b64 = frame_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(frame_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"invalid frame: {e}"}), 400

    if frame is None:
        return jsonify({"error": "failed to decode frame"}), 400

    processed, counter = process_realtime_frame(stream_id, frame)
    ok, jpeg = cv2.imencode(".jpg", processed)
    if not ok:
        return jsonify({"error": "failed to encode frame"}), 500

    return jsonify({
        "pushup": counter.pushup_cnt,
        "squat": counter.squat_cnt,
        "jj": counter.jj_cnt,
        "frame": base64.b64encode(jpeg).decode("ascii"),
    })

@app.route("/api/realtime_reset", methods=["POST"])
def realtime_reset():
    data = request.get_json(silent=True) or {}
    stream_id = data.get("stream_id")
    if stream_id:
        STREAM_STATE.pop(stream_id, None)
    return jsonify({"reset": True})

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    # Bind to 0.0.0.0 so bisa diakses dari laptop lain dalam jaringan
    app.run(debug=True, host="0.0.0.0", port=7000)
