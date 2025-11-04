from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ultralytics import YOLO
import cv2, numpy as np, os, math

app = Flask(__name__)
UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = YOLO("yolov8n-pose.pt")

def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def euclid(p1, p2):
    p1, p2 = np.array(p1), np.array(p2)
    return float(np.linalg.norm(p1 - p2))

def get_kpt_xy(kpts, idx):
    return (float(kpts[idx,0]), float(kpts[idx,1]))

def is_visible(kpts, idx, thr=0.25):
    return float(kpts[idx,2]) >= thr

class RepetitionCounter:
    def __init__(self):
        self.pushup_cnt = 0; self.pushup_state = "up"
        self.squat_cnt  = 0; self.squat_state  = "top"
        self.jj_cnt     = 0; self.jj_state     = "closed"

    def update_pushup(self, k):
        if not all(is_visible(k, i) for i in [5,6,7,8,9,10,11,12,15,16]): return
        L = angle(get_kpt_xy(k,5), get_kpt_xy(k,7), get_kpt_xy(k,9))
        R = angle(get_kpt_xy(k,6), get_kpt_xy(k,8), get_kpt_xy(k,10))
        elbow = (L+R)/2
        hipL = angle(get_kpt_xy(k,5), get_kpt_xy(k,11), get_kpt_xy(k,15))
        hipR = angle(get_kpt_xy(k,6), get_kpt_xy(k,12), get_kpt_xy(k,16))
        plank = (hipL > 155 and hipR > 155)
        down_thr, up_thr = 90, 150
        if plank:
            if self.pushup_state == "up" and elbow < down_thr:
                self.pushup_state = "down"
            elif self.pushup_state == "down" and elbow > up_thr:
                self.pushup_cnt += 1
                self.pushup_state = "up"

    def update_squat(self, k):
        if not all(is_visible(k, i) for i in [11,12,13,14,15,16]): return
        L = angle(get_kpt_xy(k,11), get_kpt_xy(k,13), get_kpt_xy(k,15))
        R = angle(get_kpt_xy(k,12), get_kpt_xy(k,14), get_kpt_xy(k,16))
        knee = (L+R)/2
        hip_y = (get_kpt_xy(k,11)[1] + get_kpt_xy(k,12)[1]) / 2
        knee_y = (get_kpt_xy(k,13)[1] + get_kpt_xy(k,14)[1]) / 2
        depth_ok = hip_y > knee_y - 5
        bottom_thr, top_thr = 90, 160
        if self.squat_state == "top" and knee < bottom_thr and depth_ok:
            self.squat_state = "bottom"
        elif self.squat_state == "bottom" and knee > top_thr:
            self.squat_cnt += 1
            self.squat_state = "top"

    def update_jj(self, k, frame_w):
        if not all(is_visible(k, i) for i in [5,6,9,10,15,16,0]): return
        wristY = (get_kpt_xy(k,9)[1] + get_kpt_xy(k,10)[1]) / 2
        headY  = get_kpt_xy(k,0)[1]
        hands_up = wristY < headY
        ankles = euclid(get_kpt_xy(k,15), get_kpt_xy(k,16))
        feet_apart = ankles > 0.25 * frame_w
        if self.jj_state == "closed" and hands_up and feet_apart:
            self.jj_state = "open"
        elif self.jj_state == "open" and (not hands_up or not feet_apart):
            self.jj_cnt += 1
            self.jj_state = "closed"

def draw_text(img, text, x, y, scale=0.9, thickness=2):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+3, cv2.LINE_AA)
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness, cv2.LINE_AA)

def process_video(in_path, out_path="static/output.mp4", img_size=640):
    cap = cv2.VideoCapture(in_path)
    assert cap.isOpened(), "Error: Could not open video file."
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W,H))

    counter = RepetitionCounter()

    while True:
        ok, frame = cap.read()
        if not ok: break
        res = model.predict(source=frame, imgsz=img_size, conf=0.25, verbose=False)[0]
        if len(res.keypoints) > 0:
            scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.array([1.0]*len(res.keypoints))
            best_i = int(np.argmax(scores))
            kpts = res.keypoints[best_i].data[0].cpu().numpy().reshape(-1,3)

            def is_standing(kpts):
                head_y = get_kpt_xy(kpts, 0)[1]
                ankle_y = (get_kpt_xy(kpts,15)[1] + get_kpt_xy(kpts,16)[1]) / 2
                return (ankle_y - head_y) > 0.5 * ankle_y

            if is_standing(kpts):
                counter.update_jj(kpts, frame_w=W)
            else:
                counter.update_pushup(kpts)
                counter.update_squat(kpts)

            frame = res.plot()
        draw_text(frame, f"Push-up: {counter.pushup_cnt}", 20, 40)
        draw_text(frame, f"Squat  : {counter.squat_cnt}", 20, 80)
        draw_text(frame, f"J. Jack: {counter.jj_cnt}",   20, 120)
        writer.write(frame)

    cap.release(); writer.release()
    return out_path, counter


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "video" not in request.files:
            return redirect(request.url)
        file = request.files["video"]
        if file.filename == "":
            return redirect(request.url)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        out_path, counter = process_video(filepath)
        return render_template(
            "index.html", 
            output_video=url_for("static", filename="output.mp4"),
            pushup=counter.pushup_cnt,
            squat=counter.squat_cnt,
            jj=counter.jj_cnt
        )
    return render_template("index.html")

@app.route("/download")
def download_file():
    return send_from_directory(app.config["UPLOAD_FOLDER"], "output.mp4", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
