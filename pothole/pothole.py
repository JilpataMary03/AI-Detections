import argparse
from ultralytics import YOLO
import cv2
import os
import numpy as np
import smtplib
from email.message import EmailMessage

# ---------------- Fixed Video Path ----------------
video_path = "potholee.mp4"   # 👈 change here if you want another input

# ---------------- Parse CLI Arguments ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--video-size", type=str, default="768x432", help="Output video size WxH")
parser.add_argument("--roi", nargs=4, type=int, metavar=('x1', 'y1', 'x2', 'y2'),
                    help="ROI coordinates: x1 y1 x2 y2 (optional)")
parser.add_argument("--box-mode", type=str, default="enable", choices=["enable", "disable"],
                    help="Enable/Disable pothole bounding box drawing")
parser.add_argument("--alert-mode", type=str, default="off", choices=["on", "off"],
                    help="Enable/Disable email alerts")
args = parser.parse_args()

# ---------------- Parse video size ----------------
out_w, out_h = map(int, args.video_size.lower().split("x"))
roi = args.roi
box_mode = args.box_mode
alert_mode = args.alert_mode

# ---------------- Email Config ----------------
EMAIL_SENDER = "jananir851@gmail.com"
EMAIL_PASS   = "gwtk ymar whdt nnqk"   # Gmail app password (not your normal login)
EMAIL_RECEIVER = "maryjilpata@gmail.com"

def send_email_alert(snapshot_path, pothole_id):
    msg = EmailMessage()
    msg["Subject"] = f"⚠️ Pothole Alert - ID {pothole_id}"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.set_content(f"A pothole (ID {pothole_id}) was detected. Snapshot attached.")

    with open(snapshot_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(snapshot_path)
        msg.add_attachment(file_data, maintype="image", subtype="jpeg", filename=file_name)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASS)
        smtp.send_message(msg)

    print(f"📩 Email alert sent for pothole ID {pothole_id}")

# ---------------- Load Model ----------------
model = YOLO("pothole.pt")

# ---------------- Video IO ----------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("pothole_detection_output.mp4", fourcc, fps, (out_w, out_h))

os.makedirs("snapshots", exist_ok=True)

# ---------------- Stable ID Tracker ----------------
active_ids = {}     # {custom_id: {"box": [x1,y1,x2,y2], "missed":0, "snaps":0}}
next_custom_id = 1
MAX_MISSED = 10     # frames before forgetting pothole
IOU_THRESH = 0.5    # overlap threshold

def iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2
    xi1, yi1 = max(x1, x1p), max(y1, y1p)
    xi2, yi2 = min(x2, x2p), min(y2, y2p)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2p - x1p) * (y2p - y1p)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

# ---------------- Processing Loop ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize FIRST (for speed)
    resized_frame = cv2.resize(frame, (out_w, out_h))

    # ✅ Always draw ROI if provided (independent of box-mode)
    if roi:
        rx1, ry1, rx2, ry2 = roi
        cv2.rectangle(resized_frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)  # Red ROI box

    # Run YOLO detection
    results = model.predict(resized_frame, conf=0.5, verbose=False)
    detections = results[0].boxes.xyxy.cpu().numpy().astype(int)

    # Mark all active potholes as missed by default
    for cid in list(active_ids.keys()):
        active_ids[cid]["missed"] += 1
        if active_ids[cid]["missed"] > MAX_MISSED:
            del active_ids[cid]

    # Process new detections
    for det in detections:
        x1, y1, x2, y2 = det
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # ROI check
        if roi:
            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                continue

        # Match with existing IDs using IoU
        matched_id = None
        for cid, data in active_ids.items():
            if iou(det, data["box"]) > IOU_THRESH:
                matched_id = cid
                active_ids[cid]["box"] = det
                active_ids[cid]["missed"] = 0
                break

        if matched_id is None:
            matched_id = next_custom_id
            active_ids[matched_id] = {"box": det, "missed": 0, "snaps": 0}
            next_custom_id += 1

        # ✅ Only detection boxes controlled by box-mode
        if box_mode == "enable":
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_frame, f"ID {matched_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save max 2 snapshots per ID
        if active_ids[matched_id]["snaps"] < 2:
            active_ids[matched_id]["snaps"] += 1
            snapshot_path = f"snapshots/pothole_ID{matched_id}_{active_ids[matched_id]['snaps']}.jpg"
            cv2.imwrite(snapshot_path, resized_frame)
            print(f"✅ Saved snapshot {active_ids[matched_id]['snaps']} for pothole ID {matched_id}")

            # Send email only once per ID (first snapshot only)
            if alert_mode == "on" and active_ids[matched_id]["snaps"] == 1:
                send_email_alert(snapshot_path, matched_id)

    # Show and save frame
    cv2.imshow("Pothole Detection", resized_frame)
    out.write(resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
