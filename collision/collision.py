import cv2
from ultralytics import YOLO
import numpy as np
import math
import time
from collections import deque, defaultdict

# ------------------------------
# CONFIG
# ------------------------------
YOLO_MODEL = "yolov8n.onnx"
SOURCE = "acc.mp4"  #your video or rtsp

RUNNING_FPS_SKIP = 1
CONF = 0.4

IOU_MATCH_THRESHOLD = 0.3
MAX_UNSEEN_FRAMES = 15

EDGE_DISTANCE_FACTOR = 0.15
IOU_COLLISION_THRESHOLD = 0.05

COLLAPSE_HEIGHT_RATIO = 0.60
COLLAPSE_ASPECT_RATIO = 0.75
COLLAPSE_FRAMES_REQUIRED = 3

VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "motorcycle", "bicycle", "bike"}

# ------------------------------
# UTILITIES
# ------------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-9)

def center(box):
    return ((box[0]+box[2])/2.0, (box[1]+box[3])/2.0)

def edge_distance(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return math.hypot(dx, dy)

# ------------------------------
# TRACKER
# ------------------------------
class Track:
    def __init__(self, tid, box, cls_name, score, frame_idx):
        self.id = tid
        self.box = box
        self.cls = cls_name
        self.score = score
        self.last_seen = frame_idx
        self.age = 0
        self.unseen = 0
        self.history = deque(maxlen=30)
        self.history.append(box)
        self.baseline_heights = deque(maxlen=10)
        self.collapse_counter = 0
        self.flag_collapsed = False
        # collision persistence counter
        self.collision_counter = defaultdict(int)

    def update(self, box, score, frame_idx):
        self.box = box
        self.score = score
        self.last_seen = frame_idx
        self.unseen = 0
        self.age += 1
        self.history.append(box)

class SimpleTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def step(self, detections, classes, scores, frame_idx):
        assigned = [-1] * len(detections)
        if len(self.tracks) > 0 and len(detections) > 0:
            iou_mat = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for t_idx, tr in enumerate(self.tracks):
                for d_idx, box in enumerate(detections):
                    iou_mat[t_idx, d_idx] = iou(tr.box, box)
            t_used, d_used = set(), set()
            while True:
                t_idx, d_idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                val = iou_mat[t_idx, d_idx]
                if val < IOU_MATCH_THRESHOLD:
                    break
                if t_idx in t_used or d_idx in d_used:
                    iou_mat[t_idx, d_idx] = 0
                    continue
                assigned[d_idx] = self.tracks[t_idx].id
                self.tracks[t_idx].update(detections[d_idx], scores[d_idx], frame_idx)
                t_used.add(t_idx); d_used.add(d_idx)
                iou_mat[t_idx, :] = 0
                iou_mat[:, d_idx] = 0
        for d_idx, box in enumerate(detections):
            if assigned[d_idx] == -1:
                tr = Track(self.next_id, box, classes[d_idx], scores[d_idx], frame_idx)
                if classes[d_idx] == "person":
                    tr.baseline_heights.append(box[3]-box[1])
                self.tracks.append(tr)
                assigned[d_idx] = self.next_id
                self.next_id += 1
        for tr in self.tracks[:]:
            if tr.last_seen != frame_idx:
                tr.unseen += 1
            if tr.unseen > MAX_UNSEEN_FRAMES:
                self.tracks.remove(tr)
        return assigned

    def get_tracks(self):
        return self.tracks

# ------------------------------
# MAIN
# ------------------------------
def main():
    model = YOLO(YOLO_MODEL)
    cap = cv2.VideoCapture(SOURCE, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except:
        pass

    tracker = SimpleTracker()
    frame_idx = 0
    last_time = time.time()

    # --------------------------
    # INCIDENT SNAPSHOT TRACKING
    # --------------------------
    incident_captured = set()
    snapshot_counter = 0
    def save_snapshot(frame, label):
        nonlocal snapshot_counter
        snapshot_counter += 1
        fname = f"snap_{snapshot_counter}_{label}.jpg"
        cv2.imwrite(fname, frame)
        print(f"[INFO] Snapshot saved: {fname}")

    # helper function for speed
    def speed(tr):
        if len(tr.history)<2: return 0
        c_prev = center(tr.history[-2])
        c_now = center(tr.history[-1])
        return math.hypot(c_now[0]-c_prev[0], c_now[1]-c_prev[1])

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_idx += 1

        detections, classes, scores = [], [], []
        if frame_idx % RUNNING_FPS_SKIP == 0:
            results = model(frame, conf=CONF, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0].item())
                name = model.names[cls]
                conf_score = float(box.conf[0].item()) if hasattr(box, 'conf') else 0.0
                if (x2-x1)<10 or (y2-y1)<10: continue
                detections.append((x1,y1,x2,y2))
                classes.append(name)
                scores.append(conf_score)

        assigned_ids = tracker.step(detections, classes, scores, frame_idx)

        persons, vehicles = [], []
        for d_idx, tr_id in enumerate(assigned_ids):
            box = detections[d_idx]
            cls_name = classes[d_idx]
            tr_obj = next((tr for tr in tracker.get_tracks() if tr.id==tr_id), None)
            if tr_obj is None: continue
            if cls_name=="person":
                cur_h = box[3]-box[1]
                if len(tr_obj.baseline_heights) < tr_obj.baseline_heights.maxlen:
                    tr_obj.baseline_heights.append(cur_h)
                persons.append((tr_obj, box))
            elif cls_name in VEHICLE_CLASSES:
                vehicles.append((tr_obj, box))

        # ----------------------------
        # VEHICLE COLLISION
        # ----------------------------
        for i in range(len(vehicles)):
            for j in range(i+1,len(vehicles)):
                t1,b1 = vehicles[i]
                t2,b2 = vehicles[j]
                avg_w = ((b1[2]-b1[0])+(b2[2]-b2[0]))/2.0
                avg_h = ((b1[3]-b1[1])+(b2[3]-b2[1]))/2.0
                edge_thresh = EDGE_DISTANCE_FACTOR*(avg_w+avg_h)/2.0
                close = iou(b1,b2)>=IOU_COLLISION_THRESHOLD or edge_distance(b1,b2)<edge_thresh
                slow = speed(t1)<2.0 and speed(t2)<2.0
                incident_id = f"VV_{t1.id}_{t2.id}"
                if close and slow:
                    t1.collision_counter[t2.id] += 1
                    t2.collision_counter[t1.id] += 1
                    if t1.collision_counter[t2.id]>=2:
                        x=min(b1[0],b2[0]);y=min(b1[1],b2[1])-10
                        cv2.putText(frame,f"VEHICLE-VEHICLE COLLISION ({t1.id},{t2.id})",(x,max(10,y)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
                        cv2.rectangle(frame,(b1[0],b1[1]),(b1[2],b1[3]),(0,255,255),2)
                        cv2.rectangle(frame,(b2[0],b2[1]),(b2[2],b2[3]),(0,255,255),2)
                        if incident_id not in incident_captured:
                            save_snapshot(frame, incident_id)
                            incident_captured.add(incident_id)
                else:
                    t1.collision_counter[t2.id]=0
                    t2.collision_counter[t1.id]=0

        # ----------------------------
        # PERSON-PERSON COLLISION
        # ----------------------------
        for i in range(len(persons)):
            for j in range(i+1,len(persons)):
                t1,b1 = persons[i]; t2,b2 = persons[j]
                avg_w = ((b1[2]-b1[0])+(b2[2]-b2[0]))/2.0
                avg_h = ((b1[3]-b1[1])+(b2[3]-b2[1]))/2.0
                edge_thresh = EDGE_DISTANCE_FACTOR*(avg_w+avg_h)/2.0
                close = iou(b1,b2)>=IOU_COLLISION_THRESHOLD or edge_distance(b1,b2)<edge_thresh
                slow = speed(t1)<2.0 and speed(t2)<2.0
                incident_id = f"PP_{t1.id}_{t2.id}"
                if close and slow:
                    t1.collision_counter[t2.id] += 1
                    t2.collision_counter[t1.id] += 1
                    if t1.collision_counter[t2.id]>=2:
                        x=min(b1[0],b2[0]);y=min(b1[1],b2[1])-10
                        cv2.putText(frame,f"PERSON-PERSON COLLISION ({t1.id},{t2.id})",(x,max(10,y)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
                        cv2.rectangle(frame,(b1[0],b1[1]),(b1[2],b1[3]),(255,0,255),2)
                        cv2.rectangle(frame,(b2[0],b2[1]),(b2[2],b2[3]),(255,0,255),2)
                        if incident_id not in incident_captured:
                            save_snapshot(frame, incident_id)
                            incident_captured.add(incident_id)
                else:
                    t1.collision_counter[t2.id]=0
                    t2.collision_counter[t1.id]=0

        # ----------------------------
        # PERSON-VEHICLE COLLISION
        # ----------------------------
        for tp,bp in persons:
            for tv,bv in vehicles:
                avg_w = ((bp[2]-bp[0])+(bv[2]-bv[0]))/2.0
                avg_h = ((bp[3]-bp[1])+(bv[3]-bv[1]))/2.0
                edge_thresh = EDGE_DISTANCE_FACTOR*(avg_w+avg_h)/2.0
                close = iou(bp,bv)>=IOU_COLLISION_THRESHOLD or edge_distance(bp,bv)<edge_thresh
                slow = speed(tp)<2.0 and speed(tv)<2.0
                incident_id = f"PV_{tp.id}_{tv.id}"
                if close and slow:
                    tp.collision_counter[tv.id] += 1
                    tv.collision_counter[tp.id] += 1
                    if tp.collision_counter[tv.id]>=2:
                        x=min(bp[0],bv[0]);y=min(bp[1],bv[1])-10
                        cv2.putText(frame,f"PERSON-VEHICLE COLLISION ({tp.id},{tv.id})",(x,max(10,y)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,128,255),2)
                        cv2.rectangle(frame,(bp[0],bp[1]),(bp[2],bp[3]),(0,128,255),2)
                        cv2.rectangle(frame,(bv[0],bv[1]),(bv[2],bv[3]),(0,128,255),2)
                        if incident_id not in incident_captured:
                            save_snapshot(frame, incident_id)
                            incident_captured.add(incident_id)
                else:
                    tp.collision_counter[tv.id]=0
                    tv.collision_counter[tp.id]=0

        # ----------------------------
        # PERSON COLLAPSE
        # ----------------------------
        for tr,b in persons:
            x1,y1,x2,y2 = b
            cur_h = y2-y1
            cur_w = x2-x1
            baseline = float(np.median(np.array(tr.baseline_heights))) if len(tr.baseline_heights)>0 else cur_h
            if cur_h>baseline*1.05:
                tr.baseline_heights.append(cur_h)
                baseline=float(np.median(np.array(tr.baseline_heights)))
            height_ratio = cur_h/(baseline+1e-9)
            aspect = cur_w/(cur_h+1e-9)
            collapsed_candidate=(height_ratio<COLLAPSE_HEIGHT_RATIO) and (aspect>COLLAPSE_ASPECT_RATIO)
            tr.collapse_counter = tr.collapse_counter+1 if collapsed_candidate else max(0,tr.collapse_counter-1)
            tr.flag_collapsed=tr.collapse_counter>=COLLAPSE_FRAMES_REQUIRED
            color=(0,0,255) if tr.flag_collapsed else (0,255,0)
            # cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            label=f"ID:{tr.id} P:{tr.score:.2f}"+("  COLLAPSED" if tr.flag_collapsed else "")
            cv2.putText(frame,label,(x1,max(15,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

            incident_id = f"C_{tr.id}"
            if tr.flag_collapsed and incident_id not in incident_captured:
                save_snapshot(frame, incident_id)
                incident_captured.add(incident_id)

        # draw vehicles
        for tr,b in vehicles:
            x1,y1,x2,y2 = b
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,200,0),2)
            cv2.putText(frame,f"ID:{tr.id} V",(x1,max(15,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,200,0),2)

        # FPS
        now=time.time()
        fps=1.0/(now-last_time) if now-last_time>0 else 0.0
        last_time=now
        cv2.putText(frame,f"FPS:{fps:.1f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        cv2.namedWindow("AI Detection",cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("AI Detection",1280,720)
        cv2.imshow("AI Detection",frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()

