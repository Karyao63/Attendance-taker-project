"""
Intelligent Attendance Tracker (GUI) Webcam + Image modes, CSV logging

Requirements:
- Python 3.9+
- pip install opencv-python face_recognition numpy pandas Pillow filelock ttkbootstrap
- install cmake and visual tools

Notes:
- Put your dataset folder named `dataset/` in the same directory as this script.
  dataset/
    John_Tan/
      1.jpg
      2.jpg
    Aisyah_Nur/
      1.png

- This program loads all images in each subfolder of `dataset/` and computes face encodings.
  Each subfolder name is used as the student name/ID.

- The GUI supports:
  * Webcam mode: live camera recognition and automatic CSV logging
  * Image mode: choose a photo file and recognize any students in it

- Output: `attendance.csv` is created/appended. Columns: date, time, name, mode, source

- Press the "Start Webcam" button to begin. Press "Stop" to stop it.
- In webcam mode, recognized students are logged only once per run (you can reset list).

"""

import os
import cv2
import face_recognition
import numpy as np
import pandas as pd
import datetime
import threading
import filelock
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
import threading


# ----- Configuration -----
DATASET_DIR = "dataset"
ATTENDANCE_CSV = "attendance.csv"
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
FACE_MATCH_TOLERANCE = 0.5  # lower is stricter; 0.4-0.6 recommended
FRAME_RATE_MS = 33  # ~30 FPS (1000ms / 30 ≈ 33.33ms)
# -------------------------


class AttendanceSystem:
    def __init__(self, dataset_dir=DATASET_DIR):
        self.dataset_dir = dataset_dir
        self.known_encodings = []
        self.known_names = []
        self.recognized_this_session = set()
        self.load_dataset()

    def load_dataset(self):
        """Load dataset images, compute face encodings, and map to student names."""
        if not os.path.exists(self.dataset_dir):
            print(f"Warning: Dataset folder not found: {self.dataset_dir}")
            return

        for student in os.listdir(self.dataset_dir):
            student_folder = os.path.join(self.dataset_dir, student)
            if not os.path.isdir(student_folder):
                continue

            for filename in os.listdir(student_folder):
                base, ext = os.path.splitext(filename.lower())
                if ext not in [".jpg", ".jpeg", ".png"]:
                    continue
                path = os.path.join(student_folder, filename)
                try:
                    img = face_recognition.load_image_file(path)
                    boxes = face_recognition.face_locations(img)
                    if len(boxes) == 0:
                        print(f"Warning: no face found in {path}, skipping")
                        continue
                    encs = face_recognition.face_encodings(img, boxes)
                    if len(encs) == 0:
                        print(f"Warning: no encodings for {path}, skipping")
                        continue
                    for e in encs:
                        self.known_encodings.append(e)
                        self.known_names.append(student)
                except Exception as e:
                    print(f"Failed to process {path}: {e}")

        print(f"Dataset loaded: {len(set(self.known_names))} students, {len(self.known_encodings)} encodings")

    def recognize_faces_in_frame(self, frame_bgr):
        """Given a BGR frame from OpenCV, return list of (name, box) for recognized faces."""
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb)
            face_encodings = face_recognition.face_encodings(rgb, face_locations)
        except Exception as e:
            print(f"Face recognition error: {e}")
            return []

        results = []
        for encoding, location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.known_encodings, encoding, tolerance=FACE_MATCH_TOLERANCE)
            name = "Unknown"
            if True in matches:
                matched_indexes = [i for i, m in enumerate(matches) if m]
                matched_names = [self.known_names[i] for i in matched_indexes]
                name = max(set(matched_names), key=matched_names.count)
            results.append((name, location))
        return results

    def log_attendance(self, name, mode, source="webcam"):
        """Append an attendance record to CSV. Avoid duplicates per session."""
        if name == "Unknown" or name in self.recognized_this_session:
            return
        self.recognized_this_session.add(name)

        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S.%f")
        row = {"date": date_str, "time": time_str, "name": name, "mode": mode, "source": source}

        with filelock.FileLock(ATTENDANCE_CSV + ".lock", timeout=10):
            df = pd.DataFrame([row])
            header = not os.path.exists(ATTENDANCE_CSV)
            df.to_csv(ATTENDANCE_CSV, mode="a", header=header, index=False)
        print(f"Logged: {row}")


class AttendanceGUI(ttkb.Window):
    def __init__(self, attendance_system: AttendanceSystem):
        super().__init__(themename="darkly")
        self.title("Intelligent Attendance System")
        self.minsize(920, 640)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.system = attendance_system
        self.cap = None
        self.video_running = False
        self.frame_count = 0
        self.process_every_n_frames = 5
        self.last_boxes = []
        self.current_frame = None  # Stores full-res frame with boxes

        self._build_ui()
        self._start_status_updater()
        self._bind_resize()

    # ------------------------------------------------------------------
    # UI: TEXT BUTTONS + SUMMARY LABEL
    # ------------------------------------------------------------------
    def _build_ui(self):
        paned = ttkb.PanedWindow(self, orient=HORIZONTAL)
        paned.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # ----- VIDEO -----
        video_frame = ttkb.Frame(paned, bootstyle="dark")
        paned.add(video_frame, weight=3)
        self.canvas = ttkb.Label(video_frame, bootstyle="dark", anchor="center")
        self.canvas.pack(fill=BOTH, expand=True)

        # ----- CONTROLS -----
        ctrl_frame = ttkb.Frame(paned, bootstyle="secondary", padding=15)
        paned.add(ctrl_frame, weight=1)

        ttkb.Label(ctrl_frame, text="Attendance Controls",
                   font=("Helvetica", 14, "bold"),
                   bootstyle="inverse-secondary").pack(anchor="w", pady=(0, 12))

        btn_cfg = dict(pady=5, fill=X)

        self.start_btn = ttkb.Button(ctrl_frame, text="Start Webcam",
                                     command=self.start_webcam, bootstyle="success")
        self.start_btn.pack(**btn_cfg)

        self.stop_btn = ttkb.Button(ctrl_frame, text="Stop",
                                    command=self.stop_webcam, state=DISABLED, bootstyle="danger")
        self.stop_btn.pack(**btn_cfg)

        self.image_btn = ttkb.Button(ctrl_frame, text="Recognize Image",
                                     command=self.open_image, bootstyle="info")
        self.image_btn.pack(**btn_cfg)

        self.reset_btn = ttkb.Button(ctrl_frame, text="Reset Session",
                                     command=self.reset_session, bootstyle="warning")
        self.reset_btn.pack(pady=(15, 2), fill=X)

        # ---- SUMMARY LABEL ----
        self.summary_label = ttkb.Label(ctrl_frame, text="Present Today: 0",
                                        font=("Helvetica", 11), bootstyle="inverse-secondary")
        self.summary_label.pack(anchor="w", pady=(10, 5))

        # ---- TREEVIEW (Name + Time) ----
        tree_frame = ttkb.Frame(ctrl_frame)
        tree_frame.pack(fill=BOTH, expand=True)

        cols = ("name", "time")
        self.tree = ttkb.Treeview(tree_frame, columns=cols, show="headings",
                                  bootstyle="dark", height=12)
        self.tree.heading("name", text="Student")
        self.tree.heading("time", text="Time")
        self.tree.column("name", width=160, anchor="w")
        self.tree.column("time", width=100, anchor="center")
        self.tree.pack(side=LEFT, fill=BOTH, expand=True)

        scroll = ttkb.Scrollbar(tree_frame, orient=VERTICAL, command=self.tree.yview,
                                bootstyle="round")
        scroll.pack(side=RIGHT, fill=Y)
        self.tree.configure(yscrollcommand=scroll.set)

        # ---- STATUS BAR ----
        self.status_var = ttkb.StringVar(value="Idle")
        status = ttkb.Label(self, textvariable=self.status_var,
                            bootstyle="inverse-dark", padding=(10, 4), anchor="w")
        status.pack(side=BOTTOM, fill=X)


    def _bind_resize(self):
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def _on_canvas_resize(self, event=None):
        if hasattr(self, "current_frame"):
            self._display_frame(self.current_frame)

    def _start_status_updater(self):
        def update():
            status = "Webcam Active" if self.video_running else "Idle"
            self.status_var.set(f"Status: {status}")
            self.after(1000, update)
        update()

    def start_webcam(self):
        if self.video_running: return
        for i in range(3):
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened(): break
        if not self.cap.isOpened():
            ttkb.dialogs.Messagebox.show_error("Cannot open webcam.", "Camera Error")
            return

        self.video_running = True
        self.start_btn.config(state=DISABLED)
        self.stop_btn.config(state=NORMAL)
        self.video_loop()

    def stop_webcam(self):
        if not self.video_running: return
        self.video_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state=NORMAL)
        self.stop_btn.config(state=DISABLED)

    def video_loop(self):
        if not self.video_running or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_webcam()
            return

        # --- Always draw last known boxes ---
        if hasattr(self, "last_boxes"):
            for name, (top, right, bottom, left) in self.last_boxes:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- Recognize only every N frames ---
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames == 0:
            # Run recognition in background
            def recognize():
                small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                results = self.system.recognize_faces_in_frame(small)
                scaled = [(name, (t*2, r*2, b*2, l*2)) for name, (t, r, b, l) in results]
                self.last_boxes = scaled  # Update shared state

                # Log attendance
                for name, _ in scaled:
                    if name != "Unknown":
                        self.system.log_attendance(name, mode="webcam")
                        self.after(0, lambda n=name: self._add_to_tree(n))

            threading.Thread(target=recognize, daemon=True).start()

        
        self.current_frame = frame.copy()
        self._display_frame(frame)

        self.after(33, self.video_loop)

    def _display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w <= 1 or h <= 1: return
        ratio = min(w / pil.width, h / pil.height)
        new_w, new_h = int(pil.width * ratio), int(pil.height * ratio)
        resized = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(resized)
        self.canvas.configure(image=imgtk)
        self.canvas.image = imgtk

    # ------------------------------------------------------------------
    # IMAGE RECOGNITION
    # ------------------------------------------------------------------
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Choose image file",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not path: return
        img = cv2.imread(path)
        if img is None:
            ttkb.dialogs.Messagebox.show_error("Cannot open image.", "File Error")
            return

        results = self.system.recognize_faces_in_frame(img)
        for name, (top, right, bottom, left) in results:
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if name != "Unknown":
                self.system.log_attendance(name, mode="image", source=os.path.basename(path))
                self._add_to_tree(name)

        self.current_frame = img.copy()
        self._display_frame(img)
        self.status_var.set(f"Image Processed: {os.path.basename(path)}")

    # ------------------------------------------------------------------
    # TREEVIEW + SUMMARY UPDATE
    # ------------------------------------------------------------------
    def _add_to_tree(self, name):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        if name not in [self.tree.item(i)["values"][0] for i in self.tree.get_children()]:
            self.tree.insert("", END, values=(name, now))
            self._update_summary()

    def _update_summary(self):
        count = len(self.tree.get_children())
        self.summary_label.config(text=f"Present Today: {count} student{'s' if count != 1 else ''}")

    def reset_session(self):
        self.system.recognized_this_session.clear()
        for i in self.tree.get_children():
            self.tree.delete(i)
        self._update_summary()
        self.status_var.set("Session reset")

    def on_close(self):
        self.stop_webcam()
        self.destroy()

if __name__ == '__main__':
    try:
        system = AttendanceSystem(DATASET_DIR)
        if not system.known_encodings:
            messagebox.showwarning("Dataset Warning", "No valid face encodings found in dataset. System will run but cannot recognize faces.")
        app = AttendanceGUI(system)
        app.mainloop()
    except Exception as e:
        messagebox.showerror("Dataset Error", f"Failed to load dataset: {e}")
        raise