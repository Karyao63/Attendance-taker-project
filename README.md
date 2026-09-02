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
