import cv2
import subprocess

# 1) 读本地视频源（可改成 /dev/video0 或本地文件）
cap = cv2.VideoCapture('/dev/video0')
ok, frame = cap.read()
if not ok:
    raise RuntimeError("无法读取视频源")

camera_input_fps = 30
h, w = frame.shape[:2]
output_fps = 10

# 2) FFmpeg 管道：从 stdin 收 raw bgr 帧，编码后推 RTSP
cmd = [
    "ffmpeg",
    "-re",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{w}x{h}",
    "-r", str(int(output_fps)),
    "-i", "-",
    "-an",
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-tune", "zerolatency",
    "-f", "rtsp",
    "-rtsp_transport", "tcp",
    "rtsp://127.0.0.1:8554/lpr"
]

proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

try:
    # 先把第一帧写进去
    proc.stdin.write(frame.tobytes())
    cnt = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # 这里做你的 OpenCV 处理
        # frame = your_process(frame)
        if cnt % 3 == 0:
            proc.stdin.write(frame.tobytes())
        cnt += 1
finally:
    cap.release()
    if proc.stdin:
        proc.stdin.close()
    proc.wait()
