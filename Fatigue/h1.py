#此处换为你自己的地址
import subprocess as sp
import cv2
rtsp_url = 'rtsp://127.0.0.1:8554/video'
cap = cv2.VideoCapture(0)
# Get video information
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
command = ['ffmpeg',
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-pix_fmt', 'bgr24',
               '-s', "{}x{}".format(width, height),
               '-r', str(fps),
               '-i', '-',
               '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p',
               '-preset', 'ultrafast',
               '-f', 'rtsp',
               rtsp_url]
p = sp.Popen(command, stdin=sp.PIPE)

while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        print("Opening camera is failed")
        break
    p.stdin.write(frame.tostring())
