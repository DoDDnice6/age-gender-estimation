import cv2

def create_video():
    video_capture=cv2.VideoCapture(0)
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('_output.avi', fourcc, 15, (w, h))

    while True:
        ret, frame = video_capture.read()
        if ret != True:
            break
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    create_video()