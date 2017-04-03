import cv2
import configparser

# TODO: better path setting option (all videos from folder)

vid_path = 'test.h264'
vid_path = 'salmon_noodles.mp4'
imgs_path = '/imgs/'

cap = cv2.VideoCapture(vid_path)

while cap.isOpened():
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.circle(gray, (700, 200), 100, 255)

    cv2.imshow('frame', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    input()

    # img_name = imgs_path + 'bla.png'
    # cv2.imwrite(img_name, frame)
    # cv2.imshow('test', frame)
    # input('enter to continue')