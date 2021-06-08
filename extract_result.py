# -*- coding: utf-8 -*-
import cv2
import sys
import os

if len(sys.argv) < 2:
    print('usage: python extract_result.py ~/Desktop/play.mp4')
    exit(1)

video = cv2.VideoCapture(sys.argv[1])
dir_name = sys.argv[1].split('.')[-2]
os.mkdir(dir_name)

video_cursor = 0

detector = cv2.ORB_create()
base_result = cv2.imread('./base_result.png', cv2.IMREAD_GRAYSCALE)
(base_kp, base_des) = detector.detectAndCompute(base_result, None)

# Set an initial value just to start first cycle of the while loop
end_flag = True
result_number = 0

while end_flag is True:
    video_cursor += 1
    end_flag, frame = video.read()

    if video_cursor % 60 != 0:
        continue

    sys.stderr.write(f'\r\033[KProgress: {int(video.get(cv2.CAP_PROP_POS_FRAMES))} / {int(video.get(cv2.CAP_PROP_FRAME_COUNT))} frames')

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    ret = {}

    grayed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (comparing_kp, comparing_des) = detector.detectAndCompute(
            grayed_frame, None)

    matches = bf.match(base_des, comparing_des)
    dist = [m.distance for m in matches]
    matchness = sum(dist) / len(dist)

    if matchness < 70.0:
        cv2.imwrite(f'{dir_name}/{result_number}.png', frame)
        # Skip frames for 1.5min
        for v in range(60 * 90):
            video.grab()
        result_number += 1

video.release()
