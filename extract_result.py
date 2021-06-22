# -*- coding: utf-8 -*-
import cv2
import sys
import os
import subprocess
from multiprocessing import Process

def process_video(process_index, output_dir):
    video_cursor = 0
    spilt_video = cv2.VideoCapture(f'./.video_tmp/{process_index}.mp4')

    detector = cv2.ORB_create()
    base_result = cv2.imread('./base_result.png', cv2.IMREAD_GRAYSCALE)
    (base_kp, base_des) = detector.detectAndCompute(base_result, None)

    # Set an initial value just to start first cycle of the while loop
    end_flag = True
    result_number = 0

    while end_flag is True:
        video_cursor += 1
        end_flag, frame = spilt_video.read()

        if video_cursor % 60 != 0:
            continue

        sys.stderr.write(f'\r\033[KProgress: {int(spilt_video.get(cv2.CAP_PROP_POS_FRAMES))} / {int(spilt_video.get(cv2.CAP_PROP_FRAME_COUNT))} frames')
        sys.stderr.flush()

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        ret = {}

        grayed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (comparing_kp, comparing_des) = detector.detectAndCompute(
                grayed_frame, None)

        matches = bf.match(base_des, comparing_des)
        dist = [m.distance for m in matches]
        matchness = sum(dist) / len(dist)

        if matchness < 70.0:
            cv2.imwrite(f'{output_dir}/{process_index}_{result_number}.png',
                    frame)
            # Skip frames for 1.5min
            for v in range(60 * 90):
                spilt_video.grab()
            result_number += 1


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python extract_result.py ~/path/to/play.mp4')
        exit(1)

    video_name = sys.argv[1]
    video = cv2.VideoCapture(video_name)
    dir_name = sys.argv[1].split('.')[-2]
    os.mkdir(dir_name)

    # Split the video to process concurrently
    os.mkdir('./.video_tmp')
    total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    subprocess.run(['ffmpeg', '-i', sys.argv[1],
            '-ss', '0',
            '-t', str(total_frame // 60 // 3 - 1),
            '-c', 'copy', './.video_tmp/0.mp4'])
    subprocess.run(['ffmpeg', '-i', sys.argv[1],
            '-ss', str(total_frame // 60 // 3),
            '-t', str(total_frame // 60 // 3 - 1),
            '-c', 'copy', './.video_tmp/1.mp4'])
    subprocess.run(['ffmpeg', '-i', sys.argv[1],
            '-ss', str(total_frame // 60 // 3 * 2),
            '-c', 'copy', './.video_tmp/2.mp4'])
    process_list = []
    for i in range(3):
        p = Process(
                target=process_video,
                kwargs={'process_index': i, 'output_dir': dir_name})
        p.start()
        process_list.append(p)

    for process in process_list:
        process.join()

    video.release()
    os.remove('./.video_tmp/0.mp4')
    os.remove('./.video_tmp/1.mp4')
    os.remove('./.video_tmp/2.mp4')
    os.rmdir('./.video_tmp')
