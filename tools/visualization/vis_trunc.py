import os
import cv2
import argparse
import natsort
from PIL import Image
import numpy as np
from tqdm import tqdm
from natsort import natsorted

# Define a color map to associate colors with object IDs
color_map = {
    i: tuple(np.random.randint(0, 256, 3).tolist()) for i in range(1, 999)
}


def to_mp4(image_dir, gt, output, log_file=None):
    gt_dict = {}
    if gt is not None:
        with open(gt, 'r') as file:
            lines = file.readlines()

        for line in lines:
            data = line.strip().split(',')
            frame_index = int(data[0])
            target_id = int(data[1])
            bb_left = int(data[2])
            bb_top = int(data[3])
            w = float(data[4])
            h = float(data[5])
            conf = int(data[6])
            cat = int(data[7])
            trunc = int(data[8])
            occl = int(data[9])

            if frame_index not in gt_dict:
                gt_dict[frame_index] = {}

            gt_dict[frame_index][target_id] = {
                'bb_left': bb_left,
                'bb_top': bb_top,
                'width': w,
                'height': h,
                'confidence': conf,
                'object_category': cat,
                'truncation': trunc,
                'occlusion': occl
            }

    image_files = sorted(os.listdir(image_dir))

    # frame index so that we can retrieve GT bounding boxes
    frame_index = 1
    pause = False

    # Get the dimensions of the first frame
    first_frame = cv2.imread(os.path.join(image_dir, image_files[0]))
    frame_height, frame_width, _ = first_frame.shape

    # Initialize the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output, fourcc, 30, (frame_width, frame_height))

    for image_file in image_files:
        if pause:
            c = cv2.waitKey(1)
            if c == 32:  # space bar pressed
                if not pause:
                    pause = True
                else:
                    pause = False
            continue
        
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        
        # plot bounding boxes
        if frame_index in gt_dict:
            frame_gt = gt_dict[frame_index]
            for target_id in frame_gt:
                x, y, w, h, conf, obj_category, truncation, occlusion = (
                    frame_gt[target_id]["bb_left"],
                    frame_gt[target_id]["bb_top"],
                    frame_gt[target_id]["width"],
                    frame_gt[target_id]["height"],
                    frame_gt[target_id]["confidence"],
                    frame_gt[target_id]["object_category"],
                    frame_gt[target_id]["truncation"],
                    frame_gt[target_id]["occlusion"],
                )
                # if conf:
                x = int(float(x))
                y = int(float(y))
                w = int(float(w))
                h = int(float(h))
    
                # Get the color for the object ID from the color_map
                color = color_map.get(target_id, (0, 0, 0))
                truncation_color = (0, 255, 0) if truncation == 0 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), truncation_color, 2)


                # cv2.putText(
                #     frame,
                #     f'ID: {target_id}',
                #     (x, y),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.5,
                #     color,
                #     thickness=2,
                # )
                
                cv2.putText(
                    frame,
                    f'Class: {obj_category}',
                    (x, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    thickness=1,
                )
                cv2.putText(
                    frame,
                    f'Frame ID: {frame_index}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),  # Red color (BGR format)
                    thickness=2,
                )
        else:
            print(f'{image_path} {frame_index}th frame has no annotation')
            
            # save log if provided
            if log_file:
                with open(log_file, 'a') as log:
                    log.write(f'{image_path} {frame_index}th frame has no annotation\n')
            
        # Write the frame to the output video
        output_video.write(frame)
        
        cv2.imshow("frame", frame)
        frame_index += 1

        c = cv2.waitKey(33)
        if c == 32:  # space bar pressed
            if not pause:
                pause = True
            else:
                pause = False
        if c & 0xFF == ord("q"):
            break

    # Release the VideoWriter and close the output video file
    output_video.release()
    cv2.destroyAllWindows()

    print(f"MP4 video file saved as {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", required=True, help="Specify the directory visdrone"
    )

    parser.add_argument(
        "--output", required=False, default="result", help="Specify the output folder for mp4 files"
    )
    
    parser.add_argument(
        "--log_file", required=False, action="store_true", help="Enable log file"
    ) 

    args = parser.parse_args()
    data_dir = ['/VisDrone2019-MOT-train/', '/VisDrone2019-MOT-val/', '/VisDrone2019-MOT-test-dev/']
    
    for d in data_dir:
        for image_dir in tqdm(natsorted(os.listdir(args.data_root + d + 'sequences'))):  # Apply natsorted to the directory list
            input_dir = args.data_root + d + 'sequences/' + image_dir
            gt = args.data_root + d + 'annotations/' + image_dir + '.txt'
            
            # Create the output directory if it doesn't exist
            output_dir = os.path.join(args.output, d[1:-1])
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, image_dir + '.mp4')
            
            log_file = None
            if args.log_file:
                log_file = os.path.join(output_dir, 'log.txt')
            
            to_mp4(input_dir, gt, output_path, log_file)

if __name__ == '__main__':
    main()