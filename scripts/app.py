# -*- coding: utf-8 -*-
"""App for POC drowsiness detection on video

@license Copyright Sebastian Correa Echeverri

@author  scorrea
"""
import time
import math
import logging
import argparse
import dlib
import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
from imutils import face_utils
from drowzee.drowsiness import Drowsiness


def parse_options(log):
    """ Function to parse arguments.

    Returns
    -------
    Arguments parsed.
    """
    parser = argparse.ArgumentParser(
        description='Script for trainig ')

    parser.add_argument('-th', '--threshold',
                        dest='threshold',
                        action='store',
                        type=float,
                        default=0.2,
                        help='eye opened threshold')
    parser.add_argument('-frame', '--frame',
                        dest='frame',
                        action='store',
                        type=int,
                        default=12,
                        help='number of frames with close eye to pop alarm')
    args = parser.parse_args()
    for key, value in sorted(vars(args).items()):
        log.info("{} = {}".format(key, value))
    return args

def closed_eye_alarm():
    """Event when eye close"""
    logger.info("Driver is Sleeping!!!!")


def landmark_to_imgpoints(landmark):
    image_points = np.array([(359, 391),     # Nose tip 34
                             (399, 561),     # Chin 9
                             (337, 297),     # Left eye left corner 37
                             (513, 301),     # Right eye right corne 46
                             (345, 465),     # Left Mouth corner 49
                             (453, 469)      # Right mouth corner 55
                            ], dtype="double")
    for (i, (x, y)) in enumerate(landmark):
        if i == 33:
            image_points[0] = np.array([x, y], dtype='double')
        elif i == 8:
            image_points[1] = np.array([x, y], dtype='double')
        elif i == 45:
            image_points[3] = np.array([x, y], dtype='double')
        elif i == 48:
            image_points[0] = np.array([x, y], dtype='double')
        elif i == 54:
            image_points[5] = np.array([x, y], dtype='double')

    return image_points


def face_orientation(frame, landmarks):
    size = frame.shape #(height, width, color_channel)

    model_points = np.array([(0.0, 0.0, 0.0),             # Nose tip
                             (0.0, -330.0, -65.0),        # Chin
                             (-225.0, 170.0, -135.0),     # Left eye left corner
                             (225.0, 170.0, -135.0),      # Right eye right corne
                             (-150.0, -150.0, -125.0),    # Left Mouth corner
                             (150.0, -150.0, -125.0)     # Right mouth corner
                            ])
    axis = np.float32([[500, 0, 0],
                       [0, 500, 0],
                       [0, 0, 500]])
    image_points = landmark_to_imgpoints(landmarks)

    # Camera internals
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="float32")

    dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points,
                                                                  image_points,
                                                                  camera_matrix,
                                                                  dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        return None

    imgpts, _ = cv2.projectPoints(axis,
                                  rotation_vector,
                                  translation_vector,
                                  camera_matrix,
                                  dist_coeffs)
    modelpts, _ = cv2.projectPoints(model_points,
                                    rotation_vector,
                                    translation_vector,
                                    camera_matrix,
                                    dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [math.radians(_) for _ in euler_angles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return imgpts, modelpts, (int(roll), int(pitch), int(yaw))


def main(args, logger):
    # initialize dlib's face detector
    logger.info("Loading facial landmark predictor...")
    DETECTOR = dlib.get_frontal_face_detector()
    PREDICTOR = dlib.shape_predictor("models/facial/shape_predictor_68_face_landmarks.dat")

    # start the video stream thread
    logger.info("Starting video stream thread...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    drowzi = Drowsiness(eye_ar_TH=args.threshold,
                        eye_ar_frames=args.frame,
                        alarm_func=closed_eye_alarm)

    # loop over frames from the video stream
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=900)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = DETECTOR(gray, 0)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            shape = PREDICTOR(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Print face rectangle
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Print landmarks
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            # Calculate face rotation
            _, _, rotate_degree = face_orientation(frame, shape)
            cv2.putText(frame,
                        "{}".format(rotate_degree),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

            # Calculate drowsiness
            frame = drowzi.calculate_drowsiness(frame, shape)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logger.info('Start Drowsiness detector')
    args = parse_options(logger)
    main(args, logger)
