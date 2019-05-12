# -*- coding: utf-8 -*-
"""Class for drowsiness detection

@license Copyright Sebastian Correa Echeverri

@author  scorrea
"""
from threading import Thread
import cv2
from imutils import face_utils
from scipy.spatial import distance as dist

(ELSTART, ELEND) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(ERSTART, EREND) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

class Drowsiness():
    """Class for detect drowsiness by eye closure"""

    def __init__(self, eye_ar_TH=0.3, eye_ar_frames=48, alarm_func=None):
        self.eye_ar_TH = eye_ar_TH
        self.eye_ar_frames = eye_ar_frames
        self.counter = 0
        self.alarm_on = False
        self.alarm_func = alarm_func

    def eye_aspect_ratio(self, eye):
        """ Compute the euclidean distances between the two sets of
        vertical eye landmarks (x, y)-coordinates.
    
        Parameters
        ----------
        eye: array[array[int]]
            coordenates for both eyes landmark

        Returns
        -------
        float with mean eye aspect ratio
        """
        # compute the euclidean distances between the two sets of eyes
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # compute the euclidean distance between the horizontal
        C = dist.euclidean(eye[0], eye[3])
        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        return ear

    def calculate_drowsiness(self, frame, landmarks):
        """Calculate drowsinness using eye aspect ratio threshold

        Parameters
        ----------
        frame: numpy array
            image conainting a face
        landmarks: array
            landmarks coords of the image face

        Returns
        -------
        numpy array frame with anotations
        """
        left_eye = landmarks[ELSTART:ELEND]
        right_eye = landmarks[ERSTART:EREND]
        left_eye_ar = self.eye_aspect_ratio(left_eye)
        right_eye_ar = self.eye_aspect_ratio(right_eye)

        ear = (left_eye_ar + right_eye_ar)/2.0

        if ear < self.eye_ar_TH:
            self.counter += 1

            if self.counter >= self.eye_ar_frames:
                if not self.alarm_on :
                    self.alarm_on  = True
                    t = Thread(target=self.alarm_func, args=None)
                    t.deamon = True
                    t.start()
                cv2.putText(frame,
                            "DROWSINESS ALERT!",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
        else:
            self.counter = 0
            ALARM_ON = False
        
        cv2.putText(frame,
                    "EAR: {:.2f}".format(ear),
                    (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        
        return frame
