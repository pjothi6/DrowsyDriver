# python drowsiness_alert.py --shape-predictor eye_predictor.dat --phone-number YOUR_PHONE_NUMBER

from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
from twilio.rest import Client
import numpy as np


EYE_AR_THRESH = 0.20 
EYE_AR_CONSEC_FRAMES = 48  

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("--phone-number", required=True,
                help="your phone number in E.164 format, e.g., +1234567890")
args = vars(ap.parse_args())

TWILIO_SID = "Enter_your_Twilio_Account_SID"
TWILIO_AUTH_TOKEN = "Enter_your_Twilio_Authentic_Token"
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

print("[INFO] camera sensor warming up...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

frame_count = 0
drowsy = False

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (sX, sY) in shape:
            cv2.circle(frame, (sX, sY), 1, (0, 0, 255), -1)

        left_eye = shape[36:42]
        right_eye = shape[42:48]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        if ear < EYE_AR_THRESH:
            frame_count += 1
            if frame_count >= EYE_AR_CONSEC_FRAMES:
                drowsy = True
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Send a phone call alert
                call = client.calls.create(
                    to=args["phone_number"],
                    from_="Enter_your_Twilio_Phone_Number",
                    url="Enter_your_Webhook_URL"
                )
        else:
            frame_count = 0
            drowsy = False

        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

