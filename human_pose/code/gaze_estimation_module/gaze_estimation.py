from tracemalloc import start
from typing import List, Optional

import torch
from torch.nn import DataParallel

from gaze_estimation_module.models.eyenet import EyeNet
import os
import numpy as np
import cv2
from gaze_estimation_module import util
import gaze_estimation_module.util.gaze
import time

from gaze_estimation_module.util.eye_prediction import EyePrediction
from gaze_estimation_module.util.eye_sample import EyeSample

torch.backends.cudnn.enabled = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dirname = os.path.dirname(__file__)
face_cascade = cv2.CascadeClassifier(os.path.join(dirname, 'lbpcascade_frontalface_improved.xml'))
print(dirname)
checkpoint = torch.load(os.path.join(dirname, 'checkpoint.pt'), map_location=device)
nstack = checkpoint['nstack']
nfeatures = checkpoint['nfeatures']
nlandmarks = checkpoint['nlandmarks']
eyenet = EyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks).to(device)
eyenet.load_state_dict(checkpoint['model_state_dict'])


def estimate_gaze_from_face_image(orig_frame, frame, face_coordinate, left_eye_boxes, right_eye_boxes, visualization):
    eye_boxes = np.concatenate([right_eye_boxes, left_eye_boxes], axis=0)
    current_face = None
    landmarks = None
    alpha = 1
    left_eye = None
    right_eye = None
    faces = face_coordinate
    orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2GRAY)
    if len(faces):
        next_face = faces[0]
        if current_face is not None:
            current_face = alpha * next_face + (1 - alpha) * current_face
        else:
            current_face = next_face

    if current_face is not None:
        next_landmarks = eye_boxes
        if landmarks is not None:
            landmarks = next_landmarks * alpha + (1 - alpha) * landmarks
        else:
            landmarks = next_landmarks



    if landmarks is not None:
        eye_samples = segment_eyes(gray, landmarks, visualization)

        eye_preds = run_eyenet(eye_samples)
        left_eyes = list(filter(lambda x: x.eye_sample.is_left, eye_preds))
        right_eyes = list(filter(lambda x: not x.eye_sample.is_left, eye_preds))

        if left_eyes:
            left_eye = smooth_eye_landmarks(left_eyes[0], left_eye, smoothing=0.1)
        if right_eyes:
            right_eye = smooth_eye_landmarks(right_eyes[0], right_eye, smoothing=0.1)
        eyes = [left_eye, right_eye]
    return frame, eyes



def draw_cascade_face(face, frame):
    (x, y, w, h) = (int(e) for e in face)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


def draw_landmarks(landmarks, frame):
    for (x, y) in landmarks:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)


def segment_eyes(frame, landmarks, visualization, ow=160, oh=96):
    eyes = []

    # Segment eyes
    for corner1, is_left in [(1, True), (0, False)]:
        x1, y1, x2, y2 = landmarks[corner1, :]
        eye_width = 1.5 * np.linalg.norm([x2-x1, y2-y1])
        if eye_width == 0.0:
            return eyes

        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        # center image on middle of eye
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-cx], [-cy]]
        inv_translate_mat = np.asmatrix(np.eye(3))
        inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

        # Scale
        scale = ow / eye_width
        scale_mat = np.asmatrix(np.eye(3))
        scale_mat[0, 0] = scale_mat[1, 1] = scale
        inv_scale = 1.0 / scale
        inv_scale_mat = np.asmatrix(np.eye(3))
        inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

        estimated_radius = 0.5 * eye_width * scale

        # center image
        center_mat = np.asmatrix(np.eye(3))
        center_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
        inv_center_mat = np.asmatrix(np.eye(3))
        inv_center_mat[:2, 2] = -center_mat[:2, 2]

        # Get rotated and scaled, and segmented image
        transform_mat = center_mat * scale_mat * translate_mat
        inv_transform_mat = (inv_translate_mat * inv_scale_mat * inv_center_mat)

        eye_image = cv2.warpAffine(frame, transform_mat[:2, :], (ow, oh))
        eye_image = cv2.equalizeHist(eye_image)
        if visualization:
            if is_left:
                eye_image = np.fliplr(eye_image)
                cv2.imshow('left eye image', eye_image)
            else:
                cv2.imshow('right eye image', eye_image)
        eyes.append(EyeSample(orig_img=frame.copy(),
                              img=eye_image,
                              transform_inv=inv_transform_mat,
                              is_left=is_left,
                              estimated_radius=estimated_radius))
    return eyes


def smooth_eye_landmarks(eye: EyePrediction, prev_eye: Optional[EyePrediction], smoothing=0.2, gaze_smoothing=0.4):
    if prev_eye is None:
        return eye
    return EyePrediction(
        eye_sample=eye.eye_sample,
        landmarks=smoothing * prev_eye.landmarks + (1 - smoothing) * eye.landmarks,
        gaze=gaze_smoothing * prev_eye.gaze + (1 - gaze_smoothing) * eye.gaze)


def run_eyenet(eyes: List[EyeSample], ow=160, oh=96) -> List[EyePrediction]:
    result = []
    for eye in eyes:
        with torch.no_grad():
            x = torch.tensor([eye.img], dtype=torch.float32).to(device)
            _, landmarks, gaze = eyenet.forward(x)
            landmarks = np.asarray(landmarks.cpu().numpy()[0])
            gaze = np.asarray(gaze.cpu().numpy()[0])
            assert gaze.shape == (2,)
            assert landmarks.shape == (34, 2)

            landmarks = landmarks * np.array([oh/48, ow/80])

            temp = np.zeros((34, 3))
            if eye.is_left:
                temp[:, 0] = ow - landmarks[:, 1]
            else:
                temp[:, 0] = landmarks[:, 1]
            temp[:, 1] = landmarks[:, 0]
            temp[:, 2] = 1.0
            landmarks = temp
            assert landmarks.shape == (34, 3)
            landmarks = np.asarray(np.matmul(landmarks, eye.transform_inv.T))[:, :2]
            assert landmarks.shape == (34, 2)
            result.append(EyePrediction(eye_sample=eye, landmarks=landmarks, gaze=gaze))
    return result