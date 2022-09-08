# core functionality
import os, shutil
import logging
import argparse, textwrap

# file dialogs, for non-cli users
import tkinter as tk
from tkinter import filedialog

# annotations
from typing import Tuple

# open-cv
import numpy as np
import cv2 as cv

# constants
MISSING_INPUT = -1
COSINE_SIMILARITY_THRESHOLD = 0.5
L2_SIMILARITY_THRESHOLD = 1

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

def visualize(input: cv.Mat, face: cv.Mat, thickness: int = 2) -> cv.Mat:
    """Draw detection of a face

    Args:
        input (cv.Mat): input image, as returned from cv.imread()
        face (cv.Mat): detected face coordinates,
                       as returned from cv.FaceDetectorYN.detect()
        thickness (int, optional): Thickness of the shape outline.
                                   Defaults to 2.

    Returns:
        cv.Mat: _description_
    """
    if face is None:
        return input

    output = input.copy()
    logging.debug("Face detected, %s, %s, %s",
                 f'top-left coordinates: ({face[0]:.0f}, {face[1]:.0f})',
                 f'box width: {face[2]:.0f}, box height: {face[3]:.0f}',
                 f'score: {face[-1]:.2f}')
    coords = face[:-1].astype(np.int32)
    cv.rectangle(
        output,
        (coords[0], coords[1]),
        (coords[0] + coords[2], coords[1] + coords[3]),
        (0, 255, 0),
        thickness)
    cv.circle(output, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
    cv.circle(output, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
    cv.circle(output, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
    cv.circle(output, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
    cv.circle(output, (coords[12], coords[13]), 2, (0, 255, 255), thickness)

    return output

def my_parser() -> argparse.Namespace:
    """This is the custom parser for cli users

    Returns:
        Tuple[str, str, float]: see help strings below.
    """
    backends = [cv.dnn.DNN_BACKEND_OPENCV,
                cv.dnn.DNN_BACKEND_CUDA]
    targets = [cv.dnn.DNN_TARGET_CPU,
               cv.dnn.DNN_TARGET_CUDA,
               cv.dnn.DNN_TARGET_CUDA_FP16]
    help_msg_backends = ("Choose one of the computation backends: "
                         "{:d}: OpenCV implementation (default); {:d}: CUDA")
    help_msg_targets = ("Choose one of the target computation devices: "
                        "{:d}: CPU (default); {:d}: CUDA; {:d}: CUDA fp16")
    parser = argparse.ArgumentParser(
        description='Find images by face.',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--image', '-i', type=str,
        help=textwrap.dedent('''\
        Path to the image containing a single face to search for.
        '''))
    parser.add_argument(
        '--folder', '-f', type=str,
        help=textwrap.dedent('''\
        Path to a folder with the images to search in.
        '''))
    parser.add_argument(
        '--score_threshold', '-s', type=float,
        default=0.95,
        help=textwrap.dedent('''\
        Filtering out faces with score < score_threshold. Default is 0.95.
        You may want to lower this value if not all images with the face
        you search for are found.
        '''))
    parser.add_argument(
        '--backend', '-b', type=int, default=backends[0],
        help=help_msg_backends.format(*backends))
    parser.add_argument(
        '--target', '-t', type=int, default=targets[0],
        help=help_msg_targets.format(*targets))
    args = parser.parse_args()
    
    return args

def config_logger(level: int = logging.DEBUG) -> None:
    fmt = '[%(levelname)s] - %(message)s'
    logging.basicConfig(level=level, format=fmt)

def is_match(image_path: str,
             ftf_features: np.ndarray,
             detector: cv.FaceDetectorYN,
             recognizer: cv.FaceRecognizerSF) -> bool:
    """This function returns whether or not there is a match between the
       face in the target image and the current analyzed image.

    Args:
        image_path (str): path to the searched image
        ftf_features (np.ndarray): target face's encoding
        detector (cv.FaceDetectorYN): face detection model
        recognizer (cv.FaceRecognizerSF): face recognition model (encoder)

    Returns:
        bool: True if there is a match, False otherwise.
    """
    # load the image to search in
    image = cv.imread(image_path)
    detector.setInputSize((image.shape[1], image.shape[0]))
    _, faces = detector.detect(image)
    if faces is None:
        logging.debug("No face was detected in %s.",
                     os.path.basename(image_path))
        return False

    # search for a match among all detected faces
    for face in faces:
        face_align = recognizer.alignCrop(image, face)
        face_features = recognizer.feature(face_align)
        cosine_score = recognizer.match(ftf_features,
                                        face_features,
                                        cv.FaceRecognizerSF_FR_COSINE)
        l2_score = recognizer.match(ftf_features,
                                    face_features,
                                    cv.FaceRecognizerSF_FR_NORM_L2)
        if (cosine_score >= COSINE_SIMILARITY_THRESHOLD
                and l2_score <= L2_SIMILARITY_THRESHOLD):
            return True

    return False

def find_images(face_img_path: str,
                folder_path: str,
                detector: cv.FaceDetectorYN,
                recognizer: cv.FaceRecognizerSF) -> None:
    """This function loads the target image, detects the face in it, encodes it,
       and searches for images that contain this face (i.e. with similar encoding)

    Args:
        face_img_path (str): path to the target image
        folder_path (str): path to a folder with images to search in
        detector (cv.FaceDetectorYN): face detection model
        recognizer (cv.FaceRecognizerSF): face recognition model (encoder)
    """
    # load the target image
    face_image = cv.imread(face_img_path)

    # set input size before inference
    detector.setInputSize((face_image.shape[1], face_image.shape[0]))

    # inference
    _, faces = detector.detect(face_image)

    if faces is None:
        logging.error(f'Cannot find a face in {face_img_path}. Exiting...')
        return
    elif len(faces) > 1:
        logging.warning(f'More than one face detected in {face_img_path}.')
        # Draw detection of the (soon to be) searched face
        face_img_with_det = visualize(face_image, faces[0])
        # Save the annotation
        logging.info('The face searched is annotated in \'find_me.jpg\'.')
        cv.imwrite('find_me.jpg', face_img_with_det)

    # we have a face to find - let's do it
    ftf = faces[0]
    # make a directory to store the matching images
    dir_name = os.path.basename(face_img_path).split('.')[0]
    logging.info(f'Searching for photos with {dir_name} in them.')
    output_dir = folder_path + '/' +  dir_name
    os.makedirs(output_dir, exist_ok=True)

    # align and crop
    ftf_align = recognizer.alignCrop(face_image, ftf)
    # extract features
    ftf_features = recognizer.feature(ftf_align)

    # scan the folder for matches
    count = 0
    for file in files(folder_path):
        logging.debug(f'Checking for a match in {file}.')
        if is_match(os.path.join(folder_path, file),
                    ftf_features,
                    detector,
                    recognizer):
            # copy the image over to the new folder
            shutil.copy2(os.path.join(folder_path, file), output_dir)
            count += 1
    if count > 0:
        logging.info(f'Found {count} photos {dir_name} appears in.')
        logging.info(f'You can find them here: {output_dir}')
    else:
        logging.info(f'No photos of/with {dir_name} were found.')
        shutil.rmtree(output_dir)

def initialize_models(threshold: int,
                      backend: int,
                      target: int) -> Tuple[cv.FaceDetectorYN,
                                            cv.FaceRecognizerSF]:
    """Initialize DNN-based face detector & face recognizer

    Args:
        threshold (int): used to filter out bounding boxes of score less than
                         the given value.
        backend (int): computation backend
        target (int): target computation device

    Returns:
        Tuple[cv.FaceDetectorYN, cv.FaceRecognizerSF]: Detector, Recognizer
    """
    # initialize FaceDetectorYN
    detector = cv.FaceDetectorYN.create(
        model='../models/face_detection_yunet_2022mar.onnx',
        config="",
        input_size=(320, 320),
        score_threshold=threshold,
        backend_id=backend,
        target_id=target
    )

    # initialize FaceRecognizerSF
    recognizer = cv.FaceRecognizerSF.create(
        model='../models/face_recognition_sface_2021dec.onnx',
        config="",
        backend_id=backend,
        target_id=target
    )

    return detector, recognizer

def find_photos_by_face() -> None:
    """This is the main function. Handles user input and setup.
    """
    config_logger(level=logging.INFO)

    args = my_parser()
    # for non-cli users
    if not args.image or not args.folder:
        tk.Tk().withdraw()
        if not args.image:
            args.image = filedialog.askopenfilename(
                title='Select the photo with the face to search for')
            if not args.image:
                logging.error("You must supply a photo with a face to search for.")
                exit(MISSING_INPUT)
        if not args.folder:
            args.folder = filedialog.askdirectory(
                title='Select the folder with the photos')
            if not args.folder:
                logging.error("You must supply a folder to search in.")
                exit(MISSING_INPUT)

    detector, recognizer = initialize_models(args.score_threshold,
                                             args.backend,
                                             args.target)
    find_images(args.image, args.folder, detector, recognizer)


if __name__ == '__main__':
    find_photos_by_face()
