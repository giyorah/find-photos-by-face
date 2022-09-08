# Find Photos by Face

This tool uses [face detection](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet) and [face recognition](https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface) technologies to help you quickly and easily identify and organize photos of friends and family.

In general, face detection distinguishes faces from other objects in an image. For example, face detection lets the tool know when there is a face in a photo so that it can further analyze it. Face recognition identifies and groups similar faces across multiple photos in your folder(s).

## Brief
All you need to do is specify the
* Path to a (frontal face) photo of a person to search for
* Path to a folder with photos to search in.

The tool will create a new folder named after the person it searches for (provided you named the person's photo accordingly). This folder will be populated with (copies of) the photos the person appears in.

## Setup
```shell
git clone https://github.com/giyorahy/find-photos-by-face.git
cd find-photos-by-face/src
```

## Usage
```
usage: find.py [-h] [--image IMAGE] [--folder FOLDER] [--score_threshold SCORE_THRESHOLD] [--backend BACKEND]
               [--target TARGET]

Find images by face.

optional arguments:
  -h, --help            show this help message and exit
  --image IMAGE, -i IMAGE
                        Path to the image containing a single face to search for.
  --folder FOLDER, -f FOLDER
                        Path to a folder with the images to search in.
  --score_threshold SCORE_THRESHOLD, -s SCORE_THRESHOLD
                        Filtering out faces with score < score_threshold. Default is 0.95.
                        You may want to lower this value if not all images with the face
                        you search for are found.
  --backend BACKEND, -b BACKEND
                        Choose one of the computation backends: 3: OpenCV implementation (default); 5: CUDA
  --target TARGET, -t TARGET
                        Choose one of the target computation devices: 0: CPU (default); 6: CUDA; 7: CUDA fp16
```


## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference

- https://ieeexplore.ieee.org/document/9318547