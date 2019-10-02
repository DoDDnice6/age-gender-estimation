
from __future__ import division, print_function, absolute_import
from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from mtcnn.mtcnn import MTCNN


import os
from timeit import time
import warnings
import sys
from deep_sort_yolov3.yolotiny import YOLO

from deep_sort_yolov3.deep_sort import preprocessing
from deep_sort_yolov3.deep_sort import nn_matching
from deep_sort_yolov3.deep_sort.detection import Detection
from deep_sort_yolov3.deep_sort.tracker import Tracker
from deep_sort_yolov3.tools import generate_detections as gdet
from deep_sort_yolov3.deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')
from PIL import Image
detector = MTCNN()

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    # capture video
    # with video_capture("TGOP.mp4") as cap:
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img




def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)

    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r)))

def detect_face(img):
    faces=detector.detect_faces(img)
    bounding_boxes=np.zeros((0,4))
    for face in faces:
        if face['confidence'] > 0.95:
            bounding_boxes=np.append(bounding_boxes,np.array([face['box'][0],face['box'][1],face['box'][2],
                                                            face['box'][3]]).reshape((1,4)),axis=0)    
    return bounding_boxes



def main():
    args = get_args()
    depth = args.depth
    k = args.width
    # weight_file = args.weight_file
    weight_file="checkpoints168/weights.48-3.62.hdf5"
    margin = args.margin
    image_dir = args.image_dir

    if not weight_file:
        weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)

    video_capture = yield_images_from_dir(image_dir) if image_dir else yield_images()
    
    
    # tracking
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    parent_dir = os.path.abspath(os.path.abspath(os.path.dirname(__file__))) 
    model_filename = parent_dir+'/deep_sort_yolov3/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True 


    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        # w = int(video_capture.get(3))
        # h = int(video_capture.get(4))
        w=640
        h=480
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('_output_123.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1 

    fps = 0.0

    
    yolo=YOLO()
    
    # print(len(list(image_generator)))
    skip_frame=1
    ind =0
    for img_ in video_capture:
        if ind%skip_frame == 0:
            t1 = time.time()
        
            # image = Image.fromarray(frame)
            image__ = Image.fromarray(img_)
            boxs = yolo.detect_image(image__)
        # print("box_num",len(boxs))
            features = encoder(img_,boxs)
            
            # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
            
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            
            # Call the tracker
            tracker.predict()
            tracker.update(detections)
            
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                cv2.rectangle(img_, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                cv2.putText(img_, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

            for det in detections:
                bbox = det.to_tlbr()
                cv2.rectangle(img_,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
                
            cv2.imshow('', img_)
            
            # if writeVideo_flag:
            #     # save a frame
            #     out.write(img_)
            #     img_ = frame_index + 1
            #     list_file.write(str(img_)+' ')
            #     if len(boxs) != 0:
            #         for i in range(0,len(boxs)):
            #             list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            #     list_file.write('\n')
                
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %f"%(fps))
            
            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            
            
            # input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img_=cv2.imread(img)
            # cv2.imshow("",img_)
            # cv2.waitKey(0)
            bounding_boxes=detect_face(img_)
            faces = np.empty((len(bounding_boxes), 64, 64, 3))
            for i,bounding_box in enumerate(bounding_boxes):
                x=int(bounding_box[0])
                y=int(bounding_box[1])
                w=int(bounding_box[2])
                h=int(bounding_box[3])


                img_h, img_w, _ = np.shape(img_)
                _x = max(int(x - margin * w), 0)
                _y = max(int(y - margin * h), 0)
                _x2 = min(int(x+w + margin * w), img_w - 1)
                _y2 = min(int(y+h + margin * h), img_h - 1)


                cv2.rectangle(img_,(x,y),(x+w,y+h),(0,155,255),2)
                cv2.rectangle(img_,(_x,_y),(_x2,_y2),(255,155,255),2)
                faces[i,:,:,:]=cv2.resize(img_[_y:_y2+1,_x:_x2+1,:],(64,64))

                # predict ages and genders of the detected faces
            try:
                results = model.predict(faces)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()

                    # draw results
                for i, bounding_box in enumerate(bounding_boxes):
                    label = "{}, {}".format(int(predicted_ages[i]),
                                                "M" if predicted_genders[i][0] < 0.5 else "F")
                    draw_label(img_, (int(bounding_box[0]), int(bounding_box[1])), label)

                cv2.imshow("result", img_)
                key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)
                if key == 27:  # ESC
                    break
            except:
                cv2.imshow("result", img_)
                key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)
                if key == 27:  # ESC
                    break
        ind=ind+1
            
       


if __name__ == '__main__':
    main()
