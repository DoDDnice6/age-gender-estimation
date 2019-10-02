from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from mtcnn.mtcnn import MTCNN

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
    parser.add_argument("--margin", type=float, default=0,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.5, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    # cv2.rectangle(image, (x, y - size[1]),
    #               (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale,
                (255, 255, 255), thickness, lineType=cv2.LINE_AA)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    # h = 640
    # w = 480

    # capture video
    with video_capture("/home/dodat/Documents/python-projects/age-gender-estimation/video/Hanh_fvb.mp4") as cap:
        # with video_capture(0) as cap:
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, h)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, w)

        while True:
            # get video frame
            ret, img = cap.read()
            # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
            # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # cv2.transpose(img, img);
            # cv2.flip(img , 1);

            # M = cv2.getRotationMatrix2D((int(height/2), int(width/2)), 270, 1)
            # img = cv2.warpAffine(img, M, (int(height), int(width)))
            if not ret:
                raise RuntimeError("Failed to capture image")
            h, w, _ = img.shape
            if max(w,h) > 960:
                
                r = 960 / max(w, h)
                yield cv2.resize(img, (int(w * r), int(h * r)))
            # yield cv2.resize(img, (h, w))


def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)
    for image_path in image_dir.glob("*.jpg"):
        print(str(image_path))
        img = cv2.imread(str(image_path), 1)
        if img is not None:
            h, w, _ = img.shape
            if max(w,h) > 960:
                
                r = 960 / max(w, h)
                yield cv2.resize(img, (int(w * r), int(h * r)))


def detect_face(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img)
    # index_confidence_max = max(enumerate(faces),key=lambda item:item[1]['confidence'])[0]
    # print(index_confidence_max)
    bounding_boxes = np.zeros((0, 4))
    for face in faces:
        if face['confidence'] > 0.95:
            bounding_boxes = np.append(bounding_boxes, np.array([face['box'][0], face['box'][1], face['box'][2],
                                                                 face['box'][3]]).reshape((1, 4)), axis=0)
    # bounding_box=faces[index_confidence_max]['box']
    return bounding_boxes


def main():
    args = get_args()
    depth = args.depth
    k = args.width
    # weight_file = args.weight_file
    weight_file = "checkpoints168/weights.48-3.62.hdf5"
    margin = args.margin
    image_dir = args.image_dir

    if not weight_file:
        weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)

    image_generator = yield_images_from_dir(
        image_dir) if image_dir else yield_images()
    # image_generator=yield_images()
    skip_frame = 1
    ind = 0

    for img_ in image_generator:
        print(img_.shape)
        if ind % skip_frame == 0:

            bounding_boxes = detect_face(img_)
            print("face number: ",len(bounding_boxes))
            faces = np.empty((len(bounding_boxes), 64, 64, 3))
            for i, bounding_box in enumerate(bounding_boxes):
                x = int(bounding_box[0])
                y = int(bounding_box[1])
                w = int(bounding_box[2])
                h = int(bounding_box[3])

                img_h, img_w, _ = np.shape(img_)
                _x = max(int(x - margin * w), 0)
                _y = max(int(y - margin * h), 0)
                _x2 = min(int(x+w + margin * w), img_w - 1)
                _y2 = min(int(y+h + margin * h), img_h - 1)

                cv2.rectangle(img_, (x, y), (x+w, y+h), (0, 155, 255), 2)
                cv2.rectangle(img_, (_x, _y), (_x2, _y2), (255, 155, 255), 2)
                faces[i, :, :, :] = cv2.resize(
                    img_[_y:_y2+1, _x:_x2+1, :], (64, 64))

                # predict ages and genders of the detected facesƯ
            try:
                results = model.predict(faces)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()

                # draw results
                for i, bounding_box in enumerate(bounding_boxes):
                    label = "{}, {}".format(int(predicted_ages[i]),
                                            "Nam" if predicted_genders[i][0] < 0.5 else "Nữ")
                    draw_label(
                        img_, (int(bounding_box[0]), int(bounding_box[1])), label)

                cv2.imshow("result", img_)
                # cv2.waitKey(0)
                # print(ind)
                key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)
                if key == 27:  # ESC
                    break
                continue
            except:
                cv2.imshow("result", img_)
                key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)
                if key == 27:  # ESC
                    break
        ind = ind+1


if __name__ == '__main__':
    main()
