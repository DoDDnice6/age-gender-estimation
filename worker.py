from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from wide_resnet import WideResNet

import settings
import helpers
import redis
import time
import json

import demo_mtcnn
import cv2

# connect to Redis server
db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)


def classify_process():
    print("* Loading model...")
    img_size = 64
    model = WideResNet(img_size, depth=16, k=8)()
    model.load_weights("checkpoints168/weights.48-3.62.hdf5")
    print("* Model loaded")

    while True:
        body = db.lpop(settings.IMAGE_QUEUE)
        if body is not None:
            body = json.loads(body.decode("utf-8"))
            image = helpers.base64_decode_image(body["image"],body["height"])

            bounding_boxes = demo_mtcnn.detect_face(image)
            margin = 0.4
            faces = np.empty((len(bounding_boxes), 64, 64, 3))
            for i, bounding_box in enumerate(bounding_boxes):
                x = int(bounding_box[0])
                y = int(bounding_box[1])
                w = int(bounding_box[2])
                h = int(bounding_box[3])
                img_h, img_w, _ = np.shape(image)
                _x = max(int(x - margin * w), 0)
                _y = max(int(y - margin * h), 0)
                _x2 = min(int(x+w + margin * w), img_w - 1)
                _y2 = min(int(y+h + margin * h), img_h - 1)
                # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 155, 255), 2)
                faces[i, :, :, :] = cv2.resize(
                    image[_y:_y2+1, _x:_x2+1, :], (64, 64))

                # cv2.rectangle(image, (_x, _y), (_x2, _y2), (0, 155, 255), 2)
                # cv2.imshow("",faces[0])
                
                # key  = cv2.waitKey(-1)
                # if key==27:
                #     break
            if len(faces) > 0:

                results = model.predict(faces)
                # results = imagenet_utils.decode_predictions(preds)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()
                predicted_ages = predicted_ages.tolist()
                predicted_genders = [gender.tolist()
                                     for gender in predicted_genders]
                print(type(predicted_genders[0]))
                print(type(predicted_ages))

                # for i, bounding_box in enumerate(bounding_boxes):
                # 	label = "{}, {}".format(int(predicted_ages[i]),"M" if predicted_genders[i][0] < 0.5 else "F")
                # 	demo_mtcnn.draw_label(image, (int(bounding_box[0]), int(bounding_box[1])), label)
                # cv2.imshow("",image)
                # cv2.waitKey(1000)
                print("done")

            # image_out = helpers.base64_encode_image(image).decode('utf-8')

            # db.set(body["id"], json.dumps(image_out))
            data = {"boudingboxes": bounding_boxes.tolist(), "ages": predicted_ages,
                    "genders": predicted_genders}
            db.set(body["id"], json.dumps(data))
        else:
            print("No data")
        # sleep for a small amount
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    classify_process()
