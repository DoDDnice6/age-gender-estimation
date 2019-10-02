from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import settings
import helpers
import flask
import redis
import uuid
import time
import json
import io
import os
import cv2
import sys

app = flask.Flask(__name__)
db = redis.StrictRedis(host=settings.REDIS_HOST,
	port=settings.REDIS_PORT, db=settings.REDIS_DB)


@app.route("/")
def homepage():
	return "Welcome to the Keras REST API!"

@app.route("/predict", methods=["POST"])
def predict():

	data = {"success": False}

	if flask.request.method == "POST":
		print("Start predict ...")
		if flask.request.files.get("video"):
			video = flask.request.files["video"].read()
			print(len(video))
		if flask.request.files.get("image"):
			image = flask.request.files["image"].read()

			npimg = np.fromstring(image, np.uint8)
			image = cv2.imdecode(npimg, 3)
			height=image.shape[1]
			print("hape img:",image.shape)

			k = str(uuid.uuid4())
			image = helpers.base64_encode_image(image).decode('utf-8')
			d = {"id": k, "image": image,"height":height}

			db.rpush(settings.IMAGE_QUEUE, json.dumps(d))


			while True:
				output = db.get(k)
				if output is not None:
					data["predictions"] = json.loads(output)
					
					db.delete(k)
					break
				# time.sleep(settings.CLIENT_SLEEP)

			data["success"] = True
		else:
			print("err")
	print(data)
	return flask.jsonify(data)

if __name__ == "__main__":
	app.run('0.0.0.0')