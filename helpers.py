# import the necessary packages
import numpy as np
import base64
import sys
import cv2

def base64_encode_image(a):
	# base64 encode the input NumPy array
	return base64.b64encode(a)

def base64_decode_image(a,h):
	img_ = base64.b64decode(a)
	img = np.frombuffer(img_, dtype=np.uint8)
	img= np.reshape(img,(-1,h,3))
	return img


if __name__ == "__main__":
	pass
	# img=cv2.imread("69300_1950-05-11_2009.jpg")
	# img=cv2.resize(img,(64,64))
	# cv2.imshow("",img)
	# cv2.waitKey(0)
	# print(img.shape)
	# jpg_as_text = base64.b64encode(img)
	# print(jpg_as_text[:80])

	# # Convert back to binary
	# jpg_original = base64.b64decode(jpg_as_text)

	# jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
	# print(jpg_as_np.shape)
	# jpg= np.reshape(jpg_as_np,(-1,64,3))
	# cv2.imshow("",jpg)
	# cv2.waitKey(0)