import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import get_meta

db = "wiki"
# db = "imdb"
mat_path = "data/{}_crop/{}.mat".format(db, db)
full_path, dob, gender, photo_taken, face_score, second_face_score, age\
    = get_meta(mat_path, db)
print("#images: {}".format(len(face_score)))
print("#images with inf scores: {}".format(np.isinf(face_score).sum()))
hist = plt.hist(face_score[face_score>0], bins=np.arange(0, 8, 0.2), color='b')
plt.xlabel("face score")
print("#images with second face scores: {}".format((~np.isnan(second_face_score)).sum()))
hist = plt.hist(second_face_score[~np.isnan(second_face_score)], bins=np.arange(0, 8, 0.2), color='b')
plt.xlabel("second face score")