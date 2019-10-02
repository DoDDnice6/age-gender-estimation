import scipy.io
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
# from mpl_toolkits.mplot3d import Axes3D

path=Path("data/wiki_crop")
data_train = scipy.io.loadmat(path.joinpath("train").joinpath("train.mat"))

print(data_train.keys())
Y = data_train['age'][0]

plt.xlabel('Tuổi')
plt.ylabel('Số lượng ảnh')
plt.title('Phân bố tuổi ')
plt.hist(Y, range(100))
plt.grid(True)
plt.show()

plt.cla()

data_test = scipy.io.loadmat(path.joinpath("test").joinpath("test.mat"))

print(data_train.keys())
Y = data_test['age'][0]

plt.xlabel('Tuổi')
plt.ylabel('Số lượng ảnh')
plt.title('Phân bố tuổi ')
plt.hist(Y, range(100))
plt.grid(True)
plt.show()