import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from numpy.random.mtrand import logistic
from sklearn.metrics import accuracy_score
from matplotlib.patches import Rectangle
from PIL import Image as img
import glob
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

Train_path = "C:\\Users\\PROFESSOR\\Desktop\\AI lab tasks\\apple"
folder = os.listdir(Train_path)

images = []
ImgCount = 0
ImagesCount_list = []
rDiagonal = []
def diagonal(img):
    
    a, b = img.size
    right_diagonal = np.zeros(a)

    k = 0
    for j in range(100):
        if(img[j][j] != 'nan'):
            right_diagonal[k] = img[j][j]
            k = k+1

    return np.sum(right_diagonal)/100


x_train = []
y_train = []
for i in folder:
    x = len(os.listdir(Train_path+"\\"+i))
    ImagesCount_list.append(x)
    for filename in os.listdir(Train_path+"\\"+i):
        if any([filename.endswith(x) for x in ['.jpg']]):
            img = image.imread(Train_path+"\\"+i+"\\"+filename)
            a = diagonal(img)
            rDiagonal.append(a)
print(rDiagonal)
print(ImagesCount_list)
print(images)

        