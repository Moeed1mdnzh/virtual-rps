import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import MaxPooling2D
from sklearn.model_selection import train_test_split

###Load the dataset

data, labels = [], []
for i, label in enumerate(["rock", "paper", "scissors"]):
  for j in range(1,1001):
    path = f"data//{label + str(j)}.jpg"
    img = cv2.imread(path)
    data.append(img)
    labels.append(i)
data, labels = np.array(data), np.array(labels)

print(data.shape, labels.shape)

###Split the data

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=42)
print(X_train.shape, y_train.shape)

###Augment the data

def zoom(image : np.ndarray, scale : float, stepSize : int) -> list:
  res = []
  H, W = image.shape[:2]
  winH, winW = int(H*scale), int(W*scale)
  for y in range(0,H,stepSize):
    for x in range(0,W,stepSize):
      if x + winW >= W or y + winH >= H:
        continue
      res.append(cv2.resize(image[y:y+winH,x:x+winW],(W,H)))
  return res


def mirror(image : np.ndarray) -> list:
  return [cv2.flip(image,i) for i in [0,1]]


def lighting(image : np.ndarray, rate : int) -> list:
  amount = np.ones(image.shape,np.uint8) * int((100*rate))
  bright = cv2.add(image,amount)
  dark = cv2.subtract(image,amount)
  return [bright,dark]

def rotate(image : np.ndarray) -> list:
  res = []
  image_center = tuple(np.array(image.shape[:2][::-1]) / 2)
  for angle in np.arange(10,351,50):
    rotated = cv2.getRotationMatrix2D(image_center,angle, 1.0)
    res.append(cv2.warpAffine(image, rotated, image.shape[:2][::-1], flags=cv2.INTER_LINEAR))
  return res

def transfer(image : np.array, limit : int) -> list:
  vol = limit * 100
  H, W = image.shape[:2]
  res = []
  pts = [np.float32([[1,0,vol],[0,1,vol]]),
      np.float32([[1,0,-vol],[0,1,vol]]),
      np.float32([[1,0,-vol],[0,1,-vol]]),
      np.float32([[1,0,vol],[0,1,-vol]]),
      np.float32([[1,0,0],[0,1,vol]]),
      np.float32([[1,0,vol],[0,1,0]]),
      np.float32([[1,0,-vol],[0,1,0]]),
      np.float32([[1,0,0],[0,1,-vol]])]
  for pt in pts:
    res.append(cv2.warpAffine(image,pt,(W,H)))
  return res

def Augment(X : np.ndarray, y : np.ndarray) -> np.ndarray:
  X_new, y_new = [], []
  for x, Y in zip(X, y):
    zoomed = zoom(x, 0.8, 75)
    mirrored = mirror(x)
    light = lighting(x, 0.9)
    rotated = rotate(x)
    shifted = transfer(x, 0.3)
    for change in [mirrored, light, rotated, shifted, zoomed]:
      for im in change:
        X_new.append(im)
        y_new.append(Y)
  X_new, y_new = np.array(X_new), np.array(y_new)
  return X_new, y_new
  

X_train, y_train = Augment(X_train, y_train)

print("After augmentation : ", X_train.shape, y_train.shape)

###Create the model

class RPSModel:
    def build(width : int, height : int, depth : int, classes : int):
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        model = Sequential()
        #First Conv
        model.add(Conv2D(64, (3, 3), use_bias=True, input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))
        #Second Conv
        model.add(Conv2D(128, (3, 3), input_shape=inputShape, use_bias=True))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))
        #Third Conv
        model.add(Conv2D(128, (3, 3), input_shape=inputShape, use_bias=True))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))
        #Top Layers
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

#Build the model

model = RPSModel.build(64, 64, 3, 3)
print(model.summary())

###Preprocess the data

X_train, X_test = X_train / 255.0, X_test / 255.0
lb = LabelBinarizer()
y_train, y_test = lb.fit_transform(y_train), lb.fit_transform(y_test)
opt = SGD(learning_rate=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=opt)

###Train the model

H = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

###Evaluate the model

performance = model.evaluate(X_test, y_test)
print(performance)

###Save the model

print("Saving the model")
model.save("rps_model.h5")

def show_details(H):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(H.epoch, H.history["loss"], label="Train loss")
    ax[0].plot(H.epoch, H.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(H.epoch, H.history["accuracy"], label="Train acc")
    ax[1].plot(H.epoch, H.history["val_accuracy"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()
    max_loss = np.max(np.array([np.max(H.history["loss"]),np.max(H.history["val_loss"])]))
    max_acc = np.max(np.array([np.max(H.history["accuracy"]),np.max(H.history["val_accuracy"])]))
    ax[0].set_yticks(np.array([0,max_loss]))
    ax[1].set_yticks(np.array([0,max_acc]))
    fig.savefig('images/performance.png')
    plt.show()

show_details(H)
