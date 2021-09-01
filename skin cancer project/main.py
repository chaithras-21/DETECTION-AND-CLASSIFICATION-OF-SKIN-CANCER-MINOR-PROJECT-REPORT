import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import pandas as pd
import seaborn as sns
import cv2
import scipy.misc
import scipy.ndimage
import imutils
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from sklearn.metrics import confusion_matrix,classification_report


data = r'C:\Users\User\Documents\skin cancer project\dataset'
benign = glob.glob(r'dataset\benign\*.jpg')
malignant = glob.glob(r'dataset\malignant\*.jpg')
print('Number of images with benign : {}'.format(len(benign)))
print('Number of images with malignant : {}'.format(len(malignant)))

#read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

benign_ = scipy.misc.imread('dataset/benign/2.jpg')
benign_ = 255 - benign_
median_filtered = scipy.ndimage.median_filter(benign_, size=3)


fig, ax = plt.subplots(1, 3, figsize=(10, 8));
plt.suptitle('SAMPLE PROCESSED IMAGE', x=0.5, y=0.8)
plt.tight_layout(1)

ax[0].set_title('ORG.', fontsize=12)
ax[1].set_title('BENIGN', fontsize=12)
ax[2].set_title('MEADIAN_FILTER', fontsize=12)

ax[0].imshow(255-benign_, cmap='gray');
ax[1].imshow(benign_, cmap='gray');
ax[2].imshow(median_filtered, cmap='gray');



min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)

# Get pointer to video frames from primary device
image = cv2.imread("dataset/benign/2.jpg")
imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
skinYCrCb = cv2.bitwise_and(image, image, mask = skinRegionYCrCb)
plt.title('CANCER')
plt.show()
plt.imshow(skinYCrCb)

font = cv2.FONT_HERSHEY_COMPLEX
img2 = cv2.imread('dataset/benign/2.jpg', cv2.IMREAD_COLOR)

# Reading same image in another
# variable and converting to gray scale.
img = cv2.imread('dataset/benign/2.jpg', cv2.IMREAD_GRAYSCALE)

# Converting image to a binary image
# ( black and white only image).
_, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

# Detecting contours in image.
contours, _= cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

# Going through every contours found in the image.
for cnt in contours :

	approx = cv2.approxPolyDP(cnt, 0.012 * cv2.arcLength(cnt, True), True)

	# draws boundary of contours.
	cv2.drawContours(img, [approx], 0, (0, 50, 255), 2)

plt.show()
plt.title('SKIN CANCER')
plt.imshow(img)

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def is_contour_bad(c):
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# the contour is 'bad' if it is not a rectangle
	return not len(approx) == 4
imag = cv2.imread("dataset/benign/2.jpg")
gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 80, 100)
# find contours in the image and initialize the mask that will be
# used to remove the bad contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
mask = np.ones(imag.shape[:2], dtype="uint8") * 255
# loop over the contours
for c in cnts:
	# if the contour is bad, draw it on the mask
	if is_contour_bad(c):
		cv2.drawContours(mask, [c], -1, 0, -1)
# remove the contours from the image and show the resulting images
image = cv2.bitwise_and(imag, imag, mask=mask)
fig, ax = plt.subplots(1, 3, figsize=(10, 8));
plt.suptitle('SAMPLE PROCESSED IMAGE', x=0.5, y=0.8)
plt.tight_layout(1)

ax[0].set_title('ORG.', fontsize=12)
ax[1].set_title('mask', fontsize=12)
ax[2].set_title('after mask', fontsize=12)

ax[0].imshow(imag, cmap='gray');
ax[1].imshow(image, cmap='gray');
ax[2].imshow(mask, cmap='gray');


lst_benign = []
for x in benign:
  lst_benign.append([x,1])
lst_malignant = []
for x in malignant:
  lst_malignant.append([x,0])
lst_complete = lst_benign + lst_malignant
random.shuffle(lst_complete)

df = pd.DataFrame(lst_complete,columns = ['files','target'])
df.head(10)
filepath_img ="dataset/malignant/*.png"
df = df.loc[~(df.loc[:,'files'] == filepath_img),:]
df.shape

plt.figure(figsize = (10,10))
sns.countplot(x = "target",data = df)
plt.title("BENING and MALIGNANT") 
plt.show()

def preprocessing_image(filepath):
  img = cv2.imread(filepath) #read
  img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) #convert
  img = cv2.resize(img,(196,196))  # resize
  img = img / 255 #scale
  return img

def create_format_dataset(dataframe):
  X = []
  y = []
  for f,t in dataframe.values:
    X.append(preprocessing_image(f))
    y.append(t)
  
  return np.array(X),np.array(y)
X, y = create_format_dataset(df)
X.shape,y.shape
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,stratify = y)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


'''CNN'''
CNN = Sequential()

CNN.add(Conv2D(32,(2,2),input_shape = (196,196,3),activation='relu'))
CNN.add(Conv2D(64,(2,2),activation='relu'))
CNN.add(MaxPooling2D())
CNN.add(Conv2D(32,(2,2),activation='relu'))
CNN.add(MaxPooling2D())

CNN.add(Flatten())
CNN.add(Dense(32))
CNN.add(Dense(1,activation= "sigmoid"))
CNN.summary()
CNN.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])
CNN.fit(X_train,y_train,validation_data=(X_test,y_test),epochs = 10,batch_size = 20)
print("Accuracy of the CNN is:",CNN.evaluate(X_test,y_test)[1]*100, "%")
history = CNN.history.history

#Plotting the accuracy
train_loss = history['loss']
val_loss = history['val_loss']
train_acc = history['acc']
val_acc = history['val_acc']
    
# Loss
plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.show()
    
# Accuracy
plt.figure()
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()

y_pred = CNN.predict(X_test)
y_pred = y_pred.reshape(-1)
y_pred[y_pred<0.5] = 0
y_pred[y_pred>=0.5] = 1
y_pred = y_pred.astype('int')
y_pred
print('\n')
classification=classification_report(y_test,y_pred)
print(classification)
print('\n')
plt.figure(figsize = (20,10))
cm = confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(confusion_matrix(y_test,y_pred),annot = True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()