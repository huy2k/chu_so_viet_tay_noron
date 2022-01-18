import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import API functions
from keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
# get the image parameters
num_train_samples = train_images.shape[0]
num_test_samples = test_images.shape[0]
height = train_images.shape[1]
width = train_images.shape[2]

# normalize the data samples
train_images = train_images.reshape((num_train_samples, height, width,1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((num_test_samples, height, width,1))
test_images = test_images.astype('float32') / 255

# encode the targers
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
# y = test_labels
test_labels = to_categorical(test_labels)

# create samples for simple hold-out validation
val_train_images = train_images[:10000]
val_train_labels = train_labels[:10000]
partial_train_images = train_images[10000:]
partial_train_labels = train_labels[10000:]

# create a network model
from keras import models
from keras import layers
num_classes = 10
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height,width,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.35))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.35))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

#train the network
history = model.fit(partial_train_images, partial_train_labels, epochs = 5, batch_size = 128, validation_data=(val_train_images, val_train_labels))

#evaluate the trained netwotk
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)





# plot training loss and validation loss
# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.plot(history.history['loss'],'-b^',label = 'Training loss')
# plt.plot(history.history['val_loss'],'-rv',label = 'Validation loss')
# plt.ylabel('Loss')
# plt.xlabel('Epochs')
# plt.legend(loc = 'upper right')

# #plot training accurancy and validation accurancy
# plt.figure(2)
# plt.plot(history.history['accuracy'],'-b>',label = 'Training accuracy')
# plt.plot(history.history['val_accuracy'],'-r<',label = 'Validation accuracy')
# plt.ylabel('Accurancy')
# plt.xlabel('Epochs')
# plt.legend(loc = 'upper left')
# plt.show()


import numpy as np
run = False
ix,iy = -1,-1
follow = 25
img = np.zeros((256,256 ,1))
## func
import cv2

def draw(event, x, y, flag, params):
    global run,ix,iy,img,follow
    if event == cv2.EVENT_LBUTTONDOWN:
        run = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if run == True:
            cv2.circle(img, (x,y), 10, (255,255,255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        run = False
        cv2.circle(img, (x,y), 10, (255,255,255), -1)
        gray = cv2.resize(img, (28, 28))
        gray = gray.reshape(1, 28,28,1)
        result = np.argmax(model.predict(gray))
        result = 'cnn : {}'.format(result)
        print(result) 
        # cv2.putText(img, org=(25,follow), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, text= result, color=(255,0,0), thickness=1)
        # follow += 25
    elif event == cv2.EVENT_RBUTTONDOWN:
        img = np.zeros((256,256,1))
        # follow = 25


##

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)

while True:    
    cv2.imshow("image", img)
   
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()


# 
# import cv2
# from google.colab.patches import cv2_imshow
# import numpy as np
# image = cv2.imread("photo_2.jpg")
# im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# im_blur = cv2.GaussianBlur(im_gray,(5,5),0)
# cv2_imshow(im_blur)
# im,thre = cv2.threshold(im_blur,90,255,cv2.THRESH_BINARY_INV)

# contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# rects = [cv2.boundingRect(cnt) for cnt in contours]

# for i in contours:
#     (x,y,w,h) = cv2.boundingRect(i)
#     cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
#     roi = thre[y:y+h,x:x+w]
#     roi = np.pad(roi,(20,20),'constant',constant_values=(0,0))
#     roi = cv2.resize(roi, (28, 28))
#     roi = cv2.dilate(roi, (3, 3))
#     img = roi.astype(np.float32)
#     roi = img/255
#     gray = roi.reshape(1,28,28,1)
#     result = np.argmax(model.predict(gray))
#     print(result)
#     cv2.putText(image,str(result), (x, y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
#     cv2_imshow(image)
# cv2.imwrite("image_pands.jpg",image)
# cv2.waitKey()
# cv2.destroyAllWindows()

