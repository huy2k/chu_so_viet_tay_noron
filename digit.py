from keras.datasets import mnist 
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
import warnings 
warnings.filterwarnings('ignore') 
from keras import models 
from keras import layers
from keras.datasets import mnist

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
train_images=train_images.reshape((60000,28*28))
train_images=train_images.astype('float32')/255
test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255

from tensorflow.keras.utils  import to_categorical
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

model=models.Sequential()
model.add(layers.Dense(256,activation='relu',input_shape=(28*28,)))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(28,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_images,train_labels,epochs=5,batch_size=128)
test_loss,test_acc=model.evaluate(test_images,test_labels)
print('test_acc:',test_acc)

#
import numpy as np
run = False
ix,iy = -1,-1
follow = 25
img = np.zeros((256,256 ,1))
### func
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
        gray = gray.reshape(1, 784)
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