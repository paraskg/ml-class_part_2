import numpy
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Reshape, AveragePooling2D, Activation
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config
config.epochs = 10

# load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.astype('float32')/255.
X_test = X_test.astype('float32')/255.

img_width = X_train.shape[1]
img_height = X_train.shape[2]
labels =["T-shirt/top","Trouser","Pullover","Dress",
    "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]

# create model
model=Sequential()
model.add(Reshape((28,28,1), input_shape=(28,28)))
#model.add(Dropout(0.5))
#model.add(Conv2D(32, (4,4), strides = 1, activation='relu',  padding='same', input_shape=(28,28)))
#model.add(Conv2D(32, (4,4), strides = 3, activation='relu', input_shape=(28,28)))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Conv2D(64, (3,3), activation='tanh',  padding='same', input_shape=(28,28)))
#model.add(Conv2D(64, (3,3), activation='tanh', input_shape=(28,28)))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Conv2D(32, (3,3), activation='relu',  padding='same', input_shape=(28,28)))
##model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28)))
#model.add(AveragePooling2D(pool_size=(2,2)))

#model.add(Flatten())
#model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
#model.add(Dense(500, activation='relu'))
#model.add(Dropout(0.2))


model.add(Conv2D(32, (3, 3), input_shape=(28,28)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

# Fully connected layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['accuracy'])


# Fit the model
model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test),
                    callbacks=[WandbCallback(data_type="image", labels=labels)])

print(model.predict(X_train[:10]))