# import package
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

class AlexNet:
    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        # init the model along with the input shape to be
        # "channel last ordering"
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using channel first ordering
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # Block 1 conv -> rlu -> pool
        model.add(Conv2D(96, (11, 11), strides=(4,4),
        input_shape=inputShape, padding="same",
        kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Dropout(0.25))


        # Block 2 conv -> relu -> pool
        model.add(Conv2D(256, (5,5), padding="same",
        kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Dropout(0.25))

        # Block 3 conv, relu, conv, relu, conv, relu
        model.add(Conv2D(384, (3,3), padding="same",
        kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(384, (3,3), padding="same",
        kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3,3), padding="same",
        kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Dropout(0.25))

        # Block 4 first set of FC, relu layers
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Block 5 second set of fc relu
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes, kernel_regularizer=l2(reg)))
        model.add(Activation("softmax"))

        # return the constructed network arch
        return model


        