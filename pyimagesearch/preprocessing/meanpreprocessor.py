# import package
import cv2

class MeanPreProcessor:
    def __init__(self, rMean, gMean, bMean):
        # store the read green and blue mean average across the training set
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def preprocess(self, image):
        # split the channel into respective Red, Green and blue channel
        (B, G, R) = cv2.split(image.astype('float32'))

        # subtract the means for each channel
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean

        # merge the channel back together and return the image
        return cv2.merge([B,G,R])
