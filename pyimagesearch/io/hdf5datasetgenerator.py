# import package
from tensorflow.keras.utils import to_categorical
import numpy as np 
import h5py

class HDF5DataSetGenerator:
    def __init__(self, dbPath, batchSize, preprocessors=None, aug=None,
     binarize=True, classes=2):
        # store the batch size, preprocessors, and data aug,
        # whether or not the label should be binarized
        # with the total num of class
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self. classes = classes

        # open the hdf5 database for reading and determine the total num
        # of entries in the db
        self.db = h5py.File(dbPath, "r")
        self.numImages = self.db['labels'].shape[0]

    def generator(self, passes=np.inf):
        # init the epoch count
        epochs = 0

        # keep looping indefinately -- the model will stop once we have 
        # reach the desired number of epochs
        while epochs < passes:
            # loop over the hdf5 datasets
            for i in np.arange(0, self.numImages, self.batchSize):
                # extract the images and lablels from the hdf5 dataset
                images = self.db['images'][i: i + self.batchSize]
                labels = self.db['labels'][i: i + self.batchSize]

                # check to see if the labels should be binarized
                if self.binarize:
                    labels =  to_categorical(labels, self.classes)

                # check to see if our preprocessors are not None
                if self.preprocessors is not None:
                    # init the preprocess images
                    procImages = []

                    # loop over the image
                    for image in images:

                        # loop over the preprocess and applly
                        # to all images
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        # update the list of processd image
                        procImages.append(image)

                    # update the images array to be the processed images
                    image = np.array(procImages)

                # if data aug exist, apply it
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images, labels,
                     batch_size=self.batchSize))

                # yield a tuple of image and labels
                yield (images, labels)

            # increment the total number of epoch

            epochs += 1

    def close(self):
        # close db
        self.db.close()

