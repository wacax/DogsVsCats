__author__ = 'wacax'

#version 0.3
#the revenge of the ng

#import libraries
import os
import cv2
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics

wd = '/home/wacax/Documents/Wacax/Kaggle Data Analysis/DogsCats/' #change this to make the code work
dataTrainDir = '/home/wacax/Documents/Wacax/Kaggle Data Analysis/DogsCats/Data/train/'
dataTestDir = '/home/wacax/Documents/Wacax/Kaggle Data Analysis/DogsCats/Data/test1/'

os.chdir(wd)

labels = ['cat.', 'dog.']
desiredDimensions = [20, 20]

#define loading and pre-processing function grayscale
def preprocessImg(animal, number, dim1, dim2, dataDir):
    imageName = '{0:s}{1:s}{2:d}{3:s}'.format(dataDir, animal, number, '.jpg')
    npImage = cv2.imread(imageName)
    npImage = cv2.cvtColor(npImage, cv2.COLOR_BGR2GRAY)
    avg = np.mean(npImage.reshape(1, npImage.shape[0] * npImage.shape [1]))
    avg = np.tile(avg, (npImage.shape[0], npImage.shape [1]))
    npImage = npImage - avg
    npImage = cv2.resize(npImage, (dim1, dim2))
    return(npImage.reshape(1, dim1 * dim2))

#m = 5000 #pet Train dataset
m = 12499 #full Train dataset
mTest = 12500 #number of images in the test set


indexesIm = np.random.permutation(m * len(labels) + len(labels))
idxImages = np.tile(range(m + 1), len(labels))
idxImages = idxImages[indexesIm]
testIndexes = range(len(indexesIm) - 1, len(indexesIm) + mTest)
y = np.append(np.tile(0, m + 1), np.tile(1, m + 1))
y = y[indexesIm]

def animalInput(theNumber):
    if theNumber == 0:
        return 'cat.'
    elif theNumber == 1:
        return 'dog.'
    else:
        return ''

#Build the sparse matrix with the preprocessed image data for both train and test data
bigMatrix = lil_matrix((len(indexesIm) + len(testIndexes), desiredDimensions[0] * desiredDimensions[1]))

for i in range(len(indexesIm)):
    bigMatrix[i, :] = preprocessImg(animalInput(y[i]), idxImages[i], desiredDimensions[0], desiredDimensions[1], dataTrainDir)

for i in range(1, mTest + 1):
    bigMatrix[testIndexes[i], :] = preprocessImg(animalInput('printNothing'), i, desiredDimensions[0], desiredDimensions[1], dataTestDir)

#Transform to csr matrix
bigMatrix = bigMatrix.tocsr()

#Reduce features to main components so that they contain 99% of variance
pca = RandomizedPCA(n_components=150, whiten = True)
pca.fit(bigMatrix)
varianceExplained = pca.explained_variance_ratio_
print(pca.explained_variance_ratio_)

variance = 0
for ii in range(len(varianceExplained)):
    variance += varianceExplained[ii]
    if variance > 0.99:
        componentIdx = ii
        break

pca = RandomizedPCA(n_components=150, whiten = True)
BigMatrixReduced = pca.fit_transform(bigMatrix, y = componentIdx)

#Divide train Matrix and Test Matrix (for which I don't have labels)
trainMatrixReduced = BigMatrixReduced[0:2*m, :]
testMatrixReduced = BigMatrixReduced[BigMatrixReduced.shape[0] - mTest:BigMatrixReduced.shape[0], :]

#Divide dataset for cross validation purposes
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    trainMatrixReduced, y, test_size=0.4, random_state=0)

#Machine Learning part
#Support vector machine model
clf = svm.SVC(probability = True, verbose = True)
clf.fit(X_train, y_train)

#prediction
predictionFromDataset = clf.predict(X_test)

correctValues = sum(predictionFromDataset == y_test)
percentage = float(correctValues)/len(y_test)

print(percentage)

#prediction probability
predictionFromDataset2 = clf.predict_proba(X_test)
predictionFromDataset2 = predictionFromDataset2[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, predictionFromDataset2)
predictionProbability = metrics.auc(fpr, tpr)

#Predict images from the test set


#Train the model with full data set
clf = svm.SVC(probability = True, verbose = True)
clf.fit(trainMatrixReduced, y)

#Prediction
#predictionFromTest = clf.predict_proba(testMatrixReduced)
predictionFromTest = clf.predict(testMatrixReduced)
#label = predictionFromTest[:, 1]
idVector = range(1, mTest + 1)

#predictionsToCsv = np.column_stack((idVector, label))
predictionsToCsv = np.column_stack((idVector, predictionFromTest))

import csv

ofile = open('predictionI.csv', "wb")
fileToBeWritten = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

for row in predictionsToCsv:
    fileToBeWritten.writerow(row)

ofile.close()