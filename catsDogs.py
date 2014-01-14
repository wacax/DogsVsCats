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

#m = 5000 #pet dataset
m = 12499 #full dataset

indexesIm = np.random.permutation(m * len(labels))
idxImages = np.tile(range(m), len(labels))
idxImages = idxImages[indexesIm]
y = np.append(np.tile(0, m), np.tile(1, m))
y = y[indexesIm]

def animalInput(theNumber):
    if theNumber == 0:
        return 'cat.'
    elif theNumber == 1:
        return 'dog.'
    else:
        return ''

#Build the sparse matrix with the preprocessed image data
lilTrainMatrix = lil_matrix((m * len(labels), desiredDimensions[0] * desiredDimensions[1]))

for i in range(m * len(labels)):
    lilTrainMatrix[i, :] = preprocessImg(animalInput(y[i]), idxImages[i], desiredDimensions[0], desiredDimensions[1], dataTrainDir)

lilTrainMatrix = lilTrainMatrix.tocsr()

#Reduce features to main components so that they contain 99% of variance
pca = RandomizedPCA(n_components=150, whiten = True)
pca.fit(lilTrainMatrix)
varianceExplained = pca.explained_variance_ratio_
print(pca.explained_variance_ratio_)

variance = 0
for ii in range(len(varianceExplained)):
    variance += varianceExplained[ii]
    if variance > 0.99:
        componentIdx = ii
        break

pca = RandomizedPCA(n_components=150, whiten = True)
trainMatrixReduced = pca.fit_transform(lilTrainMatrix, y = componentIdx)

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
#Build the sparse matrix with the preprocessed image data
mTest = 12500 #number of images in the test set
lilTestMatrix = lil_matrix((mTest, desiredDimensions[0] * desiredDimensions[1]))

for i in range(1, mTest):
    lilTestMatrix[i, :] = preprocessImg(animalInput('printNothing'), i, desiredDimensions[0], desiredDimensions[1], dataTestDir)

lilTestMatrix = lilTestMatrix.tocsr()

pca = RandomizedPCA(n_components=150, whiten = True)
testMatrixReduced = pca.fit_transform(lilTestMatrix, y = componentIdx)

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