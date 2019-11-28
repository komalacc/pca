import csv as csv 
import numpy as np
import Activation, logReg, optim, loadData

#################################################################
# reading from csv
print 'Loading Training Data'
csv_train = csv.reader(open('../data/train.csv', 'rb'))
header = csv_train.next()
data = [[map(int, row[1:]), [int(row[0])]] for row in csv_train]

train = loadData.Data()
train.loadList(data, numClasses = 10)
train.NormalizeScale(factor = 255.0)

#################################################################
# PCA of training set
print 'Performing PCA - Principal COmponent Analysis'
import npPCA
Z, U_reduced = npPCA.PCA(train.X, varRetained = 0.95, show = True)