'''
Author: Harshitha Machiraju
Date  : 20/11/2018
Title : Implementation of Naive Bayes Classifier
Description:->The input is taken as Y which has observations in rows and features are 
				represented as columns
			->The labels are taken in as the row vector t
			->The function train_bayes then does suitable transpose as required for Y and t
			and trains the classifier
			->The function test_bayes can be used for test data provided it is a row vector.
'''



import numpy as np
import pandas as pd

#Function of gaussian MLE
def MLE_gaussian(Y):
	mean=np.mean(Y)
	sigma_square=np.var(Y)
	return mean,sigma_square

#Function to calculate the pdf of x in that distribution
def prob_gaussian(x,mean,sigma_square):
	den=(np.sqrt(2*np.pi*sigma_square))**(-0.5)
	num=np.exp(((x-mean)**2)/float((-2*sigma_square)))
	return num*den

#Function to estimate the mean and variance of gaussian using MLE for each feature
def feature_parameters_for_class(Y_class):
	no_columns=Y_class.shape[1]
	feature_metric_estimates=[]
	for i in range(no_columns):
		mle_estimate=MLE_gaussian(Y_class[:,i])
		feature_metric_estimates=feature_metric_estimates+[mle_estimate]
	return feature_metric_estimates

# The main function which preprocesses the data and trains the classifier
def train_bayes(Y,t):
	#Observations are in rows and features in columns
	#t is 1000x1
	#Y becomes 1000x2
	if(Y.shape[0]<Y.shape[1]):
		Y=Y.T 
	if(t.shape[0]<t.shape[1]):
		t=t.T 

	#Matrix with the observations and corresponding labels	
	merged_data=np.concatenate((Y,t),axis=1)

	last_col=merged_data.shape[1]
	last_col=last_col-1

	#Separate the observations with class label +1 and -1
	Y_class1=merged_data[merged_data[:,last_col]==1,:]
	Y_class2=merged_data[merged_data[:,last_col]==-1,:]

	#Remove the labels column from the arrays
	Y_class1=Y_class1[:,0:last_col]#it copies till 1 less than last_col
	Y_class2=Y_class2[:,0:last_col]

	#Calculate the feature parameters for both class +1 and -1
	feature_metric_class1=feature_parameters_for_class(Y_class1)
	feature_metric_class2=feature_parameters_for_class(Y_class2)
	parameters=[feature_metric_class1,feature_metric_class2]

	#Calculate the probabilities of each class
	prob_class1=np.count_nonzero(t == 1)/float(t.shape[0])
	prob_class2=np.count_nonzero(t == -1)/float(t.shape[0])
	probs=[prob_class1,prob_class2]

	return parameters,probs

#Y_test is a row vector
def test_bayes(Y_test,parameters,probs):
	#for class 1
	prod_1=1
	prod_2=1
	for i in range(Y_test.shape[1]):
		prod_1=prod_1*prob_gaussian(Y_test[:,i],parameters[0][i][0],parameters[0][i][1])
		prod_2=prod_2*prob_gaussian(Y_test[:,i],parameters[1][i][0],parameters[1][i][1])
	prod_1=probs[0]*prod_1
	prod_2=probs[1]*prod_2
	if(prod_1>prod_2):
		return 1
	else:
		return -1


##########################################################################

#Take the inputs 'Y' and the labels 't'
Y = np.asarray(pd.read_csv('Y.csv', sep=',',header=None))
t=np.asarray(pd.read_csv('t.csv', sep=',',header=None))

# Train the bayes classifier
parameters,probs=train_bayes(Y,t)

#Test inputs
y_test_1=np.array([[1,1]]) #Ground truth 1
y_test_2=np.array([[1,-1]]) #Ground truth 1
y_test_3=np.array([[-1,1]]) #Ground truth -1
y_test_4=np.array([[-1,-1]]) #Ground truth -1

c1=test_bayes(y_test_1,parameters,probs)
c2=test_bayes(y_test_2,parameters,probs)
c3=test_bayes(y_test_3,parameters,probs)
c4=test_bayes(y_test_4,parameters,probs)

print("class for "+str(y_test_1)+" is "+str(c1))
print("class for "+str(y_test_2)+" is "+str(c2))
print("class for "+str(y_test_3)+" is "+str(c3))
print("class for "+str(y_test_4)+" is "+str(c4))









