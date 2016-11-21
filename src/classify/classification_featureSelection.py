import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import svm

''' Consortship classification'''

logReg_forward_feature_selection_errors = []
linear_svm_forward_feature_selection_errors = []
gaussian_svm_forward_feature_selection_errors = []

logReg_backward_feature_selection_errors = []
linear_svm_backward_feature_selection_errors = []
gaussian_svm_backward_feature_selection_errors = []

# Forward Feature Selection using log-reg
for j in range(2,14):
	K=5
	valid_labels=('1','0')
	attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(j))
	labels=np.genfromtxt('data_file.csv',delimiter=',',usecols=[15],dtype='str')
	cv=KFold(labels.shape[0],K)
	correct=0
	wrong=0
	for train_index, test_index in cv:
	    m = LogisticRegression()
	    m.fit(attrs[train_index,:],labels[train_index])
	    preds=m.predict(attrs[test_index,:])
	    wrong+=np.sum(labels[test_index]!=preds)
	    correct+=np.sum(labels[test_index]==preds)
	#print "Correct:",correct
	#print "Wrong:",wrong
	#print "Error rate:",float(wrong)/(correct+wrong)
	logReg_forward_feature_selection_errors.append(float(wrong)/(correct+wrong))
print logReg_forward_feature_selection_errors

# Forward Feature Selection using linear SVM
for j in range(2,14):
	K=3
	valid_labels=('1','0')
	attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(j))
	labels=np.genfromtxt('data_file.csv',delimiter=',',usecols=[15],dtype='str')
	cv=KFold(labels.shape[0],K)
	correct=0
	wrong=0
	for train_index, test_index in cv:
	    m = svm.SVC(C=1,kernel='linear')
	    m.fit(attrs[train_index,:],labels[train_index])
	    preds=m.predict(attrs[test_index,:])
	    wrong+=np.sum(labels[test_index]!=preds)
	    correct+=np.sum(labels[test_index]==preds)
	#print "Correct:",correct
	#print "Wrong:",wrong
	#print "Error rate:",float(wrong)/(correct+wrong)
	linear_svm_forward_feature_selection_errors.append(float(wrong)/(correct+wrong))
print linear_svm_forward_feature_selection_errors

'''# Forward Feature Selection using gaussian SVM
for j in range(2,14):
	K=5
	valid_labels=('1','0')
	attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(j))
	labels=np.genfromtxt('data_file.csv',delimiter=',',usecols=[15],dtype='str')
	cv=KFold(labels.shape[0],K)
	correct=0
	wrong=0
	for train_index, test_index in cv:
	    m = svm.SVC(C=1,kernel='rbf')
	    m.fit(attrs[train_index,:],labels[train_index])
	    preds=m.predict(attrs[test_index,:])
	    wrong+=np.sum(labels[test_index]!=preds)
	    correct+=np.sum(labels[test_index]==preds)
	#print "Correct:",correct
	#print "Wrong:",wrong
	#print "Error rate:",float(wrong)/(correct+wrong)
	gaussian_svm_forward_feature_selection_errors.append(float(wrong)/(correct+wrong))

# Backward Feature Selection using log-reg
for j in range(2,14):
	K=5
	valid_labels=('1','0')
	attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(14-j,14))
	labels=np.genfromtxt('data_file.csv',delimiter=',',usecols=[15],dtype='str')
	cv=KFold(labels.shape[0],K)
	correct=0
	wrong=0
	for train_index, test_index in cv:
	    m = LogisticRegression()
	    m.fit(attrs[train_index,:],labels[train_index])
	    preds=m.predict(attrs[test_index,:])
	    wrong+=np.sum(labels[test_index]!=preds)
	    correct+=np.sum(labels[test_index]==preds)
	#print "Correct:",correct
	#print "Wrong:",wrong
	#print "Error rate:",float(wrong)/(correct+wrong)
	logReg_backward_feature_selection_errors.append(float(wrong)/(correct+wrong))

# Backward Feature Selection using linear SVM
for j in range(2,14):
	K=5
	valid_labels=('1','0')
	attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(14-j,14))
	labels=np.genfromtxt('data_file.csv',delimiter=',',usecols=[15],dtype='str')
	cv=KFold(labels.shape[0],K)
	correct=0
	wrong=0
	for train_index, test_index in cv:
	    m = svm.SVC(C=1,kernel='linear')
	    m.fit(attrs[train_index,:],labels[train_index])
	    preds=m.predict(attrs[test_index,:])
	    wrong+=np.sum(labels[test_index]!=preds)
	    correct+=np.sum(labels[test_index]==preds)
	#print "Correct:",correct
	#print "Wrong:",wrong
	#print "Error rate:",float(wrong)/(correct+wrong)
	linear_svm_backward_feature_selection_errors.append(float(wrong)/(correct+wrong))

# Backward Feature Selection using Gaussian SVM
for j in range(2,14):
	K=5
	valid_labels=('1','0')
	attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(14-j,14))
	labels=np.genfromtxt('data_file.csv',delimiter=',',usecols=[15],dtype='str')
	cv=KFold(labels.shape[0],K)
	correct=0
	wrong=0
	for train_index, test_index in cv:
	    m = svm.SVC(C=1,kernel='rbf')
	    m.fit(attrs[train_index,:],labels[train_index])
	    preds=m.predict(attrs[test_index,:])
	    wrong+=np.sum(labels[test_index]!=preds)
	    correct+=np.sum(labels[test_index]==preds)
	#print "Correct:",correct
	#print "Wrong:",wrong
	#print "Error rate:",float(wrong)/(correct+wrong)
	gaussian_svm_backward_feature_selection_errors.append(float(wrong)/(correct+wrong))

print 'log_reg_forward_feature_selection_errors' ,log_reg_forward_feature_selection_errors
print 'linear_svm_forward_feature_selection_errors' ,linear_svm_forward_feature_selection_errors
print 'gaussian_svm_forward_feature_selection_errors' ,gaussian_svm_forward_feature_selection_errors

print 'log_reg_backward_feature_selection_errors' ,log_reg_backward_feature_selection_errors
print 'linear_svm_backward_feature_selection_errors' ,linear_svm_backward_feature_selection_errors
print 'gaussian_svm_backward_feature_selection_errors' ,gaussian_svm_backward_feature_selection_errors'''