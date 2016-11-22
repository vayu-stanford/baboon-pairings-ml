import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm

''' ##################################### Consortship classification & Feature Selection ##################################### '''

n_features = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

'''# Forward Feature Selection using log-reg
errors = []
recalls = []
precisions = []
f1s = []
for j in range(2,17):
	K=3
	attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(j))
	labels=np.genfromtxt('data_file.csv',delimiter=',',usecols=[16],dtype='str')
	correct=0
	wrong=0
	average_recall=0
	average_precision=0
	average_fischer=0
	average_error=0

	cv=KFold(labels.shape[0],K,shuffle = False)
	for train_index, test_index in cv:
	    m = LogisticRegression(class_weight='balanced', n_jobs=-1)
	    m.fit(attrs[train_index,:],labels[train_index])
	    preds = m.predict(attrs[test_index,:])
	    wrong += np.sum(labels[test_index] != preds)
	    correct += np.sum(labels[test_index] == preds)

	    average_recall = average_recall + 1.0/K * metrics.recall_score(labels[test_index],preds,pos_label = '1')
	    average_precision = average_precision + 1.0/K * metrics.precision_score(labels[test_index],preds,pos_label = '1')
	    average_fischer = average_fischer + 1.0/K * metrics.f1_score(labels[test_index],preds,pos_label = '1')
	    average_error = average_error + 1.0/K * float(wrong)/(correct+wrong)


	precisions.append(average_precision)
	f1s.append(average_fischer)
	recalls.append(average_recall)
	errors.append(average_error)

# Confusion matrix when using all features
confusion_matrix = metrics.confusion_matrix(labels[test_index],preds)

info = np.transpose([n_features,errors,precisions,recalls,f1s])
np.savetxt('logReg_confusion_matrix.txt',confusion_matrix,fmt='%.1f')
np.savetxt('logReg_forwardFSelection.txt',info,fmt='%.5f',header = 'features,  error,  precision,  recall,  fischer')


# Forward Feature Selection using linear SVM
errors = []
recalls = []
precisions = []
f1s = []
for j in range(2,17):
	print j
	K=3
	attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(j))
	labels=np.genfromtxt('data_file.csv',delimiter=',',usecols=[16],dtype='str')
	correct=0
	wrong=0
	average_recall=0
	average_precision=0
	average_fischer=0
	average_error=0

	cv=KFold(labels.shape[0],K,shuffle = False)
	for train_index, test_index in cv:
	    m = svm.SVC(class_weight='balanced',kernel='linear')
	    m.fit(attrs[train_index,:],labels[train_index])
	    preds = m.predict(attrs[test_index,:])
	    wrong += np.sum(labels[test_index] != preds)
	    correct += np.sum(labels[test_index] == preds)

	    average_recall = average_recall + 1.0/K * metrics.recall_score(labels[test_index],preds,pos_label = '1')
	    average_precision = average_precision + 1.0/K * metrics.precision_score(labels[test_index],preds,pos_label = '1')
	    average_fischer = average_fischer + 1.0/K * metrics.f1_score(labels[test_index],preds,pos_label = '1')
	    average_error = average_error + 1.0/K * float(wrong)/(correct+wrong)


	precisions.append(average_precision)
	f1s.append(average_fischer)
	recalls.append(average_recall)
	errors.append(average_error)

# Confusion matrix when using all features
confusion_matrix = metrics.confusion_matrix(labels[test_index],preds)

info = np.transpose([n_features,errors,precisions,recalls,f1s])
np.savetxt('linearSVM_confusion_matrix.txt',confusion_matrix,fmt='%.1f')
np.savetxt('linearSVM_forwardFSelection.txt',info,fmt='%.5f',header = 'features,  error,  precision,  recall,  fischer')'''

'''# Forward Feature Selection using Gaussian SVM
errors = []
recalls = []
precisions = []
f1s = []
for j in range(2,17):
	print j
	K=3
	attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(j))
	labels=np.genfromtxt('data_file.csv',delimiter=',',usecols=[16],dtype='str')
	correct=0
	wrong=0
	average_recall=0
	average_precision=0
	average_fischer=0
	average_error=0

	cv=KFold(labels.shape[0],K,shuffle = False)
	for train_index, test_index in cv:
	    m = svm.SVC(C=100,class_weight='balanced',kernel='poly')
	    m.fit(attrs[train_index,:],labels[train_index])
	    preds = m.predict(attrs[test_index,:])
	    wrong += np.sum(labels[test_index] != preds)
	    correct += np.sum(labels[test_index] == preds)

	    average_recall = average_recall + 1.0/K * metrics.recall_score(labels[test_index],preds,pos_label = '1')
	    average_precision = average_precision + 1.0/K * metrics.precision_score(labels[test_index],preds,pos_label = '1')
	    average_fischer = average_fischer + 1.0/K * metrics.f1_score(labels[test_index],preds,pos_label = '1')
	    average_error = average_error + 1.0/K * float(wrong)/(correct+wrong)


	precisions.append(average_precision)
	f1s.append(average_fischer)
	recalls.append(average_recall)
	errors.append(average_error)

# Confusion matrix when using all features
confusion_matrix = metrics.confusion_matrix(labels[test_index],preds)

info = np.transpose([n_features,errors,precisions,recalls,f1s])
np.savetxt('polySVM_C100_confusion_matrix.txt',confusion_matrix,fmt='%.1f')
np.savetxt('polySVM_C100_forwardFSelection.txt',info,fmt='%.5f',header = 'features,  error,  precision,  recall,  fischer')'''



'''# Backward Feature Selection using log-reg
errors = []
recalls = []
precisions = []
f1s = []
for j in range(2,17):
	K=3
	attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(16-j,16))
	print attrs
	print ' '
	labels=np.genfromtxt('data_file.csv',delimiter=',',usecols=[16],dtype='str')
	correct=0
	wrong=0
	average_recall=0
	average_precision=0
	average_fischer=0
	average_error=0

	cv=KFold(labels.shape[0],K,shuffle = False)
	for train_index, test_index in cv:
	    m = LogisticRegression(class_weight='balanced', n_jobs=-1)
	    m.fit(attrs[train_index,:],labels[train_index])
	    preds = m.predict(attrs[test_index,:])
	    wrong += np.sum(labels[test_index] != preds)
	    correct += np.sum(labels[test_index] == preds)

	    average_recall = average_recall + 1.0/K * metrics.recall_score(labels[test_index],preds,pos_label = '1')
	    average_precision = average_precision + 1.0/K * metrics.precision_score(labels[test_index],preds,pos_label = '1')
	    average_fischer = average_fischer + 1.0/K * metrics.f1_score(labels[test_index],preds,pos_label = '1')
	    average_error = average_error + 1.0/K * float(wrong)/(correct+wrong)


	precisions.append(average_precision)
	f1s.append(average_fischer)
	recalls.append(average_recall)
	errors.append(average_error)

info = np.transpose([n_features,errors,precisions,recalls,f1s])
np.savetxt('logReg_backwardFSelection.txt',info,fmt='%.5f',header = 'features,  error,  precision,  recall,  fischer')


'''# Backward Feature Selection using linear SVM
errors = []
recalls = []
precisions = []
f1s = []
for j in range(2,17):
	print j
	K=3
	attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(16-j,16))
	labels=np.genfromtxt('data_file.csv',delimiter=',',usecols=[16],dtype='str')
	correct=0
	wrong=0
	average_recall=0
	average_precision=0
	average_fischer=0
	average_error=0

	cv=KFold(labels.shape[0],K,shuffle = False)
	for train_index, test_index in cv:
	    m = svm.SVC(C=100,class_weight='balanced',kernel='linear')
	    m.fit(attrs[train_index,:],labels[train_index])
	    preds = m.predict(attrs[test_index,:])
	    wrong += np.sum(labels[test_index] != preds)
	    correct += np.sum(labels[test_index] == preds)

	    average_recall = average_recall + 1.0/K * metrics.recall_score(labels[test_index],preds,pos_label = '1')
	    average_precision = average_precision + 1.0/K * metrics.precision_score(labels[test_index],preds,pos_label = '1')
	    average_fischer = average_fischer + 1.0/K * metrics.f1_score(labels[test_index],preds,pos_label = '1')
	    average_error = average_error + 1.0/K * float(wrong)/(correct+wrong)


	precisions.append(average_precision)
	f1s.append(average_fischer)
	recalls.append(average_recall)
	errors.append(average_error)

info = np.transpose([n_features,errors,precisions,recalls,f1s])
np.savetxt('linearSVM_backwardFSelection.txt',info,fmt='%.5f',header = 'features,  error,  precision,  recall,  fischer')

'''# Backward Feature Selection using Gaussian SVM
errors = []
recalls = []
precisions = []
f1s = []
for j in range(2,17):
	print j
	K=3
	attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(16-j,16))
	labels=np.genfromtxt('data_file.csv',delimiter=',',usecols=[16],dtype='str')
	correct=0
	wrong=0
	average_recall=0
	average_precision=0
	average_fischer=0
	average_error=0

	cv=KFold(labels.shape[0],K,shuffle = False)
	for train_index, test_index in cv:
	    m = svm.SVC(C=100,class_weight='balanced',kernel='rbf')
	    m.fit(attrs[train_index,:],labels[train_index])
	    preds = m.predict(attrs[test_index,:])
	    wrong += np.sum(labels[test_index] != preds)
	    correct += np.sum(labels[test_index] == preds)

	    average_recall = average_recall + 1.0/K * metrics.recall_score(labels[test_index],preds,pos_label = '1')
	    average_precision = average_precision + 1.0/K * metrics.precision_score(labels[test_index],preds,pos_label = '1')
	    average_fischer = average_fischer + 1.0/K * metrics.f1_score(labels[test_index],preds,pos_label = '1')
	    average_error = average_error + 1.0/K * float(wrong)/(correct+wrong)


	precisions.append(average_precision)
	f1s.append(average_fischer)
	recalls.append(average_recall)
	errors.append(average_error)


info = np.transpose([n_features,errors,precisions,recalls,f1s])
np.savetxt('gaussianSVM_C100_backwardFSelection.txt',info,fmt='%.5f',header = 'features,  error,  precision,  recall,  fischer')'''