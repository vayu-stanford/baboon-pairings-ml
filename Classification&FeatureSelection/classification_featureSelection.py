import numpy as np
from scipy import interp
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix,roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import preprocess
from sklearn.tree import DecisionTreeClassifier



def plot_confusion_matrix(cm, classes,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def evaluateGaussianSVM(all_attrs,labels,f_selection):
	errors = []
	recalls = []
	precisions = []
	f1s = []
	if f_selection == False:
		i = 16
		n_features = [16]
	else:
		i = 2
		n_features = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

	for j in range(i,17):
		print j
		K=10
		#attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(j))
		attrs = all_attrs[0:len(labels),0:j]
		#attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(16-j,16))
		correct=0
		wrong=0
		average_recall=0
		average_precision=0
		average_fischer=0
		average_error=0

		'''mean_tpr = 0.0
		mean_fpr = np.linspace(0, 1, 100)
		roc_auc = 0
		lw = 2'''

		cv=KFold(labels.shape[0],K,shuffle = True)
		for train_index, test_index in cv:
		    (pre_attrs,pre_labels) = (attrs[train_index,:],labels[train_index])
		   # (pre_attrs, pre_labels) = preprocess.augment(pre_attrs,pre_labels, attr_idxs=range(15),  mult=2)
		    #m = LogisticRegression(class_weight='balanced', n_jobs=-1)
		    m = svm.SVC(C=2,class_weight='balanced',kernel='rbf')
		    m.fit(pre_attrs,pre_labels)
		    preds = m.predict(attrs[test_index,:])
		    wrong += np.sum(labels[test_index] != preds)
		    correct += np.sum(labels[test_index] == preds)

		    average_recall += 1.0/K * metrics.recall_score(labels[test_index],preds,pos_label = 1)
		    average_precision += 1.0/K * metrics.precision_score(labels[test_index],preds,pos_label = 1)
		    average_fischer += 1.0/K * metrics.f1_score(labels[test_index],preds,pos_label = 1)
		    average_error += 1.0/K * float(wrong)/(correct+wrong)

		    '''if j == 16:
		    	fpr, tpr,thresh = roc_curve(labels[test_index],preds,pos_label =1)
		    	mean_tpr += interp(mean_fpr, fpr, tpr)
    			roc_auc += auc(fpr, tpr)'''


		precisions.append(average_precision)
		f1s.append(average_fischer)
		recalls.append(average_recall)
		errors.append(average_error)

	info = np.transpose([n_features,errors,precisions,recalls,f1s])
	np.savetxt('gaussianSVM_forwardFSelection.txt',info,fmt='%.5f',header = 'features,  error,  precision,  recall,  fischer')

	'''mean_tpr /= K
	mean_tpr[-1] = 1.0
	mean_tpr[0] = 0.0
	mean_auc = auc(mean_fpr, mean_tpr)
	plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()'''

def evaluateAdaBoost(all_attrs,labels,f_selection):
	errors = []
	recalls = []
	precisions = []
	f1s = []
	if f_selection == False:
		i = 16
		n_features = [16]
	else:
		i = 2
		n_features = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
	for j in range(i,17):
		print j
		K=5
		#attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(j))
		attrs = all_attrs[0:len(labels),0:j]
		print attrs
		#attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(16-j,16))
		correct=0
		wrong=0
		average_recall=0
		average_precision=0
		average_fischer=0
		average_error=0

		cv=KFold(labels.shape[0],K,shuffle = True)
		for train_index, test_index in cv:
			#m = LogisticRegression(class_weight='balanced', n_jobs=-1)
		    #m = svm.SVC(C=0.1,class_weight='balanced',kernel='rbf')
		    (pre_attrs,pre_labels) = (attrs[train_index,:],labels[train_index])
		    #(pre_attrs, pre_labels) = preprocess.augment(pre_attrs,pre_labels, attr_idxs=range(15),  mult=5)
		    #b_m = svm.SVC(probability = True, C=0.1,class_weight='balanced',kernel='rbf')
		    b_m = DecisionTreeClassifier(max_depth = 1, min_samples_leaf = 1, class_weight = 'balanced')
		    m = AdaBoostClassifier(base_estimator = b_m,n_estimators=200,random_state=1)
		    m.fit(pre_attrs,pre_labels)
		    preds = m.predict(attrs[test_index,:])
		    wrong += np.sum(labels[test_index] != preds)
		    correct += np.sum(labels[test_index] == preds)

		    average_recall += 1.0/K * metrics.recall_score(labels[test_index],preds,pos_label = 1)
		    average_precision += 1.0/K * metrics.precision_score(labels[test_index],preds,pos_label = 1)
		    average_fischer += 1.0/K * metrics.f1_score(labels[test_index],preds,pos_label = 1)
		    average_error += 1.0/K * float(wrong)/(correct+wrong)


		precisions.append(average_precision)
		f1s.append(average_fischer)
		recalls.append(average_recall)
		errors.append(average_error)
	info = np.transpose([n_features,errors,precisions,recalls,f1s])
	np.savetxt('adaBoost_forwardFSelection.txt',info,fmt='%.5f',header = 'features,  error,  precision,  recall,  fischer')

def evaluateRandomForest(all_attrs,labels,f_selection):
	errors = []
	recalls = []
	precisions = []
	f1s = []
	if f_selection == False:
		i = 16
		n_features = [16]
	else:
		i = 2
		n_features = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
	for j in range(i,17):
		print j
		K=20
		#attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(j))
		attrs = all_attrs[0:len(labels),0:j]
		#print attrs
		#attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(16-j,16))
		correct=0
		wrong=0
		average_recall=0
		average_precision=0
		average_fischer=0
		average_error=0

		cv=KFold(labels.shape[0],K,shuffle = True)
		for train_index, test_index in cv:
			#m = LogisticRegression(class_weight='balanced', n_jobs=-1)
		    #m = svm.SVC(C=0.1,class_weight='balanced',kernel='rbf')
		    (pre_attrs,pre_labels) = (attrs[train_index,:],labels[train_index])
		    #(pre_attrs, pre_labels) = preprocess.augment(pre_attrs,pre_labels, attr_idxs=range(15),  mult=5)
		    m = RandomForestClassifier(max_features=None,random_state=1,class_weight = 'balanced')
		    m.fit(pre_attrs,pre_labels)
		    preds = m.predict(attrs[test_index,:])
		    wrong += np.sum(labels[test_index] != preds)
		    correct += np.sum(labels[test_index] == preds)

		    average_recall += 1.0/K * metrics.recall_score(labels[test_index],preds,pos_label = 1)
		    average_precision += 1.0/K * metrics.precision_score(labels[test_index],preds,pos_label = 1)
		    average_fischer += 1.0/K * metrics.f1_score(labels[test_index],preds,pos_label = 1)
		    average_error += 1.0/K * float(wrong)/(correct+wrong)


		precisions.append(average_precision)
		f1s.append(average_fischer)
		recalls.append(average_recall)
		errors.append(average_error)
	info = np.transpose([n_features,errors,precisions,recalls,f1s])
	np.savetxt('randomForest_forwardFSelection.txt',info,fmt='%.5f',header = 'features,  error,  precision,  recall,  fischer')

def SVM_ConfusionMatrix(attrs,labels):
	X_train, X_test, y_train, y_test = train_test_split(attrs, labels, test_size=0.20, random_state=1)
	m = svm.SVC(C=0.1,class_weight='balanced',kernel='rbf')
	m.fit(X_train,y_train)
	preds = m.predict(X_test)
	confusion_matrix = metrics.confusion_matrix(y_test,preds)
	np.set_printoptions(precision=2)
	class_names = ['Non-Consort','Consort']

	# Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=True,
	                      title='Gaussian SVM Normalized confusion matrix')
	plt.show()
	#np.savetxt('gaussianSVM_confusion_matrix.txt',confusion_matrix,fmt='%.1f')

def AdaBoost_ConfusionMatrix(attrs,labels):
	X_train, X_test, y_train, y_test = train_test_split(attrs, labels, test_size=0.20, random_state=1)
	#b_m = svm.SVC(probability = True, C=0.1,class_weight='balanced',kernel='linear')
	#b_m = RandomForestClassifier(max_features=None,random_state=1,class_weight = 'balanced')
	b_m = DecisionTreeClassifier(max_depth = 1, min_samples_leaf = 1, class_weight = 'balanced')
	m = AdaBoostClassifier(base_estimator = b_m,n_estimators=100,random_state=1)
	m.fit(X_train,y_train)
	preds = m.predict(X_test)
	confusion_matrix = metrics.confusion_matrix(y_test,preds)
	np.set_printoptions(precision=2)
	class_names = ['Non-Consort','Consort']

	# Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=True,
	                      title='AdaBoost Normalized confusion matrix')
	#plt.show()
	#np.savetxt('gaussianSVM_confusion_matrix.txt',confusion_matrix,fmt='%.1f')

def RandomForest_ConfusionMatrix(attrs,labels):
	X_train, X_test, y_train, y_test = train_test_split(attrs, labels, test_size=0.20, random_state=1)
	m = RandomForestClassifier(max_features=None,random_state=1,class_weight='balanced')
	m.fit(X_train,y_train)
	preds = m.predict(X_test)
	confusion_matrix = metrics.confusion_matrix(y_test,preds)
	np.set_printoptions(precision=2)
	class_names = ['Non-Consort','Consort']
	#np.savetxt('gaussianSVM_confusion_matrix.txt',confusion_matrix,fmt='%.1f')

	# Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=True,
	                      title='Random Forest Normalized confusion matrix')
	plt.show()



'Importing Data'
attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(16))
labels=np.genfromtxt('data_file.csv',delimiter=',',usecols=[16],dtype='int')
(attrs,labels)=preprocess.preprocess_data(attrs,labels, range(15), whitening = False)
#(attrs, labels) = preprocess.augment(attrs,labels, attr_idxs=range(15),  mult=10)


'Model Evaluation and Feature Selection Based on Performance Metrics'
evaluateGaussianSVM(attrs,labels,False)
#evaluateAdaBoost(attrs,labels,False)
#evaluateRandomForest(attrs,labels,False)

'Model Metrics for All Features'
#SVM_ConfusionMatrix(attrs,labels)
#AdaBoost_ConfusionMatrix(attrs,labels)
#RandomForest_ConfusionMatrix(attrs,labels)
