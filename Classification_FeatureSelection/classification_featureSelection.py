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
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn.naive_bayes import GaussianNB
import preprocess
from sklearn.tree import DecisionTreeClassifier
from graph_features import HITS, PageRank



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

def evaluateGaussianSVM(all_attrs,labels, ids,f_selection):
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
		    (pre_attrs,pre_labels,pre_ids) = (attrs[train_index,:],labels[train_index],ids[train_index,:])
		    (pre_attrs_test,pre_ids_test) = (attrs[test_index,:],ids[test_index,:])
		    #(pre_attrs, pre_attrs_test ) = HITS(pre_ids, pre_attrs, pre_labels, pre_ids_test, pre_attrs_test)
		   # (pre_attrs, pre_attrs_test ) = PageRank(pre_ids, pre_attrs, pre_labels, pre_ids_test, pre_attrs_test)

		   # (pre_attrs, pre_labels) = preprocess.augment(pre_attrs,pre_labels, attr_idxs=range(15),  mult=2)
		    #m = LogisticRegression(class_weight='balanced', n_jobs=-1)
		    m = svm.SVC(C=10,class_weight='balanced',kernel='rbf')
		    m.fit(pre_attrs,pre_labels)
		    preds = m.predict(pre_attrs_test)
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

	#info = np.transpose([n_features,errors,precisions,recalls,f1s])
	#np.savetxt('gaussianSVM_forwardFSelection.txt',info,fmt='%.5f',header = 'features,  error,  precision,  recall,  fischer')

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

def evaluateAdaBoost(all_attrs,labels, ids,f_selection):
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
		    (pre_attrs,pre_labels, pre_ids) = (attrs[train_index,:],labels[train_index],ids[train_index,:])
		    (pre_attrs_test,pre_ids_test) = (attrs[test_index,:],ids[test_index,:])
		    #(pre_attrs, pre_attrs_test ) = HITS(pre_ids, pre_attrs, pre_labels, pre_ids_test, pre_attrs_test)
		    #(pre_attrs, pre_attrs_test ) = PageRank(pre_ids, pre_attrs, pre_labels, pre_ids_test, pre_attrs_test)

		    #(pre_attrs, pre_labels) = preprocess.augment(pre_attrs,pre_labels, attr_idxs=range(15),  mult=5)
		    #b_m = svm.SVC(probability = True, C=0.1,class_weight='balanced',kernel='rbf')
		    b_m = DecisionTreeClassifier(max_depth = 1, min_samples_leaf = 1, class_weight = 'balanced')
		    #b_m = LogisticRegression(class_weight='balanced', n_jobs=-1)
		    m = AdaBoostClassifier(base_estimator = b_m,n_estimators=200,random_state=1)
		    m.fit(pre_attrs,pre_labels)
		    preds = m.predict(pre_attrs_test)
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
	np.savetxt('gaussianSVM_confusion_matrix.txt',confusion_matrix,fmt='%.1f')

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

def evaluateGaussianSVM_final(all_attrs,labels, ids, f_selection_start=2, exclude_idxs=[]):
	errors = []
	recalls = []
	precisions = []
	f1s = []
	aucs = []
	n_features = []
	include_features = range(f_selection_start -1)
	for idx in exclude_idxs:
		if idx in include_features:
			include_features.remove(idx)
	
	best_fischer = 0
	m = svm.SVC(C=10,class_weight='balanced',kernel='rbf')
		    
	for j in range(f_selection_start-1,all_attrs.shape[1]):
		if(j not in exclude_idxs):
			feature_idxs = include_features+[j]
		else:
			feature_idxs = include_features
		K=10
		#attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(j))
		attrs = all_attrs[0:len(labels),feature_idxs]
		#attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(16-j,16))
		correct=0
		wrong=0
		average_recall=0
		average_precision=0
		average_fischer=0
		average_error=0
		average_auc = 0

		'''mean_tpr = 0.0
		mean_fpr = np.linspace(0, 1, 100)
		roc_auc = 0
		lw = 2'''

		cv=KFold(labels.shape[0],K,shuffle = True)
		for train_index, test_index in cv:
		    (pre_attrs,pre_labels,pre_ids) = (attrs[train_index,:],labels[train_index],ids[train_index,:])
		    (pre_attrs_test,pre_ids_test) = (attrs[test_index,:],ids[test_index,:])
		    #(pre_attrs, pre_attrs_test ) = HITS(pre_ids, pre_attrs, pre_labels, pre_ids_test, pre_attrs_test)
		    (pre_attrs, pre_attrs_test ) = PageRank(pre_ids, pre_attrs, pre_labels, pre_ids_test, pre_attrs_test)


		    m.fit(pre_attrs,pre_labels)
		    preds = m.predict(pre_attrs_test)
		    wrong += np.sum(labels[test_index] != preds)
		    correct += np.sum(labels[test_index] == preds)

		    average_recall += 1.0/K * metrics.recall_score(labels[test_index],preds,pos_label = 1)
		    average_precision += 1.0/K * metrics.precision_score(labels[test_index],preds,pos_label = 1)
		    average_fischer += 1.0/K * metrics.f1_score(labels[test_index],preds,pos_label = 1)
		    average_auc += 1.0/K * metrics.roc_auc_score(labels[test_index],preds)
		    average_error += 1.0/K * float(wrong)/(correct+wrong)

		    '''if j == 16:
		    	fpr, tpr,thresh = roc_curve(labels[test_index],preds,pos_label =1)
		    	mean_tpr += interp(mean_fpr, fpr, tpr)
    			roc_auc += auc(fpr, tpr)'''

        
		print(j)
		print(average_fischer)
		print(best_fischer)
		if(average_fischer > best_fischer):
			print("Boo!")
			precisions.append(average_precision)
			f1s.append(average_fischer)
			aucs.append(average_auc)
			recalls.append(average_recall)
			errors.append(average_error)
			n_features.append(j)
			best_fischer = average_fischer
			include_features.append(j)


	info = np.transpose([n_features,errors,precisions,recalls,f1s,aucs])
	np.savetxt('gaussianSVM_forwardFSelection.txt',info,fmt='%.5f',header = 'features,  error,  precision,  recall,  fischer, auc')
	
	X_train, X_test, y_train, y_test = train_test_split(all_attrs[0:len(labels),include_features], labels, test_size=0.20, random_state=1)
	m.fit(X_train,y_train)
	preds = m.predict(X_test)

	info = np.transpose([y_test,preds])
	np.savetxt('gaussianSVM_predictionsVSTrueLabels.txt',info,fmt='%.5f',header = 'True Labels, Predictions')


	final_recall =  metrics.recall_score(y_test,preds,pos_label = 1)
	final_precision =  metrics.precision_score(y_test,preds,pos_label = 1)
	final_fischer =  metrics.f1_score(y_test,preds,pos_label = 1)
	final_auc =  metrics.roc_auc_score(y_test,preds)
	fpr, tpr,thresholds = roc_curve(y_test,preds,pos_label= 1)
	print('recall ', final_recall)
	print('precision ', final_precision)
	print('fischer ', final_fischer)
	print('auc ', final_auc)

	confusion_matrix = metrics.confusion_matrix(y_test,preds)
	np.set_printoptions(precision=2)
	class_names = ['Non-Consort','Consort']

	#Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=True,
	                      title='Gaussian SVM Normalized confusion matrix')
	plt.show()


	# Plot ROC Curve
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % final_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Gaussian SVM ROC')
	plt.legend(loc="lower right")
	plt.show()

	
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

def evaluateAdaBoosting_final(all_attrs,labels, ids, f_selection_start=2, exclude_idxs=[]):
	errors = []
	recalls = []
	precisions = []
	f1s = []
	aucs = []
	n_features = []
	include_features = range(f_selection_start -1)
	for idx in exclude_idxs:
		if idx in include_features:
			include_features.remove(idx)
	
	best_fischer = 0
	b_m = DecisionTreeClassifier(max_depth = 1, min_samples_leaf = 1, class_weight = 'balanced')
	#b_m = LogisticRegression(class_weight='balanced', n_jobs=-1)
	m = AdaBoostClassifier(base_estimator = b_m,n_estimators=500,random_state=1)

	for j in range(f_selection_start-1,all_attrs.shape[1]):
		if(j not in exclude_idxs):
			feature_idxs = include_features+[j]
		else:
			feature_idxs = include_features
		K=20
		#attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(j))
		attrs = all_attrs[0:len(labels),feature_idxs]
		#attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(16-j,16))
		correct=0
		wrong=0
		average_recall=0
		average_precision=0
		average_fischer=0
		average_error=0
		average_auc = 0

		'''mean_tpr = 0.0
		mean_fpr = np.linspace(0, 1, 100)
		roc_auc = 0
		lw = 2'''

		cv=KFold(labels.shape[0],K,shuffle = True)
		for train_index, test_index in cv:
		    (pre_attrs,pre_labels,pre_ids) = (attrs[train_index,:],labels[train_index],ids[train_index,:])
		    (pre_attrs_test,pre_ids_test) = (attrs[test_index,:],ids[test_index,:])
		    (pre_attrs, pre_attrs_test ) = HITS(pre_ids, pre_attrs, pre_labels, pre_ids_test, pre_attrs_test)
		    (pre_attrs, pre_attrs_test ) = PageRank(pre_ids, pre_attrs, pre_labels, pre_ids_test, pre_attrs_test)

		    m.fit(pre_attrs,pre_labels)
		    
		    preds = m.predict(pre_attrs_test)
		    wrong += np.sum(labels[test_index] != preds)
		    correct += np.sum(labels[test_index] == preds)

		    average_recall += 1.0/K * metrics.recall_score(labels[test_index],preds,pos_label = 1)
		    average_precision += 1.0/K * metrics.precision_score(labels[test_index],preds,pos_label = 1)
		    average_fischer += 1.0/K * metrics.f1_score(labels[test_index],preds,pos_label = 1)
		    average_auc += 1.0/K * metrics.roc_auc_score(labels[test_index],preds)
		    average_error += 1.0/K * float(wrong)/(correct+wrong)

		    '''if j == 16:
		    	fpr, tpr,thresh = roc_curve(labels[test_index],preds,pos_label =1)
		    	mean_tpr += interp(mean_fpr, fpr, tpr)
    			roc_auc += auc(fpr, tpr)'''

        
		print(j)
		print(average_fischer)
		print(best_fischer)
		if(average_fischer > best_fischer):
			print("Boo!")
			precisions.append(average_precision)
			f1s.append(average_fischer)
			aucs.append(average_auc)
			recalls.append(average_recall)
			errors.append(average_error)
			n_features.append(j)
			best_fischer = average_fischer
			include_features.append(j)


	info = np.transpose([n_features,errors,precisions,recalls,f1s,aucs])
	np.savetxt('adaBoost_forwardFSelection.txt',info,fmt='%.5f',header = 'features,  error,  precision,  recall,  fischer, auc')

	X_train, X_test, y_train, y_test = train_test_split(all_attrs[0:len(labels),include_features], labels, test_size=0.20, random_state=1)
	m.fit(X_train,y_train)
	preds = m.predict(X_test)

	info = np.transpose([y_test,preds])
	np.savetxt('adaBoost_predictionsVSTrueLabels.txt',info,fmt='%.5f',header = 'True Labels, Predictions')


	final_recall =  metrics.recall_score(y_test,preds,pos_label = 1)
	final_precision =  metrics.precision_score(y_test,preds,pos_label = 1)
	final_fischer =  metrics.f1_score(y_test,preds,pos_label = 1)
	final_auc =  metrics.roc_auc_score(y_test,preds)
	fpr, tpr,thresholds = roc_curve(y_test,preds,pos_label= 1)
	print('recall ', final_recall)
	print('precision ', final_precision)
	print('fischer ', final_fischer)
	print('auc ', final_auc)

	confusion_matrix = metrics.confusion_matrix(y_test,preds)
	np.set_printoptions(precision=2)
	class_names = ['Non-Consort','Consort']

	#Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=True,
	                      title='AdaBoost Normalized confusion matrix')
	plt.show()

	# Plot ROC Curve
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % final_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('AdaBoost ROC')
	plt.legend(loc="lower right")
	plt.show()



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

def getTestTrainMetrics(attrs,labels,ids):

	all_attrs = np.concatenate((attrs,ids),axis = 1)
	X_train_pre, X_test_pre, y_train, y_test = train_test_split(all_attrs,labels, test_size=0.20, random_state=1)
	pre_ids = X_train_pre[:,-2:]
	X_train = X_train_pre[:,:-2]
	pre_ids_test = X_test_pre[:,-2:]
	X_test = X_test_pre[:,:-2]
	#(X_train, X_test) = HITS(pre_ids, X_train, y_train, pre_ids_test, X_test)
	#(X_train, X_test) = PageRank(pre_ids, X_train, y_train, pre_ids_test, X_test)

	#X_train = X_train[:,-2:]
	#X_test = X_test[0:,-2:]

	svm_m = svm.SVC(C=10,class_weight='balanced',kernel='rbf',probability=False,random_state = 1)
	svm_m.fit(X_train,y_train)
	SVM_test_preds = svm_m.predict(X_test)
	SVM_train_preds = svm_m.predict(X_train)
	print 'svm test error', 1-metrics.accuracy_score(y_test,SVM_test_preds)
	print 'svm test f1', metrics.f1_score(y_test,SVM_test_preds,pos_label = 1)
	print 'svm train error', 1-metrics.accuracy_score(y_train,SVM_train_preds)
	print 'svm train f1', metrics.f1_score(y_train,SVM_train_preds,pos_label = 1)

	b_m = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 1, class_weight = 'balanced',max_features=None,random_state = 1)
	adaBoost_m = AdaBoostClassifier(base_estimator = b_m,n_estimators=8,random_state=1,algorithm = 'SAMME.R')
	adaBoost_m.fit(X_train,y_train)
	adaBoost_test_preds = adaBoost_m.predict(X_test)
	adaBoost_train_preds = adaBoost_m.predict(X_train)
	print 'adaboost test error', 1-metrics.accuracy_score(y_test,adaBoost_test_preds)
	print 'adaboost test f1', metrics.f1_score(y_test,adaBoost_test_preds,pos_label = 1)
	print 'adaboost train error', 1-metrics.accuracy_score(y_train,adaBoost_train_preds)
	print 'adaboost train f1', metrics.f1_score(y_train,adaBoost_train_preds,pos_label = 1)

	randomForest_m = RandomForestClassifier(n_estimators = 13, max_features='auto', max_depth=4, random_state=1,class_weight='balanced')
	randomForest_m.fit(X_train,y_train)
	randomForest_test_preds = randomForest_m.predict(X_test)
	randomForest_train_preds = randomForest_m.predict(X_train)
	print 'random forest test error', 1-metrics.accuracy_score(y_test,randomForest_test_preds)
	print 'random forest test f1', metrics.f1_score(y_test,randomForest_test_preds,pos_label = 1)
	print 'random forest train error', 1-metrics.accuracy_score(y_train,randomForest_train_preds)
	print 'random forest train f1', metrics.f1_score(y_train,randomForest_train_preds,pos_label = 1)

	'''gradientBoosting_m = GradientBoostingClassifier(subsample = 0.5,warm_start = False,learning_rate = 0.1,n_estimators = 10,random_state=1,max_depth = 10)
	gradientBoosting_m.fit(X_train,y_train)
	gradientBoosting_test_preds = gradientBoosting_m.predict(X_test)
	gradientBoosting_train_preds = gradientBoosting_m.predict(X_train)
	print 'gradient boosting test error', 1-metrics.accuracy_score(y_test,gradientBoosting_test_preds)
	print 'gradient boosting test f1', metrics.f1_score(y_test,gradientBoosting_test_preds,pos_label = 1)
	print 'gradient boosting train error', 1-metrics.accuracy_score(y_train,gradientBoosting_train_preds)
	print 'gradient boosting train f1', metrics.f1_score(y_train,gradientBoosting_train_preds,pos_label = 1)

	gaussianNB_m = GaussianNB()
	gaussianNB_m.fit(X_test,y_test)
	gaussianNB_test_preds = gaussianNB_m.predict(X_test)
	gaussianNB_train_preds = gaussianNB_m.predict(X_train)
	print 'NB test error', 1-metrics.accuracy_score(y_test,gaussianNB_test_preds)
	print 'NB test f1', metrics.f1_score(y_test,gaussianNB_test_preds,pos_label = 1)
	print 'NB train error', 1-metrics.accuracy_score(y_train,gaussianNB_train_preds)
	print 'NB train f1', metrics.f1_score(y_train,gaussianNB_train_preds,pos_label = 1)

	votingClassifier_m = VotingClassifier(estimators=[('svm',svm_m),('randForest',randomForest_m),('adaBoost',adaBoost_m)],weights = [1,1,1],n_jobs = 1, voting='hard')
	#votingClassifier_m = VotingClassifier(estimators=[('svm',svm_m)],n_jobs = 1, voting='hard')
	votingClassifier_m.fit(X_train,y_train)
	votingClassifier_test_preds = votingClassifier_m.predict(X_test)
	votingClassifier_train_preds = votingClassifier_m.predict(X_train)
	print 'voting classifier test error', 1-metrics.accuracy_score(y_test,votingClassifier_test_preds)
	print 'voting classifier test f1', metrics.f1_score(y_test,votingClassifier_test_preds,pos_label = 1)
	print 'voting classifier train error', 1-metrics.accuracy_score(y_train,votingClassifier_train_preds)
	print 'voting classifier train f1', metrics.f1_score(y_train,votingClassifier_train_preds,pos_label = 1)'''


def feature_select(m, all_attrs,labels, ids, model_name='unknown_model',  exclude_idxs=[]):
	errors = []
	recalls = []
	precisions = []
	f1s = []
	aucs = []
	n_features = []
	include_features = []
        remaining_features = range(all_attrs.shape[1])
        converged = False
	
        i= 0
        best_fischer = 0
        while len(remaining_features)>0 and not converged:
            print('Iteration %d of feature selection ' % i)
            average_recall=0
            average_precision=0
            average_fischer=0
            average_error=0
            average_auc = 0
            next_feature=-1

            for j in remaining_features:
                print('Testing feature %d' % j)
                feature_idxs = include_features+[j]
                K=10
                #attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(j))
                attrs = all_attrs[0:len(labels),feature_idxs]
                #attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(16-j,16))
                correct=0
                wrong=0
                temp_average_recall=0
                temp_average_precision=0
                temp_average_fischer=0
                temp_average_error=0
                temp_average_auc = 0

                '''mean_tpr = 0.0
                mean_fpr = np.linspace(0, 1, 100)
                roc_auc = 0
                lw = 2'''

                cv=KFold(labels.shape[0],K,shuffle = True)
                for train_index, test_index in cv:
                    (pre_attrs,pre_labels,pre_ids) = (attrs[train_index,:],labels[train_index],ids[train_index,:])
                    (pre_attrs_test,pre_ids_test) = (attrs[test_index,:],ids[test_index,:])
                    #(pre_attrs, pre_attrs_test ) = HITS(pre_ids, pre_attrs, pre_labels, pre_ids_test, pre_attrs_test)
                    #(pre_attrs, pre_attrs_test ) = PageRank(pre_ids, pre_attrs, pre_labels, pre_ids_test, pre_attrs_test)


                    m.fit(pre_attrs,pre_labels)
                    preds = m.predict(pre_attrs_test)
                    wrong += np.sum(labels[test_index] != preds)
                    correct += np.sum(labels[test_index] == preds)

                    temp_average_recall += 1.0/K * metrics.recall_score(labels[test_index],preds,pos_label = 1)
                    temp_average_precision += 1.0/K * metrics.precision_score(labels[test_index],preds,pos_label = 1)
                    temp_average_fischer += 1.0/K * metrics.f1_score(labels[test_index],preds,pos_label = 1)
                    temp_average_auc += 1.0/K * metrics.roc_auc_score(labels[test_index],preds)
                    temp_average_error += 1.0/K * float(wrong)/(correct+wrong)

                    '''if j == 16:
                        fpr, tpr,thresh = roc_curve(labels[test_index],preds,pos_label =1)
                        mean_tpr += interp(mean_fpr, fpr, tpr)
                        roc_auc += auc(fpr, tpr)'''

                if(temp_average_fischer > average_fischer):
                    average_recall=temp_average_recall
                    average_precision=temp_average_precision
                    average_fischer=temp_average_fischer
                    average_error=temp_average_error
                    average_auc = temp_average_auc
                    next_feature = j

            if(average_fischer > best_fischer):
                    print('%d is next feature added' % next_feature)
                    print('Average F1 score is now %f' % average_fischer)
                    precisions.append(average_precision)
                    f1s.append(average_fischer)
                    aucs.append(average_auc)
                    recalls.append(average_recall)
                    errors.append(average_error)
                    n_features.append(next_feature)
                    best_fischer = average_fischer
                    include_features.append(next_feature)
                    remaining_features.remove(next_feature)
            else:
                print('Converged after %d' % i)
                converged = True

	info = np.transpose([n_features,errors,f1s,aucs])
	np.savetxt(model_name+'_forwardFSelection.txt',info,fmt='%.5f',header = 'features,  error, fischer, auc')
	
	X_train, X_test, y_train, y_test = train_test_split(all_attrs[0:len(labels),include_features], labels, test_size=0.20, random_state=1)
	m.fit(X_train,y_train)
	preds = m.predict(X_test)

	#info = np.transpose([y_test,preds])
	#np.savetxt(model_name+'_predictionsVSTrueLabels.txt',info,fmt='%.5f',header = 'True Labels, Predictions')


	'''final_recall =  metrics.recall_score(y_test,preds,pos_label = 1)
	final_precision =  metrics.precision_score(y_test,preds,pos_label = 1)
	final_fischer =  metrics.f1_score(y_test,preds,pos_label = 1)
	final_auc =  metrics.roc_auc_score(y_test,preds)
	fpr, tpr,thresholds = roc_curve(y_test,preds,pos_label= 1)
	print('recall ', final_recall)
	print('precision ', final_precision)
	print('fischer ', final_fischer)
	print('auc ', final_auc)'''

	confusion_matrix = metrics.confusion_matrix(y_test,preds)
	np.set_printoptions(precision=2)
	class_names = ['Non-Consort','Consort']

	#Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=True,
	                      title= model_name + 'Normalized confusion matrix')
	return include_features
	#plt.show()


def getTestTrainMetricsSingle(m, attrs,labels,ids, model_name):

	all_attrs = np.concatenate((attrs,ids),axis = 1)
	X_train_pre, X_test_pre, y_train, y_test = train_test_split(all_attrs,labels, test_size=0.20, random_state=1)
	pre_ids = X_train_pre[:,-2:]
	X_train = X_train_pre[:,:-2]
	pre_ids_test = X_test_pre[:,-2:]
	X_test = X_test_pre[:,:-2]
	#(X_train, X_test) = HITS(pre_ids, X_train, y_train, pre_ids_test, X_test)
	#(X_train, X_test) = PageRank(pre_ids, X_train, y_train, pre_ids_test, X_test)

	#X_train = X_train[:,-2:]
	#X_test = X_test[0:,-2:]
	m.fit(X_train,y_train)
	test_preds = m.predict(X_test)
	train_preds = m.predict(X_train)
	print model_name+' test error', 1-metrics.accuracy_score(y_test,test_preds)
	print model_name+' test f1', metrics.f1_score(y_test,test_preds,pos_label = 1)
	print model_name+ ' train error', 1-metrics.accuracy_score(y_train,train_preds)
	print model_name+ ' train f1', metrics.f1_score(y_train,train_preds,pos_label = 1)



'Importing Data'
attrs=np.loadtxt('data_file.csv',delimiter=',',usecols=range(16))
labels=np.genfromtxt('data_file.csv',delimiter=',',usecols=[16],dtype='int')
ids = np.loadtxt('rawdata.csv',delimiter=',',usecols=range(2),skiprows = 1,dtype='str')
(preprocessed_attrs, preprocessed_labels, preprocessed_ids)=preprocess.preprocess_data(attrs,labels, ids=ids, whitening = False)


'Model Evaluation and Feature Selection Based on Performance Metrics'
#evaluateGaussianSVM(attrs,labels, ids,True)
#evaluateAdaBoost(attrs,labels, ids, True)
#evaluateRandomForest(attrs,labels,False)
#evaluateGaussianSVM_final(attrs,labels, ids, f_selection_start=2, exclude_idxs=[])
#evaluateGaussianSVM_final(attrs,labels, ids, f_selection_start=16, exclude_idxs=[10,11,12,13,14,15])
#evaluateGaussianSVM_final(attrs,labels, ids, f_selection_start=16, exclude_idxs=[10,11,12,13,14,15])

#evaluateAdaBoosting_final(attrs,labels, ids, f_selection_start=2, exclude_idxs=[])
#evaluateGaussianSVM_final(attrs,labels, ids, f_selection_start=2, exclude_idxs=[])

'Model Metrics for All Features'
#SVM_ConfusionMatrix(attrs,labels)
#AdaBoost_ConfusionMatrix(attrs,labels)
#RandomForest_ConfusionMatrix(attrs,labels)


b_m = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 1, class_weight = 'balanced',max_features=None,random_state = 1)
m=AdaBoostClassifier(base_estimator = b_m,n_estimators=8,random_state=1,algorithm = 'SAMME.R')
ada_features= feature_select( m, attrs,labels, ids, 'adaBoost');
#print 'ada_features'
#print ada_features
getTestTrainMetricsSingle(m, attrs[:,ada_features],labels,ids, 'adaBoost')



'''
m = RandomForestClassifier(n_estimators = 13, max_features='auto', max_depth=4, random_state=1,class_weight='balanced')
rf_features = feature_select( m, attrs,labels, ids, 'randomForest')
#print 'rf_features'
#print rf_features
getTestTrainMetricsSingle(m, attrs[:,rf_features],labels,ids, 'randomForest')
#getTestTrainMetricsSingle(m, attrs,labels,ids, 'randomForest_feature')

m = svm.SVC(C=10,class_weight='balanced',kernel='rbf');
svm_features = feature_select(m, preprocessed_attrs, preprocessed_labels, preprocessed_ids, 'gaussian_SVM');
#svm_features = [0,1,2,3,4,5,6,7,8]
#print 'svm_features'
#print svm_features
#print preprocessed_attrs[:,svm_features]
#print preprocessed_attrs
getTestTrainMetricsSingle(m, preprocessed_attrs[:,svm_features], preprocessed_labels, preprocessed_ids, 'gaussian_SVM')
#getTestTrainMetricsSingle(m, preprocessed_attrs, preprocessed_labels, preprocessed_ids, 'gaussian_SVM')
'''

#getTestTrainMetrics(attrs,labels,ids)
