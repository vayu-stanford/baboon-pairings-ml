from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import numpy as np
class GraphModel:
    def __init__(self,graph,model_type):
        self.model_type=model_type
        if model_type=='general':#Learns one model for all gorillas, simplest way to go about it
            self.model=GeneralModel(graph)

    def predict(self,attrs):
        return self.model.predict(attrs)
    
    def predict_with_stats(self,attrs,labels):
        preds=self.predict(attrs)
        pred_labels=[]
        true_labels=[]
        for key in labels:
            assert len(labels[key])==len(preds[key])
            for i in range(len(labels[key])):
                pred_labels.append(preds[key][i])
                true_labels.append(labels[key][i])
        pred_labels=np.array(pred_labels)
        true_labels=np.array(true_labels)
        print confusion_matrix(true_labels,pred_labels)
        print "Accuracy: ",accuracy_score(true_labels,pred_labels)
        
        

class GeneralModel:
    """Learns one model for all gorillas, doesn't attempt to leverage the graph structure at all"""
    def __init__(self,graph):
        self.model=LogisticRegression()
        (attrs,labels)=graph.get_attrs_and_labels_general()
        self.model.fit(attrs,labels)

    def predict(self,attrs):#attrs is a dict of (id1,id2) to attrs
        preds={}
        for key in attrs:
            preds[key]=self.model.predict(np.array(attrs[key]))
        return preds

"""class SpecificModel:
    #Learns one model for all gorillas, doesn't attempt to leverage the graph structure at all
    def __init__(self,graph):
        (attrs,labels)=graph.get_attrs_and_labels_general()
        models={}
        attrs_by_node={}
        labels_by_node={}
        for node1,node2 in attrs:
            if node1 not in attrs:
                attrs_by_node[node1]=[]
                labels_by_node[node1]=[]
            if node2 not in attrs:
                attrs_by_node[node2]=[]
                labels_by_node[node1]=[]
            attrs_by_node[node1].append(attrs[(node1,node2)])
            attrs_by_node[node2].append(attrs[(node1,node2)])
            labels_by_node[node1].append(labels[(node
        for node in attrs_by_node:
            models[node]=LogisticRegression()
            models[node].fit(np.array(
            model=LogisticRegression()
        self.model.fit(attrs,labels)

    def predict(self,attrs):#attrs is a dict of (id1,id2) to attrs
        preds={}
        for key in attrs:
            preds[key]=self.model.predict(np.array(attrs[key]))
        return preds


"""
if __name__=='__main__':
    from Graph import Graph
    g=Graph('../../data/rawdata.csv')
    (graphs,test_classes,test_attrs)=g.create_k_subgraphs()
    
    GM=GraphModel(graphs[0],'general')
    print GM.predict_with_stats(test_attrs[0],test_classes[0])
