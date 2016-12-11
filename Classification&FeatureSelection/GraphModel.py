from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import numpy as np
from numpy.linalg import norm
import math
class GraphModel:
    def __init__(self,graph,model_type,options=[]):
        self.model_type=model_type
        if model_type=='general':#Learns one model for all gorillas, simplest way to go about it
            self.model=GeneralModel(graph,options)
        if model_type=='specific':#Each gorilla learns its own model
            self.model=SpecificModel(graph,options)
        if model_type=='locallyweighted':#Each gorilla learns its own model
            self.model=LocallyWeightedModel(graph,options)
        if model_type=='globaledge':#Each gorilla learns its own model
            self.model=GlobalEdgeModel(graph,options)
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
        print "F1 Score: ",f1_score(true_labels,pred_labels)
        

class GeneralModel:
    """Learns one model for all gorillas, doesn't attempt to leverage the graph structure at all"""
    def __init__(self,graph,options):
        self.model=LogisticRegression()
        (attrs,labels)=graph.get_attrs_and_labels_general()
        self.model.fit(attrs,labels)

    def predict(self,attrs):#attrs is a dict of (id1,id2) to attrs
        preds={}
        for key in attrs:
            preds[key]=self.model.predict(np.array(attrs[key]))
        return preds

class SpecificModel:
    #Learns one model for all gorillas, doesn't attempt to leverage the graph structure at all
    def __init__(self,graph,options):
        (attrs,labels)=graph.get_attrs_and_labels_specific()
        self.options=options
        self.models={}
        for node1,node2 in attrs:
            if len(labels[(node1,node2)])==1:
                continue
            print labels[(node1,node2)]
            self.models[(node1,node2)]=LogisticRegression()
            self.models[(node1,node2)].fit(attrs[(node1,node2)],labels[(node1,node2)])
            

    def predict(self,attrs):#attrs is a dict of (id1,id2) to attrs
        preds={}
        for node1,node2 in attrs:
            preds=self.model[(node1,node2)].predict(np.array(attrs[node1]))
            preds[(node1,node2)]=pred
        return preds

class LocallyWeightedModel:
    #Doesn't actually learn a model, instead it essentially does locally weighted regression
    def __init__(self,graph,options):
        self.t=options[0]
        self.graph=graph       
        self.prior=graph.prior

    def predict(self,attrs):#attrs is a dict of (id1,id2) to attrs
        preds={}
        num_failed=0
        num_succeeded=0
        for node1,node2 in attrs:
            examples1=self.graph.get_neighs(node1)
            examples2=self.graph.get_neighs(node2)
            examples=examples1+examples2
            preds[(node1,node2)]=[]            

            for attr in attrs[(node1,node2)]:
                total_weight=0.0
                class0_score=0.0
                class1_score=0.0
                num_class0=0
                num_class1=0
                for example in examples:
                    d=self.dist(example[0],attrs[(node1,node2)])
                    weight=math.exp(-d**2/float(self.t**2))
                    if example[1]==0:
                        class0_score+=weight
                        num_class0+=1
                    else:
                        num_class1+=1
                        class1_score+=weight
                if num_class0!=0 and (num_class1==0 or class0_score/num_class0>class1_score/num_class1):
                    preds[(node1,node2)].append(0)
                else:
                    preds[(node1,node2)].append(1)
            preds[(node1,node2)]=np.array(preds[(node1,node2)])
        print "FAILED, SUCCEEDED:",num_failed,num_succeeded
        return preds

    def dist(self,attr1,attr2):
        return norm(attr1-attr2)

class GlobalEdgeModel:
    #Doesn't actually learn a model, instead it essentially does locally weighted regression
    def __init__(self,graph,options):
        self.graph=graph

    def predict(self,attrs):#attrs is a dict of (id1,id2) to attrs
        preds={}
        j=0
        for node1,node2 in attrs:
            preds[(node1,node2)]=[]            
            print j
            for attr in attrs[(node1,node2)]:
                preds[(node1,node2)].append(self.graph.dist_to_example(attr))
            preds[(node1,node2)]=np.array(preds[(node1,node2)])
            j+=1
        return preds

    def dist(self,attr1,attr2):
        return norm(attr1-attr2)


if __name__=='__main__':
    from Graph import Graph
    g=Graph('../../data/rawdata.csv')
    (graphs,test_classes,test_attrs)=g.create_k_subgraphs()    
#    g.print_stats()
#    graphs[0].print_stats()
    GM=GraphModel(graphs[0],'globaledge',[.5])
    GM.predict_with_stats(test_attrs[0],test_classes[0])

#    GM=GraphModel(graphs[0],'general',[.5])
#    GM.predict_with_stats(test_attrs[0],test_classes[0])
