import numpy as np
from numpy.linalg import norm
from numpy.linalg import eig
import random
from Graph import *
#Returns a dictionary of the HITS score for each node.
#HITS is generally supposed to work in a directed graph, so to modify it to work in an undirected setting I
#am not using hubs, because hubs and authorities are essentially the same in the undirected setting. 
#So I am just updating authorities, using authorities. Also, I am only utilizing edges where consorting appears.
#So failed attempts do not influence the HITS score for a node.
def HITS(ids,attrs,labels,test_ids,test_attrs):
    edge_list,edge_class=construct_edges(ids,labels)
    authorities={}
    for node in edge_list:
        authorities[node]=1.0
    for i in range(100):
        total_auth=0.0
        new_authorities={}
        for node in edge_list:
            new_authorities[node]=sum([authorities[neigh] for neigh in edge_list[node] if 1 in edge_class[(node,neigh)]])
            total_auth+=authorities[node]
        for node in edge_list:
            new_authorities[node]/=(total_auth**.5)
        authorities=new_authorities
    train_attrs=add_attrs(attrs,ids,authorities)
    test_attrs=add_attrs(test_attrs,test_ids,authorities)
    return (train_attrs,test_attrs)

#Regular pagerank, beta is 1 minus the teleport probability, so set it to 1 to never teleport
#ids is a list of tuples where the first element is the first baboons id, second is the second baboons id
#1 indicating consort, 0 indicating non-consort
#Note that like HITS we only utilize the successful edges, but unlike in HITS we utilize the number of successes. E.g.
#A successfully connecting with B twice means that it gets twice the amount of rank from B then if it did it once.
#Also, note that like HITS we are treating this as undirected.
#ids, attrs, labels 
def PageRank(ids,attrs,labels,test_ids,test_attrs,beta=.9):
    edge_list,edge_class=construct_edges(ids,labels)
    #Constructing weighted adjacency matrix
    M=np.zeros((len(edge_list),len(edge_list)))
    indices={}
    counter=0
    for node in edge_list:
        if node not in indices:
            indices[node]=counter
            counter+=1
        num_edges=0
        for neigh in edge_list[node]:
            num_edges+=edge_class[(node,neigh)].count(1)
        for neigh in edge_list[node]:
            if neigh not in indices:
                indices[neigh]=counter
                counter+=1
            if num_edges>0:
                M[indices[node],indices[neigh]]=edge_class[(node,neigh)].count(1)/float(num_edges)
    M=beta*M+(1-beta)/float(M.size)
    values,vectors=eig(M.T)
    largest_index=0
    for i in range(len(values)):
        if values[i]>values[largest_index]:
            largest_index=i
    principle_vector=vectors[:,largest_index]
    principle_vector/=sum(principle_vector)
    ranks={}
    for node in indices:
        ranks[node]=principle_vector[indices[node]]
    train_attrs=add_attrs(attrs,ids,ranks)
    test_attrs=add_attrs(test_attrs,test_ids,ranks)
    return (train_attrs,test_attrs)

def add_attrs(attrs, ids,ranks):
    new_attrs=np.zeros((attrs.shape[0],2))
    for i in range(attrs.shape[0]):
        new_attrs[i,:]=np.array([ranks[ids[i][0]],ranks[ids[i][1]]])
    print new_attrs.shape
    print attrs.shape
    return np.concatenate((attrs,new_attrs),axis=1)

def construct_edges(ids, labels):
    features=[]
    for i in range(len(ids)):
        features.append((ids[i][0],ids[i][1],labels[i]))
    edge_list={}
    edge_class={}
    counter=0
    for entry in features:
        if entry[0] not in edge_list:
            edge_list[entry[0]]=[]
        if entry[1] not in edge_list:
            edge_list[entry[1]]=[]
        if entry[1] not in edge_list[entry[0]]:
            edge_list[entry[0]].append(entry[1])
        if entry[0] not in edge_list[entry[1]]:
            edge_list[entry[1]].append(entry[0])
        if (entry[0],entry[1]) not in edge_class:
            edge_class[(entry[0],entry[1])]=[]
            edge_class[(entry[1],entry[0])]=[]
        edge_class[(entry[0],entry[1])].append(entry[2])
        edge_class[(entry[1],entry[0])].append(entry[2])
    size=0
    return (edge_list,edge_class)

if __name__=='__main__':
    random.seed(10)
    g=open('../../data/rawdata.csv')
    ids=[]
    labels=np.array([])
    attrs=np.zeros((12141,16))
    first=True
    i=0
    for line in g:
        if first:
            first=False
            continue
        ids.append(line.strip().split(',')[:2])
        labels=np.append(labels,int(line.strip().split(',')[3]))
        attrs[i,:]=np.array(line.strip().split(',')[4:])
        i+=1
    print attrs.shape
    page_rank=PageRank(ids,attrs,labels,ids,attrs)
    print page_rank[0].shape
    print page_rank[0][0,:]

