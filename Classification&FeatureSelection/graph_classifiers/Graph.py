import numpy as np
import random
random.seed(10)
class Graph:
    def __init__(self,filename=None,edge_class=None,attrs=None,id_to_gender=None):
        if filename!=None:
            self.init_from_file(filename)
        elif edge_class!=None and attrs!=None and id_to_gender!=None:
            self.init_from_dicts(edge_class,attrs,id_to_gender)
        else:
            print "ERROR! CONSTRUCTOR INCORRECTLY CALLED"
            print "FILENAME: ",filename
            print "EDGE_CLASS: ",edge_class
            print "ATTRS:",attrs
            print "ID_TO_GENDER:",id_to_gender
            exit(1)
        failed=0
        total=0
        for node in self.edge_class:
            for event in self.edge_class[node]:        
                total+=1
                if event==0:
                    failed+=1
        self.prior=float(failed)/total

    def init_from_dicts(self,edge_class,attrs,id_to_gender):
        self.edge_class=edge_class
        self.attrs=attrs
        self.id_to_gender=id_to_gender
        self.edge_list={}
        added_edge_class={}
        for node1,node2 in self.edge_class:
            added_edge_class[(node2,node1)]=self.edge_class[(node1,node2)]#Because the input is not symmetric
            self.attrs[(node2,node1)]=self.attrs[(node1,node2)]#Because the input is not symmetric
            self.attr_length=len(self.attrs[(node1,node2)][-1])
            if node1 not in self.edge_list:
                self.edge_list[node1]=[]
            if node2 not in self.edge_list:
                self.edge_list[node2]=[]
            if node1 not in self.edge_list[node2]:
                self.edge_list[node2].append(node1)
                self.edge_list[node2].append(node2)
        self.edge_class.update(added_edge_class)

    def init_from_file(self,filename):
        f=open(filename,'r')
        self.edge_list={}#Maps
        self.id_to_gender={}#Maps ID to 'Male' or 'Female'
        self.edge_class={}#Maps tuple of (id1, id2) to list of results, 0=failed, 1=consorted, 2=conceived
        self.attrs={}#Maps tuple of (id1,id2) to list of numpy array of attrs, each one correspons to the same class in self.edge_class
        first=True
        for line in f:
            if first:
                first=False
                continue
            line=line.strip().split(',')
            if line[0] not in self.edge_list:
                self.edge_list[line[0]]=[]
                self.id_to_gender[line[0]]='Female'
            if line[1] not in self.edge_list:
                self.edge_list[line[1]]=[]
                self.id_to_gender[line[1]]='Male'
            if (line[0],line[1]) not in self.edge_class:
                self.edge_class[(line[0],line[1])]=[]
                self.edge_class[(line[1],line[0])]=[]
                self.attrs[(line[0],line[1])]=[]
                self.attrs[(line[1],line[0])]=[]
            if line[1] not in self.edge_list[line[0]]:
                self.edge_list[line[0]].append(line[1])
                self.edge_list[line[1]].append(line[0])
            self.edge_class[(line[0],line[1])].append(int(line[3]))#consort=1, nothing=0
            self.edge_class[(line[1],line[0])].append(int(line[3]))
            self.attrs[(line[1],line[0])].append(map(float,np.array(line[4:])))
            self.attrs[(line[0],line[1])].append(map(float,np.array(line[4:])))
            self.attr_length=len(self.attrs[(line[1],line[0])][-1])
    """Returns a list of K graph objects, and a list of K test attrs, and labels, however, every node is left with still at least one known edge between every node, e.g. if there are at 5 edges between node A and node B, then the training graph will still have at least 1 edge between node A and node B"""
    def create_k_subgraphs(self,k=4):
        #First keeping one edge between each pair of nodes
        test_classes=[{} for i in range(k)]
        test_attrs=[{} for i in range(k)]
        train_classes=[{} for i in range(k)]
        train_attrs=[{} for i in range(k)]
        for edge1,edge2 in self.edge_class:            
            if edge2>edge1:#Don't double count edges
                continue
            indices=range(len(self.edge_class[(edge1,edge2)]))
            random.shuffle(indices)
            for i in range(k):
                test_classes[i][(edge1,edge2)]=[]
                test_attrs[i][(edge1,edge2)]=[]
                train_classes[i][(edge1,edge2)]=[]
                train_attrs[i][(edge1,edge2)]=[]
                for j in range(len(indices)):
                    if (j-i)%k==0:
                        test_classes[i][(edge1,edge2)].append(self.edge_class[(edge1,edge2)][indices[j]])
                        test_attrs[i][(edge1,edge2)].append(self.attrs[(edge1,edge2)][indices[j]])
                    else:
                        train_classes[i][(edge1,edge2)].append(self.edge_class[(edge1,edge2)][indices[j]])
                        train_attrs[i][(edge1,edge2)].append(self.attrs[(edge1,edge2)][indices[j]])
                if train_attrs[i][(edge1,edge2)]==[]:
                    train_attrs[i][(edge1,edge2)]=test_attrs[i][(edge1,edge2)]
                    train_classes[i][(edge1,edge2)]=test_classes[i][(edge1,edge2)]
                    del test_attrs[i][(edge1,edge2)]
                    del test_classes[i][(edge1,edge2)]
                    
        graphs=[]
        for i in range(k):
            graphs.append(Graph(edge_class=train_classes[i],attrs=train_attrs[i],id_to_gender=self.id_to_gender))
        return (graphs, test_classes,test_attrs)

    def get_attrs_and_labels_general(self):
        #Returns a 2 tuple of all (attrs,labels) both are numpy arrays, does not take advantage of the graph structure at all
        attrs=np.zeros((sum([len(self.edge_class[node]) for node in self.edge_class])/2,self.attr_length))
        labels=np.zeros((sum([len(self.edge_class[node]) for node in self.edge_class])/2))
        i=0
        for node1,node2 in self.edge_class:
            if node1<=node2:#So no duplicates
                continue
            assert len(self.edge_class[(node1,node2)])==len(self.attrs[(node1,node2)]),'MISMATCH ON ATTRS AND LABELS SIZE'
            for j in range(len(self.edge_class[(node1,node2)])):
                
                attrs[i,:]=self.attrs[(node1,node2)][j]
                labels[i]=self.edge_class[(node1,node2)][j]
                i+=1
        return (attrs,labels)

    def get_attrs_and_labels_specific(self):
        #Returns a 2 tuple of all (attrs,labels) both are numpy arrays, does not take advantage of the graph structure at all
        attrs={}
        labels={}
        i=0
        for node1,node2 in self.edge_class:
            if node1<=node2:#So no duplicates
                continue
            if (node1,node2) not in attrs:
                attrs[(node1,node2)]=np.zeros((0,self.attr_length))
                labels[(node1,node2)]=np.zeros((0))
            assert len(self.edge_class[(node1,node2)])==len(self.attrs[(node1,node2)]),'MISMATCH ON ATTRS AND LABELS SIZE'
            for j in range(len(self.edge_class[(node1,node2)])):
                attrs[(node1,node2)]=np.append(attrs[(node1,node2)],self.attrs[(node1,node2)][j])
                labels[(node1,node2)]=np.append(labels[(node1,node2)],self.edge_class[(node1,node2)][j])
            attrs[(node1,node2)]=np.reshape(attrs[(node1,node2)],(len(self.edge_class[(node1,node2)]),self.attr_length))
        return (attrs,labels)

    
    def get_neighs(self, node):
        attrs=[]
        labels=[]
        for neigh in self.id_to_gender:
            if (node,neigh) in self.attrs:
                for attr,label in zip(np.array(self.attrs[(node,neigh)]),self.edge_class[(node,neigh)]):
                    attrs.append(attr)
                    labels.append(label)
                    
        return zip(attrs,labels)
        
    def print_stats(self):
        print "Num Nodes:",len(self.edge_list)
        print "Num Edges:",sum([len(self.edge_class[node]) for node in self.edge_class])/2
        print "Num Unique Edges:",len(self.edge_class)
        print "Num Females:",sum([1 for node in self.id_to_gender if self.id_to_gender[node]=='Female'])
        print "Num Males:",sum([1 for node in self.id_to_gender if self.id_to_gender[node]=='Male'])
        failed=0
        conceive=0
        consort=0
        for node in self.edge_class:
            for event in self.edge_class[node]:
                if event==0:
                    failed+=1
                if event==1:
                    consort+=1
                if event==2:
                    conceive+=1
        print "Num Failed:",failed/2
        print "Num Consorts:",consort/2
        print "Num Conceptions:",conceive/2
    

if __name__=='__main__':
    g=Graph(filename='../../data/rawdata.csv')
    g.print_stats()
    print ""
#    g.id_to_gender={"abc":"Male","def":"Female"}
#    g.edge_list={"abc":["def"],"def":['abc']}
#    g.edge_class={('abc','def'):[0,2,1,2,1,2],('def','abc'):[0,2,1,2,1,2]}
#    g.attrs={('abc','def'):[('atts0','attsother0'),('atts1','attsother1'),('atts2','attsother2'),('atts3','attsother3'),('atts4','attsother4'),('atts5','attsother5')],('def','abc'):[('atts0','attsother0'),('atts1','attsother1'),('atts2','attsother2'),('atts3','attsother3'),('atts4','attsother4'),('atts5','attsother5')]}
    (train_graphs,test_classes,test_attrs)=g.create_k_subgraphs()
    g.get_attrs_and_labels_specific()
                        
