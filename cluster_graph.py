###############################################################################
# cluster tree data structure implementation
# author: Jinjing Zhou, Xiaocheng Li
# date: Jan 1st, 2016
###############################################################################
import sys
from factors import *
import numpy as np

class ClusterGraph:
    def __init__(self, numVar=0):
        '''
        var - list: index/names of variables
        domain - list: the i-th element represents the domain of the i-th variable; 
                     for this programming assignments, all the domains are [0,1]
        varToCliques - list of lists: the i-th element is a list with the indices 
                     of cliques/factors that contain the i-th variable
        nbr - a dictionary: if factor[i] and factor[j] shares variable(s), then nbr[(i,j)]=variables(s)
        factor: a list of Factors
        sepset: a dictionary, sepset[(i,j)] is a list of variables shared by 
                factor[i] and factor[j]
        messages: a dictionary to store the messages, keys are (src, dst) pairs, values are 
                the Factors of sepset[src][dst]. Here src and dst are the indices for factors.
        '''
        self.var = [None for _ in range(numVar)]
        self.domain = [None for _ in range(numVar)]
        self.varToCliques = [[] for _ in range(numVar)]
        self.nbr = {}
        self.factor = []
        self.sepset = {}
        self.messages = {}
    
    def evaluateWeight(self, assignment):
        '''
        param - assignment: the full assignment of all the variables
        return: the multiplication of all the factors' values for this assigments, i.e. the joint probability of the entire assignment
        '''
        a = np.array(assignment, copy=False)
        output = 1.0
        for f in self.factor:
            output *= f.val[tuple(a[f.scope])]
        return output
    
    def getInMessage(self, src, dst):
        '''
        param - src: the source factor/clique index
        param - dst: the destination factor/clique index
        return: Factor with var set as sepset[src][dst]
        
        In this function, the message will be initialized as an all-one vector if 
        it is not computed and used before. 
        In order to prevent memory leakage, remove the inMsg.name unless doing debugging
        '''
        if (src, dst) not in self.messages:
            inMsg = Factor()
            inMsg.scope = self.sepset[(src,dst)]
            inMsg.card = [len(self.domain[s]) for s in inMsg.scope]
            inMsg.val = np.ones(np.prod(inMsg.card))
            #inMsg.name='message_{%s to %s}'% (src,dst)
            inMsg.name=None
            self.messages[(src, dst)] = inMsg
        return self.messages[(src, dst)]

    def runParallelLoopyBP(self, iterations): 
        '''
        param - iterations: the number of iterations you do loopy BP
        
        The factors are set in this part. For other models, the factors can be customerized to fit various objectives.
        For general purpose, there will be unary factors and pairwise factors, factors with larger scope are not recommended, becase
        it has the risk of making the model untractable. However, feel free to modify if necessary.

        The message name is commented out to prevent memory leakage, but for debuggin, feel free to enable it. Be sure to test in small models
        '''
        M=len(self.var)
        N=len(self.factor)
        record=[float('inf')]*5
        for iter in range(iterations):
        ###############################################################################
            convergeIndex=0
            total=0
            newMessages={}
            #unaries i->j
            for i,unary in enumerate(self.factor[:M]):
                for j in self.nbr[i]:
                    message=unary
                    for inm in self.nbr[i]:
                        if inm!=j:
                            message=message.multiply(self.getInMessage(inm,i))
                    #message.name='message_(%s to %s)_iter%s'%(i,j,iter)
                    message.name=None
                    newMessages[(i,j)]=message.normalize()
                    convergeIndex+=sum(abs(newMessages[(i,j)].val-self.getInMessage(i,j).val))
                    total+=sum(self.getInMessage(i,j).val)
            #parity checks
            for j,parity in enumerate(self.factor[M:]):
                j+=M
                for i in self.nbr[j]:
                    message=parity
                    for inm in self.nbr[j]:
                        if inm!=i:
                            message=message.multiply(self.getInMessage(inm,j))
                    message=message.marginalize_all_but(self.sepset[(j,i)])
                    #message.name='message_(%s to %s)_iter%s'%(j,i,iter)
                    message.name=None
                    newMessages[(j,i)]=message.normalize()
                    convergeIndex+=sum(abs(newMessages[(i,j)].val-self.getInMessage(i,j).val))
                    total+=sum(self.getInMessage(i,j).val)
            if convergeIndex!=0:
                record[:-1]=record[1:]
                record[-1]=1.0*convergeIndex/total
                sys.stdout.write('difference: '+str(record[-1])+'-->')
                sys.stdout.flush() 
            if sum(record)<0.03:
                print 'it is converged'
                break
            '''
            res= self.getMarginalMAP()
            sys.stdout.write(str(sum(res))+'-->')
            sys.stdout.flush() 
            if sum(res)==0:
                print 'To save time, we stop here!'
                break   
            '''
            self.messages=newMessages

            
        ###############################################################################


    def estimateMarginalProbability(self, var):
        '''
        param - var: a single variable index
        return: the marginal probability of the var
        
        e.g.
        >>> cluster_graph.estimateMarginalProbability(0)
        >>> [0.2, 0.8]
    
        '''
        ###############################################################################  
        marginal=Factor(self.factor[var])
        InM=[_ for _ in self.nbr[var]]
        for inm in InM:
            marginal=marginal.multiply(self.messages[(inm,var)])
        marginal=marginal.normalize()
        return marginal.normalize()
        ###############################################################################
    

    def getMarginalMAP(self):
        '''
        return the marginal MAP assignments for all variables.

        '''
        ###############################################################################
        output = np.zeros(len(self.var))
        for i,v in enumerate(self.var):
            marginal=list(self.estimateMarginalProbability(v).val)
            output[i]=marginal.index(max(marginal))
        return output
        ###############################################################################  
