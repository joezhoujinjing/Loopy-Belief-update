###############################################################################
# Example of picture recovery using low density parity check basedf on clique tree & factors learn and inference model (loopy belief update)
# author: Jinjing Zhou, Xiaocheng Li
# date: Jan 1st, 2016
###############################################################################

## Utility code for PA3
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from cluster_graph import *
from factors import *
import random

def loadLDPC(name):
    """
    :param - name: the name of the file containing LDPC matrices
  
    return values:
    G: generator matrix
    H: parity check matrix
    """
    A = sio.loadmat(name)
    G = A['G']
    H = A['H']
    return G, H

def loadImage(fname, iname):
    '''
    :param - fname: the file name containing the image
    :param - iname: the name of the image
    (We will provide the code using this function, so you don't need to worry too much about it)  
  
    return: image data in matrix form
    '''
    img = sio.loadmat(fname)
    return img[iname]


def applyChannelNoise(y, p):
    '''
    :param y - codeword with 2N entries
    :param p channel noise probability
  
    return corrupt message yhat  
    yhat_i is obtained by flipping y_i with probability p 
    '''
    ###############################################################################
    yhat=y
    for _ in yhat:
        if random.random()>0.95:
            _[0]=1-_[0]
    ###############################################################################
    return yhat


def encodeMessage(x, G):
    '''
    :param - x orginal message
    :param[in] G generator matrix
    :return codeword y=Gx mod 2
    '''
    return np.mod(np.dot(G, x), 2)


def constructClusterGraph(yhat, H, p):
    '''
    :param - yhat: observed codeword
    :param - H parity check matrix
    :param - p channel noise probability

    return G clusterGraph
   
    You should consider two kinds of factors:
    - M unary factors 
    - N each parity check factors
    '''
    N = H.shape[0]
    M = H.shape[1]
    G = ClusterGraph(M)
    domain = [0, 1]
    G.nbr = [[] for _ in range(M+N)]
    G.nbr={}
    G.sepset={}
    G.var=range(M)
    G.domain=[domain]*M
    noise=[p,1-p,1-p,p]
    #unary factors
    for i in xrange(M):
        unary=Factor(f=None, scope=[G.var[i],'y%s'%(i)], card=[2,2], val=None, name="Unary_%s"%(i))
        unary.val= np.zeros(unary.card)
        for j,a in enumerate(indices_to_assignment(range(np.prod(unary.card)), unary.card)):
            unary.val[tuple(a)]=noise[j]
        ob=yhat[i][0]
        unary=unary.observe('y%s'%(i),ob).marginalize_all_but([G.var[i]])
        unary.name="Unary_%s"%(i)
        G.factor.append(unary)
    #parity check factors
    for i in xrange(N):
        parity=Factor()
        parity.scope=[G.var[_] for _ in range(len(H[i])) if H[i][_]==1]
        parity.name='Parity_%s_(%s)'%(M+i,parity.scope)
        parity.card=[2]*len(parity.scope)
        parity.val= np.zeros(parity.card)
        for j,a in enumerate(indices_to_assignment(range(np.prod(parity.card)), parity.card)):
            parity.val[tuple(a)]=1 if sum(a)%2==0 else 0
        G.factor.append(parity)
    
    #G.varToCliques and G.nbr
    for i in range(M):
        G.varToCliques[i].append(i)
    for j in range(M,M+N):
        for _ in G.factor[j].scope:
            G.varToCliques[_].append(j)
            if j not in G.nbr:
                G.nbr[j]=[_]
            else:
                G.nbr[j].append(_)
            if _ not in G.nbr:
                G.nbr[_]=[j]
            else:
                G.nbr[_].append(j)

    for i,vToC in enumerate(G.varToCliques):
        for _ in vToC:
            for __ in vToC:
                if _!=__:
                    if (_,__) not in G.sepset:
                        G.sepset[(_,__)]=[i]
                    else:
                        G.sepset[(_,__)].append(i)
    return G
    ##############################################################


def main(error,iterations):
    '''
    param - error: the transmission error probability
    '''
    ##############################################################
    G, H = loadLDPC('ldpc36-1600.mat')
    img = loadImage('images.mat', 'cs242')
    
    x=np.reshape((img.T).flatten(),(1600,1))
    y = encodeMessage(x, G)
    plt.subplot(131),plt.imshow(np.reshape(encodeMessage(x,G),np.array([80,40]))),plt.title('ORIGINAL')
    y=applyChannelNoise(y,1-error)
    plt.subplot(132),plt.imshow(np.reshape(y,np.array([80,40]))),plt.title('CORRUPTED')
    print 'Graph construction begins...'
    Graph=constructClusterGraph(y,H,1-error)
    print 'Graph constructed, loopy belief update begins...'
    Graph.runParallelLoopyBP(iterations)
    recover=Graph.getMarginalMAP()
    plt.subplot(133),plt.imshow(np.reshape(recover,np.array([80,40]))),plt.title('RECOVERRED')
    plt.show()
    ################################################################

main(0.06,30)
