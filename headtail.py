"""
headtail.py

Module for approximating ccdh in a graph stream.

Implementation of algorithm introduced in http://arxiv.org/abs/1506.02574

Olivia Simpson
UCSD 2015
"""

import random
import snap
import math
import sys
import scipy.io
import numpy as np
from numpy import linalg
import pickle
from time import time
# import pyhash


"""
I. Accessing the graphs
"""

"""
SNAP dataset methods

If using a graph from the SNAP collection, these will be faster.
"""

def load_SNAP_graph(graph_input_file, simple=False):
    G = snap.LoadEdgeList(snap.PUNGraph, graph_input_file, 0, 1)
    n = G.GetNodes()
    #convert to a simple graph if simple set to True
    if simple:
        snap.DelSelfEdges(G) #remove self-loops
        m = G.GetEdges()
    else:
        m = G.GetEdges()
    return (G, n, m)


def GetDegs(G):
    '''return degree vector as a dictionary'''
    #use SNAP function to get degrees of vertices
    OutDegV = snap.TIntPrV()
    snap.GetNodeOutDegV(G[0], OutDegV)
    #create lookup table
    ODV = {}
    for pair in OutDegV:
        ODV[pair.GetVal1()] = pair.GetVal2()
    return ODV


def GetDegFreq(G):
    '''
    compute true # vertices of degree d in graph using built-in SNAP function
    Output: vector of (degree, number of nodes of such degree) pairs
    '''
    TF = snap.TIntPrV()
    snap.GetDegCnt(G[0], TF)

    nd = {}
    for d in TF:
        nd[d.GetVal1()] = d.GetVal2()
    return nd


"""
General datasets
"""

def load_graph_from_file(file):
    '''
    Load an (undirected) edge list from a txt file.
    Store the graph as an array of edges, and compute the number of nodes
    and the number of edges.

    Return the tuple (edglist, n, m)
    '''
    G = [] #edge list

    with open(file, 'r') as source:
        for line in source:
            if line[0] != '#':
                edge = [int(n) for n in line.split() if n.isdigit()]
                G.append(edge)

    n = num_nodes(G)
    m = num_edges(G)

    return (G, n, m)

def permute_graph_from_file(file):
    '''
    Randomly permute edge list
    '''
    with open(file, 'r') as source:
        data = [ (random.random(), line) for line in source if line[0] != '#' ]
    data.sort()

    G = [] #edge list
    for _, line in data:
        edge = [int(n) for n in line.split() if n.isdigit()]
        # edge = tuple(edge) #for undirected?
        G.append(edge)

    n = num_nodes(G)
    m = num_edges(G)

    return (G, n, m)

def pickle_graph(G, pkl_file):
    data = open(pkl_file, 'wb')
    pickle.dump(G[0], data)
    data.close()

def load_pickled_graph(pkl_file):
    data = open(pkl_file, 'rb')
    G = pickle.load(data)
    data.close()
    return G

def num_nodes(G):
    nodes = [ e[0] for e in G ]
    nodes.extend([ e[1] for e in G ])
    nodes = set(nodes)
    return len(nodes)

def num_edges(G):
    return len(G)

"""
"""

def get_node_set(G):
    if type(G[0]) == snap.PUNGraph: ## SNAP graph
        nodes = [ e.GetSrcNId() for e in G[0].Edges() ]
        nodes.extend([ e.GetDstNId() for e in G[0].Edges() ])
        nodes = set(nodes)
        return nodes
    else: ## edge list
        nodes = [ e[0] for e in G[0].Edges ]
        nodes.extend([ e[1] for e in G[0] ])
        nodes = set(nodes)
        return nodes


def compute_degrees(G):
    '''
    Return all node degrees in a dictionary
    '''
    if type(G[0]) == snap.PUNGraph: ## SNAP graph
        return GetDegs(G)
    else: ## edge list
        ODV = {}
        for u,v in G[0]:
            if u in ODV:
                ODV[u] += 1
            else:
                ODV[u] = 1
            if v in ODV:
                ODV[v] += 1
            else:
                ODV[v] = 1
        return ODV


def compute_deg_frequencies(G):
    '''
    Return all degree frequencies in a dictionary
    '''
    if type(G[0]) == snap.PUNGraph: ## SNAP graph
        return GetDegFreq(G)
    else: ## edge list
        ODV = compute_degrees(G)
        nd = {}
        for v in ODV:
            if ODV[v] in nd:
                nd[ODV[v]] += 1
            else:
                nd[ODV[v]] = 1
        return nd


def compute_histogram(dic):
    '''
    Compute the histogram of elements in dic
    h[e] = #occurences of e in dic (dic[.] = e)
    '''
    h = {}
    for key in dic:
        if dic[key] not in h:
            h[dic[key]] = 1
        else:
            h[dic[key]] += 1
    return h


def dic_to_list(d):
    dic_l = []
    for k in sorted(d):
        dic_l.append([k, d[k]])

    return dic_l

def list_to_dic(l):
    list_d = {}
    for e in l:
        list_d[e[0]] = e[1]

    return list_d


def numpy_to_matlab(M, mat_out_file, mat_var_name):
    ## M is a numpy matrix
    scipy.io.savemat(mat_out_file, mdict={mat_var_name : M})


def write_to_file(data, outfile):
    ## the data is a dictionary
    f = open(outfile, 'w')
    for key in sorted(data):
        f.write(str(key)+'\t'+str(data[key])+'\n')
    f.close()


"""
"""

# PRF
def hasher(obj, which_hash):
    try:
        return hasher_store[which_hash][obj]
    except KeyError:
        value = np.random.random()
        hasher_store[which_hash][obj] = value
        return value

hasher_store = []

def initialize_hashers(num_hash):
    for i in range(num_hash):
        hasher_store.append({})

"""
"""

"""
II. Computing the ccdh 
"""

def headtail(G, eps=0.5, s=None, shead=None, stail=None):
    '''
    Compute the estimated ccdh for a graph loaded in memory.
    Input:
        G,      a loaded graph
        eps,    head/tail threshold parameter
        s,      total number of vertices sampled
        shead,  number of vertices sampled for the head estimator
        stail,  number of vertices sampled for the tail estimator
    Output:
        estimate of the ccdh 
    '''
    initialize_hashers(1)
    n = G[1]
    m = G[2]

    snapgraph = False
    if type(G[0]) == snap.PUNGraph:
        snapgraph = True
        EdgeList = G[0].Edges()
    else:
        EdgeList = G[0]

    randlist = np.random.random(2*m)
    randlist = list(randlist)

    ## straighten out the space allotments
    if (s is not None) and (shead is not None) and (stail is not None):
        assert s == stail + shead

    if (shead is None) and (stail is None):
        shead = s/2.0
        stail = s/2.0
    elif shead is None:
        shead = s-stail
    elif stail is None:
        stail = s-shead

    ph = float(shead)/n
    pt = float(stail)/(2*m)

    head_sample_node_counts = {}
    tail_sample_node_counts = {}

    #collect samples! (update)
    for e in EdgeList:
        if snapgraph:
            (u, v) = (e.GetSrcNId(), e.GetDstNId())
        else:
            (u, v) = (e[0], e[1])

            #update counts for
            #tail of distribution
            if u in tail_sample_node_counts:
                tail_sample_node_counts[u] += 1
            if v in tail_sample_node_counts:
                tail_sample_node_counts[v] += 1
            #and head of distribution
            if u in head_sample_node_counts:
                head_sample_node_counts[u] += 1
            if v in head_sample_node_counts:
                head_sample_node_counts[v] += 1

            #decide if we are following these nodes in the stream
            #for the head of the distribution
            alpha1 = hasher(u, 0)
            if alpha1 < ph:
                if u not in head_sample_node_counts:
                    head_sample_node_counts[u] = 1
            beta1 = hasher(v, 0)
            if beta1 < ph:
                if v not in head_sample_node_counts:
                    head_sample_node_counts[v] = 1

            ## decide if we sample these nodes for the tail of the
            ## distribution
            ## add u with probability pt
            try:
                alpha2 = randlist.pop()
            except IndexError:
                alpha2 = np.random.random()
            if alpha2 < pt:
                if u not in tail_sample_node_counts:
                    tail_sample_node_counts[u] = 1
            ## add v with probability pt
            try:
                beta2 = randlist.pop()
            except IndexError:
                beta2 = np.random.random()
            if beta2 < pt:
                if v not in tail_sample_node_counts:
                    tail_sample_node_counts[v] = 1

    #print 'size of head estimator sample set', len(head_sample_node_counts), '=', 1.0*len(head_sample_node_counts)/n, '* n'
    #print 'size of tail estimator sample set', len(tail_sample_node_counts), '=', 1.0*len(tail_sample_node_counts)/n, '* n'
    samplesize = len(head_sample_node_counts)+len(tail_sample_node_counts)
    #print 'total number of samples:', samplesize, '=', 1.0*samplesize/n, '* n'
    #print 'total number of samples:', samplesize, '=', 1.0*samplesize/m, '* m'
    #print

    #raw_head_sample = head_sample_node_counts.copy()
    #raw_tail_sample = tail_sample_node_counts.copy()

    ## (estimate)

    ## compute expected degree
    # print 'use binary search to invert the function and use d(c) everywhere'
    expdegs = {}
    for v in tail_sample_node_counts:
        c = tail_sample_node_counts[v]
        if c in expdegs:
            d = expdegs[c]
        else:
            d = int(round(expdeg(c, pt)))
            expdegs[c] = d
        tail_sample_node_counts[v] = d

    ## divide all counts above the tail
    ## by probability of being sampled
    # print '\ndivide by probability'
    tail_counts_hist = compute_histogram(tail_sample_node_counts)
    tail_nd_approx = {}
    for r in sorted(tail_counts_hist):
        tail_nd_approx[r] = round( ( 1.0*tail_counts_hist[r] )/( 1-((1-pt)**r) ) )

    ## compute the head
    head_counts_hist = compute_histogram(head_sample_node_counts)
    head_nd_approx = {}
    hor_thresh = (3*math.log(1./eps))/(ph*(eps**2))
    hor_thresh_degree = 0
    for d in sorted(head_counts_hist):
        nd = head_counts_hist[d]*(1.0/ph)
        head_nd_approx[d] = nd
        if nd >= hor_thresh:
            hor_thresh_degree = d

    ## compute the ccdh
    head_ccdh = ccdh(head_nd_approx)
    tail_ccdh = ccdh(tail_nd_approx)
    ccdh_approx = {}
    for d in range(hor_thresh_degree+1):
        ccdh_approx[d] = head_ccdh[d]
    for d in range(hor_thresh_degree+1, max(tail_ccdh)+1):
        ccdh_approx[d] = tail_ccdh[d]

    #return ccdh_approx, raw_head_sample, head_nd_approx, raw_tail_sample, tail_nd_approx, samplesize
    return ccdh_approx

def headtail_stream(graph_input_file, n, m, eps=0.5, s=None, shead=None, stail=None):
    '''
    Compute the estimated ccdh for a graph read line-by-line from file.
    Input:
        graph_input_file,      the graph file path
        n,      number of vertices in the graph
        m,      number of edges in the graph
        eps,    head/tail threshold parameter
        s,      total number of vertices sampled
        shead,  number of vertices sampled for the head estimator
        stail,  number of vertices sampled for the tail estimator
    Output:
        estimate of the ccdh 
    '''
    initialize_hashers(1)

    randlist = np.random.random(2*m)
    randlist = list(randlist)

    ## straighten out the space allotments
    if (s is not None) and (shead is not None) and (stail is not None):
        assert s == stail + shead

    if (shead is None) and (stail is None):
        shead = s/2.0
        stail = s/2.0
    elif shead is None:
        shead = s-stail
    elif stail is None:
        stail = s-shead

    ph = float(shead)/n
    pt = float(stail)/(2*m)

    head_sample_node_counts = {}
    tail_sample_node_counts = {}

    #collect samples! (update)
    with open(graph_input_file, 'r') as source:
        for line in source:
            if line[0] != '#':
                [u,v] = [int(node) for node in line.split() if node.isdigit()]

                #update counts for
                #tail of distribution
                if u in tail_sample_node_counts:
                    tail_sample_node_counts[u] += 1
                if v in tail_sample_node_counts:
                    tail_sample_node_counts[v] += 1
                #and head of distribution
                if u in head_sample_node_counts:
                    head_sample_node_counts[u] += 1
                if v in head_sample_node_counts:
                    head_sample_node_counts[v] += 1

                #decide if we are following these nodes in the stream
                #for the head of the distribution
                alpha1 = hasher(u, 0)
                if alpha1 < ph:
                    if u not in head_sample_node_counts:
                        head_sample_node_counts[u] = 1
                beta1 = hasher(v, 0)
                if beta1 < ph:
                    if v not in head_sample_node_counts:
                        head_sample_node_counts[v] = 1

                ## decide if we sample these nodes for the tail of the
                ## distribution
                ## add u with probability pt
                try:
                    alpha2 = randlist.pop()
                except IndexError:
                    alpha2 = np.random.random()
                if alpha2 < pt:
                    if u not in tail_sample_node_counts:
                        tail_sample_node_counts[u] = 1
                ## add v with probability pt
                try:
                    beta2 = randlist.pop()
                except IndexError:
                    beta2 = np.random.random()
                if beta2 < pt:
                    if v not in tail_sample_node_counts:
                        tail_sample_node_counts[v] = 1

    #print 'size of head estimator sample set', len(head_sample_node_counts), '=', 1.0*len(head_sample_node_counts)/n, '* n'
    #print 'size of tail estimator sample set', len(tail_sample_node_counts), '=', 1.0*len(tail_sample_node_counts)/n, '* n'
    samplesize = len(head_sample_node_counts)+len(tail_sample_node_counts)
    #print 'total number of samples:', samplesize, '=', 1.0*samplesize/n, '* n'
    #print 'total number of samples:', samplesize, '=', 1.0*samplesize/m, '* m'
    #print

    #raw_head_sample = head_sample_node_counts.copy()
    #raw_tail_sample = tail_sample_node_counts.copy()

    ## (estimate)

    ## compute expected degree
    # print 'use binary search to invert the function and use d(c) everywhere'
    expdegs = {}
    for v in tail_sample_node_counts:
        c = tail_sample_node_counts[v]
        if c in expdegs:
            d = expdegs[c]
        else:
            d = int(round(expdeg(c, pt)))
            expdegs[c] = d
        tail_sample_node_counts[v] = d

    ## divide all counts above the tail
    ## by probability of being sampled
    # print '\ndivide by probability'
    tail_counts_hist = compute_histogram(tail_sample_node_counts)
    tail_nd_approx = {}
    for r in sorted(tail_counts_hist):
        tail_nd_approx[r] = round( ( 1.0*tail_counts_hist[r] )/( 1-((1-pt)**r) ) )

    ## compute the head
    head_counts_hist = compute_histogram(head_sample_node_counts)
    head_nd_approx = {}
    hor_thresh = (3*math.log(1./eps))/(ph*(eps**2))
    hor_thresh_degree = 0
    for d in sorted(head_counts_hist):
        nd = head_counts_hist[d]*(1.0/ph)
        head_nd_approx[d] = nd
        if nd >= hor_thresh:
            hor_thresh_degree = d

    ## compute the ccdh
    head_ccdh = ccdh(head_nd_approx)
    tail_ccdh = ccdh(tail_nd_approx)
    ccdh_approx = {}
    for d in range(hor_thresh_degree+1):
        ccdh_approx[d] = head_ccdh[d]
    for d in range(hor_thresh_degree+1, max(tail_ccdh)+1):
        ccdh_approx[d] = tail_ccdh[d]

    #return ccdh_approx, raw_head_sample, head_nd_approx, raw_tail_sample, tail_nd_approx, samplesize
    return ccdh_approx

def expcount(d, p):
    '''
    The count computed as d(v)-E[loss(v) | v is sampled]

    c = dp + p - 1 + (1-p)^{d+1} / p(1 - (1-p)^d)
    '''
    num = d*p + p - 1 + ((1-p)**(d+1))
    den = p*(1-((1-p)**d))
    return num/den

def expdeg(c, p):
    '''
    Use binary search to invert the expcount(d, p) function.

    Return expected degree based on expected loss.
    '''
    return expdegBS(c, p, c, 2*c)

def expdegBS(c, p, low, high):
    guess = (low+high)/2.0
    deval = expcount(guess, p)
    if round(deval) == round(c):
        return guess
    elif round(deval) > round(c):
        return expdegBS(c, p, low, guess)
    elif round(deval) < round(c):
        return expdegBS(c, p, guess, high)
    else:
        print 'something is wrong...'
        return

"""
III. ccdh 
"""

def ccdhterm(nd, breakpoint):
    ## 1 - CUMMULATIVE DEGREE FUNCTION
    ## Compute the total number of vertices of degree >= d in nd
    # return sum([row[1] for row in nd if row[0] >= breakpoint])
    return sum([nd[d] for d in nd if d >= breakpoint])

def ccdh(nd):
    ## compute the full vector of CDF values
    ccdh = {}
    maxd = int(np.ceil(max(nd)))
    for d in range(maxd+1):
        ccdh[d] = ccdhterm(nd, d)
    return ccdh

"""
IV. Relative Hausdoff distance
"""

def rel_hausdorff(ccdh_A, ccdh_B, deg_err, pointwise_max=True):
    '''
    Relative Hausdorff distance between two ccdhs.
    Input:
        ccdh_A, ccdh_B, two ccdhs
        deg_err,        relative distance between degrees (epsilon in the
                        definition)
        pointwise_max,  set to True to output a pointwise maximum delta for each
                        degree.  If set to false, the output will be two lists
                        of the RH distance from ccdh_A to ccdh_B, and from 
                        ccdh_B to ccdh_A.
    Output:
        RH distance at all scales.  Either two lists or one depending on the
        pointwise_max flag.
    '''
    ccdhs = []
    ccdhs.append(ccdh_A)
    ccdhs.append(ccdh_B)

    freq_errs = []
    freq_errs.append([])
    freq_errs.append([])
    for i in [0,1]:          # Choosing one distribution to compare with other
        for deg in ccdhs[i]: # Looping over points in ccdhs[i]
            freq = ccdhs[i][deg]
            if deg == 0 or freq == 0:
                continue
            min_err = 100   ## error capped at 100
            if deg in ccdhs[1-i]:
                rel_err = abs(freq - ccdhs[1-i][deg])/freq
                if rel_err < min_err:
                    min_err = rel_err
            for d_ in np.arange(np.ceil((1-deg_err)*deg), np.floor((1+deg_err)*deg)):
                if d_ in ccdhs[1-i]:
                    ## get relative error
                    rel_err = abs(freq - ccdhs[1-i][d_])/freq
                    if rel_err < min_err:
                        min_err = rel_err
            freq_errs[i].append([deg, min_err])
    print 'summative RH distance:', max( deg_err, max([point[1] for point in freq_errs[0]]), max([point[1] for point in freq_errs[1]]) )

    if pointwise_max:    
        rh_max = {}
        first = list_to_dic(freq_errs[0])
        second = list_to_dic(freq_errs[1])
        maxdeg = max(max(first), max(second))
        for d in range(1, int(maxdeg)+1):
            if (d in first) and (d in second):
                rh_max[d] = max(first[d], second[d])
            elif d in first:
                rh_max[d] = first[d]
            elif d in second:
                rh_max[d] = second[d]
        return rh_max
    
    else:    
        return freq_errs
