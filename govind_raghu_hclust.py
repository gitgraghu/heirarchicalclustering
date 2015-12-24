import sys
import numpy as np
import itertools
import heapq
from collections import defaultdict

class Cluster:
    def __init__(self, features, pointid=None, numelements=1):
        self.X = np.array(map(float, features))
        self.clusters = []
        self.parent = None
        self.pointid = pointid
        self.numelements = numelements

def distance(clusterpair):
    return np.linalg.norm(clusterpair[0].X - clusterpair[1].X)

def getclusterelements(root, elements):
    if(root.pointid!=None):
        elements.append(root.pointid)
        return

    getclusterelements(root.clusters[0], elements)
    getclusterelements(root.clusters[1], elements)

def mergecluster(clusterpair):
    cluster1 = clusterpair[0]
    cluster2 = clusterpair[1]
    numelements = cluster1.numelements + cluster2.numelements
    X = (cluster1.X*cluster1.numelements + cluster2.X*cluster2.numelements)/numelements
    mergedcluster = Cluster(X, numelements=numelements)
    mergedcluster.clusters.append(cluster1)
    mergedcluster.clusters.append(cluster2)
    cluster1.parent = mergedcluster
    cluster2.parent = mergedcluster
    return mergedcluster

def isnotmerged(clusterpair):
    cluster1 = clusterpair[0]
    cluster2 = clusterpair[1]
    if(cluster1.parent == None and cluster2.parent == None):
        return True
    return False

if __name__ == '__main__':

    # Read command line arguments
    datafilename = sys.argv[1]
    K = int(sys.argv[2])
    datafile = open(datafilename, 'r')

    # Read lines of data file and initialize clusters
    clusters = []
    goldset = defaultdict(list)
    linenum = 0
    for line in datafile:
        data = line.rstrip('\n').split(',')
        clusters.append(Cluster(data[:-1], linenum))
        classification = data[-1]
        goldset[classification].append(linenum)
        linenum += 1

    # Create heap from pair distances
    distanceq = []
    for pair in itertools.combinations(clusters, 2):
        distanceq.append((distance(pair),pair))
    heapq.heapify(distanceq)

    # Merge clusters that are min distance for every iteration
    while len(clusters) > 1:
        mindistancepair  = heapq.heappop(distanceq)[1]
        if(isnotmerged(mindistancepair)):
            clusters.remove(mindistancepair[0])
            clusters.remove(mindistancepair[1])
            newcluster = mergecluster(mindistancepair)
            for j in range(len(clusters)):
                heapq.heappush(distanceq, (distance((newcluster,clusters[j])), (newcluster, clusters[j])))
            clusters.append(newcluster)

    # Get K clusters from tree
    root = clusters[0]
    k = K - 1
    cqueue = []
    cqueue.append(root)
    while(k > 0):
        node = cqueue.pop(0)
        for cluster in node.clusters:
            cqueue.append(cluster)
        k = k - 1

    # Get cluster elements from each cluster as a list
    clustersfound = []
    pairs = set()
    for cluster in cqueue:
        elements = []
        getclusterelements(cluster, elements)
        elements.sort()
        clustersfound.append(elements)
        pairs.update(itertools.combinations(elements,2))

    # Get gold pairs
    goldpairs =set()
    for clustername in goldset.keys():
        cluster = goldset[clustername]
        goldpairs.update(itertools.combinations(cluster, 2))

    # Calculate Precision and Recall
    correct = pairs.intersection(goldpairs)
    P = float(len(correct))/len(pairs)
    R = float(len(correct))/len(goldpairs)

    # Print results
    print P
    print R
    for cluster in clustersfound:
        print cluster
