import networkx as nx
from random import random
import matplotlib.pyplot as plt
import itertools
import math
import numpy as np

def main():
    G = nx.Graph()
    numNodes = int(raw_input("Enter the number of nodes to be included in the graph: "))
    edgeList = np.array([x for x in xrange(numNodes)])
    print "Print 1 for YES and 0 for NO"
    for x in xrange(numNodes-1):
        for y in xrange(1,numNodes):
            if x==y:
                continue
            print "Node between {} and {} ?: ".format(x,y)
            user = int(raw_input())
            if user == 1:
                G.add_edge(x,y)
            else:
                continue
    nx.draw(G, with_labels = True)
    plt.savefig("InitialGraph.png")

if __name__ == "__main__":
    main()
