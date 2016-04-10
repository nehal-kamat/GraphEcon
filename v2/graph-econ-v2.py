import networkx as nx
from random import random
import matplotlib.pyplot as plt
import itertools
import math
import numpy as np

class LinearUtility:
    def __init__(self, weights):
        self.weights = np.array(map(float, weights))

    def bestPurchase(self, price_vec, budget):
        i = np.argmax(self.weights/np.array(map(float,price_vec)))
        x = np.zeros(price_vec.shape)
        x[i] = budget/price_vec[i]

        return x

class NonLinearUtility:
    """"Return the max utility, used for both linear and non linear"""
    def __init__(self, weights):
        self.weights = weights

    def bestPurchase(self, price_vec, budget):
        quantity=[]
        for n in range(0, len(price_vec)):
            quantity.append(budget/price_vec[n])
        utilities= [eval(self.weights, {'math': math, '__builtins__': None}, {'x': value})
                for value, self.weights in zip(quantity, self.weights)]
        i = np.argmax(utilities)
        x = np.zeros(price_vec.shape)
        x[i]=utilities[i]

        return x

class market:
    def __init__(self, G, utility, endowment):
        self.G = G.copy()
        self.utility = utility
        self.endowment = endowment

    def checkClear(self, prices):
        self.prices = prices
        for n in self.G:
            self.G.node[n]['prices'] = prices[n]
        budget = {}
        for x in range(self.G.number_of_nodes()):
            budget[x] = sum(self.prices[x]*self.endowment[x])


def main():
    G = nx.Graph()
    numNodes = 3 #int(raw_input("Enter the number of nodes to be included in the graph: "))
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
    plt.show()
    #plt.savefig("InitialGraph.png")

    endowment = {0:np.array([1,2]), 1:np.array([1,1]), 2:np.array([2,1])}
    utility = {0:LinearUtility(np.array([1,0])), 1:LinearUtility(np.array([1,1])), 2:LinearUtility(np.array([0,1]))}

    mkt = market(G, utility, endowment)

    prices = {0:np.array([2,1]), 1:np.array([2,2]), 2:np.array([1,2])}
    mkt.checkClear(prices)

if __name__ == "__main__":
    main()
