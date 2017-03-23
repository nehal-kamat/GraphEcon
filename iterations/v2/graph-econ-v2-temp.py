import networkx as nx
from random import random
import matplotlib.pyplot as plt
from itertools import product
import math
import numpy as np

class LinearUtility:
    def __init__(self, weights):
        self.weights = weights

    def bestPurchase(self, price_vec, budget):
        price_vec = np.array(price_vec)
        best_index = []
        candidate_set = np.array(map(float, self.weights))/np.array(map(float,price_vec))
        tmp = np.argmax(candidate_set)
        temp = candidate_set[tmp]
        best_index = np.where(candidate_set==temp)[0].tolist()

        if len(best_index)==1:
            x = np.zeros(price_vec.shape)
            x[tmp] = budget/price_vec[tmp]
            return x.tolist()
        else:
            rng = budget/price_vec[tmp]
            num = [p for p in range(0, rng+1)]
            combinations = product(num, repeat=len(best_index))
            x = [c for c in combinations if reduce(lambda x, y: x + y, c) == rng]
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

        """searching for neighbours having the best prices for goods"""
        best_prices = {}
        best_neighbours = {}
        neighbor_track={}
        keys={}
        for n in self.G:
            best_prices[n] = {}
            best_neighbours[n] = {}
            neighbor_track[n] = []
            keys[n] = []
            for good in range(0, len(self.endowment[n])):
                #best_prices[n][good] = []
                best_neighbours[n][good] = []
                min_price = self.prices[n][good]
                #best_prices[n][good][min_price] = []
                best_prices[n][good] = min_price
                #best_prices[n][good][min_price].append(n)
                best_neighbours[n][good].append(n)
                for neighbor in self.G.neighbors(n):
                    #if price of good X of neighbor y == min->add to list (tie)
                    if self.prices[neighbor][good] == min_price:
                        #best_prices[n][good][min_price].append(neighbor)
                        best_neighbours[n][good].append(neighbor)
                    #if price of good X of neighbor y < create new list
                    if self.prices[neighbor][good] < min_price:
                        min_price = self.prices[neighbor][good]
                        #best_prices[n][good]={}
                        best_neighbours[n][good] = []
                        #best_prices[n][good][min_price] = []
                        #best_prices[n][good][min_price].append(neighbor)
                        best_prices[n][good] = min_price
                        best_neighbours[n][good].append(neighbor)

            for good in best_prices[n]:
                keys[n].append(best_prices[n][good])

                #self.G.node[n]['best neighbours'] = best_prices[n][good][min_price]

        print "budget: {}\nbest prices: {}\nbest neighbours: {}".format(budget, best_prices, best_neighbours)

        best_purchase = {}
        for n in self.G:
            best_purchase[n] = {}
            best_purchase[n] = self.utility[n].bestPurchase(keys[n], budget[n])

        print "best purchase candidates: ", best_purchase

"""
        for n in best_purchase:
            if len(best_purchase[n]) > len(self.endowment[n]):
                feasibility = {}
                for t in best_purchase[n]:
                    feasibility[t] = []
                    for item in t:
                        if item > all(best_prices[n][item][key for key in best_prices[n][item]]):
                            feasibility[t].append(0)
                        else:
                            feasibility.append(1)
                print feasibility
"""

def main():
    G = nx.Graph()
    numNodes = 3 #int(raw_input("Enter the number of nodes to be included in the graph: "))
    G.add_edges_from([(0,1),(1,2)])
    """
    # ask user for presence or absence of edges between pairs of nodes
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
    """
    nx.draw(G, with_labels = True)
    plt.show()
    #plt.savefig("InitialGraph.png")

    endowment = {0:np.array([1,2]), 1:np.array([1,1]), 2:np.array([2,1])}
    utility = {0:LinearUtility([1,0]), 1:LinearUtility([1,1]), 2:LinearUtility([0,1])}

    mkt = market(G, utility, endowment)

    prices = {0:np.array([2,1]), 1:np.array([2,2]), 2:np.array([1,2])}
    mkt.checkClear(prices)

if __name__ == "__main__":
    main()
