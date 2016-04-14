import networkx as nx
from random import random
import matplotlib.pyplot as plt
import itertools
import math
import numpy as np

class LinearUtility:
    def __init__(self, weights):
        self.weights = weights

    def bestPurchase(self, price_vec, budget):
        price_vec = np.array(price_vec)
        #i.append(np.argmax(np.array(map(float, self.weights))/np.array(map(float,price_vec))))

        best_index = []
        candidate_set = np.array(map(float, self.weights))/np.array(map(float,price_vec))
        tmp = np.argmax(candidate_set)
        temp = candidate_set[tmp]
        best_index = np.where(candidate_set==temp)
        best_index = best_index[0].tolist()
        #best_index = ','.join(map(int, best_index[0][0]))
        #best_index = np.array(best_index.tolist())
        print best_index
        if len(best_index)==1:
            x = np.zeros(price_vec.shape)
            x[tmp] = budget/price_vec[tmp]
            return x.tolist()
        else:
            rng = budget/price_vec[tmp]
            num = [p for p in range(0, rng+1)]
            combinations = itertools.product(num, repeat=len(best_index))
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
        neighbor_track={}
        keys={}
        for n in self.G: #iterate though the nodes of graph
            best_prices[n] = {}
            neighbor_track[n] = []
            keys[n] = []
            for good in range(0, len(self.endowment[n])): #iterate though the goods of a node
                best_prices[n][good]={}
                min_price = self.prices[n][good]
                best_prices[n][good][min_price] = []
                best_prices[n][good][min_price].append(n)
                for neighbor in self.G.neighbors(n): #iterate though the nodes of graph
                    if self.prices[neighbor][good] == min_price: #if price of good X of neighbor y == min->add to list (tie)
                        best_prices[n][good][min_price].append(neighbor)
                    if self.prices[neighbor][good] < min_price: #if price of good X of neighbor y < create new list
                        min_price = self.prices[neighbor][good]
                        best_prices[n][good]={}
                        best_prices[n][good][min_price] = []
                        best_prices[n][good][min_price].append(neighbor)


                for key in best_prices[n][good]:
                    keys[n].append(key)

                self.G.node[n]['best neighbours'] = best_prices[n][good][min_price]

                    #neighbor_track[n].append(best_prices[n][good][key])

            #print neighbor_track[n]

        print "budget: {}\nbest prices: {}".format(budget, best_prices)

        best_purchase = {}
        for n in self.G:
            best_purchase[n] = {}
            best_purchase[n] = self.utility[n].bestPurchase(keys[n], budget[n])

        print best_purchase

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
        for n in self.G:
            combn = list(itertools.product(*neighbor_track[n]))
            best_purchase = []
            for item in combn:
                pv = []
                for i, ii in enumerate(item):
                    pv.append(self.G.node[ii]['prices'][i])
                best_purchase.append(self.utility[n].bestPurchase(pv, budget[n]))
                #best_purchase.sort()
            print best_purchase
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
