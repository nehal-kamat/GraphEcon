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

class Market:
    def __init__(self, G, utility, endowment):
        self.G = G.copy()
        self.utility = utility
        self.endowment = endowment

    def bestPrices(self, prices):
        self.prices = prices
        for n in self.G:
            self.G.node[n]['prices'] = prices[n]
        budget = {}
        for x in range(self.G.number_of_nodes()):
            budget[x] = sum(self.prices[x]*self.endowment[x])
            self.G.node[x]['budget'] = budget[x]

        """searching for neighbours having the best prices for goods"""
        best_prices = {}
        best_neighbours = {}
        keys={}
        for n in self.G:
            best_prices[n] = {}
            best_neighbours[n] = {}
            keys[n] = []
            for good in range(0, len(self.endowment[n])):
                best_neighbours[n][good] = []
                best_neighbours[n][good].append(n)
                min_price = self.prices[n][good]
                best_prices[n][good] = min_price
                for neighbor in self.G.neighbors(n):
                    #if price of good X of neighbor y == min->add to list (tie)
                    if self.prices[neighbor][good] == min_price:
                        best_neighbours[n][good].append(neighbor)
                    #if price of good X of neighbor y < create new list
                    elif self.prices[neighbor][good] < min_price:
                        min_price = self.prices[neighbor][good]
                        best_neighbours[n][good] = []
                        best_neighbours[n][good].append(neighbor)
                        best_prices[n][good] = min_price

            for good in best_prices[n]:
                keys[n].append(best_prices[n][good])

            self.G.node[n]['best prices'] = [best_prices[n][good] for good in best_prices[n]]
            self.G.node[n]['best neighbours'] = [best_neighbours[n][good] for good in best_neighbours[n]]

        # format of best_prices, best_neighbours : {node: {good : ...}}
        print "budget: {}\nbest prices: {}\nbest neighbours: {}".format(budget, best_prices, best_neighbours)

        # return keys, budget

        best_purchase = {}
        for n in self.G:
            best_purchase[n] = {}
            best_purchase[n] = self.utility[n].bestPurchase(keys[n], budget[n])
        print "purchase candidates: ", best_purchase

        # return best_purchase

        final_purchase = {}
        for n in self.G:
            final_purchase[n] = []
            print type(best_purchase[n][0])
            if type(best_purchase[n][0])!=tuple:
                bn = zip(best_purchase[n], self.G.node[n]['best neighbours'])
                bn = [list(x) for x in bn]
                for item in bn:
                    temp=[]
                    for ii in item[1]:
                        if item[0] > self.endowment[ii][bn.index(item)]:
                            temp.append(self.endowment[ii][bn.index(item)])
                            diff = item[0] - self.endowment[ii][bn.index(item)]
                            # print diff
                            item[0]=diff
                            if ii==len(item[1])-1 and diff!=0:
                                remaining = diff
                        else:
                            temp.append(item[0])
                    final_purchase[n].append(temp)

            elif type(best_purchase[n][0])==tuple:
                for tup in best_purchase[n]:
                    bn = zip(tup, self.G.node[n]['best neighbours'])
                    bn = [list(x) for x in bn]
                    print bn
                    remaining = [0]*len(bn)
                    temp=[]
                    for item in bn:
                        for ii in item[1]:
                            if item[0] > self.endowment[ii][bn.index(item)]:
                                temp.append(self.endowment[ii][bn.index(item)])
                                diff = item[0] - self.endowment[ii][bn.index(item)]
                                remaining[bn.index(item)]=diff
                                item[0]=diff
                                # if ii==len(item[1])-1 and diff!=0:
                                #     remaining[bn.index(item)]=diff
                            else:
                                temp.append(item[0])
                    print temp
                    print remaining
                    if remaining==[0 for x in range(len(remaining))]:
                        print "yes"
                        final_purchase[n].append(temp)
                        # print rem

        print "final purchase: {}".format(final_purchase)

    # def bestPurchase(self, keys, budget):


def main():
    G = nx.Graph()
    numNodes = 3 #int(raw_input("Enter the number of nodes to be included in the graph: "))
    G.add_edges_from([(0,1),(1,2)])
    nx.draw(G, with_labels = True)
    plt.show()
    #plt.savefig("InitialGraph.png")

    endowment = {0:np.array([1,2]), 1:np.array([1,1]), 2:np.array([2,1])}
    utility = {0:LinearUtility([1,0]), 1:LinearUtility([1,1]), 2:LinearUtility([0,1])}

    mkt = Market(G, utility, endowment)

    prices = {0:np.array([2,1]), 1:np.array([2,2]), 2:np.array([1,2])}

    mkt.bestPrices(prices)
    # best_prices, budget     = mkt.bestPrices(prices)
    # best_purchase           = mkt.bestPurchase(best_prices, budget)

if __name__ == "__main__":
    main()
