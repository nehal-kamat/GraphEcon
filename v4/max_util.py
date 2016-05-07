# this is an auxiliary class to calculate the maximum utility for each node in the graph
import numpy as np
import networkx as nx

from random import random
from itertools import product, permutations, combinations

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
        # print "budget: {}\nbest prices: {}\nbest neighbours: {}".format(budget, best_prices, best_neighbours)

        # return keys, budget

        best_purchase = {}
        for n in self.G:
            best_purchase[n] = {}
            best_purchase[n] = self.utility[n].bestPurchase(keys[n], budget[n])
        # print "purchase candidates: ", best_purchase

        # return best_purchase

        final_purchase = {}
        for n in self.G:
            final_purchase[n] = []
            # print type(best_purchase[n][0])
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
                    # print bn
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
                    # print temp
                    # print remaining
                    if remaining==[0 for x in range(len(remaining))]:
                        # print "yes"
                        final_purchase[n].append(temp)
                        # print rem

        # print "final purchase: {}".format(final_purchase)

        max_util = {}
        for n in self.G:
            max_util[n] = sum([float(item) for sublist in final_purchase[n] for item in sublist])

        return max_util
