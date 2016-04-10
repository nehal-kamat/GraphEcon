#import numpy.random as npr
import random
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import math
import numpy as np


class Utility(object):
    """"Return the max utility, used for both lineal and non lineal"""
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


class LinearUtility:
    def __init__(self, weights):
        self.weights = np.array(map(float, weights))

    def bestPurchase(self, price_vec, budget):
        i = np.argmax(self.weights/np.array(map(float,price_vec)))
        x = np.zeros(price_vec.shape)
        x[i] = budget/price_vec[i]

        return x


def gen_data(G, numItems):
    numNodes = nx.number_of_nodes(G)

    for i in xrange(numNodes):
        ch = [lambda x : math.sqrt(x), lambda y : random.randint(0, numItems)*y, lambda z : 4 * math.sqrt(z)]
        utilVector = [random.choice(ch) for x in xrange(numItems)]
        #[random.choice([0,1]) for x in range(numItems)] #G.node[i]['utility'] = random.sample(xrange(num_items), num_items)
        G.node[i]['util'] = utilVector
        #G.node[i]['price_vec'] = [float(random.randint(1,10)) for x in range(numItems)] #[random.sample(xrange(1,10),numItems)]]
        #G.node[i]['endow'] = random.sample(xrange(1,10),numItems)
        G.node[i]['price_vec'] = np.array(map(int, raw_input().split(',')))
        G.node[i]['endow'] = np.array(map(int, raw_input().split(',')))
        G.node[i]['budget'] = sum([x*y for x,y in zip(G.node[i]['price_vec'], G.node[i]['endow'])])
        G.node[i]['expenditure'] = [0] * numItems


def is_better_price(best_prices, price_vec_n, item):
    if best_prices[item] > price_vec_n[item]:
        return True
    return False

def checkClearing(G, best_prices, tracking):
    bundle = []
    for i in G.nodes_iter():
        print "Enter utility of each good for node {}: ".format(i)
        utility_vec = np.array(map(int, raw_input().split(',')))
        U = LinearUtility(utility_vec)
        print best_prices[i]
        print U.bestPurchase(np.array(best_prices[i]), G.node[i]['budget'])
        #print bundle[i]

def market(G):
    """Checks whether equilibrium exists for given network"""
    tracking = {} # make dictionary of dictionary to keep track of item -> neighbour
    best_prices = {}
    for i in G.nodes_iter():
        nList = G.neighbors(i)
        print 'Node: ', i
        print 'Node {}\'s prices for goods: {}'.format(i,G.node[i]['price_vec'])
        print 'Node {}\'s endowment: {}'.format(i,G.node[i]['endow'])
        print 'Node {}\'s budget: {}'.format(i,G.node[i]['budget'])
        print 'Node {}\'s neighbours: {}'.format(i,nList)

        best_prices[i] = list(G.node[i]['price_vec']) # default best prices belong to node i itself
        tracking[i] = {}
        for item in range(0, len(G.node[i]['endow'])):
            tracking[i][item] = i
            for nn in nList:
                if is_better_price(best_prices[i], G.node[nn]['price_vec'], item): # checking if neighbour has lesser or equal price for given item
                    best_prices[i][item] = G.node[nn]['price_vec'][item]
                    tracking[i][item] = nn

        print 'best prices: ', best_prices[i]
        print 'tracking: ', tracking[i]
        print '---'

    checkClearing(G, best_prices, tracking)


def main():
    numTraders = int(raw_input("Enter the number of traders : "))
    numItems = int(raw_input("Enter the number of items : "))
    G = nx.fast_gnp_random_graph(numTraders, 0.5)
    nx.draw(G, with_labels = True)
    plt.show()
    gen_data(G, numItems)
    market(G)

if __name__ == "__main__":
    main()
