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

                

class LinearUtility(Utility):

    def __init__(self, weights):
        self.weights = weights

    
    def bestPurchase(self, price_vec, budget):
        #Calculate best purchase for linear, given a price vector and the budget
        #return the index of the maximum
        i = np.argmax(self.weights/price_vec)
        x = np.zeros(price_vec.shape)
        x[i] = budget/price_vec[i]
        
        return x
       
  
    '''
    def bestPurchase(self, price_vec, budget):
        #other method, returns max utility
        for n in range(0, len(price_vec)):
            price_vec[n]=budget/price_vec[n]
        x = np.zeros(price_vec.shape)
        index = np.argmax(price_vec)
        x[index]=price_vec[index]*self.weights[index]
       
        return x
    '''
   

class NonLinearUtility(Utility):

    def __init__(self, weights):
        self.weights = weights

    def bestPurchase(self, values, budget):
        #self.price_vec = price_vec
        #self.budget = budget
        
        return [eval(self.weights, {'math': math, '__builtins__': None}, {'x': value})
                for value, self.weights in zip(values, self.weights)]
       
        
        
###########
def gen_data(G, numItems):
    numNodes = nx.number_of_nodes(G)
    
    for i in xrange(numNodes):
        ch = [lambda x : math.sqrt(x), lambda y : random.randint(0, numItems)*y, lambda z : 4 * math.sqrt(z)]
        utilVector = [random.choice(ch) for x in xrange(numItems)]
        #[random.choice([0,1]) for x in range(numItems)] #G.node[i]['utility'] = random.sample(xrange(num_items), num_items)
        G.node[i]['util'] = utilVector 
        G.node[i]['price_vec'] = [float(random.randint(1,10)) for x in range(numItems)] #[random.sample(xrange(1,10),numItems)]]
        G.node[i]['endow'] = random.sample(xrange(1,10),numItems)
        G.node[i]['earning'] = [x*y for x,y in zip(G.node[i]['price_vec'], G.node[i]['endow'])]
        G.node[i]['expenditure'] = [0] * numItems

###########
def check_rationality(G, neighbor, bestNeighbour, i,  idx):
    if G.node[neighbor]['endow'][idx]/float(G.node[i]['earning'][idx]) < G.node[bestNeighbour]['endow'][idx]/float(G.node[i]['earning'][idx]):
        return bestNeighbour
    else:
        return neighbor
        

###########
def find_best_price(G, i, idx, nList):
    bestPrice = G.node[i]['price_vec'][idx]
    bestNeighbour = i
    
    for neighbor in nx.all_neighbors(G, i):
        print 'neighbor : %d | Price vector : %d' %(neighbor, G.node[neighbor]['price_vec'][idx])
        
        if G.node[neighbor]['price_vec'][idx] < bestPrice:
            bestPrice = G.node[neighbor]['price_vec'][idx]
            bestNeighbour = neighbor
        
        elif G.node[neighbor]['price_vec'][idx] == bestPrice:
            bestNeighbour = check_rationality(G, neighbor, bestNeighbour, i, idx)
            bestPrice = G.node[bestNeighbour]['price_vec'][idx]

    print 'best price of commodity %s is %d from trader %d' %(idx, bestPrice, bestNeighbour)
    
    return bestNeighbour, bestPrice

############
def market_status(G, numTraders, numItems):
    totalPurchased = 0
    totalSold = 0
    for i in G.nodes_iter():
        totalSold += sum(G.node[i]['endow'])

    print 'total items sold : ', totalSold

    for i in G.nodes_iter():
        nList = G.neighbors(i)
        print 'trader : ', i
        #print 'utility ', G.node[i]['util']
        print 'price vector ', G.node[i]['price_vec']
        print 'endowment ', G.node[i]['endow']
        print 'earning ', G.node[i]['earning']
        print 'list of neighbours : %s' %nList

        idx = 0
        for item in G.node[i]['util']:
            if item != 0:
                #idx = G.node[i]['util'].index(item)
                print 'index ', idx
                bestN, bestP = find_best_price(G, i, idx, nList)
                G.node[i]['expenditure'][idx] = bestP * G.node[bestN]['endow'][idx]
                idx += 1
            else:
                idx += 1
        print 'expenditure : ', G.node[i]['expenditure']
        print 'total earning : %.2f | total expenditure : %.2f' %(sum(G.node[i]['earning']), sum(G.node[i]['expenditure']))
        
        print


#############
def main():
    numTraders = int(raw_input("Enter the number of traders : "))
    numItems = int(raw_input("Enter the number of items : "))
    print
    G = nx.fast_gnp_random_graph(numTraders, 0.6)
    gen_data(G, numItems)
    nx.draw(G, with_labels = True)
    plt.show()
    market_status(G, numTraders, numItems)

##############
if __name__ == "__main__":
    main()