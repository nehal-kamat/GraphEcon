import networkx as nx
from random import random
import matplotlib.pyplot as plt
from itertools import product, permutations, combinations
import math
import numpy as np
import numdifftools as nd
from scipy import optimize


class Utility:
    def __init__(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

class LinearUtility:
    def __init__(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

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
        self.ngoods = 2
        self.nnodes = len(self.G)

    def getOptPlans(self, price_vec):
        self.prices = price_vec
        prices = self.prices
        nnodes = self.nnodes
        ngoods = self.ngoods
        endowment = self.endowment
        utility = self.utility

        mat = {}
        price_mat = {}
        util_mat = {}

        adj = np.array([[1,1,0],[1,1,1],[0,1,1]])

        for i in range(nnodes):
            mat[i] = np.zeros((nnodes, ngoods))
            price_mat[i] = np.zeros((nnodes, ngoods))
            util_mat[i] = np.zeros((nnodes, ngoods))
            for j in range(nnodes):
                for k in range(ngoods):
                    if adj[i][j] == 1:
                        mat[i][j][k] = endowment[j][k]
                        price_mat[i][j][k] = price_vec[j][k]
                        wght = utility[i].get_weights()
                        util_mat[i][j][k] = wght[k]

        # print mat
        # print price_mat
        # print util_mat

        end_rav = [np.ravel(mat[i], order='F') for i in range(nnodes)]
        end_rav = [item for sublist in end_rav for item in sublist]
        self.end_rav = end_rav

        # print self.end_rav

        prices_rav = [np.ravel(price_mat[i], order='F') for i in range(nnodes)]
        prices_rav = [item for sublist in prices_rav for item in sublist]
        self.prices_rav = prices_rav

        util_rav = [np.ravel(util_mat[i], order='F') for i in range(nnodes)]
        util_rav = [item for sublist in util_rav for item in sublist]
        self.util_rav = util_rav

        def l(i,j,k):
            avg = len(self.end_rav) / nnodes
            countj = j
            countk = k*nnodes
            counti = avg*i

            # out = []
            # last = 0.0
            # while last < len(self.end_rav):
            #     out.append(self.end_rav[int(last):int(last+avg)])
            #     last += avg
            # avg2 = len(out[i]) / ngoods
            # out2 = []
            # last2 = 0.0
            # while last2 < len(out[i]):
            #     out2.append(out[i][int(last2):int(last2+avg2)])
            #     last2 +=avg2

            return counti + countk + countj

        # func = [(endowment[i][k]**2 - 2*endowment[i][k]*sum([l(j,i,k) for j in range(nnodes)]) + (sum([l(j,i,k) for j in range(nnodes)]))**2) for i in range(nnodes) for k in range(ngoods)]

        fun = lambda x: sum([(sum([x[l(j,i,k)] for j in range(nnodes)]))**2 - 2*endowment[i][k]*sum([x[l(j,i,k)] for j in range(nnodes)]) + endowment[i][k]**2 for i in range(nnodes) for k in range(ngoods)])

        # print fun
        temp = []
        for i in range(nnodes):
            for k in range(ngoods):
                for j in range(nnodes):
                    temp.append(l(j,i,k))

        print temp

        A = np.zeros((nnodes, len(self.end_rav)))
        A2 = np.zeros((nnodes, len(self.end_rav)))
        B = np.array([4.0, 4.0, 4.0])
        B2 = np.array([-2.0, -2.0, -2.0])

        # print B.shape

        div = len(self.end_rav) / nnodes

        start = 0
        for i in range(0, nnodes):
            A[i][start : start + div] = self.prices_rav[start : start + div]
            start = start + div

        start = 0
        for i in range(0, nnodes):
            A2[i][start : start + div] = self.util_rav[start : start + div]
            start = start + div

        # print A
        # print A2

        def hessian(x):
            """
            Calculate the hessian matrix with finite differences
            Parameters:
               - x : ndarray
            Returns:
               an array of shape (x.dim, x.ndim) + x.shape
               where the array[i, j, ...] corresponds to the second derivative x_ij
            """
            x_grad = np.gradient(x)
            hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
            for k, grad_k in enumerate(x_grad):
                # iterate over dimensions
                # apply gradient again to every component of the first derivative.
                tmp_grad = np.gradient(grad_k)
                for l, grad_kl in enumerate(tmp_grad):
                    hessian[k, l, :, :] = grad_kl

            return hessian


        H = np.zeros((18, 18), int)
        np.fill_diagonal(H, 2)

        for i in range(len(temp)):
            t = len(temp) / (ngoods * nnodes)
            start = 0.0
            for ii in range(nnodes * ngoods):
                chunk = temp[int(start):int(start+3)]
                comb = list(combinations(chunk, 2))
                for item in comb:
                    H[item[0]][item[1]] = 2
                    H[item[1]][item[0]] = 2
                start += t

        print H


        c = np.zeros(18)
        c0 = 8
        x0 = np.ones(18)
        cons = ({'type':'ineq', 'fun':lambda x: B - np.dot(A,x)},
                {'type':'eq', 'fun':lambda x: B2 - np.dot(A2,x)})

        bnds = ((0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None))

        # def fun(x):
            # return (np.dot(x.T, np.dot(H, x))+ np.dot(c, x) + c0)

        fun = lambda x: 0.5 * np.dot(x.T, np.dot(H, x))+ np.dot(c, x) + c0

        res_cons = optimize.minimize(fun, x0, hess = H, bounds = bnds, constraints=cons)

        print res_cons.x[6]

def main():
    G = nx.Graph()
    numNodes = 3 #int(raw_input("Enter the number of nodes to be included in the graph: "))
    G.add_edges_from([(0,1),(1,2)])
    # nx.draw(G, with_labels = True)
    # plt.show()
    #plt.savefig("InitialGraph.png")

    endowment = {0:np.array([1,2]), 1:np.array([1,1]), 2:np.array([2,1])}
    utility = {0:LinearUtility([1,0]), 1:LinearUtility([1,1]), 2:LinearUtility([0,1])}
    # utility = {0:Utility([1,0]), 1:Utility([1,1]), 2:Utility([0,1])}

    mkt = Market(G, utility, endowment)

    prices = {0:np.array([2,1]), 1:np.array([2,2]), 2:np.array([1,2])}
    budget = {}
    for n in G:
        budget[n] = sum(endowment[n]*prices[n])
    bp = {}
    for n in G:
        bp[n] = utility[n].bestPurchase(prices[n], budget[n])
    # print bp
    # mkt.bestPrices(prices)
    mkt.getOptPlans(prices)
    # best_prices, budget     = mkt.bestPrices(prices)
    # best_purchase           = mkt.bestPurchase(best_prices, budget)

if __name__ == "__main__":
    main()
