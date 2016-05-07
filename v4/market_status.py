import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from random import random
from scipy import optimize
from max_util import MaxUtil
from itertools import product, permutations, combinations

class Market:
    def __init__(self, G, endowment, utility, ngoods):
        self.G = G.copy()
        self.utility = utility
        self.endowment = endowment
        self.ngoods = ngoods
        self.nnodes = len(self.G)

    def checkClear(self, price_vec):
        self.prices = price_vec
        prices = self.prices
        nnodes = self.nnodes
        ngoods = self.ngoods
        endowment = self.endowment
        utility = self.utility

        adj = np.zeros((nnodes, nnodes))
        for i in range(nnodes):
            for j in range(nnodes):
                if i == j:
                    adj[i][j] = 1
                if j in self.G.neighbors(i):
                    adj[i][j] = 1

        div = nnodes * ngoods
        def l(i,j,k):
            countj = j
            countk = k*nnodes
            counti = div * i
            return counti + countk + countj

        end_mat = {}
        price_mat = {}
        util_mat = {}

        for i in range(nnodes):
            end_mat[i] = np.zeros((nnodes, ngoods))
            price_mat[i] = np.zeros((nnodes, ngoods))
            util_mat[i] = np.zeros((nnodes, ngoods))
            wght = utility[i].get_weights()
            for k in range(ngoods):
                for j in range(nnodes):
                    if adj[i][j] == 1:
                        end_mat[i][j][k] = endowment[j][k]
                        price_mat[i][j][k] = price_vec[j][k]
                        util_mat[i][j][k] = wght[k]

        budget = {}
        for n in self.G:
            budget[n] = sum(endowment[n]*prices[n])

        self.end_rav = [item for sublist in [np.ravel(end_mat[i], order='F') for i in range(nnodes)] for item in sublist]

        self.prices_rav = [item for sublist in [np.ravel(price_mat[i], order='F') for i in range(nnodes)] for item in sublist]

        self.util_rav = [item for sublist in [np.ravel(util_mat[i], order='F') for i in range(nnodes)] for item in sublist]

        max_util = MaxUtil(self.util_rav, self.end_rav, self.prices_rav, self.G, budget, ngoods, nnodes)
        max_utility = max_util.get_max_util()

        fun = lambda x: sum([(sum([x[l(j,i,k)] for j in range(nnodes)]))**2 - 2*endowment[i][k]*sum([x[l(j,i,k)] for j in range(nnodes)]) + endowment[i][k]**2 for i in range(nnodes) for k in range(ngoods)])

        temp = [l(j,i,k) for i in range(nnodes) for k in range(ngoods) for j in range(nnodes)]

        dimension  = nnodes * nnodes * ngoods
        A = np.zeros((nnodes, dimension))
        A2 = np.zeros((nnodes, dimension))
        B = np.array([budget[n] for n in self.G])
        B2 = np.array([max_utility[n] for n in self.G])

        start = 0
        for i in range(0, nnodes):
            A[i][start : start + div] = self.prices_rav[start : start + div]
            start = start + div

        start = 0
        for i in range(0, nnodes):
            A2[i][start : start + div] = self.util_rav[start : start + div]
            start = start + div

        H = np.zeros((dimension, dimension), int)
        np.fill_diagonal(H, 2)

        for i in range(len(temp)):
            t = len(temp) / (ngoods * nnodes)
            start = 0.0
            for ii in range(nnodes * ngoods):
                chunk = temp[int(start):int(start+nnodes)]
                comb = list(combinations(chunk, 2))
                for item in comb:
                    H[item[0]][item[1]] = 2
                    H[item[1]][item[0]] = 2
                start += t

        c = np.zeros(18)
        c0 = 8
        x0 = np.ones(18)
        cons = ({'type':'ineq', 'fun':lambda x: B - np.dot(A,x), 'jac':lambda x: -A},
                {'type':'eq', 'fun':lambda x: B2 - np.dot(A2,x), 'jac':lambda x: -A2})

        bnds = ((0, None),) * dimension

        res_cons = optimize.minimize(fun, x0, bounds = bnds, constraints=cons)

        if res_cons.fun == float(0):
            print "Market Cleared!"
        else:
            print "Distance from equilibrium: {}".format(res_cons.fun)

        print "Optimal plans:",
        print [format(ii, '.2f') for ii in res_cons.x]
