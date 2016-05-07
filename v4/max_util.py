from itertools import product, permutations, combinations

class MaxUtil():
    def __init__(self, util_rav, end_rav, prices_rav, G, budget, ngoods, nnodes):
        self.util_rav = util_rav
        self.end_rav = end_rav
        self.prices_rav = prices_rav
        self.G = G
        self.budget = budget
        self.ngoods = ngoods
        self.nnodes = nnodes

    def get_max_util(self):
        max_util = {}

        def l(i,j,k):
            countj = j
            countk = k*self.nnodes
            counti = div * i
            return counti + countk + countj

        div = self.nnodes * self.ngoods
        for n in self.G:
            temp = []
            combs = []
            mx = 0
            mxu = 0
            for j in self.G:
                for k in range(self.ngoods):
                    if self.util_rav[l(n,j,k)] == 1:
                        temp.append(l(n,j,k))
            for ii in range(len(temp)):
                combs.append(list(combinations(temp, ii+1)))
            combs = [item for sublist in combs for item in sublist]
            maxu = 0
            for item in combs:
                s = 0
                for ii in item:
                    s += self.end_rav[ii]*self.prices_rav[ii]
                if s <= self.budget[n] and s > maxu:
                    maxu = sum([self.util_rav[ii]*self.end_rav[ii] for ii in item])
            max_util[n] = maxu

        return max_util
