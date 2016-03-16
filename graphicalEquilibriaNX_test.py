#!~/Public/anaconda2/bin/python

import numpy.random as npr
import random
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import math


"""
In this function, we're generating random data for each node
utility, price vector and endowment are vectors have num_items number of random entries
"""

def gen_data(G, num_nodes, num_items):
	for i in xrange(num_nodes):
		#G.node[i]['utility'] = random.sample(xrange(num_items), num_items)
		G.node[i]['utility'] = [random.choice([0,1]) for x in range(num_items)]
		print 'utility ', G.node[i]['utility']
			
		G.node[i]['price vector'] = random.sample(xrange(1,10),num_items)
		print 'price vector ', G.node[i]['price vector']
			
		G.node[i]['endowment'] = random.sample(xrange(1,25),num_items)
		print 'endowment ', G.node[i]['endowment']

"""
In this function, we're checking whether or not the market has cleared
Market clearing condition is : total number of items bought = total number of items sold
"""

def check_price(index, ns):
	minimum_price = min([G.node[nn]['price vector'][index] for nn in ns])
	print minimum_price

def get_endowment(G, i, num_items):
	print 'number of items : ', num_items
	G.node[i]['endowment'] = list(map(int,raw_input("Enter endowment as space-separated values : ").split(' ')))
	return G.node[i]['endowment']

def get_price_vector(G, i, num_items):
	print 'number of items : ', num_items
	G.node[i]['price vector'] = list(map(int,raw_input("Enter price vector as space-separated values : ").split(' ')))
	return G.node[i]['price vector']

def get_utility_function(G, i, num_items):
	print 'number of items : ', num_items
	G.node[i]['utility'] = list(map(int,raw_input("Enter utility as space-separated values : ").split(' ')))
	return G.node[i]['utility']
	
def utility_function(vector, uf):
	final_util = 0
	util_multi = []
	for i in range(len(vector)):
		util_multi.append(uf[i]*(vector[i]))
		#print uf[i], vector[i], uf[i](vector[i])
		final_util += uf[i]*(vector[i])
	return final_util, util_multi

def market_status(G, num_nodes, num_items):
	"""
	for i in xrange(num_nodes):
		print  sum(G.node[i]['endowment'])
	"""
	
	total_items_bought = 0
	keep_track = set()
	pos = []
	
	for i in G.nodes_iter():
		earning = {}
		expenditure = {}
		_earning = []
		_expenditure = []
		consumption_plans = []
		max_utility_plan = []
		temp_expenditure = []
		print '---'
		print 'node ',i
		G.node[i]['endowment'] = get_endowment(G, i, num_items)
		G.node[i]['price vector'] = get_price_vector(G, i, num_items)
		G.node[i]['utility'] = get_utility_function(G, i, num_items)

		earning[i] = [a*b for a,b in zip(G.node[i]['price vector'],G.node[i]['endowment'])]
		expenditure[i] = [a*b for a,b in zip(G.node[i]['price vector'],G.node[i]['endowment'])]
		print 'p[i]*e[i] = ',sum(earning[i])
		ns = G.neighbors(i)
		
		print 'neighbors : ',ns
		for nn in ns:
			print 'neighbor node', nn
			G.node[nn]['endowment'] = get_endowment(G, nn, num_items)
			G.node[nn]['price vector'] = get_price_vector(G, nn, num_items)
			#G.node[i]['utility'] = get_utility_function(G, i, num_items)
		for nn in ns:
			expenditure[nn] = [a*b for a,b in zip(G.node[nn]['price vector'],G.node[nn]['endowment'])]
		#print 'expenditure of all neighbor nodes : ',expenditure
		#print ''
		for key,value in expenditure.iteritems():
			temp = value
			temp_expenditure.append(temp)
		#print _expenditure
		_expenditure = list(itertools.combinations(temp_expenditure,num_items))
		_expenditure2 = []
		for sublist in _expenditure:
			_expenditure2.append(list(itertools.product(*sublist)))
		
		for item in _expenditure2:
			for subitem in item:
				if sum(subitem)<=sum(earning[i]):
					consumption_plans.append(subitem)

		#print 'candidate set consumption plans\n', consumption_plans
		#print list(itertools.product(*_expenditure))
		
		
		ch = [lambda x : math.sqrt(x), lambda y : random.randint(0, num_items)*y, lambda z : 4 * math.sqrt(z)]
		temp = [random.choice(ch) for x in xrange(num_items)]

		max_util = {}
		if len(consumption_plans)!=0:
			for item in consumption_plans:
				util_value, util_mult = utility_function(item,G.node[i]['utility'])
				max_util[item]=util_value
			maxx= max(max_util.iterkeys(), key=(lambda key: max_util[key]))
			print util_mult
			print '\nmax utility plan : ', maxx

			for item in maxx:
				#print item
				for key,value in expenditure.iteritems():
					#print item2
					if item in value:
						keep_track.add(key)
						#print keep_track
						total_items_bought += G.node[key]['endowment'][value.index(item)]
						print total_items_bought

		else:
			print 'Not in equilibrium!'
			break

		print '---'
	total_items_sold = sum([sum(G.node[x]['endowment']) for x in G.nodes_iter()])
	print 'Total commodities sold : ',total_items_sold
	print keep_track
	print total_items_bought
		
				#check_price(G.node[i]['utility'].index(item), ns)
	#pass
def main():
	num_nodes = int(raw_input("Enter the number of nodes in the graph : "))
	G = nx.Graph()
	for i in range(num_nodes):
		G.add_node(i)
	G.add_edge(0,1)
	G.add_edge(1,2)

	#G = nx.fast_gnp_random_graph(num_nodes, 0.9) # No trade restrictions
	num_items = int(raw_input("Enter the number of items : "))
	nx.draw(G, with_labels = True)
	plt.show()
	gen_data(G, num_nodes, num_items)

	plt.savefig('initial graph.png')
	#node_data(G, num_nodes)
	market_status(G, num_nodes, num_items)
	

	"""
	for item in G.nodes_iter():
		print G.node[item]['utility'] 
		print G.node[item]['price vector']
		print G.node[item]['endowment']
	"""
	#nx.draw_networkx_labels(G,labels='utility')


if __name__ == "__main__":
	main()
	
