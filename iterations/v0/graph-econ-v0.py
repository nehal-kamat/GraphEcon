#!/usr/bin/python

def check_prices(a,b,i,j):
	if utility_function[a][j]==1 and utility_function[b][j]!=1:
		return b
	elif utility_function[b][j]==1 and utility_function[a][j]!=1:
		return a
	elif utility_function[b][j]==1 and utility_function[a][j]==1:
		return i
	elif endowment[a][j]*local_price_vector[a][j] < endowment[b][j]*local_price_vector[b][j]:
		return a
	else:
		return b

num_traders = 3

endowment = {1:(1,2),2:(1,1),3:(2,1)}
local_price_vector = {1:(2,1),2:(2,2),3:(1,2)}
utility_function = {1:(1,0),2:(1,1),3:(0,1)}

#wealth earned after selling all endowments
earning = {i:sum(tuple(map(lambda x,y:x*y,endowment[1],local_price_vector[1]))) for i in range(1,4)}

print earning

#total number of items sold
total_sell_count = sum([endowment[i][0]+endowment[i][1] for i in endowment])

print "total items sold is %i" %total_sell_count

total_buy_count = {}

#calculating number of items bought across the graph
for i in xrange(1, num_traders+1):
	count = 0
	if i == 1:
		for j in xrange(2):
			if utility_function[i][j] == 1:
				count += endowment[2][j]
		total_buy_count[i] = count
	elif i == 2:
		for j in xrange(2):
			if utility_function[i][j] == 1:
				best = check_prices(1,3,i,j)
				count += endowment[best][j]
			total_buy_count[i] = count
	elif i == 3:
		for j in xrange(2):
			if utility_function[i][j] == 1:
				count += endowment[2][j]
		total_buy_count[i] = count

#total number of items bought
total_buy = sum([total_buy_count[i] for i in total_buy_count])

print "total items bought is %i" %total_buy

if total_buy != total_sell_count:
	print "Market hasn't cleared!"

#for i in xrange(1, num_traders+1):

