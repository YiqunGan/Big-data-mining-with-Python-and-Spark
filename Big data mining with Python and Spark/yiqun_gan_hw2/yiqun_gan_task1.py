import math
import itertools
from pyspark import SparkContext
import sys
import numpy as np
import pandas as pd
import collections
import bisect
import json
class FirstList(collections.UserList):
    def __lt__(self, other):
        return self[0].__lt__(other)
def get_item_dict(baskets):
	# Assign each item (artist_id) an integer to be used as index in the matrix
	item_dict = {}
	for basket in baskets:
		items = basket[1] #basket[0] is user_id, basket[1] is a list of artist_id
		for item in items:
			if item not in item_dict:
				#len(item_dict) is the size of dictionary
				#When adding the first item, it evaluates to 0, adding the second item, it evaluates to 1
				#So the range of assigned integers is [0, #items-1]
				item_dict[item] = len(item_dict)
	return item_dict
def get_item_counter(baskets):
	item_counter = collections.Counter()
	for basket in baskets:
		items = basket[1]
		item_counter.update(items)
	return item_counter

def inverse_dict(d):
	# {key: value} will become {value: key}
	return {v: k for k, v in d.items()}

def tuple_wrapper(s):
	if type(s) is not tuple:
		s = (s, )
	return s

def get_possible_k(item_dict, k):
	possible_k = {}
	for pair in itertools.combinations(item_dict.keys(), 2):
		pair_set = set()
		for i in range(2):
			pair_set = pair_set.union(tuple_wrapper(pair[i]))
		if len(pair_set) == k:
			possible_k[frozenset(pair_set)] = [pair[0], pair[1]]
	return possible_k
def tuple_list_method(baskets, support, item_dict=None, k=2):
    if item_dict is None:
        item_dict = get_item_dict(baskets)
    else:
        # Only used in Q3, Q4, Q5
        # item_dict has been computed -> it's aprior method. Filter baskets to remove infrequent items
        # When k=2, infrequent single items have been removed from baskets. When k>=3, item_dict won't include single items
        # baskets will be modified!
        if k == 2:
            for i in range(len(baskets)):
                basket = baskets[i]
                items = basket[1]
                items_filterd = [item for item in items if item in item_dict]
                baskets[i] = (basket[0], items_filterd)

    item_dict_inv = inverse_dict(item_dict)
    n = len(item_dict)

    # Only used in Q3, Q4, Q5
    if k >= 3:
        possible_k = get_possible_k(item_dict, k)

    tuples = [] # Storage space is allocated every time a new pair is occurred, similar to LinkedList

    # Key logic: Tuple List Method
    for basket in baskets:
        items = basket[1]
        for kpair in itertools.combinations(items, k):
            # kpair is a k element tuple, kpair[i] is item (string)
            if k >= 3:
                pair_set = frozenset(kpair)

                # Now kpair is a 2 element pair
                kpair = possible_k.get(pair_set, None)
                if kpair is None:
                    continue

            i = item_dict[kpair[0]]
            j = item_dict[kpair[1]]

            if i > j:
                j, i = i, j

            # Convert 2D index to 1D index
            # idx don't have to be continuous in this case. The only thing we care is their relative order.
            # We could use simple C style index calculation, or continue using the index method in q1
            # idx = int((n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1)
            idx = i*n+j

            # Core idea: tuples are sorted in increasing order, so they could be efficiently located by binary search
            # Example: x = [1, 4, 12, 20, 30, 50] If we want to find the index of item 30 and manipulate it, it could be found efficiently by binary search. bisect.bisect_left(x, 30)
            # Binary search takes O(log(n)) time, much faster than traversing the list, which takes O(n).
            # This is the benefit of keeping the list sorted. When adding items, we want to keep it sorted, finding the insertion index also takes O(log(n)) time. bisect.bisect_left(x, 25)
            # Checkout https://docs.python.org/3.6/library/bisect.html
            insert_idx = bisect.bisect_left(tuples, idx)

            # The insertion index is at the end of the list, i.e. the new item is larger than all items in the list
            if insert_idx >= len(tuples):
                tuples.append(FirstList([idx, 1]))
            else:
                tp = tuples[insert_idx]

                # This pair is already in the tuple list. Increase it's count (second element) by 1
                if tp[0] == idx:
                    tp[1] += 1
                else:
                    # This pair is not yet in the tuple list. Add a new tuple, the format is: (1D index, count)
                    tuples.insert(insert_idx, FirstList([idx, 1]))

    # Extract results
    frequent_itemset_list = []
    for tp in tuples:
        count = tp[1]

        # Convert 1D index to 2D index
        # If you use different indexing method, this also needs to be changed
        i = tp[0] // n
        j = tp[0] % n

        item_i = item_dict_inv[i]
        item_j = item_dict_inv[j]

        # This implementation is ready for k>=3
        item_all = set()
        for item in (item_i, item_j):
            item_all = item_all.union(tuple_wrapper(item))

        item_all = tuple(sorted(list(item_all)))

        # apply support threshold
        if count >= support:
            frequent_itemset_list.append((item_all, count))

    frequent_itemset_list = sorted(frequent_itemset_list, key=lambda x: [-x[1]] + list(x[0]))
    return frequent_itemset_list
# Wrap in a function to be reused in Q3
def triangular_matrix_method(baskets, support, item_dict=None, k=2):
	if item_dict is None:
		item_dict = get_item_dict(baskets)  #item -> integer
	else:
		# Only used in Q4, Q5
		# item_dict has been computed -> it's aprior method. Filter baskets to remove infrequent items
		# When k=2, infrequent single items have been removed from baskets. When k>=3, item_dict won't include single items
		# baskets will be modified!
		if k == 2:
			for i in range(len(baskets)):
				basket = baskets[i]
				items = basket[1]
				items_filterd = [item for item in items if item in item_dict]
				baskets[i] = (basket[0], items_filterd)

	item_dict_inv = inverse_dict(item_dict) #integer -> item. Inverse dict will be used when printing results
	n = len(item_dict)

	# Only used in Q4, Q5
	if k >= 3:
		possible_k = get_possible_k(item_dict, k)

	# Storage space is pre-allocated. Similiar to ArrayList
	# Convert 2D index to 1D index
	# Conversion logic: https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
	tri_matrix = [0] * (n * (n-1) // 2) # n * (n-1) always be even for n >= 2, use true division to make it a int

	# Key logic: Upper Triangular Matrix Method
	for basket in baskets:
		# Take a basket (user), iterate all items (artist)
		items = basket[1]

		# Checkout https://docs.python.org/3.6/library/itertools.html#itertools.combinations
		# Equivalent to a double loop, but more concise
		for kpair in itertools.combinations(items, k):
			# kpair is a k element tuple, kpair[i] is item (string)
			if k >= 3:
				pair_set = frozenset(kpair)

				# Now kpair is a 2 element pair
				kpair = possible_k.get(pair_set, None)
				if kpair is None:
					continue

			# i, j is integer index
			i = item_dict[kpair[0]]
			j = item_dict[kpair[1]]

			# Keep sorted in upper triangular order
			if i > j:
				j, i = i, j

			# Convert 2D index to 1D index
			idx = int((n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1)
			# Increase count by 1
			tri_matrix[idx] += 1

	# Extract results
	frequent_itemset_list = []
	for idx in range(len(tri_matrix)):
		# Convert 1D index to 2D index
		i = int(n - 2 - math.floor(math.sqrt(-8*idx + 4*n*(n-1)-7)/2.0 - 0.5))
		j = int(idx + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2)

		count = tri_matrix[idx]
		item_i = item_dict_inv[i]
		item_j = item_dict_inv[j]

		# Keep sorted in ascii order. item_i, item_j are strings or tuple of strings
		# This implementation is ready for k>=3
		item_all = set()
		for item in (item_i, item_j):
			item_all = item_all.union(tuple_wrapper(item))

		item_all = tuple(sorted(list(item_all)))

		# apply support threshold
		if count >= support:
			frequent_itemset_list.append((item_all, count))

	# First sorted by the occurrence count in decreasing order
	# Then sort by ascii order of the first item, in ascending order
	# Then sort by ascii order of the second item, in ascending order
	frequent_itemset_list = sorted(frequent_itemset_list, key=lambda x: [-x[1]] + list(x[0]))
	return frequent_itemset_list

def get_dict_from_frequent(frequent_list):
	item_dict = {}
	for item in frequent_list:
		item_dict[item] = len(item_dict)
	return item_dict

def aprior_all_method(baskets, support, method, son=False, total_baskets=0):
	# Used by Q5: SON
	if type(baskets) is not list:
		baskets = list(baskets) #baskets are list now
	if son:
		support = math.floor(support*len(baskets)/total_baskets)
	

	item_counter = get_item_counter(baskets)
	itemsets_1 = sorted([(k, v) for k, v in item_counter.items() if v >= support], key=lambda x: x[1], reverse=True)
	frequent_1 = [x[0] for x in itemsets_1]

	itemsets_list = [itemsets_1]
	frequent_list = frequent_1
	frequent_last = frequent_1

	k = 2
	max_k = frequent_1[-1]
	while True:
		# get a dictionary of current frequent items
		# Note: only frequent item pairs from the last pass is needed
		item_dict = get_dict_from_frequent(frequent_last)

		# baskets will be modfied!
		itemsets = method(baskets, support, item_dict, k=k)
		if len(itemsets) > 0:
			frequent_last = [x[0] for x in itemsets]
			frequent_list += frequent_last
			itemsets_list.append(itemsets)
			k += 1
		else:
			break  
		if len(frequent_last)<=1:
			break
	return itemsets_list, frequent_list
def get_dict_from_frequentitem(itemsets_list):
	itemsets_dict = {}
	for size in itemsets_list:
		for item in size:
			if type(item[0]) is not tuple:
				itemsets_dict[tuple_wrapper(item[0])] = item[1]
			else:
				itemsets_dict[item[0]] = item[1]
	
	return itemsets_dict

def print_all_frequent_itemsets(itemsets_list):
	n_itemsets = len(itemsets_list)
	for i in range(n_itemsets-1, -1, -1):
		print(f'Frequent itemsets of size: {i+1}')
		print_frequent_itemsets(itemsets_list[i])
def print_frequent_itemsets(itemsets):
	for frequent_itemset in itemsets:
		print(frequent_itemset)

	print(f'Total: {len(itemsets)}')

if __name__ == '__main__':
	filename = sys.argv[1]
	output = sys.argv[2]
	interest = float(sys.argv[3])
	support = int(sys.argv[4])
	ratings = pd.read_csv(filename, encoding = 'utf8')
	rating_filter = ratings[ratings['rating'] ==5]
	basket_name = None
	basket = []
	baskets = []
	for index,row in rating_filter.iterrows():
		user_id, movie_id = int(row.userId), int(row.movieId)
		if user_id != basket_name:
			if basket_name is not None:
				baskets.append((basket_name, basket))
			basket = [movie_id]
			basket_name = user_id
		else:
			basket.append(movie_id)
	baskets.append((basket_name, basket)) 
	#print(baskets)
	total_baskets = len(baskets)
	itemsets_list, frequent_list = aprior_all_method(baskets, support, tuple_list_method, True, total_baskets)
	#print(itemsets_list)
	#item_counter = get_item_counter(baskets)
	#itemsets_1 = sorted([(k, v) for k, v in item_counter.items() if v >= support], key=lambda x: x[1], reverse=True)
	#frequent_1 = [x[0] for x in itemsets_1]
	#frequent_list = frequent_1
	itemsets_dict = get_dict_from_frequentitem(itemsets_list)
	answers = []
	#print(frequent_list)
	for pair in itertools.combinations(frequent_list,2):
		pair = list(pair)
		if type(pair[0]) is not tuple:
			pair[0] = tuple_wrapper(pair[0])
		if type(pair[1]) is not tuple:
			pair[1] = tuple_wrapper(pair[1])
		if abs(len(pair[0]) - len(pair[1]))== 1:
			if len(pair[0])>len(pair[1]):
				a = pair[0]
				b = pair[1]
			else:
				a = pair[1]
				b = pair[0]
			if set(b).issubset(set(a)):
				movie_j = list(set(a) - set(b))[0]
				interests = itemsets_dict[a]/itemsets_dict[b] - itemsets_dict[tuple_wrapper(movie_j)]/total_baskets
				if interests>= interest:
					associate = sorted(list(set(b)))
					answer = [associate, movie_j,interests ,itemsets_dict[a]]
					answers.append(answer)
	answers_sorted = sorted(answers, key=lambda x: (-abs(x[2]), -x[3], x[0], x[1]))
	#print(answers_sorted)

	with open(output,'w') as f:
		json.dump(answers_sorted,f)

	
