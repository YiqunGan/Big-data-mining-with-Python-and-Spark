import math
import os
import numpy as np
import pandas as pd
import scipy.sparse 
import sys
import itertools
import json


def tokenize(movies):

#   add a new column to the movies, which contains lists of words from genres
    tokenlist=[]
    for index,row in movies.iterrows():
        tokenlist.append(tokenize_string(row.genres))
    movies['tokens']=tokenlist
    return tokenlist

def tokenize_string(my_string):
#     split by |
    return my_string.split('|');

def get_shingle_hash(hash_size, tokens_list):
    # The hash table could be stored as a sparse matrix. However, I did a few benchmark, they are not as efficient as store non-zero entries in a list
    # So keep use the same layout as in q1
    shingles = [] #list of set

    for tokens in tokens_list:

        shingle = set()
        for token in tokens:
            #frozenset to make it order independent and hashable
            
            token_hash = hash(token) % hash_size

            shingle.add(token_hash)
        shingles.append(shingle)

    # return None to keep the same signature as q1
    return shingles
def get_shingle_dict(tokens_list):
    #db mappings each shingle to an unique index. shingle->index
    db = {}
    shingles = [] #list of set

    for tokens in tokens_list:
        
        shingle = set()
        for token in tokens:
            #frozenset to make it order independent and hashable

            #same idea in week2, get_item_dict
            if token not in db:
                db[token] = len(db)

            shingle.add(db[token])
        shingles.append(shingle)

    return shingles, db

def get_hash_coeffs(br):
    #hash(x) = (a*x + b) % c
    #a, b are random integers less than maximum value of x.
    #Here I choose a, b in range [0, 2**10). Because x is in range [0, 2**20), this choice of a, b keep a*x+b inside the range of int32 [0, 2**32), be more efficient. In python you don't need to worry about overflow, in language like C, exceeding the range of int32 will cause overflow.
    #c is a prime number greater than 2**20. Look it up from http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
    rnds = np.random.choice(2**10, (2, br), replace=False)
    c = 1048583
    return rnds[0], rnds[1], c

def min_hashing(shingles, hash_coeffs, br):
    count = len(shingles)

    (a, b, c) = hash_coeffs
    a = a.reshape(1, -1)
    M = np.zeros((br, count), dtype=int) #Its layout same as slide 56. col are docs, row are signature index
    
    for i, s in enumerate(shingles):
        row_idx = np.asarray(list(s)).reshape(-1, 1)
        m = (np.matmul(row_idx, a) + b) % c
        m_min = np.min(m, axis=0) #For each hash function, minimum hash value for all shingles
        M[:, i] = m_min

    return M

def LSH(M, b, r, band_hash_size):
    count = M.shape[1]

    bucket_list = []
    for band_index in range(b):
        # The hash table for each band is stored as a sparse matrix! Learn basics about sparse matrix here: https://docs.scipy.org/doc/scipy/reference/sparse.html
        # However, I did a few benchmark, lil_matrix, dok_matrix are claimed to be efficient for incremental consturction, they are not as efficient as store indices of non-zero entries in a list
        # So instead of having a matrix, just two arrays to store the indices
        # But we need to image there is a matrix. Its layout same as slide 56. Cols are documents, rows are hash of signature index
        row_idx = []
        col_idx = []

        row_start = band_index * r
        for c in range(count):
            v = M[row_start:(row_start+r), c]
            v_hash = hash(tuple(v.tolist())) % band_hash_size
            row_idx.append(v_hash)
            col_idx.append(c)

        # It's a binary matrix. Set to True at these indices.
        data_ary = [True] * len(row_idx)

        # Convert to row based sparse matrix for fast processing later
        m = scipy.sparse.csr_matrix((data_ary, (row_idx, col_idx)), shape=(band_hash_size, count), dtype=bool)
        bucket_list.append(m)

    return bucket_list

def find_similiar(shingles, movies_index, query_index, threshold, bucket_list, M, b, r, band_hash_size):
    # Step 1: Find candidates
    candidates = set()

    for band_index in range(b):
        row_start = band_index * r
        v = M[row_start:(row_start+r), query_index]
        v_hash = hash(tuple(v.tolist())) % band_hash_size

        m = bucket_list[band_index]
        bucket = m[v_hash].indices #Sparse sparse matrix method: get indices of nonzero elements
        #print(f'Band: {band_index}, candidates: {bucket}')
        candidates = candidates.union(bucket)

        intersect = candidates & set(movies_index)

    #print(f'Found {len(intersect)} candidates')

    # Step 2: Verify similarity of candidates
    sims = {}
    # Since the candidates size is small, we just evaluate it on k-shingles matrix, or signature matrix for greater efficiency
    query_set = shingles[query_index]
    for col_idx in intersect:
        col_set = shingles[col_idx]

        sim = len(query_set & col_set) / len(query_set | col_set) # Jaccard Similarity
        if sim >= threshold:
            sims[col_idx] =sim

    return sims

def print_similar_items(sims, query_index, threshold):
    print(f'Reviews similiar to review #{query_index} with similarity greater than {threshold}')
    for sim in sims:
        print(sim)

    print(f'Total: {len(sims)}')


if __name__ == '__main__':

    movies_filename = sys.argv[1]
    ratings_train_filename = sys.argv[2]
    ratings_test_filename = sys.argv[3]
    output_filename = sys.argv[4]

    movies = pd.read_csv(movies_filename)

    movies_tokenized = tokenize(movies)
    hash_size = 2**20
    shingles = get_shingle_hash(hash_size,movies_tokenized)

    b = 30
    r = 3
    br = b*r

    band_hash_size = 2**16

    hash_coeffs = get_hash_coeffs(br)

    M = min_hashing(shingles, hash_coeffs, br) #col are docs, row are signature index
    bucket_list = LSH(M, b, r, band_hash_size) #list of sparse matrix
    #print(M)
    threshold = 0.3
    query_index = 5



    ratings_train = pd.read_csv(ratings_train_filename, encoding = 'utf8')
    ratings_test = pd.read_csv(ratings_test_filename, encoding = 'utf8')

    result = []

    movies_dict = {}
    for index, row in movies.iterrows():
        movies_dict[row['movieId']] = index
    user_id = -1

    result = []
    for index, row in ratings_test.iterrows():
        #print(row['userId'])

        if user_id != row['userId']:
            movies_list = list(ratings_train.loc[ratings_train['userId']== row['userId']]['movieId'])
            movies_rating_list = list(ratings_train.loc[ratings_train['userId']== row['userId']]['rating'])
        #print(index)

        query_index = movies_dict[row['movieId']]

        movies_index = [ movies_dict[movies_id] for movies_id in movies_list]
        #print(movies_index)
        #print(shingles)
        sims = find_similiar(shingles,movies_index, int(query_index), threshold, bucket_list, M, b, r, band_hash_size)
        #print(sims)
        sim_ratings = [ movies_rating_list[i] for i,v in enumerate(movies_list) if movies_dict[v] in sims]
        #sim_movie = [v for i,v in enumerate(movies_list) if movies_dict[v] in sims]
        #sim_jac = [sims[movies_dict[v]] for i,v in enumerate(movies_list) if movies_dict[v] in sims]
        #weight = [sim_jac[v]/sum(sim_jac) for v in range(len(sim_jac))]
        
        #print(sim_movie)
        if len(sim_ratings) == 0 :
            rating_curr = np.mean(movies_rating_list)
        else:
            #rating_curr = np.sum([sim_ratings[i]*weight[i] for i in range(len(sim_ratings))])
            rating_curr = np.mean(sim_ratings)
        if rating_curr>=0.5:
            result.append(rating_curr)
        else:
            rating_curr = 0.5
            result.append(0.5)
        user_id = row['userId']
        
        #print(rating_curr)

    predict = np.array(result)

    ratings_test.rating = predict

    #truth_rating = pd.read_csv('ml-latest-small/ratings_test_truth.csv')

    #truth = np.array(truth_rating.rating)

    #error = np.square(np.subtract(predict , truth)).mean()

    #print(error)

    ratings_test.to_csv(output_filename)






