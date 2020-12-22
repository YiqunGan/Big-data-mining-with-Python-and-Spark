import math
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from numpy.linalg import norm
import sys

def MSE(R,P,Q,lamda):
    ratings = R.data
    count = len(ratings)
    err = 0 
    for i in range(count):
        rating=ratings[i]
        r = R.row[i]
        c = R.col[i]
        if rating>0:
            err+=pow(rating-np.dot(P[r,:],Q[:,c]),2)+lamda*(pow(norm(P[r,:]),2)+pow(norm(Q[:,c]),2))
    mse = err/count
    return mse

#R = P * Q, k components
def SGD(R, P, Q, K, cof,l,steps):
       
    for step in range(steps):
        for i in range(len(R.data)):
            rating=R.data[i]
            r = R.row[i]
            c = R.col[i]
            if rating>0:
                err=rating-np.dot(P[r,:],Q[:,c])
                P[r,:]+=2*cof*(err*Q[:,c]-l*P[r,:])
                Q[:,c]+=2*cof*(err*P[r,:]-l*Q[:,c])
        mse = MSE(R,P,Q,l)
        if mse<0.8:
            break
    return P,Q


if __name__ == '__main__':

    movies_filename = sys.argv[1]
    ratings_train_filename = sys.argv[2]
    ratings_test_filename = sys.argv[3]
    output_filename = sys.argv[4]

    ratings_train = pd.read_csv(ratings_train_filename, encoding = 'utf8')
    ratings_test = pd.read_csv(ratings_test_filename, encoding = 'utf8')

    ratings_matrix = ratings_train.pivot(index = 'userId', columns = 'movieId',values = 'rating').fillna(0)

    users = list(ratings_matrix.index)
    movies = list(ratings_matrix.columns)

    user_ratings_mean = np.mean(ratings_matrix, axis = 1)
    movie_ratings_mean = np.mean(ratings_matrix, axis = 0)

    R = coo_matrix(ratings_matrix.values)
    M,N=R.shape
    K=6
    P=np.random.rand(M,K)
    Q=np.random.rand(K,N)

    P,Q=SGD(R,P, Q, K=6,cof=0.0007,l=0.02, steps=100)

    all_ratings = np.matmul(P,Q)
    all_ratings_df = pd.DataFrame(all_ratings, columns = movies, index = users)

    user_mean_ratings = {}
    movie_mean_ratings = {}
    for index, row in ratings_test.iterrows():
            
        # find the movieId that train user_id equals the test user_id 
        #mlist = list(ratings_train.loc[ratings_train['userId'] == row['userId']]['movieId'])

        # movies rating by that user
        mrlist = list(ratings_train.loc[ratings_train['userId'] == row['userId']]['rating'])
        if row['movieId'] not in movie_mean_ratings:
            urlist = list(ratings_train.loc[ratings_train['movieId']== row['movieId']]['rating'])

        user_mean_ratings[row['userId']] = np.mean(mrlist)
        if len(urlist)== 0:
            movie_mean_ratings[row['movieId']] = 0
        else:
            movie_mean_ratings[row['movieId']] = np.mean(urlist)
    result = []
    for index, row in ratings_test.iterrows():
        user_id = int(row['userId'])
        movie_id = int(row['movieId'])
        #+ user_ratings_bias[user_id-1] +movie_bias[movie_id]
        try:
            pred = all_ratings_df.loc[user_id][movie_id]
            #pred = rating_global_mean + all_ratings_df.loc[user_id][movie_id]
        except KeyError:
            pred = user_mean_ratings[user_id]
        
        result.append(pred)
        #print(pred)

    predict = np.array(result)
    #truth_rating = pd.read_csv('ml-latest-small/ratings_test_truth.csv')
    #truth = np.array(truth_rating.rating)
    #error = np.square(np.subtract(predict , truth)).mean()
    #print(error)

    ratings_test.rating = predict

    ratings_test.to_csv(output_filename)
