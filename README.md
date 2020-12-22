# Big-data-mining-with-Python-and-Spark
DS553 homework and project
## Assignment1:
work with two datasets (Gamergate.json and tweets).The Gamergate.json contains metadata for the gamergate twitter dataset which is a data collected from twitter.
Tasks:

Task1

A. How many tweets are in this dataset? 
B. How many unique users are in this dataset? Hint: each user has its unique user id. 
C. Identify top 3 users with most followers. 
D. Each tweet has an associated date to when it is created. Identify the number of tweets that are created on Tuesday. 

Task2

A. Each tweet has a retweet count associated to it. What is the mean retweet count for tweets in this dataset? (on average, how many retweets each tweet gets) 
B. What is the maximum retweet count? 
C. What is the standard deviation for the retweet counts? 

Task3

A. What is the most frequent word in this file with what frequency? 
B. How many times the word “mindless” appears in the file? (
C. how many tweet chunks are there in this file? 

## Assignment2:
work with MovieLens dataset. Practive with A-Prior, MinHash, Locality Sensitive Hashing and different types of recommendation systems.
Tasks:

Task1

Find association rules in baskets using A-Prior algorithms, such that |interest| >= I, and support >=S.

Task2

Build an efficient content-based recommendation system, that using LSH to find similar items.

Task3

Build a model-based recommendation system based on matrix factorization.

## Assignment3:
Tasks:

Task1

Creating Retweet Network and Analyzing it 
A. Given a json file (similar in format to Gamergate.json) create the retweet network for it and save the network as a gexf.
B. How many nodes does this network have?
C. How many edges does this network have? 
D. Which user’s tweets get the most retweets? We need the screen name of this user. 
E. What is the number of retweets the user in part D received? 
F. Whichuserretweetsthemost?Weneedthescreennameofthisuser.
G. What is the number of retweets the user in part F did? 

Task2

Community Detection
A. Partition the graph into communities that maximize the modularity objective unsupervisely.
B. For this part, you will take the two largest communities detected in part A and train a Multinomial Naïve Bayes classifier based on TFIDF features of the nodes in the detected largest communities and text used by them.
C. Train a Multinomial Naïve Bayes classifier this time using the count- vectorizer features and repeat the task in part B. Create a new txt file and report the communities in it following the same format as in part B. 

Task3

Using Gephi and Visualization 

## Assignment4
Tasks

Task1

2-way spectral graph partition on a small graph 

Task 2

4-way spectral graph partition on a small graph

Task 3

k-way spectral graph partition on a large graph 

Task 4

Node classification based on spectral embedding

Task 5 

Spectral graph partition on non-graph data

Task 6 

Identify important nodes in a graph via page rank

## Final report

Music platforms usually didn’t provide a rating system for users like movies or books. Predicting whether a song is favored by a user is mainly decided by their repetitive listening behavior. It is a classification problem as we’d like to predict if the user likes a song or not. KKBOX, Asia’s leading music streaming service, currently uses a collaborative filtering based algorithm in their recommendation system. We are interested in exploring whether hybrid collaborative filtering models or other machine learning of classification models could lead to better results in music recommendation.


For this project, we are going to build  a music recommendation with KKBox dataset with the hybrid recommendation method. For recommendation system parts, we separately use three methods, user-based, item-based, and (SVD) model based to get independent recommendation systems first, then we adopt the optimized hybrid weighted method to combine three of them to recommend. In addition, we apply machine learning to regard the recommendation as a classifier and utilize the adaboost and catboost to obtain a strong classifier. In this process, we apply many machine learning methods, such as decision tree, random forest, and XGBoost. Finally, we use different evaluation methods to evaluate the performance of our models and both methods could yield good performance which  are  significant promotions for the original single method.

