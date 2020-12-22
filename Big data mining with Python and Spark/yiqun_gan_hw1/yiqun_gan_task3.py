import json
import sys
from pyspark import SparkContext

if __name__ == '__main__':

    tweets_path = sys.argv[1]
    output_path = sys.argv[2]
    sc = SparkContext("local","PySpark Tutorial")
    tweets= sc.textFile(tweets_path)

    chunk_count = tweets.count()

    words = tweets.flatMap(lambda x:x.split(" "))

    max_word = [0]*2
    frequent_words = words.map(lambda word:(word, 1)).reduceByKey(lambda a, b: a+b).sortBy(lambda x:x[1],False).take(5)
    max_word[0] = frequent_words[0][0]
    max_word[1] = frequent_words[0][1]

    mindless = words.map(lambda word:(word, 1)).reduceByKey(lambda a, b: a+b).filter(lambda x:x[0]=='mindless').take(1)
    mindless_count = mindless[0][1]

    output = {
        "max_word": max_word,
        "mindless_count": mindless_count,
        "chunk_count": chunk_count
    }

    with open(output_path,'w') as f:
        json.dump(output, f)
