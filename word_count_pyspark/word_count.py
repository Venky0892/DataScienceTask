# Import the library 

# import findspark
# findspark.init()
import pyspark
from pyspark.sql import SparkSession
import argparse
import os 
import pandas as pd

os.environ['PYSPARK_PYTHON'] = "./environment/bin/python"


def filtering_data(load_file:str, filter_word:str) -> None:
    """
    Given a file_path to filter the line starts with 'BG:' and write it to a new file location - filtered_path 
    for furthur processing.
    """
    writeFile = open('updated_.txt', 'w')
    with open(load_file, encoding = "ISO-8859-1") as f:
        for line in f:
            try:
                if not (line.startswith(filter_word)): 
                    writeFile.write(line)
            except UnicodeDecodeError:
                pass

def word_count(text_file):
    """
    Transformation operation that flattens the RDD/DataFrame (array/map DataFrame columns) 
    after applying the function on every element and returns a count of words.
    """
    counts = text_file.flatMap(lambda line: line.split(" ")) \
                            .map(lambda word: (word, 1)) \
                           .reduceByKey(lambda x, y: x + y)

    return counts
    

def create_df(data_df: list):
    """
    The input contain list of tuples and returns a dataframe
    
    Args:
        data_df: List of tuple containing words and their counts

    Returns:
       df:  A panda's dataframe for future transformation
    """
    df = pd.DataFrame(data_df, columns=['word', 'count'])

    return df

def create_csv(df: pd.DataFrame, count_location:str):
    df.to_csv(count_location + "word_count.csv")

    
def main(args):
    """
    Main function. It starts the spark session and loads the datasets and 
    counts the number of words occurance with only line starting with 'BG'

    Args:
        args (argparse.Namespace): arguments parsed from the command line
    """

    

    filtering_data(args.load_file, args.filter_word)

    #Loading the data file
    text_file = sc.textFile('updated_.txt')

    #counting the no of word occurrences
    total_occurence = word_count(text_file)

    # Printing the words and their counts
    output = total_occurence.collect()

    #Creating a list of tuples for words and occurence 
    data_df = list()
    for (word, count) in output:
        data_df.append((word, count))
        print("%s: %i" % (word, count))

    # Creating a dataframe and saving the result in a CSV
    df = create_df(data_df)

    create_csv(df, args.count_location)


if __name__ == "__main__":

    """
    Starting a spark cluster
    """

    # Create Spark session and context
    spark = spark = SparkSession.builder\
                    .master("local")\
                    .appName('word_count')\
                    .getOrCreate()
    sc=spark.sparkContext
    parser = argparse.ArgumentParser(
        description= "Finding the number of words in the dataset"
    )

    parser.add_argument(
        "--load_file",
        type = str,
        default= 'biographies.txt',
        help= "File location"
    )


    parser.add_argument(
        "--filter_word",
        type = str,
        default= 'BG:',
        help = "Filter word"
    )

    parser.add_argument(
        "--count_location",
        type = str,
        help= "result location path"
    )

    args = parser.parse_args()
    main(args)


