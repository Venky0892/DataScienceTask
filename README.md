
# DataScience Interview Tasks

#### This interview consist of three questions
    1. Movie categorization

    You are given millions of movies and a list of thousands of movie categories (names
    only e.g. “Sci Fi Movies”, “Romantic Movies”). Your task is to assign each movie to at
    least one of the movie categories. Each movie has a title, description and poster.
    ● How would you solve this problem?
    ● How would you verify its quality?
    ● How would you handle the case of adding or removing a category?
    ● How would you handle the case of adding or removing a movie?
    DELIVERABLES:
    ● Description about the chosen approach and its pros and cons (no code required)
    ● Short discussion about alternative approaches you might have considered and
    their pro and cons_

    2. Word count in PySpark

    The goal of this task is to count the words of a given dataset.
    The tasks to do are:
    a. Download the data set over which to run word count from the following link:
    https://s3.amazonaws.com/products-42matters/test/biographies.list.gz
    b. Implement a PySpark program that counts the number of occurrences of each
    word in the provided file. Only lines starting with the “BG:” should be considered,
    and a whitespace tokenizer should be used for tokenizing the text.
    DELIVERABLES:
    ● Code
    ● Documentation that explains how to run the code
    ● The result of the word count

    3. Movies view estimations in Python

    The goal of this task is to implement a model in python to estimate the number of views
    a movie has.
    You have the following data available:
    ● Movies list of 250 movies:
    https://github.com/WittmannF/imdb-tv-ratings/blob/master/top-250-movie-ratings.
    csv which contains the position of 250 movies as well as their rating count and
    other information
    ● A limited number of movie views data
    ○ Forrest Gump: 10000000 views
    ○ The Usual Suspects: 7500000 views
    ○ Rear Window: 6000000 views
    ○ North by Northwest: 4000000 views
    ○ The Secret in Their Eyes: 3000000 views
    ○ Spotlight: 1000000 views

    DELIVERABLES:
    ● Code
    ● Documentation that explains how to run the code
    ● Description about the chosen approach and its pros and cons
    ● Short discussion about alternative approaches you might have considered and
    their pro and cons


## Documentation




## Run Locally

Clone the project

```bash
  git clone https://github.com/Venky0892/DataScienceTask.git
  pip install -r requirements.txt  # install
```

## For Task 2 : Word count in PySpark

cd word_count_pyspark

```bash
python word_count.py  # Default filter word: 'BG:'
```
```bash

python word_count.py --load_file <file_location:str>, --filter_word <filter_word:str> # choose filter word if needed
```
Result of word count
```bash
cd word_count_pyspark/word_count.csv
```

Note : Should require JDK@8
## For Task 3 : Movie View Estimation

cd movie_views

```bash
python movie.py --load_file <file_location:str>, --model_type <model_name> # model_name =['linear_regression', 'Xgboost_regression', 'Random_forest_reg'], Default: It take all model and estimate views for each one of them!
```
Description about the chosen approach and its pros and cons /
Short discussion about alternative approaches and
their pro and cons

```bash
cd movie_views/ description_approch.txt
```



## 🚀 About Me
Machine Learning and Deep Learning enthusiast and aspiring Data Scientist.

