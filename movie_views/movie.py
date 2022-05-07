import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def data_read(load_file):
    """
    Args:
        load_file : Input File location
    """

    return pd.read_csv(load_file) # Reading the csv file which contains info about movie

def preprocessing_data(data):
    """
    Args:
       Data : Dataframe to be processed and updating the values.
    """

    # Creating a column Views to update the given value 
    data["views"] = ""

    # Update the given values to the new columns \
    data.loc[data['Title'] == 'Forest Gump', 'views'] = 10000000
    data.loc[data['Title'] == 'The Usual Suspects', 'views'] = 7500000
    data.loc[data['Title'] == 'Rear Window', 'views'] = 6000000
    data.loc[data['Title'] == 'North by Northwest', 'views'] = 4000000
    data.loc[data['Title'] == 'The Secret in Their Eyes', 'views'] = 3000000
    data.loc[data['Title'] == 'Spotlight', 'views'] = 1000000

    # Preprocessing the data 
    df = data.replace(',', '', regex = True) # Removing the commas to make str object to int
    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True) # Replacing Nan values to empty cells 

    return df 

def creatingdf(df) -> pd.DataFrame:
    """
    Args:
        Non_nan_df : Dataframe
    """

    # Create a seperate Dataframe to store values which has Nan
    non_nan_df = df[df.isna().any(axis=1)] # Nan dataframe to predict

    # Excluding the Nan values 
    new = df.dropna() 

    return new, non_nan_df


def dep_ind_variable(new):
    """
    Args:
        new : Dataframe to be processed to split into dependant and independant variables  
    """
    y = new['views']  # Dependant Variable  
    x = new.drop(['ID', 'Title', 'Year', 'views'], axis=1) # Independant variables
    x = x.apply(pd.to_numeric, errors='ignore') # Changing all the values to numeric 
    return x, y

def linear_regression(x, y, non_nan_df):
    """
    Args:
        X : Independant Variable 
        Y : Dependant Variable 
        Non_nan_df : Dataframe which contains the views to be estimated 
    """
    # Linear regression
    print(x)
    print(y)
    reg = LinearRegression().fit(x, y)
    Id_list = non_nan_df['ID'].to_list()
    

    # Storing id of movies to estimate it views 
    df_pred = pd.DataFrame(columns = ['ID','Title','Year', 'Rating', 'Rating_count', 'views', 'Linear_reg_views'])
    for i in Id_list:
        test1 = non_nan_df[(non_nan_df['ID'] == i)]
        test = test1[['Rating ', 'Rating_count']].apply(pd.to_numeric, errors='ignore')
        test1['Linear_reg_views'] = int(reg.predict(test))
        df_pred.loc[i] = test1.values[0]

    return df_pred

def XGB_regressor(x, y, non_nan_df):
    
    """
    Args:
        X : Independant Variable 
        Y : Dependant Variable 
        Non_nan_df : Dataframe which contains the views to be estimated 
    """

    # XGBoost Regressor 
    xg_reg = xgb.XGBRegressor(objective ='reg:linear', n_estimators=1)
    xg_reg.fit(x,y)
    Id_list = non_nan_df['ID'].to_list()

    # Storing id of movies to estimate it views 
    df_pred = pd.DataFrame(columns = ['ID','Title','Year', 'Rating', 'Rating_count', 'views', 'XG_Boost_views'])
    for i in Id_list:
        test1 = non_nan_df[(non_nan_df['ID'] == i)]
        test = test1[['Rating ', 'Rating_count']].apply(pd.to_numeric, errors='ignore')
        test1['XG_Boost_views'] = int(xg_reg.predict(test))
        df_pred.loc[i] = test1.values[0]
    
    return df_pred

def random_forest_regression(x, y, non_nan_df):

    """
    Args:
        X : Independant Variable 
        Y : Dependant Variable 
        Non_nan_df : Dataframe which contains the views to be estimated 
    """
    #Random Forest regressor
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(x, y)
    Id_list = non_nan_df['ID'].to_list()

    # Storing id of movies to estimate it views 
    df_pred = pd.DataFrame(columns = ['ID','Title','Year', 'Rating', 'Rating_count', 'views', 'Random_Forest_views'])

    for i in Id_list:
        test1 = non_nan_df[(non_nan_df['ID'] == i)]
        test = test1[['Rating ', 'Rating_count']].apply(pd.to_numeric, errors='ignore')
        test1['Random_Forest_views'] = int(regr.predict(test))
        df_pred.loc[i] = test1.values[0]

    return df_pred

def model(x, y, non_nan_df, model_type = str) -> pd.DataFrame:

    """
    Args:
        X : Independant Variable 
        Y : Dependant Variable 
        Model_type : Type of model to be appplied
    
    """
    # applying linear regression 
    if model_type == 'linear_regression':
        df_pred = linear_regression(x, y, non_nan_df)
    
    if model_type == 'Xgboost_regression':
        df_pred = XGB_regressor(x, y, non_nan_df)

    if model_type == 'Random_forest_reg':
        df_pred = random_forest_regression(x, y, non_nan_df)

    if model_type == 'all':
        reg = LinearRegression().fit(x, y)
        xg_reg = xgb.XGBRegressor(objective ='reg:linear', n_estimators=1)
        xg_reg.fit(x,y)
        regr = RandomForestRegressor(max_depth=2, random_state=0)
        regr.fit(x, y)
        Id_list = non_nan_df['ID'].to_list()
        # Storing id of movies to estimate it views 
        df_pred = pd.DataFrame(columns = ['ID','Title','Year', 'Rating', 'Rating_count', 'views', 'Linear_reg_views', 'XG_Boost_views','Random_Forest_views'])
        for i in Id_list:
            test1 = non_nan_df[(non_nan_df['ID'] == i)]
            test = test1[['Rating ', 'Rating_count']].apply(pd.to_numeric, errors='ignore')
            test1['Linear_reg_views'] = int(reg.predict(test))
            test1['XG_Boost_views'] = int(xg_reg.predict(test))
            test1['Random_Forest_views'] = int(regr.predict(test))
            df_pred.loc[i] = test1.values[0]

    return df_pred  

def main(args):
    """
    Main function. It loads the datasets and run the required model
    to pred to outcome

    Args:
        args (argparse.Namespace): arguments parsed from the command line
    """
    # Read file 
    data = data_read(args.load_file)

    df =  preprocessing_data(data)

    new, non_nan_df = creatingdf(df)

    x, y = dep_ind_variable(new)

    df_pred = model(x, y, non_nan_df, args.model_type)

    # Storing the final estimated value in pred.csv
    df_pred.to_csv( ROOT / "Pred.csv") 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description= "Estimation of movie views "
    )

    parser.add_argument(
        "--load_file",
        type = str,
        default= ROOT / "Top_250_Movie.csv",
        help= "Loading the default file location of this experiment"
    )

    parser.add_argument(
        "--model_type",
        type = str,
        default='all',
        choices=['linear_regression', 'Xgboost_regression', 'Random_forest_reg', 'all'],
        help=' linear_regression, Xgboost_regression, or Random_forest_reg (default: %(default)s)'
    )
    args = parser.parse_args()
    main(args)

