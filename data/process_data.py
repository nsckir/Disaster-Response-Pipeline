import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads the messages and categories from csv files

    Args:
        messages_filepath: path of the csv file with messages (string)
        categories_filepath: path of the csv file with categories (string)

    Returns:
        df: merged dataframe (pandas.Dataframe)
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """Cleans the data

    Args:
        df: merged dataframe with messages and  categories (pandas.Dataframe)

    Returns:
        df: cleaned dataframe. (pandas.Dataframe)
    """

    # split each category in a column and extract column names
    categories = df['categories'].str.split(';', expand=True)
    row = categories.loc[0].str.split('-')
    category_colnames = [x[0] for x in row]
    categories.columns = category_colnames
    print(f'Categories has {categories.shape[0]} rows and {categories.shape[1]} columns')

    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

        # drop constant columns
        if categories[column].nunique() == 1:
            categories.drop(column, axis=1, inplace=True)
            print(f'Dropping category {column} because it is constant')

    print(f'Categories has {categories.shape[0]} rows and {categories.shape[1]} columns')

    # Category 'related' contains 0, 1 and 2 whereas all other 35 categories are binary.
    # f1_macro score in GridSearchCV can't deal with multi target multiclass variables, therefore we will transform
    # 'related' to one hot encoded
    one_hot = []
    for column in categories.columns:
        if categories[column].nunique() > 2:
            print(f'Creating one hot encoded columns for category {column} ')
            one_hot.append(column)

    categories = pd.get_dummies(categories, columns=one_hot, drop_first=True, prefix=one_hot)
    print(f'Categories has {categories.shape[0]} rows and {categories.shape[1]} columns')

    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)

    print(f'Dropping {df.duplicated().sum()} duplicates')
    df = df.drop_duplicates()
    print(f'Dropping {df[categories.columns].isna().sum(axis=1).sum()} rows with missing categories')
    df = df.dropna(subset=categories.columns)
    print(f'Final dataframe has {df.shape[0]} rows and {df.shape[1]} columns')

    return df


def save_data(df, database_filename):
    """Saves the cleaned data to a sqlite database

    Args:
        df: cleaned dataframe. (pandas.Dataframe)
        database_filename: path to the sqlite database

    Returns:
    """

    engine = create_engine(''.join(['sqlite:///', database_filename]))
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    """Loads messages and categories from csv files, cleans the data and saves it to a sqlite database

    Args:

    Returns:
    """

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' 
              'datasets as the first and second argument respectively, as ' 
              'well as the filepath of the database to save the cleaned data ' 
              'to as the third argument. \n\nExample: python process_data.py ' 
              'disaster_messages.csv disaster_categories.csv ' 
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
