import sys
import pandas as pd
from sqlalchemy import create_engine 


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge disaster data into dataframe.

    Parameters
    ----------
    messages_filepath : str
        Path to disaster messages data.
    categories_filepath : str
        Path to disaster categories data.    
    Returns
    -------
    df: pd.DataFrame
        Complete data from messages and categories.
    """
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    df = messages_df.merge(categories_df, on='id')
    return df

def clean_data(df):
    """
    Perform data cleaning on disaster data.

    Parameters
    ----------
    df : pd.DataFrame
        Combined disaster data.

    Returns
    -------
    df : pd.DataFrame
        Cleaned disaster data.
    """

    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0, :]
    category_colnames = row.str.slice(stop=-2)
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str.slice(-1)
        categories[column] = categories[column].astype(int)
    df = df.drop('categories', axis='columns')
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    df['related'] = df['related'].replace(2, 1)
    df = df.drop('child_alone', axis=1)
    return df

def save_data(df, database_filename):
    """
    Save data to a database.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned disaster data
    database_filename : str
        Path to store database
    
    Returns
    -------
    None
    """
    conn = create_engine(f"sqlite:///{database_filename}")
    df.to_sql('DisasterResponse', con=conn, if_exists='replace', index=False)

def main():
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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()