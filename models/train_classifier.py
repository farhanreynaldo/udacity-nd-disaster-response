import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.multioutput import MultiOutputClassifier
import pickle

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def load_data(database_filepath):
    """
    Load clean data from database.

    Parameters
    ----------
    database_filepath : str
        database path to clean dataframe.

    Returns
    -------
    X : pd.Series
        Feature input for model.
    Y : pd.DataFrame
        Target for model.
    category_names : list
        Target class name.
    """
    conn = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql('DisasterResponse', con=conn)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()
    return X, Y, category_names

def tokenize(text):
    """
    Create clean token from messages as 
    feature engineering.

    Parameters
    ----------
    text : str
        Messages to be tokenized.

    Returns
    -------
    clean_tokens : list
        clean tokens from text.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
 
    return clean_tokens

def build_model():
    """
    Build model pipeline with hyperparameter tuning.

    Parameters
    ----------
    None

    Returns
    -------
    cv : GridSearchCV
        Model object from pipeline and 
        hyperparameter tuner.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),
    
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # specify parameters for grid search
    parameters = {
        'clf__estimator__n_estimators': [50],
        'clf__estimator__learning_rate': [1]
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model by f1 score, precision and recall 
    for each category.

    Parameters
    ----------
    model : GridSearchCV
        Model object from pipeline and 
        hyperparameter tuner.
    X_test : pd.Series
        Feature from test data.
    Y_test : pd.DataFrame
        Target from test data.
    category_names : list
        Messages to be tokenized.

    Returns
    -------
    None
    """
    Y_prediction = model.predict(X_test)
    Y_prediction_df = pd.DataFrame(Y_prediction, columns=category_names)
    
    for col in category_names:
        print(f"category:{col}")
        print(classification_report(Y_test[col], Y_prediction_df[col]))
        print('------------------------------------------------------')
    
    accuracy = (Y_prediction == Y_test).mean().mean()
    print(f"Accuracy: {accuracy:.2%}")

def save_model(model, model_filepath):
    """
    Save model to filepath as pickle.

    Parameters
    ----------
    model : GridSearchCV
        Model object from pipeline and 
        hyperparameter tuner.
    model_filepath : str
        Path to model output.

    Returns
    -------
    None
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()