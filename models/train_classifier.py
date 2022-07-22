import pickle
import re
import sys

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from xgboost import XGBClassifier

url_regex = r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'
url_regex_comp = re.compile(url_regex, re.IGNORECASE)

letters_regex = r'[^a-zA-Z0-9]'
letters_regex_comp = re.compile(letters_regex, re.IGNORECASE)

nltk.download(['punkt',
               'wordnet',
               'averaged_perceptron_tagger',
               'omw-1.4',
               'stopwords'])

np.random.seed(42)


def load_data(database_filepath):
    """Loads the data  from a sqlite database

    Args:
        database_filepath: path of the sqlite database

    Returns:
        X: features (numpy array)
        Y: target categories (numpy array)
        categories: names of categories (list)
    """
    engine = create_engine(''.join(['sqlite:///', database_filepath]))
    df = pd.read_sql_table('DisasterResponse', engine)

    categories = df.columns[4:]

    X = df['message'].values
    Y = df[categories].values

    return X, Y, categories


def tokenize(text):
    """Text tokenizer. Removes urls, punctuation.

    Args:
        text: text (string)

    Returns:
        tokens: clean tokens (list)
    """

    text = url_regex_comp.sub("urlplaceholder", text)
    text = letters_regex_comp.sub(" ", text)

    tokens = word_tokenize(text)
    tokens = [WordNetLemmatizer().lemmatize(tok).lower().strip() for tok in tokens]

    return tokens


def build_model():
    """Builds a predictive model using pipeline and grid search.

    Args:

    Returns:
        cv: grid search object (sklearn.model_selection.GridSearchCV)

    """
    params = [
        {
            'vect__max_df': [0.5, 1.0],
            # 'vect__max_features': [500, 1500, 3000],
            # 'vect__min_df': [1, 5, 10],

            'clf__estimator': [LogisticRegression()],
            'clf__estimator__class_weight': ['balanced'],
            # 'clf__estimator__C': [0.1, 1, 10],
            'clf__estimator__n_jobs': [-1],
            'clf__estimator__max_iter': [1000],
        },

        {
            # 'vect__max_df': [0.5, 1.0],
            # 'vect__max_features': [500, 1500, 3000],
            # 'vect__min_df': [1, 5, 10],

            'clf__estimator': [XGBClassifier()],
            'clf__estimator__scale_pos_weight': [10],
            'clf__estimator__n_jobs': [-1],
            'clf__estimator__objective': ['binary:logistic'],
            'clf__estimator__eval_metric': ['auc'],
        },
    ]

    stop_words = set(tokenize(' '.join(stopwords.words('english'))))

    count_vectorizer = CountVectorizer(tokenizer=tokenize, stop_words=stop_words)

    tf_idf = TfidfTransformer(smooth_idf=True)

    pipeline = Pipeline([
        ('vect', count_vectorizer),
        ('tf_idf', tf_idf),
        ('clf', MultiOutputClassifier(LogisticRegression()))
    ])

    cv = GridSearchCV(pipeline,
                      param_grid=params,
                      scoring='f1_macro',
                      n_jobs=-1
                      )
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Prints out the results of the model evaluation

    Args:
        model: trained model
        X_test: test set features (numpy array)
        Y_test: test set targets (numpy array)
        category_names: category names (list)

    Returns:

    """
    Y_pred = model.predict(X_test)

    print(classification_report(Y_test, Y_pred, target_names=category_names))
    report = classification_report(Y_test, Y_pred, target_names=category_names, output_dict=True)
    report_df = pd.DataFrame.from_dict(report, orient='index')

    return report_df


def save_results(results_df, database_filename):
    """Saves test results to a sqlite database

    Args:
        results_df: test results. (pandas.Dataframe)
        database_filename: path to the sqlite database

    Returns:
    """

    engine = create_engine(''.join(['sqlite:///', database_filename]))
    results_df.to_sql('TestResults', engine, index=True, if_exists='replace')


def save_model(model, model_filepath):
    """Prints out the results of the model evaluation

    Args:
        model: trained model
        model_filepath: path for the model

    Returns:

    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    """Loads data from sqlite database, create train and test set, trains a model on the train set,
    evaluates it on the test set and saves the trained model.

    Args:

    Returns:

    """
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
        report_df = evaluate_model(model, X_test, Y_test, category_names)

        print('Saving results...\n    DATABASE: {}'.format(database_filepath))
        save_results(report_df, database_filepath)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
