import json

import joblib
import pandas as pd
import plotly
from flask import Flask
from flask import render_template, request
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar
from plotly.graph_objs import Scatter
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)
report = pd.read_sql_table('TestResults', engine).set_index('index')

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals

    # 10 most imbalanced categories
    categories = df.columns[4:]
    pos_class = df[categories].mean().sort_values()[:10]*100
    sorted_categories = pos_class.index[:10]

    # Correlation of class imbalance and f1 score
    f1 = report.loc[categories, 'f1-score'].sort_index()
    imbalance = df[categories].mean().apply(lambda x: max(x, 1 - x)/min(x, 1 - x)).sort_index()

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=sorted_categories,
                    y=pos_class
                )
            ],

            'layout': {
                'title': '10 most imbalanced categories',
                'yaxis': {
                    'title': "% of the messages"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },

        {
            'data': [
                Scatter(
                    x=imbalance,
                    y=f1,
                    mode='markers',
                    hovertext=imbalance.index


                )
            ],

            'layout': {
                'title': 'Correlation of category imbalance and f1 test score',
                'yaxis': {
                    'title': "f1 test score"
                },
                'xaxis': {
                    'title': "imbalance (majority class/minority class)"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
