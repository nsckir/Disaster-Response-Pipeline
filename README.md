# Disaster Response Pipeline Project

This project uses supervised machine learning to predict the categories of messages during disasters.
The dataset contains approx. 26k messages and 36 categories.
Each message is labeled by hand whether it falls in each of the categories. 

In the first step the data is read from csv files, cleaned and stored in a sqlite database. 

In the second step text feature extraction such as TF-IDF is used to create a training set
for a machine learning algorithm. The parameters of the machine learning pipeline are optimized using
grid search. 

Many of the categories are highly imbalanced. To account for this the `class_weight`  and `scale_pos_weight`
parameters of the logistic regression and XGBoost are used, respectively.
During cross validation the metric `f1_macro` is used to select the best estimator.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
