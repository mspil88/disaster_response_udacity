import sys
import re
from collections import defaultdict
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sqlalchemy import inspect
from sqlalchemy import create_engine
import pickle
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet'])
import numpy as np

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    data = pd.read_sql_table("message_categories", engine)
    X = data.message
    y = data.drop(['id', 'message', 'original', 'genre'], axis=1)  
    return X, y

def tokenize(text):
    
    stops = stopwords.words("english")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    lemmatised_tokens = [lemmatizer.lemmatize(i).lower().strip() for i in tokens if i not in stops]

    return lemmatised_tokens

def metrics_dataframe(y_true, y_pred):
    
    df = pd.DataFrame(columns = ['metric', 'precision', 'recall', 'f1_score'])
    container = []
    y_pred = pd.DataFrame(y_pred, columns = y_true.columns)
    
    for i in y_true.columns:
        container.append([j.split('      ') for j in classification_report(y_true = y_true.loc[:,i], 
                                                                           y_pred = y_pred.loc[:,i]).split('\n')])

    for i in container:
        for j in i:
            if 'avg / total' in j:
                df = df.append({'metric':j[0], 'precision': float(j[1]), 'recall': float(j[2]), 
                                'f1_score': j[3]}, ignore_index=True)
    
    df['f1_score'] = df['f1_score'].apply(lambda x: float(x.split()[0]))
    
    return df


def average_metric(y_true, y_pred):
    
    df = pd.DataFrame(columns = ['metric', 'precision', 'recall', 'f1_score'])
    container = []
    y_pred = pd.DataFrame(y_pred, columns = y_true.columns)
    
    for i in y_true.columns:
        container.append([j.split('      ') for j in classification_report(y_true = y_true.loc[:,i], 
                                                                           y_pred =   y_pred.loc[:,i]).split('\n')])

    for i in container:
        for j in i:
            if 'avg / total' in j:
                df = df.append({'metric':j[0], 'precision': float(j[1]), 'recall': float(j[2]), 
                                'f1_score': j[3]}, ignore_index=True)
    
    df['f1_score'] = df['f1_score'].apply(lambda x: float(x.split()[0]))
    metric = df['f1_score'].mean()
    
    return metric

class MultiModelOptimiser:
    def __init__(self, pipeline_list, model_params, folds, score, n_jobs=None):
        self.pipeline = pipeline_list
        self.model_params = model_params
        self.gs_results = {}
        self.folds = folds
        self.score = score
        self.n_jobs = n_jobs
        self.gs_best_params = defaultdict(list)
        self.best_params = None
        self.best_params_per_model = None
        self.model_array = []


    def fit(self, X_train, y_train):
        #nested_fold = KFold(n_splits=5, shuffle=True, random_state=0)
        model_count = 0
        for i in self.model_params:
            clf = i['clf'][0]
            #print(clf)
            i.pop('clf')
            pipe = Pipeline(self.pipeline+[('clf', clf)])
            gs = GridSearchCV(estimator=pipe, param_grid=i, cv=self.folds, scoring=self.score, n_jobs=self.n_jobs)
            model_count +=1
            print(f'estimating model: {model_count}')
            gs.fit(X_train, y_train)
            self.gs_results[str(clf)] = gs
            self.gs_best_params[str(clf)] = [gs.best_params_, gs.best_score_]
            self.model_array.append(clf)
            #print(cross_val_score(gs, X_train, y_train, scoring='roc_auc', cv=nested_fold))


    @property
    def summarise_scores(self):
        container = []
        for i in self.gs_results:
            parameters = self.gs_results[i].cv_results_['params']
            cv_scores = []
            for j in range(self.folds):
                split_increment = f"split{j}_test_score"
                cv_scores.append(self.gs_results[i].cv_results_[split_increment].reshape(len(parameters), 1))
            combined_scores = np.hstack(cv_scores)

            for a, b in zip(parameters, combined_scores):
                container.append((i, a, np.mean(b), np.min(b), np.max(b), np.std(b)))

        df = pd.DataFrame(container,
                          columns=['estimator', 'parameters', 'mean_cv_error', 'min_cv_error', 'max_cv_error',
                                   'std_cv_error']).sort_values('mean_cv_error', ascending=False)

        grouped = df.loc[df.groupby('estimator')['mean_cv_error'].idxmax()]
        self.best_params = df[df['mean_cv_error'] == max(df['mean_cv_error'])][['estimator', 'parameters']].to_dict(orient='list')
        self.best_params_per_model = grouped[['estimator', 'parameters']]

        return df
    
    
    def refit_optimal_model(self, X_train, y_train, X_test, y_test, score):
        clf = [i for i in self.model_array if str(i) == self.best_params['estimator'][0]][0]
        params = {k: v for k, v in self.best_params['parameters'][0].items()}
        pipe = Pipeline(self.pipeline + [('clf', clf)])#
        pipe.set_params(**params)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        score = score(y_true=y_test, y_pred=y_pred)
        print(f'The optimal parameters are: {params}') 
        print(f'Test set score: {score}')


def build_model():
    average_f1 = make_scorer(average_metric)
    
    pipe = [
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer())]
    
    params = [{ 'vect__ngram_range' : ((1,1), (1,2)),
            #'vect__max_df' : (0.75, 1.0),
            'clf': [MultiOutputClassifier(RandomForestClassifier())],
            'clf__estimator__n_estimators' : [10, 20],
            'clf__estimator__min_samples_split': [2, 5]},      
             ]
    
    m = MultiModelOptimiser(pipe, params, 5, average_f1, 1)
    return m
    
def evaluate_model(model, X_train, y_train, X_test, y_test, score):
    model.refit_optimal_model(X_train, y_train, X_test, y_test, score)
    return model
    

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        X, Y = load_data(database_filepath)
        #print(X.shape)
        #print(Y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        print(model.summarise_scores)
        
        print('Evaluating model and saving final model')
        model = evaluate_model(model, X_train, y_train, X_test, y_test, average_metric)

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