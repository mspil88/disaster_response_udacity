# disaster_response_udacity

## Project Purpose
The goal of this project is to classify disaster response messages in order to empower emergency workers to better understand the help required as each message comes through. By producing categories such as 'shelter' and 'food', derived from classification algorithms, emergency workers can better cut through the noise in order to arrive more quickly at the appropriate response.

## How
This project includes a full end to end ETL and ML pipeline to extract and process the message data which is stored in a SQLite database, and then run this data through an ML pipeline to optimise the best model and parameters based on an averaged F1 score. This modelling interface creates the ability to optimise across multiple models and associated hyperparameters all though this search was scaled back due to slow runtimes on the server. Finally the model results are encapsulated in a Flask powered interactive dashboard which enpowers emergency works to enter in messages and get fast classifications.

## Code
The project is arranged into three key folders:

### App
run.py - receives model pkl file and sqlite database as inputs.
templates - folder containing html and css files for the dashboard

### Data
process_data.py - the ETL script which injests category and message csv files, processes them and then commits to a SQLite database.

### Models
train_classifier.py - receives the processes data and passes through the ML pipeline. Key here is the collapsing of multiple F1 scores into one metric in the 'average_f1_score' function and the 'MultiModelOptimiser' class which supports the passing of multiple models and parameters in one interface.


