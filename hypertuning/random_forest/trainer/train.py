# Project imports
import argparse

import hypertune
import pandas as pd
# GKE imports
from google.cloud import storage
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Create the argument parser for each parameter plus the job directory
parser = argparse.ArgumentParser()

parser.add_argument(
    '--job-dir',  # Handled automatically by AI Platform
    help='GCS location to write checkpoints and export models',
    required=True
)
parser.add_argument(
    '--n_estimators',  # Specified in the config file
    help='Number of trees in forest',
    default=100,
    type=int
)
parser.add_argument(
    '--criterion', # Specified in the config file
    help='Function to measure the quality of a split',
    default='gini',
    type=str
)
parser.add_argument(
    '--max_features',  # Specified in the config file
    help='Max number of features per tree',
    default="auto",
    type=str
)
parser.add_argument(
    '--max_depth',  # Specified in the config file
    help='Max depth per tree.',
    default=1000,
    type=int
)
parser.add_argument(
    '--min_samples_split',  # Specified in the config file
    help='Minimum number of samples required to split an internal node',
    default=2,
    type=int
)
parser.add_argument(
    '--min_samples_leaf',  # Specified in the config file
    help='Minimum number of samples required to be at a leaf node',
    default=1,
    type=int
)
parser.add_argument(
    '--bootstrap',  # Specified in the config file
    help='The penalty (aka regularization term) to be used',
    default=True,
    type=bool
)

args = parser.parse_args()

bucket = storage.Client().bucket("ml-final-314816-aiplatform")
blob = bucket.blob('train.csv')
blob.download_to_filename('train.csv')

with open('./train.csv', 'r') as df_train:
    X = pd.read_csv(df_train, index_col=0)

blob = bucket.blob('labels.csv')
blob.download_to_filename('labels.csv')
with open('./labels.csv', 'r') as df_labels:
    y = pd.read_csv(df_labels, index_col=0)

y = y.loc[X.index].cancelation

model = RandomForestClassifier(n_estimators=args.n_estimators,
                               max_features=args.max_features,
                               max_depth=args.max_depth,
                               min_samples_split=args.min_samples_split,
                               min_samples_leaf=args.min_samples_leaf,
                               bootstrap=args.bootstrap)

score = cross_val_score(model, X, y, cv=5).mean()

hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='accuracy',
    metric_value=score,
    global_step=1000
)

model_filename = 'model.joblib'
joblib.dump(model, model_filename)

job_dir = args.job_dir.replace('gs://', '')

bucket_id = job_dir.split('/')[0]
bucket_path = job_dir.lstrip(f'{bucket_id}/')
bucket = storage.Client().bucket(bucket_id)
blob = bucket.blob(f'{bucket_path}/{model_filename}')
blob.upload_from_filename(model_filename)
