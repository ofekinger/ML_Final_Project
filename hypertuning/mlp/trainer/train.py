# Project imports
import argparse

import hypertune
import pandas as pd
# GKE imports
from google.cloud import storage
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.model_selection import cross_val_score

# Create the argument parser for each parameter plus the job directory
parser = argparse.ArgumentParser()

parser.add_argument(
    '--job-dir',  # Handled automatically by AI Platform
    help='GCS location to write checkpoints and export models',
    required=True
)
parser.add_argument(
    '--hidden_layer_sizes',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--activation',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--solver',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--alpha',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--batch_size',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--learning_rate',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--learning_rate_init',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--power_t',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--max_iter',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--shuffle',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--random_state',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--tol',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--warm_start',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--momentum',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--nesterovs_momentum',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--early_stopping',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--validation_fraction',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--beta_1',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--beta_2',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--epsilon',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--n_iter_no_change',
    help='',
    default='',
    type=int
)
parser.add_argument(
    '--max_fun',
    help='',
    default='',
    type=int
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

mlp_cf = MLPClassifier(hidden_layer_sizes=100,
                       activation='relu',
                       solver='adam',
                       alpha=0.0001,
                       batch_size='auto',
                       learning_rate='constant',
                       learning_rate_init=0.001,
                       power_t=0.5,
                       max_iter=200,
                       shuffle=True,
                       random_state=None,
                       tol=0.0001,
                       warm_start=False,
                       momentum=0.9,
                       nesterovs_momentum=True,
                       early_stopping=False,
                       validation_fraction=0.1,
                       beta_1=0.9,
                       beta_2=0.999,
                       epsilon=1e-08,
                       n_iter_no_change=10,
                       max_fun=15000)

score = cross_val_score(mlp_cf, X, y, cv=5, scoring='roc_auc').mean()

hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='accuracy',
    metric_value=score,
    global_step=1000
)

model_filename = 'model.joblib'
joblib.dump(mlp_cf, model_filename)

job_dir = args.job_dir.replace('gs://', '')

bucket_id = job_dir.split('/')[0]
bucket_path = job_dir.lstrip(f'{bucket_id}/')
bucket = storage.Client().bucket(bucket_id)
blob = bucket.blob(f'{bucket_path}/{model_filename}')
blob.upload_from_filename(model_filename)
