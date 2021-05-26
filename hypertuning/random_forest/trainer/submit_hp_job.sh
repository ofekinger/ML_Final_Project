#!/bin/bash
BUCKET_NAME="ml-final-314816-aiplatform"
JOB_NAME="rf_binary_class_hp_$(date +"%Y%m%d_%H%M%S")"
JOB_DIR="gs://$BUCKET_NAME/hp_job_dir"
TRAINER_PACKAGE_PATH="./training_job_folder/trainer"
MAIN_TRAINER_MODULE="trainer.train"
HPTUNING_CONFIG="training_job_folder/trainer/hptuning_config.yaml"
RUNTIME_VERSION="1.9"
PYTHON_VERSION="3.5"
REGION="eu-west2"
SCALE_TIER=STANDARD_1

gcloud ai-platform jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --package-path $TRAINER_PACKAGE_PATH \
  --module-name $MAIN_TRAINER_MODULE \
  --region $REGION \
  --runtime-version=$RUNTIME_VERSION \
  --python-version=$PYTHON_VERSION \
  --scale-tier $SCALE_TIER \
  --config $HPTUNING_CONFIG

# Optional command to stream the logs in the console
gcloud ai-platform jobs stream-logs $JOB_NAME
