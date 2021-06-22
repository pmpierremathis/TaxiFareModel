from ml_flow_test import EXPERIMENT_NAME
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import haversine_vectorized, compute_rmse
from sklearn.model_selection import train_test_split

from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient

class Trainer():
    MLFLOW_URI = "https://mlflow.lewagon.co/"
    EXPERIMENT_NAME = "[FR] [Strasbourg] [pmpierremathis] TaxiFareModel + v1"
    
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return self.pipeline
               
    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        return self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)

        # log into ML Flow
        self.mlflow_log_param('model', 'Linear_Regression')
        self.mlflow_log_metric("RMSE", rmse)

        return rmse
    
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.EXPERIMENT_NAME)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.EXPERIMENT_NAME).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    taxi_ds = get_data()
    taxi_ds = clean_data(taxi_ds)
    X_train, X_test, y_train, y_test = train_test_split(taxi_ds.drop(columns='fare_amount'), taxi_ds['fare_amount'], test_size=0.2)
    trainer = Trainer(X_train, y_train)
    trainer.run()
    taxi_score = trainer.evaluate(X_test, y_test)
    print(f"RMSE : {taxi_score}")