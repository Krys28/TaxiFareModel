# imports (vérfier les imports dans corrections)
import pandas as pd
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.utils import haversine_vectorized, compute_rmse


class Trainer:
    def __init__(self, X, y):
        """
        X: pandas DataFrame
        y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    # Implement set_pipeline() function
    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        """returns a pipelined model"""

        dist_pipe = Pipeline(
            [("dist_trans", DistanceTransformer()), ("stdscaler", StandardScaler())]
        )
        time_pipe = Pipeline(
            [
                ("time_enc", TimeFeaturesEncoder("pickup_datetime")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preproc_pipe = ColumnTransformer(
            [
                (
                    "distance",
                    dist_pipe,
                    [
                        "pickup_latitude",
                        "pickup_longitude",
                        "dropoff_latitude",
                        "dropoff_longitude",
                    ],
                ),
                ("time", time_pipe, ["pickup_datetime"]),
            ],
            remainder="drop",
        )
        self.pipeline = Pipeline(
            [("preproc", preproc_pipe), ("linear_model", LinearRegression())]
        )

    def run(self):
        """set and train the pipeline"""
        # c une instance de ma classe mettre dans une variable une instance (fabric objet)
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    X = df.drop("fare_amount", axis=1)
    y = df["fare_amount"]
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # train
    trainer = Trainer(X_train, y_train)
    trainer.run()
    # evaluate
    trainer.evaluate(X_test, y_test)
