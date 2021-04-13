import os

import mysql.connector
import numpy as np
import pandas as pd
import sqlalchemy


def load_table_Y():
    engine = sqlalchemy.create_engine('mysql+pymysql://root:password@localhost:3306/data_ecg')
    dbConnection = engine.connect()

    Y = pd.read_sql("select * from y_data", dbConnection)

    pd.set_option('display.expand_frame_repr', False)

    dbConnection.close()

    return Y

def load_table_X():
    engine = sqlalchemy.create_engine('mysql+pymysql://root:password@localhost:3306/data_ecg')
    dbConnection = engine.connect()

    X_ = pd.DataFrame()
    for div_col in range(1, 14):
        X = pd.read_sql(f"select * from x_data_part{str(div_col)}", dbConnection)
        X_ = X_.join(X.set_index('id'), on='id')

    dbConnection.close()
    
    return X_

def insert_results(X):
    engine = sqlalchemy.create_engine('mysql+pymysql://root:password@localhost:3306/data_ecg')
    X.to_sql(name='results', con=engine,if_exists='append', index=False)
