import sqlalchemy


def __save_X(X):
    engine = sqlalchemy.create_engine('mysql+pymysql://root:password@localhost:3306/data_ecg')

    X = X.loc[:, ~X.columns.str.startswith('Unnamed')]
    X = X[X.diagnostic_superclass < 2]

    i_start = 0
    i_max = 995
    for div_col in range(1, 14):
        X_part = X.iloc[:, i_start:(i_max*div_col) - 1]
        X_part['id'] = X['id']
        i_start = i_max*div_col - 1
        print(X_part.columns)
        X_part.to_sql(name=f'x_data_part{div_col}', con=engine,if_exists='append', index=False)

def __save_Y(Y):
    engine = sqlalchemy.create_engine('mysql+pymysql://root:password@localhost:3306/data_ecg')
    Y.to_sql(name='y_data', con=engine,if_exists='append', index=False)


def save_vector_mysql(X, Y):
    __save_Y(Y)
    __save_X(X)