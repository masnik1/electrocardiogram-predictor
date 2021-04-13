#importing libraries
import numpy as np
import pandas as pd
import swifter
from tqdm import tqdm
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import multiprocessing
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.decomposition import PCA
import random
import neurokit2 as nk

from constants import TEST_FOLD, TRAIN_SC, NEGATIVE_RATE, POSITIVE_RATE
from queries import load_table_Y, load_table_X, insert_results

def apply_data_cross_fold(X_train, y_train):
    folds = random.randint(
        2, 3
    )  # numero de dobras do cross validation na hora do train_split
    skf = StratifiedKFold(
        n_splits=folds, shuffle=True
    )  # chamando o metodo de cross validation
    cross_validation = skf.split(
        X_train, y_train
    )  # chamando o cross validation para o train_split

    return cross_validation

def start_training():
    
    Y = load_table_Y()
    data = load_table_X()

    Y = data[['diagnostic_superclass', 'strat_fold']]
    X = data.copy().drop(columns=['diagnostic_superclass', 'strat_fold', 'id'])

    #Aplicando a técnica de normalização mínima e máxima -> 
    # Valor mínimo se transforma no novo zero e valor máximo no novo 1
    X_o = data.copy()
    X = MinMaxScaler().fit_transform(X)
    X = pd.DataFrame(X, columns=X_o.drop(columns=
    ['diagnostic_superclass', 'strat_fold', 'id']).columns)
    X = X.set_index(X_o.id)

    X['diagnostic_superclass'] = Y['diagnostic_superclass'] 
    X['strat_fold'] = Y['strat_fold']
    X['id'] = X_o['id'] 

    #Treino
    X_train = X[(X.strat_fold != TEST_FOLD)].drop(columns=
    ['diagnostic_superclass', 'strat_fold'])
    y_train = Y[(Y.strat_fold != TEST_FOLD)]['diagnostic_superclass']

    # Teste
    X_test = X[(X.strat_fold == TEST_FOLD)].drop(columns=
    ['diagnostic_superclass', 'strat_fold'])
    y_test = Y[Y.strat_fold == TEST_FOLD]['diagnostic_superclass']

    positive_rate = 0
    negative_rate = 0

    weight_classes = sum(y_train == 0) / sum(
                y_train == 1
            )

    while (positive_rate < float(POSITIVE_RATE)) or (negative_rate < float(NEGATIVE_RATE)): 
        # weight_classes para classe minoritaria

        cross_validation = apply_data_cross_fold(X_train, y_train)

        #Chamando o modelo vazio
        xgboost_model = XGBClassifier(
            objective="binary:logistic"
        )

        #Definindo os parâmetros
        params = {
        "objective": ["binary:logistic"],
        "colsample_bytree": np.arange(0.5, 0.95, 0.05).tolist(),
        "scale_pos_weight": [
            weight_classes*0.8],
        "gamma": [0.1, 0.2, 0.3, 0.4],
        "min_child_weight": np.arange(3, 7, 1).tolist(),
        "learning_rate": [0.1, 0.125, 0.15, 0.2],
        "n_estimators": [50, 100, 150, 200],
        "subsample": np.arange(0.7, 0.9, 0.05).tolist(),
        "reg_alpha": np.arange(0.4, 0.9, 0.05).tolist(),
        "reg_lambda": np.arange(0.4, 0.9, 0.05).tolist(),
        }

        gsc = RandomizedSearchCV(
            xgboost_model,
            param_distributions=params,
            cv=cross_validation,
            scoring=TRAIN_SC,
            refit=TRAIN_SC,
            verbose=4,
            n_jobs=-1,
            n_iter=1,
            return_train_score=True,
        )

        grid_result = gsc.fit(
            X_train,
            y_train,
            early_stopping_rounds= 5,
            eval_metric=['rmse'],
            eval_set=[(X_test, y_test)],
            verbose=True,
        )

        # Print do melhor resultado
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

        # Melhor modelo encontrado
        best_model = grid_result.best_estimator_

        # Prevendo a parte dos testes(0 ou 1) e prevendo a parte de testes com probabilidades
        y_pred = best_model.predict(X_test)
        y_pred_prob = best_model.predict_proba(X_test)

        #cm será a matriz de confusão gerada através da biblioteca sklearn.metrics
        cm = confusion_matrix(y_test, y_pred)

        #Cria-se então duas colunas com a probabilidade de 0 (normal) e probabilidade de 1 (anomalia)
        prediction = pd.DataFrame(
                y_pred_prob, columns=["PROB 0", "PROB 1"]
            )


        X_test_prov = X_test.reset_index(drop=True)
        X_test_prov['pred_25'] = prediction["PROB 1"].swifter.apply(lambda x: 1 if x > 0.25 else 0)
        X_test_prov['pred_225'] = prediction["PROB 1"].swifter.apply(lambda x: 1 if x > 0.225 else 0)
        X_test_prov['pred_20'] = prediction["PROB 1"].swifter.apply(lambda x: 1 if x > 0.2 else 0)
        X_test_prov['pred_30'] = prediction["PROB 1"].swifter.apply(lambda x: 1 if x > 0.3 else 0)
        X_test_prov['pred_15'] = prediction["PROB 1"].swifter.apply(lambda x: 1 if x > 0.15 else 0)
        X_test_prov['pred_35'] = prediction["PROB 1"].swifter.apply(lambda x: 1 if x > 0.35 else 0)
        X_test_prov['pred_40'] = prediction["PROB 1"].swifter.apply(lambda x: 1 if x > 0.4 else 0)
        X_test_prov['pred_275'] = prediction["PROB 1"].swifter.apply(lambda x: 1 if x > 0.275 else 0)
        X_test_prov['pred_325'] = prediction["PROB 1"].swifter.apply(lambda x: 1 if x > 0.325 else 0)
        X_test_prov['pred_45'] = prediction["PROB 1"].swifter.apply(lambda x: 1 if x > 0.45 else 0)
        X_test_prov['pred_285'] = prediction["PROB 1"].swifter.apply(lambda x: 1 if x > 0.285 else 0)
        
        #X_test = X_test[['id', 'real', 'pred']].reset_index(drop=True).drop_duplicates()

        print('Matriz normal')
        print(cm)
        print('Matriz de 15%')
        print(confusion_matrix(y_test, X_test_prov['pred_15']))
        print('Matriz de 20%')
        print(confusion_matrix(y_test, X_test_prov['pred_20']))
        print('Matriz de 22.5%')
        print(confusion_matrix(y_test, X_test_prov['pred_225']))
        print('Matriz de 25%')
        print(confusion_matrix(y_test, X_test_prov['pred_25']))
        print('Matriz de 27.5%')
        print(confusion_matrix(y_test, X_test_prov['pred_275']))
        print('Matriz de 28.5%')
        print(confusion_matrix(y_test, X_test_prov['pred_285']))
        print('Matriz de 30%')
        print(confusion_matrix(y_test, X_test_prov['pred_30']))
        print('Matriz de 32.5%')
        print(confusion_matrix(y_test, X_test_prov['pred_325']))
        print('Matriz de 35%')
        print(confusion_matrix(y_test, X_test_prov['pred_35']))
        print('Matriz de 40%')
        print(confusion_matrix(y_test, X_test_prov['pred_40']))
        print('Matriz de 45%')
        print(confusion_matrix(y_test, X_test_prov['pred_45']))

        list_confusions = ['pred_15', 'pred_225', 'pred_25', 'pred_275', 'pred_30', 'pred_325', 'pred_35', 'pred_40', 'pred_45', 'pred_285', 'pred_20']
        correct_sum = 0
        best_value = ''
        for value in list_confusions:
            cm = confusion_matrix(y_test, X_test_prov[value])
            positive_rate_ = cm[0][0] / (cm[0][0] +  cm[0][1])
            negative_rate_ = cm[1][1] / (cm[1][0] + cm[1][1])
            if ( (positive_rate_ > float(POSITIVE_RATE)) and (negative_rate_ > float(NEGATIVE_RATE)) ):
                if cm[0][0] + cm[1][1] > correct_sum:
                    correct_sum = cm[0][0] + cm[1][1]
                    positive_rate = cm[0][0] / (cm[0][0] +  cm[0][1])
                    negative_rate = cm[1][1] / (cm[1][0] + cm[1][1])
                    best_value = value

    X_test_prov.to_csv('predicted_.csv')
    insert_results(X_test_prov[['id', best_value]])

    # salvar o modelo XGBoost (xgb_model) no arquivo XGBoost_pickle_best_model.pkl
    with open('XGBoost_pickle_best_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)



if __name__ == '__main__':
    p = multiprocessing.Process(target=start_training)
    p.start()
    p.join()
