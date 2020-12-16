   #xbg_code
import codecs

import pickle
import random
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import swifter
import optuna
import optuna.integration.lightgbm as lgb
from optuna import trial
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.preprocessing import MinMaxScaler

from config import (N_ITER_SEARCH, SCORING_METHOD, SCORING_RATE_IND,
                    TRUE_POSITIVE_RATE_IND, URL_LOADDATA, URL_PERSIST, TRUE_OVERFITTING_RATE)
from constants import IndicatorsColumns, KeyspaceType, PickleNames
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from time import mktime
from sklearn.preprocessing import MultiLabelBinarizer
import multiprocessing
from sklearn.model_selection import cross_val_predict

def convert_to_timestamp(x):
    if type(x) is pd.Timestamp:
        return mktime(x.to_pydatetime().timetuple())
    elif len(x)> 1:
        return mktime(pd.to_datetime(x).to_pydatetime().timetuple())
    else:
        return x

def csv_encoding(dataset_name):
    import chardet
    with open(f"{dataset_name}.csv", 'rb') as rawdata:
        result = chardet.detect(rawdata.read(10000))
    return result['encoding']

def train_xgboost(tenant_name, response):

    df2 = pd.read_csv('dados_tratados_para_treinamento_09110700.csv', sep=',', encoding=csv_encoding("dados_tratados_para_treinamento_09110700"), parse_dates=['data_monitoramento'], 
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d'))

    del df2['Unnamed: 0']
    #del df2['status_teste']
    #del df2['data_ultimo_teste']
    df2['numero_sintomas'] = df2.iloc[:, :16].gt(0).sum(axis=1)
   
    original, df_ori  = df2.copy(), df2.copy()

    for i in df2.columns:
        if  i in ['status_teste','lideranca_id', 'id', 'empresa', 'bairro', 'cidade']:
            df2[i] = df2[i].astype(
                                'category')  #transforma as colunas em type category para atribuir um numero a cada palavra
            df2[i] = df2[i].cat.codes  #atribui um numero para cada palavra
            df2[i] = df2[i].astype(float)  #transforma o dataset em tipo float

    print(df2)
    company = df2.copy()

    specifity, true_negative_rate, true_positive_rate, start_timer, initial_TPRATE, initial_loop = 0, 0, 0, datetime.now().minute, float(TRUE_POSITIVE_RATE_IND), True
    while ((true_negative_rate < float(
            SCORING_RATE_IND
    )) or (true_positive_rate < initial_TPRATE) or (specifity <= float(TRUE_OVERFITTING_RATE))):  #o loop realiza a normalizaçao e separacao do dataset toda vez que é chamado -> pois assim garante que em cada loop está pegando um dataset aleatorio diferente
        print(specifity)
        original = df_ori.copy()
        # normalizando dados entre 0 a 1, se for escolhido normalizar:
        data = company.copy()
        data['data_monitoramento'] = data['data_monitoramento'].swifter.apply(convert_to_timestamp)
        data['data_ultimo_teste'] = data['data_ultimo_teste'].swifter.apply(convert_to_timestamp)
        data = data.sort_values(by='data_monitoramento')
        for i in data.columns:
            data[i] = data[i].astype(float)  #transforma o dataset em tipo float
        np_scaled = MinMaxScaler().fit_transform(pd.DataFrame(data))
        df_normalized = pd.DataFrame(np_scaled, columns=data.columns)
        df_normalized =data.copy()
        df_normalized[response] = company[response]
        train_df = df_normalized

        treino = train_df.head(int(len(train_df)*.8))
        #teste = train_df.tail(int(len(train_df)*.2))
        teste = train_df.copy()

        #separando os arquivos X e Y de treino e teste
        try:
            X_train = treino.drop(columns=response)
            X_test = teste.drop(columns=response)
        except:
            pass

        y_train = treino[response]
        y_test = teste[response]
        #fim separaçao para teste ser exatamente 30% dos processos
        # reseta o index apenas para organizar
        original.reset_index(level=0, inplace=True)
        # reseta o index apenas para organizar
        X_train.reset_index(level=0, inplace=True)
        # reseta o index apenas para organizar
        X_test.reset_index(level=0, inplace=True)
        del [X_train['index']]  # deleta o index antigo
        # original acaba virando apenas quem está no dataset de teste, para comparara os resultados
        original = original[original['index'].isin(X_test['index'])]
        # mantem o index antigo do original e do test para comparar eles depois nas planmilhas
        # peso para classe minoritaria
        try:
            peso = (sum(train_df[response] == 0) / sum(train_df[response] == 1))
        except:
            peso = 1
        #cria um dataframe vazio temporario para armazenar o index antes de alguma alteração
        temp = pd.DataFrame()
        temp['index'] = X_test['index']
        # tira a coluna index do treino para não interferir no treino do modelo. ela está armazenada no temp para depois bater os dados do dataset original com o x_test
        del [X_test['index']]

        folds = random.randint(4,6)  # numero de dobras do cross validation na hora do treino
        skf = StratifiedKFold(
            n_splits=folds,
            shuffle=True)  #chamando o metodo de cross validation
        xgb = XGBClassifier(
            objective="binary:logistic")  #chamando o xgboost classifier

        cv2 = skf.split(X_train,
                        y_train)  #chamando o cross validation para o treino
        #setando parametros para o RandomSearch buscar os melhores parâmetros
        import re
        X_train = X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        X_test = X_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test)
        #paramter_tuning using optuna

        def objective(trial):
            param = {
        "objective": "binary",
        "metric": "binary",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
            
            lightgbm_tuna = lgb.train(param, train_data, valid_sets=[test_data], early_stopping_rounds=30)

            best_model = lightgbm_tuna.predict(X_test)
            # prevendo a parte de teste

            return lightgbm_tuna

        import optuna
        import functools
        lb_best_para_list =[]
        pred_list = []
        pred_prob_list = []
        index_list = []
        study = optuna.create_study(direction="maximize")
        best_model = study.optimize(objective,n_trials=5)

        #previsao a parte de teste como probabilidades
        y_pred_prob = best_model.predict_proba(X_test)

        #chamando a matriz de confusão do teste
        cm = confusion_matrix(y_test, y_pred)
        rc = recall_score(y_test, y_pred, average='binary')
        print(cm)
        #pegando os TN e FN para calcular a taxa de acerto encontrada no modelo de teste
        TN = cm[0][0]
        FP = cm[0][1]
        TP = cm[1][1]
        FN = cm[1][0]
        specifity = round(TN / (TN+FP), 2)
        true_negative_rate = round(TN / (TN + FN), 2)
        true_positive_rate = round(TP / (TP+FP), 2)
        print(f'specifity: {specifity} + true_negative_rate:{true_negative_rate} + true_positive_rate:{true_positive_rate}')
        prediction = pd.DataFrame(y_pred_prob, columns=[
        'PROB 0', 'PROB 1'
        ]) 
        print(prediction)

        #codificando o pickle para enviar pelo POST
        codec = codecs.encode(pickle.dumps(grid_result.best_estimator_), "base64").decode()

        prediction = pd.DataFrame(y_pred_prob, columns=[
            'PROB 0', 'PROB 1'
        ])  #salvando as previsoes de probabilidade num dataset
        pred_df = prediction  #carregando esse dataset de previsao

        #atribuindo as PROB 0 e PROB 1 para o dataset de X_test -> dataset usado para testar o treino
        X_test['PROB 0'] = pred_df['PROB 0']
        X_test['PROB 1'] = pred_df['PROB 1']

        #recebe o index do dataset temporario, que armazenou o index do X_test antes do teste ou treino
        X_test['index'] = temp['index']

        #deixa os valores em ordem de index, menor para maior, para ficar igual ao dataset original
        X_test.sort_values(by='index', inplace=True)
        X_test = X_test.set_index('index')

        #dataset original com valores reais recebe as colunas de PROBABILIDADE
        original['PROB 0'] = X_test['PROB 0']
        original['PROB 1'] = X_test['PROB 1']

        #atribuindo a previsao para 0 ou 1 baseado em qual probabilidade é maior
        original.loc[original['PROB 0'] > original['PROB 1'], 'PREVISTO'] = 0
        original.loc[original['PROB 0'] < original['PROB 1'], 'PREVISTO'] = 1

        #Deixa response e PREVISTO no mesmo formato -> inteiro
        original['PREVISTO'] = original['PREVISTO'].astype(int)
        original[response] = original[response].astype(int)

        #se o response for igual ao previsto, ACERTOU? recebe "sim", se não recebe "não"
        original.loc[original[response] == original['PREVISTO'],
                    'ACERTOU?'] = "SIM"
        original.loc[original[response] != original['PREVISTO'],
                    'ACERTOU?'] = "NÃO"

        prob_media_1 = original['PROB 1'].mean()

        just_names_and_stat = original[['id', 'PREVISTO']].copy() 
        just_names_and_stat = just_names_and_stat.drop_duplicates()
        #numero de casos previsto no mes X:
        number_predicted_cases_on_month = just_names_and_stat['PREVISTO'].sum()
        #numero de funcionarios que responderam no mes X :
        total_monthly_answers = just_names_and_stat['id'].nunique()

        original.to_csv(f'full_original_respostas.csv')

if __name__== "__main__":
    #Basta adicionar o diretorio onde seus arquivos csv estão
    clusters = 5
    #p = threading.Thread(target=train_xgboost, args=("teste", "status_covid",))
    p = multiprocessing.Process(target=train_xgboost("teste", "status_covid"))
    p.start()
    p.join()