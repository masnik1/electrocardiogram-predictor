import pandas as pd
import numpy as np
import swifter
import wfdb
import ast
import neurokit2 as nk
import chardet
from tqdm import tqdm
import multiprocessing

from constants import SAMPLING_RATE
from queries import save_vector_mysql

path = 'C:\\Users\\dell\\Documents\\TCC\\TCC\\ia_predict_ecg\\'
file_infos = 'ptbxl_database.csv'

def get_encoding(file_name):
    """
    GET THE ENCODING OF THE DATASET - IF IT IS ASCII DECODE AS UTF-8
    """
    with open(f"{file_name}", 'rb') as rawdata:
        result = chardet.detect(rawdata.read(30000))
        if result['encoding'] == 'ascii':
            result['encoding'] = 'utf-8'
    return result['encoding']


def load_raw_data(df, SAMPLING_RATE, path):
    """
    LOAD THE RAW WAVEFORM (.HEA e .DAT) DATA INTO ARRAYS
    rdsamp reads signal files for the specified record and writes the samples as decimal numbers on the standard output.
    """
    df = df.head(2)
    if SAMPLING_RATE == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def aggregate_diagnostic(y_dic, agg_df):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

def null_columns(Y):
    """
    Getting just the columns where there's not plenitude of null values
    """
    null_column = (Y.isnull().mean() * 100)
    list_columns = []
    for index in range(0, len(null_column)):
        if int(null_column[0]) < 1:
            list_columns.append(null_column.index[index])
    Y = Y[list_columns]

    return Y

def process_X_values(X, Y):

    dfs = []

    Y["diagnostic_superclass"] = Y["diagnostic_superclass"].swifter.apply(lambda x: 0 
    if x == "NORM" else (1 if x == "MI" else (2 if x == "STTC" 
    else (3 if x == "HYP" else 4) ) ) )

    for v in tqdm(range(0, len(X))):
        temp = pd.DataFrame(X[v], 
        columns=['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])

        for value in temp.columns:
            ecg = np.array(temp[value])
            try:
                signals = nk.ecg_clean(ecg, sampling_rate=250, method='pantompkins1985')
            except:
                signals = nk.ecg_clean(ecg, sampling_rate=100, method='pantompkins1985')

            temp[value] = signals

        temp['id'] = Y.iloc[v].patient_id
        s =  temp.groupby('id').cumcount().add(1)
        temp = (temp.set_index(['id',s])
        .unstack()
        .sort_index(axis=1, level=1)
        )
        temp['diagnostic_superclass'] = Y.iloc[v].diagnostic_superclass
        temp['strat_fold'] = Y.iloc[v].strat_fold
        temp['id'] = Y.iloc[v].patient_id
        dfs.append(temp)

    data = pd.concat(dfs)
    data = data[~np.isnan(data.id)]
    return data

def extract_treat_load():
    # load and convert annotation data
    Y = pd.read_csv(f'{path}{file_infos}', encoding=get_encoding(
        path+file_infos), index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Carregando dados de diagnóstico
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.swifter.apply(aggregate_diagnostic, agg_df = agg_df)
    Y = Y[['patient_id', 'age', 'sex', 'height', 'weight', 'recording_date', 'diagnostic_superclass', 'strat_fold', 'filename_lr', 'filename_hr']]
    Y["diagnostic_superclass"] = Y["diagnostic_superclass"].str[0]

    X = load_raw_data(Y, SAMPLING_RATE, path)
    X = process_X_values(X, Y)

    return X, Y


def call_ETL():
    "MAIN CODE AREA"
    X, Y = extract_treat_load()
    save_vector_mysql(X, Y)

    return X, Y

call_ETL()