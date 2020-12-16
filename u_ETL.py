import pandas as pd
import numpy as np
import swifter
import wfdb
import ast
import chardet
import multiprocessing

path = 'ia_predict_ecg/'
file_infos = 'ptbxl_database.csv'
sampling_rate = 100
test_fold = 10

def get_encoding(file_name):
    """
    GET THE ENCODING OF THE DATASET - IF IT IS ASCII DECODE AS UTF-8
    """
    with open(f"{file_name}", 'rb') as rawdata:
        result = chardet.detect(rawdata.read(30000))
        if result['encoding'] == 'ascii':
            result['encoding'] = 'utf-8'
    return result['encoding']


def load_raw_data(df, sampling_rate, path):
    """
    LOAD THE RAW WAVEFORM (.HEA e .DAT) DATA INTO ARRAYS
    rdsamp reads signal files for the specified record and writes the samples as decimal numbers on the standard output.
    """
    if sampling_rate == 100:
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

    X = load_raw_data(Y, sampling_rate, path)

    # Train
    X_train = X[np.where(Y.strat_fold != test_fold)] #19634 - 90%
    y_train = Y[(Y.strat_fold != test_fold)]
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)] #2203 - 10%
    y_test = Y[Y.strat_fold == test_fold]

    return X_train, y_train, X_test, y_test


def main():
    "MAIN CODE AREA"
    X_train, y_train, X_test, y_test = extract_treat_load()


if __name__ == "__main__":
    p = multiprocessing.Process(target=main)
    p.start()
    p.join()
