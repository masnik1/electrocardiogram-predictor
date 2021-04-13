"""
MAIN CODE AREA
ECG PREDICTOR v.1.0
Author: paulo masnik
github: https://github.com/masnik1/ia_ecg_predictor
"""

from u_ETL.U_ETL import call_ETL
from u_ML.u_ML_train import start_training



if __name__ == '__main__':
    call_ETL()
    start_training()