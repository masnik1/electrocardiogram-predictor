# iECG Predictor - Detection of Myocardial Infarction using a gradient boosting model

This is my thesis of graduation for my bacheloors in Mechatronics and Robotics at Federal Institute of Santa Catarina in Brazil.
The goal of this project was to use a classification algorithm, in the end I chose XGBoost, to classify waves from a dataset of electrocardiograms and outputting the result of a detection of possible myocardial infarcion or not in a patient.

Some interesting libs and packages I used include wfdb:
https://wfdb.readthedocs.io/en/latest/

Which is a native Python waveform-database (WFDB) package. A library of tools for reading, writing, and processing WFDB signals and annotations. This was needed so I could convert the waveforms on the electrocardiograms to a dataset for XGBoost use. 

Also the project was made using an microsservices approach. Which each task would be ran trhough a different endpoint, ETL for data aquisition, SQL injection and reading, training, retraining and predictions.

Some used techniques in this project consisted of PCA for reduction of the dimentions of dataset
RandomizedSearchCV for the random selection of parameters for a Gradient Tunning of training
StratifiedKFold for applying Cross Validation
MinMaxScaler for putting every dataset feature into the same scale (for example patient age with patient heartbeat)

This work was published in International Journal of Computer Applications (0975 – 8887) - Volume 183 – No. 9, June 2021:
https://www.ijcaonline.org/archives/volume183/number9/masnik-2021-ijca-921384.pdf
