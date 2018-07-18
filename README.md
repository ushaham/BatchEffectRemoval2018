

Repository for the paper "???" by Uri Shaham.

Data:
* Data files should appear in a data folder specified in data_path.
* Source and training data should be in CSV format with observations in rows and markers in columns without headers. 
* Their filenames should be: source_train_data.csv and target_train_data.csv. Test data files are optional. 
* If not supplied, the train files will also use for testing. 
* Otherwise, they should follow the same format and be named source_test_data.csv and target_test_data.csv.

Training:
* The training script is calibrate.py. 
* It should be supplied the path to the data folder, and optionally also hyperparameter setting.
* The data is saved in output/calibrated_data_org_scale.
* For visualization and evaluation purposes, we also save the data in output/calibrated data. Note that this data is not in its original scale - it is processed using log transformation and z-scoring.


Usage example:
CUDA_VISIBLE_DEVICES=0 python calibrate.py --code_dim 15 --beta 0.1 --gamma 100. --delta .1 --data_path \Data --model cytof_basic
--experiment_name c10_beta0.1_gamma100.0_delta.1_cytof_basic


In addition, we provide an evaluation script that examins the reconstruction errors, plots the calibrated data, examines correlation coefficients and calculated MMD estimates.

Usage example:
python evaluate_calibration.py



Any questions should be referred to Uri Shaham, uri.shaham@yale.edu.
