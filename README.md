

Repository for the paper "???" by Uri Shaham.

Data:
Data files should appear in a data folder specified in data_path.
Source and training data should be in CSV format with observations in rows and markers in columns without headers. Their filenames should be: source_train_data.csv and target_train_data.csv. Test data files are optional. If not supplied, the train files will also use for testing. Otherwise, they should follow the same format and be named source_test_data.csv and target_test_data.csv.

Usage examples:
CUDA_VISIBLE_DEVICES=0 python calibrate.py --code_dim 10 --beta 0.1 --gamma 1. --delta 10. --data_path \Data --model cytof_basic
--experiment_name c10_beta0.1_gamma1.0_delta10.0_cytof_basic

python evaluate_calibration.py



Any questions should be referred to Uri Shaham, uri.shaham@yale.edu.
