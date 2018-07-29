

Repository for the paper "Batch Effect Removal via Batch-Free Encoding" by Uri Shaham.

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


The following usage examples were used to obtain the results reported on the manuscript:

CUDA_VISIBLE_DEVICES=0 python calibrate.py --data_type "cytof" --model "mlp" \
--n_epochs 1000 --AE_type "VAE" --code_dim 15 --beta .2 --gamma 10. --delta .1 \
--data_path './Data'  --experiment_name c15_beta.2_gamma10.0_delta.1_cytof_mlp

CUDA_VISIBLE_DEVICES=0 python calibrate.py --n_epochs 200 --data_type "other" \
--data_path './Data/scRNA-seq' --use_test False --model "mlp" --code_dim 20 \
--beta .1 --gamma .5 --delta .1 \
--experiment_name c20_beta.25_gamma1.0_delta.5_scRNA-seq_mlp



In addition, we provide an evaluation script that examins the reconstruction errors, plots the calibrated data, examines correlation coefficients and calculated MMD estimates.

Usage example:
python evaluate_calibration.py



Any questions should be referred to Uri Shaham, uri.shaham@yale.edu.
