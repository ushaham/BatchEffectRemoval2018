

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

# calibration of cytof data
CUDA_VISIBLE_DEVICES=0 python calibrate.py --data_type "cytof" --model "mlp" \
--n_epochs 1000 --AE_type "VAE" --code_dim 15 --beta .2 --gamma 10. --delta .1 \
--data_path './Data'  --experiment_name c15_beta.2_gamma10.0_delta.1_cytof_mlp

# calibration of scRNA-seq data
CUDA_VISIBLE_DEVICES=0 python calibrate.py --n_epochs 500 --data_type "other" \
--data_path './Data/scRNA-seq' --use_test False --model "mlp" --code_dim 20 \
--beta .02 --gamma .4 --delta .02 \
--experiment_name c20_beta.02_gamma.4_delta.02_scRNA-seq_mlp



In addition, we provide an evaluation script that examins the reconstruction errors, plots the calibrated data, examines correlation coefficients and calculated MMD estimates.

Usage examples:

# after running calibration of cytof data
python evaluate_calibration.py --use_test True data_type "cytof" 

# after running calibration of scRNA-seq data
python evaluate_calibration.py --use_test False data_type "other" 


Important remark:
it is important to tune the hyperparameters specified in calibrate.py to get a satisfying result.
In particular, the parameters beta,gamma and delta need to be tuned carefully.
To help the user do that, it is highly recommended to:
(i) use evaluate_calibration.py, and verify that the reconstructions and 
calibrations are both of high quality
(ii) look at the tensorboard plots for each run.



Any questions should be referred to Uri Shaham, uri.shaham@yale.edu.
