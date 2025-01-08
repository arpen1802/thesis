# Electric Vehicle charging behaviour prediction


- download the data by running
    aws s3 cp s3://q658166-thesis/ProcessedData/processed_data.parquet ./processed_data.parquet
    aws s3 cp s3://q658166-thesis/ProcessedData/removed_outliers.parquet ./processed_data.parquet
    
Best models
    Energy need with DNN: logs/dnn-soc/version_107,   model: checkpoints-dnn/best_checkpoint_soc.ckpt
    Plugin duration with DNN: logs/dnn-dur/version_8, model: checkpoints-dnn/best_checkpoint_dur-v2.ckpt
    Energy need with LSTM: logs/lstm-soc/version_41,  model: checkpoints-lstm/best_checkpoint_soc-v38.ckpt
    Plugin duration with LSTM: logs/lstm-dur/version_0,  model: checkpoints-lstm/best_checkpoint_dur.ckpt