nohup bash -c "CUDA_VISIBLE_DEVICES=6 python Retrain_Evaluation/Retrain/retrain.py" > Retrain_Evaluation/micro_sam_cache/train/train_large_all.log 2>&1 &

nohup bash -c "CUDA_VISIBLE_DEVICES=5 python Retrain_Evaluation/Retrain/retrain.py" > Retrain_Evaluation/micro_sam_cache/train/train_large.log 2>&1 &