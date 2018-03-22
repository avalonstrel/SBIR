python test.py --data_root /home/lhy/png --dataset_type tuberlin  --name attention_pretrain_tuberlin_test --model cls_model  --trained_model checkpoints/attention_pretrain_sketchy --start_epoch_label epoch_29 --feature_model attention --feat_size 512 --phase test  --scale_size 225 --image_type GRAY --batch_size 100 --gpu_ids 2 --retrieval_now \
2>&1 |tee -a log/pretrain_tuberlin_cls.log

