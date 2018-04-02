python test.py --data_root /home/lhy/datasets/TUBerlin --dataset_type tuberlin  --name attention_pretrain_tuberlin_test --model cls_model --n_labels 250 --n_attrs 250  --trained_model checkpoints/new_experiment/attention_pretrain_imgenet_canny --start_epoch_label epoch_5 --feature_model attention --feat_size 512 --phase test  --scale_size 225 --image_type GRAY --sketch_type GRAY --gpu_ids 2 --retrieval_now \
2>&1 |tee -a log/pretrain_tuberlin_cls.log

