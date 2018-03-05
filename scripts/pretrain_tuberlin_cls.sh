python train.py --data_root /home/lhy/png --dataset_type tuberlin  --name attention_pretrain_tuberlin --model cls_model --continue_train --trained_model checkpoints/attention_pretrain_sketchy --start_epoch_label epoch_29 --feature_model attention --feat_size 512 --phase train --num_epoch 30 --n_labels 250 --n_attrs 250 --scale_size 225 --image_type GRAY --batch_size 100 --gpu_ids 2 --retrieval_now --random_crop --flip \
2>&1 |tee -a log/pretrain_tuberlin_cls.log

