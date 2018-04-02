python train.py --data_root /home/lhy/datasets/TUBerlin --dataset_type tuberlin  --name new_experiment/attention_train_ft_tuberlin_from_imagenet --model cls_model --continue_train --trained_model checkpoints/new_experiment/attention_pretrain_imagenet_hed --start_epoch_label epoch_5 --feature_model attention --feat_size 512 --phase train --num_epoch 30 --n_labels 250 --n_attrs 250 --scale_size 225 --image_type GRAY --batch_size 100 --gpu_ids 1 --retrieval_now --random_crop --flip \
2>&1 |tee -a log/train_ft_tuberlin_cls_from_imagenet_hed.log

