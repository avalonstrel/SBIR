python train.py --data_root /home/lhy/datasets/TUBerlin --dataset_type tuberlin  --name attention_train_ft_tuberlin_from_coco --model cls_model --continue_train --trained_model checkpoints/new_experiment/attention_pretrain_coco_canny --start_epoch_label epoch_10 --feature_model attention --feat_size 512 --phase train --num_epoch 30 --n_labels 250 --n_attrs 250 --scale_size 225 --image_type GRAY --batch_size 100 --gpu_ids 2 --retrieval_now --random_crop --flip \
2>&1 |tee -a log/train_ft_tuberlin_cls_from_coco.log

