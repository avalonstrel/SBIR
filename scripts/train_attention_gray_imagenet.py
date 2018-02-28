python train.py --data_root /home/lhy/ILSVRC2012 --dataset_type imagenet --continue_train --trained_model_path checkpoints/attention_sketchy_gray --start_epoch_label epoch_0 --name attention_imagenet_gray --model tripletsiamese --feature_model attention --loss_type 'triplet|combine_cls,two_loss' --feat_size 512 --phase train --num_epoch 20 --n_labels 1000 --n_attrs 1000 --scale_size 225 --image_type GRAY --batch_size 50 --gpu_ids 3 --save_mode --retrieval_now --is_relu --is_bn \
2>&1 |tee -a log/train_sketchy_imagenet.log
