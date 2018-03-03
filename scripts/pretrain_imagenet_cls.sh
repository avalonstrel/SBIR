python train.py --data_root /home/lhy/ILSVRC2012 --dataset_type imagenet  --name attention_pretrain_imagenet --model cls_model --feature_model attention --feat_size 512 --phase train --num_epoch 30 --n_labels 1000 --n_attrs 1000 --scale_size 225 --image_type GRAY --batch_size 100 --gpu_ids 2 --retrieval_now --random_crop --flip\
2>&1 |tee -a log/pretrain_imagenet_edge_cls.log
