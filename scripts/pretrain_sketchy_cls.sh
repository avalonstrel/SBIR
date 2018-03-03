python train.py --data_root /home/lhy/datasets/sketchy/rendered_256x256/256x256/photo --dataset_type sketchy  --name attention_pretrain_sketchy --model cls_model --feature_model attention --feat_size 512 --phase train --num_epoch 30 --n_labels 125 --n_attrs 125 --scale_size 225 --image_type GRAY --batch_size 100 --gpu_ids 2 --retrieval_now \
2>&1 |tee -a log/pretrain_sketchy_edge_cls.log
