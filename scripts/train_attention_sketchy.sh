python train.py --data_root /home/lhy/datasets/sketchy/rendered_256x256/256x256/photo/ --dataset_type sketchy  --name attention_sketchy --model tripletsiamese --feature_model attention --feat_size 512 --phase train --num_epoch 20 --n_labels 125 --n_attrs 125 --scale_size 225 --batch_size 50 --gpu_ids 0,1,2 --save_mode --retrieval_now \
2>&1 |tee -a log/train_sketchy.log
