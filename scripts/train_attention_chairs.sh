python train.py --data_root /home/lhy/sbir_cvpr2016_release/sbir_cvpr2016/chairs --dataset_type sketchx  --model tripletsiamese --feature_model attention --feat_size 512 --phase train --num_epoch 20 --n_labels 15 --n_attrs 15 --scale_size 225 --batch_size 50 --gpu_ids 0,1,2 --save_mode --image_type GRAY--retrieval_now \
2>&1 |tee -a log/train_chairs.log
