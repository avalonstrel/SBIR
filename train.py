import os
import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from models.denseloss_model import *
from util.util import *
from datasets.base_dataset import CustomDatasetDataLoader
from models.base_model import create_model

def train():
    print('Initialize Parameters...')
    opt = TrainOptions().parse()
    print('Load data...')
    train_data_loader = CustomDatasetDataLoader(opt)
    opt.phase = 'test'
    test_data_loader = CustomDatasetDataLoader(opt)
    opt.phase = 'train'
    data_loader_size = len(train_data_loader)
    print('Construct Model...')
    model = create_model(opt)
    print(opt.model)
    model.train()
    opt.save_to_file()
    total_steps = 0
    print('Start Training...')
    for epoch in range(opt.start_epoch, opt.start_epoch + opt.num_epoch):
        epoch_start_time = time.time()
        epoch_steps = 0
        batch_start_time = time.time()
        for batch_idx, batch_data in enumerate(train_data_loader):

            epoch_steps += opt.batch_size
            total_steps += opt.batch_size
            iter_start_time = time.time()
            model.optimize(batch_data)

            if batch_idx % opt.print_freq == 0:
                batch_end_time = time.time()
                now_size = opt.batch_size * (batch_idx+1)
                print('Train Epoch: {} [{}/{} ({:.2f}%)] Time:{:.6f} \t{}'.format(epoch, 
                                                                    now_size, data_loader_size, now_size / data_loader_size * 100.0, 
                                                                    batch_end_time - batch_start_time,
                                                                    model.generate_message(model.result_record)))
                batch_start_time = time.time()

            if total_steps % opt.print_val_freq == 0 and (not opt.dataset_type in ['sketchy', 'imagenet'] or opt.model == 'cls_model') :
                val_start_time = time.time()
                now_size = opt.batch_size * (batch_idx+1)
                for i, batch_test_data in enumerate(test_data_loader):
                    model.test(batch_test_data, opt.retrieval_now)
                if not opt.retrieval_now:
                    model.test_features = model.combine_features(model.test_features)
                    model.retrieval_evaluation(model.test_features, model.test_result_record['total']['loss_value'].avg, model.test_features['labels'])
                val_end_time = time.time()
                print('Validation Epoch: {} [{}/{} ({:.2f}%)] Time:{:.6f} \t{}'.format(epoch, 
                                                                    now_size, data_loader_size, now_size / data_loader_size * 100.0, 
                                                                    val_end_time - val_start_time,
                                                                    model.generate_message(model.test_result_record)))
                model.reset_test_features()
                model.reset_test_records()
            if total_steps % opt.save_latest_freq == 0:
                print('Save Model at latest epoch {} total steps {}.'.format(epoch, total_steps))
                model.save_model('total_{}'.format(total_steps))
                model.save_model('latest', True)
            model.reset_records()
        if not opt.dataset_type in ['sketchy', 'imagenet'] or  opt.model == 'cls_model':
            for i, batch_test_data in enumerate(test_data_loader):
                model.test(batch_test_data, opt.retrieval_now)

            if not opt.retrieval_now:
                model.test_features = model.combine_features(model.test_features)
                model.retrieval_evaluation(model.test_features, model.test_result_record['total']['loss_value'].avg, model.test_features['labels'])

        if epoch % opt.save_epoch_freq == 0:
            print('Save Model at epoch {}.'.format(epoch))
            model.save_model('epoch_{}'.format(epoch))
            model.save_model('latest', True)
        
        epoch_end_time = time.time()

        print('End of epoch {} / {}, Time:{:.6f}, \t {}'.format(epoch, opt.start_epoch + opt.num_epoch, 
                                                                epoch_end_time - epoch_start_time,
                                                                model.generate_message(model.test_result_record)))
        model.reset_features()
        model.reset_test_features()
        model.reset_records()
        model.reset_test_records()

train()
