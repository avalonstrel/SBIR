import os
import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from models.denseloss_model import *
from util.util import *
from datasets.base_dataset import CustomDatasetDataLoader
from models.base_model import create_model

def train():
    opt = TrainOptions().parse()
    train_data_loader = CustomDatasetDataLoader(opt)
    opt.phase = 'test'
    test_data_loader = CustomDatasetDataLoader(opt)
    opt.phase = 'train'
    data_loader_size = len(train_data_loader)

    model = create_model(opt)
    model.train()

    total_steps = 0
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
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Time:{%4f} \t{}'.format(epoch, 
                                                                    now_size, data_loader_size, now_size / data_loader_size * 100.0, 
                                                                    batch_end_time - batch_start_time,
                                                                    model.generate_message(model.result_record)))
                batch_start_time = time.time()

            if total_steps % opt.print_val_freq == 0:
                val_start_time = time.time()
                now_size = opt.batch_size * (batch_idx+1)
                for i, batch_test_data in enumerate(test_data_loader):
                    model.test(batch_test_data)
                val_end_time = time.time()
                print('Validation Epoch: {} [{}/{} ({:.0f}%)] Time:{%4f} \t{}'.format(epoch, 
                                                                    now_size, data_loader_size, now_size / data_loader_size * 100.0, 
                                                                    val_end_time - val_start_time,
                                                                    model.generate_message(model.test_result_record)))
                model.reset_test_record()
            if total_steps % opt.save_latest_freq == 0:
                print('Save Model at latest epoch {} total steps {}.'.format(epoch, total_steps))
                model.save_model('total_{}'.format(total_steps))
                model.save_model('latest')
            model.reset_record()

        if epoch % opt.save_epoch_freq == 0:
            print('Save Model at epoch {}.'.format(epoch))
            model.save_model('epoch_{}'.format(epoch))
            model.save_model('latest')
        for i, batch_test_data in enumerate(test_data_loader):
            model.test(batch_test_data)
        epoch_end_time = time.time()

        print('End of epoch {} / {}, Time:{%4f}, \t {}'.format(epoch, opt.start_epoch + opt.num_epoch, 
                                                                epoch_end_time - epoch_start_time,
                                                                model.generate_message(model.test_result_record)))
        model.reset_test_record()






def test():
    opt = TestOptions().parse()
    test_data_loader = CustomDatasetDataLoader(opt)
    model = create_model(opt)

    model.train(False)

    for i, batch_test_data in enumerate(test_data_loader, start=0):
        model.test(batch_test_data)
    print('Test Result:{}'.format(model.generate_message(model.test_result_record)))



