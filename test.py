import os
import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from models.denseloss_model import *
from util.util import *
from datasets.base_dataset import CustomDatasetDataLoader
from models.base_model import create_model

def test():
    opt = TestOptions().parse()
    test_data_loader = CustomDatasetDataLoader(opt)
    model = create_model(opt)
    
    model.train(False)
    val_start_time = time.time()
    for i, batch_test_data in enumerate(test_data_loader, start=0):
        model.test(batch_test_data)
    if not opt.retrieval_now:
        model.test_features = model.combine_features(model.test_features)
        model.retrieval_evaluation(model.test_features, model.test_result_record['total']['loss_value'].avg, model.test_features['labels'])
    print('Test Result:{}'.format(model.generate_message(model.test_result_record)))
    val_end_time = time.time()
    print('Validation Epoch: Time:{:.6f} \t{}'.format('Final', 
                                                                     
                                                                    val_end_time - val_start_time,
                                                                    model.generate_message(model.test_result_record)))

    model.reset_test_features()
    model.reset_test_records()

test()