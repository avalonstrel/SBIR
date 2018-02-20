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

    for i, batch_test_data in enumerate(test_data_loader, start=0):
        model.test(batch_test_data)
    
    print('Test Result:{}'.format(model.generate_message(model.test_result_record)))

test()