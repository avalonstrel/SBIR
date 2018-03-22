from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--trained_model', type=str, required=True, help='Load which model to test')
        self.parser.add_argument('--start_epoch_label', type=str, default='latest', help='Start epoch for continue training')
        self.parser.add_argument('--loss_type', type=str, default='triplet|cls|attr,three_loss', help='The loss for training')        
        self.parser.add_argument('--loss_rate', type=str, default='3.0,0.5,0', help='The loss rate for different loss')
        #self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the start epoch model')

        self.update()

    def parse_specific_args(self):
        BaseOptions.parse_specific_args(self)
        self.opt.loss_type, self.opt.loss_flag = self.opt.loss_type.split(',')
        self.opt.loss_type = self.opt.loss_type.split('|')

        self.opt.is_train = False
