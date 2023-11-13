import Main
import argparse
import os

""" Experiment Setting """ 
# ARGUMENT
parser = argparse.ArgumentParser(description='Calibration-free driver drowsiness classification')
parser.add_argument('--data_root', default='/DATASET_DIR/', help="name of the data folder") # DATASET_DIR/
parser.add_argument('--run_code_folder', default='')
parser.add_argument('--save_root', default='./MODEL_SAVE_DIR/', help="where to save the models and tensorboard records") # MODEL_SAVE_DIR
parser.add_argument('--result_dir', default="", help="save folder name") 
parser.add_argument('--total_path', default="", help='total result path')
parser.add_argument('--cuda', type=bool, default=True, help='cuda')
parser.add_argument('--cuda_num', type=int, default=0, help='cuda number')
parser.add_argument('--device', default="", help='device')

parser.add_argument('--n_classes', type=int, default=0, help='num classes')
parser.add_argument('--n_channels', type=int, default=0, help='num channels')
parser.add_argument('--n_timewindow', type=int, default=0, help='timewindow')
parser.add_argument('--track_running', type=bool, default=True, help='track mean, var of batchnorm for inference') 

# BATCH NORMALIZATION
parser.add_argument('--norm_type', default="", help='bn, sabn, multibn, multisabn, multiibn')
parser.add_argument('--logit', default="", help='max, avg')
parser.add_argument('--num_domain', default=0, help='# bn')
parser.add_argument('--inferbn', default='sbn', help='sbn, abn')

parser.add_argument('--loss', default="CELoss", help='type of loss')
parser.add_argument('--f_res', default=False, help='where to apply the residual identification')

parser.add_argument('--optimizer', default="Adam", help='optimizer')
parser.add_argument('--lr', type=float, default=0, metavar='LR', help='learning rate (default: 0.01)') 
parser.add_argument('--weight_decay', type=float, default=0, help='the amount of weight decay in optimizer') 
parser.add_argument('--scheduler', default="CosineAnnealingLR", help='scheduler')
parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='input batch size of each subject for training (default: 16)') 
parser.add_argument('--total_batch_size', type=int, default=160, metavar='N', help='total batch size used for training (default: 160)') 
parser.add_argument('--valid_batch_size', type=int, default=1, metavar='N', help='valid batch size for training (default: 1)') 
parser.add_argument('--test_batch_size', type=int, default=1, metavar='N', help='input batch size for testing (default: 1)')
parser.add_argument('--num_workers', type=int, default=0, metavar='N', help='number worker') 

parser.add_argument('--subject_group', type=int, default=3, metavar='N', help='subject_group for parallel running') 
parser.add_argument('--steps', type=int, default=0, help='Number of steps') 
parser.add_argument('--checkpoint_freq', type=int, default=50, help='Checkpoint every N steps')
parser.add_argument('--seed', type=int, default=2020, help='seed') 

parser.add_argument('--dataset_name', default='B', help='Dataset name: B')
parser.add_argument('--model_name', default='', help='trained model name')
parser.add_argument('--mode', default='train', help='train, infer')

parser.add_argument('--eval_metric', default=["acc", "bacc", "f1", "recall"], help='evaluation metric for model selection ["loss", "acc", "bacc", "f1"]')
parser.add_argument('--metric_dict', default={"loss": 0,"acc": 1, "bacc": 2, "f1": 3, "recall": 4}, help='total evaluation metric')
parser.add_argument('--train_valid_class_balance', default=False, help='train, valid class balance, default: False')

parser.add_argument('--tensorboard', default=[], help='tensorboard writter')

args = parser.parse_args()
args=vars(args)

args["run_code_folder"]=os.path.realpath(__file__) # folder name of running code


subjectList=['S01', 'S02', 'S03', 'S04', 
             'S05', 'S06', 'S07', 'S08', 
             'S09', 'S10', 'S11'] 

model_name='MoResnet8'

## Set hyperparameter
"""'''  Dataset   '''"""
args["n_classes"]=2
args['n_channels']=30
args['n_timewindow']=384
args['lr']=0.002
args['weight_decay']=0 
args['steps']=800 
args['optimizer']="Adam"
args['data_root']=args['data_root']+"/"
subjectList=subjectList
 
###################################################################################
args['loss']='CELoss'
args['norm_type']='multibn'
args['logit']='avglogit'   
Main.main(subjectList, args, model_name)