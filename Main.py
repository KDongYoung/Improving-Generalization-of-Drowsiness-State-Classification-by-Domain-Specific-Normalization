import random
import numpy as np
import os
import pandas as pd
from datetime import datetime

from Data_Load.DataLoader import init_dataset
from Trainer import Trainer

def Experiment(args, subject_id, subjectList):
  
    # make a directory to save results, models
    args['total_path'] = args['save_root'] + str(args['seed']) + "_" + str(args['steps']) + "/" + args['result_dir']

    if not os.path.isdir(args['total_path']):
        os.makedirs(args['total_path'])
        for metric in args['eval_metric']:
            os.makedirs(f"{args['total_path']}/models/{metric}")

    # save ARGUEMENT
    with open(args['total_path'] + '/args.txt', 'a') as f:
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        f.write("\n"+str(args)+"\n\n")

    # connect GPU/CPU
    import torch.cuda
    args['cuda'] = torch.cuda.is_available()
    # check if GPU is available, if True chooses to use it
    args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## fix seed
    seed = args['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args['cuda']:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    if len(subjectList)==1:
        args['mode']="infer"
        print("No Source Subject.... Change to Inference Phase")

    if args['mode']=="train":
        valid_best = start_Training(args, subjectList, subject_id, seed) # Train
        args['mode']="infer"
        t_loss, t_acc, t_bacc, t_f1, t_preci, t_rocauc, t_recall, t_timecost = start_Inference(args, subjectList, subject_id, seed) # Leave-one-subject-out 
        args['mode']="train"
    if args['mode']=="infer":
        t_loss, t_acc, t_bacc, t_f1, t_preci, t_rocauc, t_recall, t_timecost = start_Inference(args, subjectList, subject_id, seed) # Leave-one-subject-out 
        valid_best = [0]*len(args['eval_metric'])
   
    return valid_best, t_loss, t_acc, t_bacc, t_f1, t_preci, t_rocauc, t_recall, t_timecost

def start_Training(args, subjectList, subject_id, seed): # train
    flatten_subjectList=sum(subjectList,[])
    num_domain=len(flatten_subjectList)-1 # train subject
    
    # MODEL
    model = load_model(args, num_domain)
    if args['cuda']:
        model.cuda(device=args['device']) # connect DEVICE
    
    train_loaders, valid_loader, test_loader, _ = init_dataset(args, subject_id, flatten_subjectList, seed)
        
    trainer = Trainer(args, subjectList, flatten_subjectList, subject_id, model)
    valid_best = trainer.training(train_loaders, valid_loader, test_loader) # train, valid
    
    return valid_best
    
def start_Inference(args, subjectList, subject_id, seed): # prediction  
    flatten_subjectList=sum(subjectList,[])
    num_domain=len(flatten_subjectList)-1  

    t_loss, t_acc, t_bacc, t_f1, t_preci, t_rocauc, t_recall, t_timecost = [], [], [], [], [], [], [], []
    
    test_loader = init_dataset(args, subject_id, flatten_subjectList, seed)
        
    for metric in args['eval_metric']:
        # MODEL
        best_model = load_model(args, num_domain)    
        if args['cuda']: best_model.cuda(device=args['device']) # connect DEVICE
       
        best_trainer=Trainer(args, subjectList, flatten_subjectList, subject_id, best_model)
        
        loss, acc, bacc, f1, preci, rocauc, recall, cost = best_trainer.prediction(metric, test_loader)
        cost = np.mean(cost[1:])
        
        t_loss.append(loss)
        t_acc.append(acc)
        t_bacc.append(bacc)
        t_f1.append(f1)
        t_preci.append(preci)
        t_rocauc.append(rocauc)
        t_recall.append(recall)
        t_timecost.append(cost)
        
    return t_loss, t_acc, t_bacc, t_f1, t_preci, t_rocauc, t_recall, t_timecost
    
def load_model(args, num_domain): # load model
    if args['model_name']=="eegnet4":
        from Networks.Model_EEGNet import eegnet_4
        model=eegnet_4(args, num_domain)
    elif args['model_name']=="eegnet8":
        from Networks.Model_EEGNet import eegnet_8
        model=eegnet_8(args, num_domain)
    elif args['model_name']=="deepconvnet":
        from Networks.Model_DeepConvNet import DeepConvNet
        model=DeepConvNet(args, num_domain)
    elif args['model_name']=="reresnet8":
        from Networks.Model_ResNet1DRe import resnet_8
        model=resnet_8(args, num_domain)
    elif args['model_name']=="reresnet18":
        from Networks.Model_ResNet1DRe import resnet_18
        model=resnet_18(args, num_domain)     
    else:
        model=None
        print("Model not loaded.....")
        
    return model


'''
###########################################################################################
#  Main
###########################################################################################
'''    

def main(subjectList, args, model_name):

    args['model_name']=model_name

    # exp_type=f"{model_name}_{args['loss']}_{args['norm_type']}"
    
    if args['logit']=="":
        exp_type=f"{model_name}_{args['loss']}_{args['norm_type']}_{args['f_res']}" 
    else:    
        exp_type=f"{model_name}_{args['loss']}_{args['norm_type']}_{args['logit']}_{args['f_res']}"

    args['result_dir']=exp_type
    result_path = args['save_root'] + args['dataset_name'] +"_" + str(args['seed']) + "_" + str(args['steps'])  + "/Results"
        
    if not os.path.isdir(result_path):
        for metric in args['eval_metric']:
            os.makedirs(f"{result_path}/{metric}")
        
    before_sbj_num=0
    for i in range(args['subject_group']):
        before_sbj_num+=len(subjectList[i])
    
    for id in range(len(subjectList[args['subject_group']])):
        print(str(args['dataset_name']) + " / " + str(args['seed']) +  " / " + exp_type)   
        print("~"*25 + ' Test Subject ' + subjectList[args['subject_group']][id] + " " + "~"*25)

        valid_best, loss, acc, bacc, f1, preci, auc, recall, time_cost = Experiment(args, before_sbj_num+id, subjectList)

        if args["mode"]=="train":
            valid_best=[valid_best[args["metric_dict"][i]] for i in args["eval_metric"]]
      
        total_perf = np.array([[subjectList[args['subject_group']][id]]*len(args['eval_metric']), valid_best, loss, acc, bacc, f1, preci, recall, auc, time_cost])
      
        print("> "*20)
        # save the performance result into '.txt' file
        for metric_i, metric_name in enumerate(args['eval_metric']):
            print(f"{metric_name} Valid: {valid_best[metric_i]:.2f}, TEST_SUBJECT: {subjectList[args['subject_group']][id]}, ACCURACY: {acc[metric_i]:.4f}%, PRECISION: {preci[metric_i]:.4f}%, RECALL: {recall[metric_i]:.4f}%")
            with open(result_path+"/"+metric_name+"/"+metric_name+"_"+str(args['seed'])+"_"+exp_type+'_Accuracy.txt', 'a') as f:
                f.write(f"valid_best: {valid_best[metric_i]:.2f}, {subjectList[args['subject_group']][id]}, acc: {acc[metric_i]:.4f}, bacc: {bacc[metric_i]:.4f}, f1: {f1[metric_i]:.4f}, precision: {preci[metric_i]:.4f}, recall: {recall[metric_i]:.4f}\n") # save test performance   
        
        # save the performance result into '.csv' file
        df=pd.DataFrame(total_perf)
        for metric_i, metric_name in enumerate(args['eval_metric']):
            part=df.iloc[:, metric_i]
            df_part=part.to_frame()
        
            print(df_part)
            df_part.to_csv(result_path+"/"+metric_name+"/"+metric_name+"_"+str(args['seed'])+"_"+exp_type+".csv", mode="a", index=False)