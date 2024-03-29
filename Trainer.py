from torch.utils.tensorboard import SummaryWriter
import os
import time
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, roc_auc_score, recall_score
import warnings
import importlib
import torch.nn.functional as F
warnings.filterwarnings('ignore')

from utils.util import to_one_hot

class Trainer():
    def __init__(self, args, subjectList, flatten_subjectList, subject_id, model):
        self.subjectList=subjectList
        self.subject_id=subject_id
        self.args=args
        self.model=model
        self.flatten_subjectList=flatten_subjectList
        self.writer = SummaryWriter(f"{self.args['total_path']}/{self.flatten_subjectList[self.subject_id]}") # tensorboard, log directory
        self.args["tensorboard"]=self.writer
        self.num_domain=len(self.flatten_subjectList)-1
        self.batch=self.args["batch_size"]
        
        self.total_domain_label=[]
        self.metric_dict={"loss":0, "acc":1, "bacc":2, "f1":3, "recall":4}

        self.set_optimizer()
        self.set_learning_rate_Scheduler()

    '''
    ###########################################################################################
    #  Initialize parameter
    ###########################################################################################
    '''
    
    def set_optimizer(self):
        if self.args['optimizer']=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        elif self.args['optimizer']=='AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        elif self.args['optimizer']=='SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args['lr'], momentum=self.args['momentum'], weight_decay=self.args['weight_decay'], nesterov=self.args['nestrov'])
    
    def set_learning_rate_Scheduler(self):
        if self.args['scheduler']=="CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args['steps']//self.args['checkpoint_freq'])
           
    '''
    ###########################################################################################
    #  Train and Predict
    ###########################################################################################
    '''
    def training(self, train_loaders, valid_loader, test_loader):
        train_minibatches = zip(*train_loaders) # gather same number of samples from each subject (domain-balanced mini-batch)

        best_score = [100.0, 0.0, 0.0, 0.0, 0.0]
        start_step=0
        self.total_lr=[]
        self.losses=[]
        
        # Load Loss
        Loss = importlib.import_module('utils.Loss.'+self.args['loss'])
        self.criterion = getattr(Loss, self.args['loss'])(**self.args)
        
        for step in range(start_step, self.args['steps']):
            minibatches_device = [(x.to(self.args['device']), y.to(self.args['device']), z.to(self.args['device'])) for x,y,z in next(train_minibatches)] # data, class label, original domain label
            
            self.lr2 = self.scheduler.get_last_lr()[0]
                      
            self.train(minibatches_device, step) # train

            if (step % self.args['checkpoint_freq'] == 0) or (step == self.args['steps'] - 1):
                valid_score = self.eval("valid", valid_loader, step) # valid 
                self.eval("test", test_loader, step) # test              
                self.scheduler.step()    
                
                for metric in self.args['eval_metric']: 
                    # compare the validation score with the best_score (from the previous step)
                    best_score[self.metric_dict[metric]] = self.compare_metric_save_model(metric, best_score[self.metric_dict[metric]], valid_score[self.metric_dict[metric]])
                                        
                start_step = step + 1
            
        self.writer.close()

        return best_score

    # Prediction
    def prediction(self, metric, test_loader):
        ''' Leave-one-subject-out, Test the best model '''
        print("== "*10, "Testing", "== "*10)
        self.model.load_state_dict(torch.load(os.path.join(self.args['total_path'], 'models', metric ,"{}_bestmodel").format(
            self.flatten_subjectList[self.subject_id]), map_location=self.args['device']))
        if self.args['cuda']: 
            self.model.cuda(device=self.args['device'])

        loss, acc, bacc, f1score, precision, roc_auc, recall, cost = self.eval("test", test_loader, self.args['steps']+1)
        return loss, acc, bacc, f1score, precision, roc_auc, recall, cost
            
    '''
    ###########################################################################################
    #  Train
    ###########################################################################################
    '''
    def train(self, minibatches, step):
        self.model.train()
        
        data = torch.cat([x for x,_,_ in minibatches])
        target = torch.cat([y for _,y,_ in minibatches])
        
        
        domain = torch.cat([z-1 for _,_,z in minibatches])
        
        # one hot coding
        onehot_target = to_one_hot(target, self.args['n_classes']).to(self.args['device'])
        
        self.optimizer.zero_grad()

        if "multi" in self.args['norm_type']:
            output, reweighted_target=self.model(data, onehot_target, domain)
            loss = self.criterion(output, reweighted_target)      

        else:
            output, reweighted_target=self.model(data, onehot_target)
            
            loss = self.criterion(output, reweighted_target)  
            
        pred = output.argmax(dim=1, keepdim=True)        
        
        loss.backward()
        
        self.optimizer.step()

        self.losses.append(loss)

        if step % self.args['checkpoint_freq'] == 0: 
            l=sum(self.losses)/len(self.losses)
            self.losses=[]
            acc=accuracy_score(reweighted_target.argmax(dim=1).cpu().numpy(), pred.cpu().numpy())
            print('Train Epoch: {}\t Loss: {:.4f}\t ACC: {:.4f}'.format(step+1, l, acc))
            self.write_tensorboard("train", step, acc=acc, loss=l, lr=self.lr2)

    '''
    ###########################################################################################
    #  Evaluation
    ###########################################################################################
    '''    
    ## EVALUATE 
    def eval(self, phase, loader, step=0):
        self.model.eval()
        lossfn = torch.nn.CrossEntropyLoss() 
        
        time_cost=[]
        outputs=[]
        targets=[]
        preds=[]
        
        with torch.no_grad(): 
            for datas in loader:
                s_time=time.time()
                data, target, domain = datas[0].to(self.args['device']), datas[1].to(self.args['device'], dtype=torch.int64), datas[2].to(self.args['device'], dtype=torch.int64)
                
                if "multi" in self.args['norm_type']:
                    domain = torch.tensor(range(self.num_domain)).to(device=data.device)
                    data = torch.cat([data for _ in range(self.num_domain)])
                    output, _ = self.model(data, domain=domain)

                else: 
                    output, _ = self.model(data)
                
                if "multi" in self.args['norm_type'] and not "sel" in self.args['norm_type']:
                    if self.args['logit']=="avglogit":
                        logit=F.softmax(output)
                        output = logit.mean(0, keepdim=True)
                        
                e_time=time.time()
                time_cost.append(e_time-s_time)
                
                outputs.append(output)
                preds.append(output.argmax(dim=1, keepdim=False)) 
                targets.append(target)
            
        outputs=torch.cat(outputs)
        preds=torch.cat(preds)
        targets=torch.cat(targets)
                    
        loss = lossfn(outputs, targets) 
        
        targets=targets.cpu().numpy()
        preds=preds.cpu().numpy()

        acc=accuracy_score(targets, preds)
        bacc=balanced_accuracy_score(targets,preds)
        f1=f1_score(targets,preds)
        preci=precision_score(targets,preds, zero_division=0)
        roc_auc=roc_auc_score(targets,preds)
        recall=recall_score(targets,preds)

        print(phase.capitalize(),'Loss: {:.4f}, Acc: {:.4f}%, Bal Acc: {:.4f}%, F1: {:.4f}, Preci: {:.4f}, Recall: {:.4f}'
                    .format(loss, acc, bacc, f1, preci, recall))

        self.write_tensorboard(phase, step, loss, acc, bacc, f1, preci, roc_auc, recall)

        if phase=="valid":
            return loss.item(), acc, bacc, f1, recall
        elif phase=="test":
            return loss.item(), acc, bacc, f1, preci, roc_auc, recall, time_cost

    '''
    ###########################################################################################
    #  Etc.
    ###########################################################################################
    '''
    ############## compare valid_score and evaluation metric and save the best model ##################
    def compare_metric_save_model(self, eval_metric, best_score, valid_score):
        ## compare validation accuracy of this epoch with the best accuracy score
        if eval_metric=="loss":
            ## if validation loss <= best loss, then save model(.pt)
            if best_score >= valid_score:
                best_score = valid_score
                torch.save(self.model.state_dict(), os.path.join(self.args['total_path'], 'models',eval_metric,"{}_bestmodel".format(self.flatten_subjectList[self.subject_id])))
        else:
            ## if validation accuracy >= best accuracy, then save model(.pt)
            if valid_score >= best_score:
                best_score = valid_score
                torch.save(self.model.state_dict(), os.path.join(self.args['total_path'], 'models',eval_metric,"{}_bestmodel".format(self.flatten_subjectList[self.subject_id])))
        
        return best_score
        
    ########################### Tensorboard ###########################
    def write_tensorboard(self, phase, step, loss=0, acc=0, bacc=0, f1=0, preci=0, roc_auc=0, sensitivity=0, lr=0):

        if phase=='train':
            self.writer.add_scalar(f'{phase}/lr2', lr, step)
            self.writer.add_scalar(f'{phase}/acc', acc, step)
            self.writer.add_scalar(f'{phase}/loss', loss, step)
        else:
            self.writer.add_scalar(f'{phase}/loss', loss, step)
            self.writer.add_scalar(f'{phase}/acc', acc, step)
            self.writer.add_scalar(f'{phase}/balanced_acc', bacc, step)  
            self.writer.add_scalar(f'{phase}/f1score', f1, step)
            self.writer.add_scalar(f'{phase}/precision', preci, step)
            self.writer.add_scalar(f'{phase}/roc_auc', roc_auc, step) 
            self.writer.add_scalar(f'{phase}/recall_sensitivity', sensitivity, step) 
