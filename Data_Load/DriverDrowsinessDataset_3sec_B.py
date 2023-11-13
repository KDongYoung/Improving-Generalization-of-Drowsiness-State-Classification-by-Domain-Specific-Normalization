from torch.utils.data import Dataset
from scipy import io
import numpy as np

''' dataset 생성 '''
class DriverDrowsiness():
    def __init__(self, args, root_path, SUBJECT_LIST, test_envs):
        if root_path is None:
            raise ValueError('Data directory not specified!')

        self.datasets=[]
        self.subjectList=SUBJECT_LIST
            
        drowsy_num=0
        alert_num=0
        ORI_DATA=io.loadmat(root_path+'unbalanced_dataset.mat') 

        x=ORI_DATA["EEGsample"]
        subIdx=ORI_DATA["subindex"]
        label=ORI_DATA["substate"]

        label.astype(int)
        subIdx.astype(int)
        subjnum=len(self.subjectList)

        for i in range(1, subjnum+1):
            
            if i>test_envs+1:
                j=i-1
            else:
                j=i
            
            idx=np.where(subIdx==i)[0]
            self.datasets.append(EEGDataset(x[idx], label.squeeze()[idx], j))
            a=np.sum(label[idx]==0)
            d=np.sum(label[idx]==1)

            drowsy_num+=d
            alert_num+=a
        print("Total Alert:Drowsy =",alert_num,":",drowsy_num)
        
        with open(args['total_path'] + '/info.txt', 'a') as f:
            f.write(f"Dataset Name {args['dataset_name']}, Alert:Drowsy = {alert_num}:{drowsy_num}\n")

    def __getitem__(self, index):
        return self.datasets[index] # one subject at a time

    def __len__(self):
        return len(self.datasets)

"""
Make a EEG dataset
X: EEG data
Y: class score
"""
class EEGDataset(Dataset):
    def __init__(self, X, y, subj_id):
        self.X = X
        self.y = y
        self.len = len(self.y)
        self.subj_id = subj_id
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        X = self.X[idx].astype('float32')  # for only eeg
        y = self.y[idx].astype('int64') 
        
        X=np.expand_dims(X,axis=0) # (1, channel, time) batch shape
    
        return X, y, self.subj_id