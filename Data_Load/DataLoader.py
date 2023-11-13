import torch
import numpy as np
import Data_Load.DriverDrowsinessDataset_3sec_B as DriverDrowsinessDataset_3sec_B
import torch
from collections import Counter
    
def init_dataset(args, subj_idx, SUBJECT_LIST, seed):
    test_envs = np.r_[subj_idx].tolist() 

    dataset = DriverDrowsinessDataset_3sec_B.DriverDrowsiness(args, args['data_root'], SUBJECT_LIST, test_envs[0]) # sbj의 데이터 불러오기

    ############################################################### for Trainning: Train, Valid, Test loader
    if args['mode'] == "train": 
        train_envs = list(range(len(SUBJECT_LIST)))
        train_envs.remove(subj_idx)
        
        with open(args['total_path'] + '/info.txt', 'a') as f:
            f.write(f"Source subject: {train_envs}\nTarget subject: {test_envs}\n")
        
        train_dataset=[dataset[i] for i in train_envs]
        test_set=[dataset[i] for i in test_envs] 
        
        # in_split: train set / out_split: val set
        in_splits = [] # TRAIN SET
        out_splits = [] # VALID SET
        holdout_fraction = 0.1 # train:valid = 9:1

        for _, env in enumerate(train_dataset): # train_dataset, dataset
            if args['train_valid_class_balance']:
                out, in_ = class_split_dataset(env, args['n_classes'], holdout_fraction, seed)  
            else:
                out, in_ = split_dataset(env, holdout_fraction, seed) 
                
            # for class balance
            in_weights = make_weights_for_balanced_classes(in_)
            out_weights = make_weights_for_balanced_classes(out)
            in_splits.append((in_, in_weights))
            out_splits.append((out, out_weights))

        train_loaders = [InfiniteDataLoader(
                            dataset=env,
                            weights=env_weights,
                            batch_size=args['batch_size'],
                            num_workers=args['num_workers'])
                        for i, (env, env_weights) in enumerate(in_splits)] # make a train loader for each source subject 

        val_set = torch.utils.data.ConcatDataset([env for i, (env, _) in enumerate(out_splits)])
        valid_loader = torch.utils.data.DataLoader(val_set,batch_size=args['valid_batch_size'], shuffle=False, pin_memory=True, 
                                                   num_workers=args['num_workers']) # valid loader
        
        test_set = torch.utils.data.ConcatDataset(test_set)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['test_batch_size'], shuffle=False, pin_memory=True, 
                                                  num_workers=args['num_workers'])

        return train_loaders, valid_loader, test_loader, test_set
    
    ############################################################### for Inference: only Test loader
    elif args['mode'] == "infer": 
        test_set=[dataset[i] for i in test_envs] 
        
        test_set = torch.utils.data.ConcatDataset(test_set)
        test_loader = torch.utils.data.DataLoader(test_set,batch_size=args['test_batch_size'], shuffle=False, pin_memory=True, 
                                                  num_workers=args['num_workers'])

        return test_loader

'''
#############################################################
Infinite_DataLoader
#############################################################
'''
"""Infinite Dataloder for each subject
Reference:
      Gulrajani et al. In Search of Lost Domain Generalization. ICLR 2021.
"""

class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=True,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=True)

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


# get weights (use the number of samples)
def make_weights_for_balanced_classes(dataset):  
    counts = Counter()
    classes = []
    for y in dataset:
        y = int(y[1])
        counts[y] += 1 
        classes.append(y) 
    n_classes = len(counts)

    weight_per_class = {}
    for y in counts: 
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


'''
#############################################################
Split dataset
#############################################################
'''
## split dataset into train, valid
class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

## split dataset into train, valid in a balanced way for classes
def class_split_dataset(dataset, n_classes, n, seed=0):
    classes=list(range(n_classes))
    c_idx=[[],[]] # 2 class

    for idx, y in enumerate(dataset): 
        if int(y[1]) == classes[0]: # class1
            c_idx[0].append(idx)
        elif int(y[1]) == classes[1]: # class2
            c_idx[1].append(idx)
    
    np.random.RandomState(seed).shuffle(c_idx[0])
    np.random.RandomState(seed).shuffle(c_idx[1])

    keys_1 = c_idx[0][:int(n*len(c_idx[0]))] + c_idx[1][:int(n*len(c_idx[1]))] # valid
    keys_2 = c_idx[0][int(n*len(c_idx[0])):] + c_idx[1][int(n*len(c_idx[1])):] # train

    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2) # valid, train

## split dataset into train, valid (randomly train valid split)
def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    n=int(len(dataset)*n)
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)
