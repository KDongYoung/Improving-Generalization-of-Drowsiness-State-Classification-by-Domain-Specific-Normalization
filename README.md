# Improving-Generalization-of-Drowsiness-State-Classification-by-Domain-Specific-Normalization
This is an official repo for Improving Generalization of Drowsiness State Classification by Domain-Specific Normalization (Proc. Int. Winter Conf. Brain-Computer Interface, 2024) [\[Paper\]]()

## Description

We propose a practical generalized framework for classifying driver drowsiness states to improve accessibility and convenience by separating the normalization process for each driver. We considered the statistics of each domain separately since they vary among domains. Moreover, we gathered samples from all the subjects/domains in a domain-balanced and class-balanced manner and composed a mini-batch.


## Getting Started

### Environment Requirement

Clone the repo:

```bash
git clone https://github.com/KDongYoung/Improving-Generalization-of-Drowsiness-State-Classification-by-Domain-Specific-Normalization.git
```

Install the requirements using `conda`:

```terminal
conda create -n EEG_AUG python=3.8.13
conda activate EEG_AUG
pip install -r requirements.txt
```

IF using a Docker, use the recent image file ("pytorch:22.04-py3") uploaded in the [\[NVIDIA pytorch\]](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) when running a container


## Data Preparation

First, create a folder `${DATASET_DIR}` to store the data of each subject.

Download the unbalanced dataset available in the paper titled "EEG-Based Cross-Subject Driver Drowsiness Recognition With an Interpretable Convolutional Neural Network" published in *IEEE Transactions on Neural Networks and Learning Systems* in 2022.

(Ref: J. Cui, Z. Lan, O. Sourina and W. Müller-Wittig, "EEG-Based Cross-Subject Driver Drowsiness Recognition With an Interpretable Convolutional Neural Network," in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2022.3147208.)

Unbalanced dataset available in [\[Dataset Download Link\]](https://figshare.com/articles/dataset/EEG_driver_drowsiness_dataset_unbalanced_/16586957)

The directory structure should look like this:

```
${DATASET_DIR}
	|--${unbalanced_dataset.mat}
```

### Training from scratch

```shell script
# train
python TotalMain.py --mode train
# test
python TotalMain.py --mode infer
```

The (BEST model for each SUBJECT and the tensorboard records) are saved in `${MODEL_SAVE_DIR}/{seed}_{step}/{model_name}` by default

The results are saved in text and csv files in `${MODEL_SAVE_DIR}/{seed}_{step}/{Results}/{evaluation metric}` by default

-> The BEST models are saved separately in each folder based on the evaluation metric used to select the model for validation.

The result directory structure would look like this:

```
${MODEL_SAVE_DIR}
    ${seed}_{step}
	|--${model_name}
	    |--${models}
	    	|--${evaluation metric}
	    |--${tensorboard records}
        |--${Results}
	    |--${evaluation metric}
	    	|--${csv file}
		|--${txt file}
```

### Evaluation

**The average results (%) for drowsiness classification:**
| Model                      | Accuracy | F1-score | Precision | Recall | AUROC | 
| -------------------------- | -------- | -------- | --------- | ------ | ----- |  
| Baseline                   |    |     |     |     |      | 
| Baseline + normalization (ours)   |    |     |     |     |      | 


## Citation

```
ss
```

--------------

If you have further questions, please contact dy_kim@korea.ac.kr

