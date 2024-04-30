---
icon: terminal
tags: [guide]
order: 40
---

# 1. Prepare Fine-tuning

Preparing a PyTorch script execution environment on MoAI Platform is not much different than on a typical GPU server.

## Check PyTorch Installation

After connecting to the container via SSH, run the following to check whether PyTorch is installed in the current conda environment.

```
$ conda list torch
...
# Name                    Version                   Build  Channel
torch                     1.13.1+cu116.moreh24.2.0          pypi_0    pypi
...
```

The “Version” shows both the PyTorch version and the MoAI version to run it. In the example above, you have version 24.2.0 of MoAI installed, running PyTorch version 1.13.1+cu116.

If the message “conda: command not found” is displayed, the torch package is not listed, or the version name does not include “moreh” even if the torch package exists, visit ***[Prepare Fine-tuning on MoAI Platform](https://www.notion.so/Prepare-Fine-tuning-on-MoAI-Platform-6063868a82ce4f6daeb1eceabf886b2a?pvs=21)*** document to find instructions to set up the conda environment.

## Check if PyTorch is working

Execute the following to check that the torch package is imported properly and the MoAI Accelerator is recognized. If you have any problem during the process, take action according to the documentation ***(troubleshooting 문서 추가 예정).***

```bash
$ python
Python 3.8.18 (default, Sep 11 2023, 13:40:15)
[GCC 11.2.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
...
>>> torch.cuda.device_count()
1
>>> torch.cuda.get_device_name()
[2024-04-16 19:17:45.714] [info] Requesting resources for MoAI Accelerator from the server...
[2024-04-16 19:17:45.752] [info] Initializing the worker daemon for MoAI Accelerator
[2024-04-16 19:17:47.409] [info] [1/1] Connecting to resources on the server (192.168.110.00:24158)...
[2024-04-16 19:17:47.452] [info] Establishing links to the resources...
[2024-04-16 19:17:47.636] [info] MoAI Accelerator is ready to use.
'MoAI Accelerator'
>>> quit()
```

## Install required Pyton packages

Execute the following to install the thrid party Python packages required to run the script prior to the training.

```bash
$ pip install transformers==4.34.0 datasets==2.14.5 loguru==0.7.2
```

## Download training script

Download the Pytorch script for training from GitHub repository by running the following. 

In this tutorial we will use the `train_gpt.py` script in the `tutorial` directory.

```bash
$ sudo apt-get install git
$ git clone https://github.com/moreh-dev/quickstart.git
$ cd quickstart
~/quickstart$ ls tutorial
...  train_gpt.py  ...
```

## Download training data

Hugging Face provides a variety of publicly available datasets that can be used for model fine-tuning as well as model checkpoints. 

In this tutorial, we will use [mlabonne/Evol-Instruct-Python-26k](https://huggingface.co/datasets/mlabonne/Evol-Instruct-Python-26k) dataset.  This dataset consists of given question conditions and Python code written to match the given question conditions. 

In order to download the training data, we will download the publicly available dataset that is released on Hugging Face using `prepare_gpt_dataset.py` which can be found inside the `dataset` directory and we will preprocess so that it could be ready for the fine-tuning.

```bash
~/quickstart$ python dataset/prepare_gpt_dataset.py
```

The preprocessed dataset is then saved as `gpt_dataset.pt` . 

We can load the saved dataset as following. 

```python
dataset = torch.load("gpt_dataset.pt")
```

