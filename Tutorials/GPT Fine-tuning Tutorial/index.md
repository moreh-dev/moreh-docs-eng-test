---
icon: terminal
tags: [guide]
order: 800
---

# GPT Fine-tuning

This tutorial introduces an example of fine-tuning a GPT-based model released as open source in  [Hugging Face](https://huggingface.co/) using MoAI Platform. The tutorial will help you learn how to use AMD GPU clusters with MoAI Platform and demonstrate the performance and automatic parallelization benefits.

## Overview

GPT is a language model architecture using only transformer decoder structure, and was first released through GPT-1 by OpenAI(https://openai.com/) in 2018. Soon after its first release, OpenAI increased the dataset size and model parameters used in pre-training and developed GPT-2, GPT-3 and GPT-4 models, of which the models released as open source are GPT-1 and GPT-2. 

Since the basic architecture of GPT is open source, you can a variety of GPT-based models in Hugging Face in addition to the models developed by OpenAI.

In this tutorial, we will fine-tune the [Cerebras-GPT-13B](https://huggingface.co/cerebras/Cerebras-GPT-13B) model using MoAI Platform for code generation task.  

## Before we start,

Please obtain a container or virtual machine on the MoAI Platform from your infrastructure provider and receive instructions on how to connect to it via SSH.

- KT Cloud Hyperscale AI Computing (https://cloud.kt.com/solution/hyperscaleAiComputing/)

Or, if you wish to have a trial conatiner or GPU resources, please contact Moreh.

***(Moreh 연락처 정보 추가 예정)***

After connecting via SSH, run the `moreh-smi` command to check whether MoAI Accelerator is displayed properly. The device name may show up differently depending on what system you use.  If any problem occurs at this stage, please contact your infrastructure provider or refer to the guide in the documentation. ***(나중에 적당한 문서 링크 연결 예정)*** 

### Check MoAI Accelerator

You need to choose MoAI Accelerator in appropriate size to train an sLLM, such as the GPT model that we will walk you through in this tutorial. First, let’s check the MoAI Accelerator currently in use using the `moreh-smi` cmmand.

You will find a detailed description of the specific MoAI Accelerator settings required for the fine-tuning in “[3. Execute Training](https://www.notion.so/3-Execute-Training-29586ff79d72480c8d01015f678dc9bc?pvs=21)”.

```bash
$ moreh-smi
11:40:36 April 16, 2024
+-------------------------------------------------------------------------------------------------+
|                                                Current Version: 24.2.0  Latest Version: 24.2.0  |
+-------------------------------------------------------------------------------------------------+
|  Device  |        Name         |     Model    |  Memory Usage  |  Total Memory  |  Utilization  |
+=================================================================================================+
|  * 0     |   MoAI Accelerator  |  Large.256GB  |  -             |  -             |  -           |
+-------------------------------------------------------------------------------------------------+
```