---
layout: post
title: How to run experiments with LLMs with an external NVIDIA GPU
date: 2025-10-15 12:00:00
description: Tutorial on how to rent an external NVIDIA GPU to run experiments with open-source models
tags: llms vlms vast.ai transformers machine learning neural nets unsloth.ai
categories: 
---
This tutorial explains how to run experiments with open-source foundation models (e.g.,[Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) with external NVIDIA GPUs. All you need is €5–10 and an account on [vast.ai](https://vast.ai/). Alternatively, you could use services like [runpod.io](https://www.runpod.io/), or [lambda.ai](https://lambda.ai/). In this tutorial, I’ll walk you through setting up a GPU-powered instance on vast.ai.

*Why choose this approach?* With a small investment, you can work directly with core LLM libraries and experiment with open-source models. For example, I own a MacBookAir; however, essential libraries such as unsloth don't support Apple chips, nor are many other libraries optimized for them. In my experience, Google Colab is less reliable when it comes to GPU availability (even for paid options). It’s also limited to notebooks, which restricts flexibility. Using a remote instance with GPUs is also helpful if you want to prepare your scripts before submitting heavy jobs on a high performance cluster. Finally, setting this up is a fun way to gain some hands-on ML Ops experience.

## Create account on vast.ai and add credit

After having created an account on vast.ai, charge your credit with 5€-10€. 

{% include figure.liquid loading="eager" path="assets/img/experiment_llms/charge_credit.png" class="img-fluid mx-auto d-block" width="70%" %}**Fig. 1:** Account page on vast.ai

## Rent instance

You can rent an instance immediately by clicking on rent. There are a few technical details to keep in mind to ensure the model runs smoothly. For this example, I want to inference from and fine-tune a [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) model, using [**unsloth**](https://unsloth.ai/). **unsloth** is a convenient Python library for fine-tuning models; it serves as a wrapper around the PEFT (parameter efficient fine-tuning) package, which is part of the PyTorch and HuggingFace universe.

The instance from vast.ai should therefore have an NVIDIA GPU and a fitting Linux distribution.

I recommend the following specifications:

- **GPU**: RTX 3060, RTX A4000 or other RTX; if more compute is needed an A100. The GPUs should have more than >12GB
- **RAM**: Usually 2xVRAM is best, but at least 16GB should suffice
- **HDD**: 100GB
- **CPU**: whatever comes with the instance
- **Bandwith**: At least 100 Mb/s, since you’ll need to download large libraries and models.

Each instance can use a different template. For LLM experiments, it’s easiest to start with the PyTorch template, which comes with CUDA, cuDNN, and many essential libraries pre-installed. This template runs inside an unprivileged Docker container.

{% include figure.liquid loading="eager" path="assets/img/experiment_llms/template.png" class="img-fluid mx-auto d-block" width="70%" %}**Fig. 2:** Template on vast.ai

A few tips for selecting the right instance:

- Check that the instance’s resources (CPUs, RAM) aren’t already heavily used. Some servers share hardware, and it’s possible to find instances on vast.ai where CPU cores are busy.
- I've had problems with several instances; just delete them and search for a different one.
- As soon as the instance is being stopped, the system continues to exist and vast.ai will charge some money everyday for storage costs (not much though).
- Once you stop an instance, someone else can rent its GPU. You won’t be able to restart your instance until the other user finishes, which can take hours or even days.

## Connect to your instance

You’ll connect to the instance via `ssh`. The example below uses macOS with `zsh`, but the setup is similar on other systems — mostly the file paths differ.

Open your terminal and go to your hidden `.ssh` directory:

```bash
cd ~/.ssh
```

If the directory doesn’t exist, create it in your home folder and set the correct permissions:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
```

Next we want to create a key, which we will call "vast_ai", by running:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/vast_ai -C "Name or E-Mail address"
```

from `/.ssh`.

You'll be asked to assign a password.

The key pair consists of a public and a private file. We want to open:

```bash
cat vast_ai.pub
```

and copy the publicly accessible content. We insert the content on our vast.ai account page and save it.

{% include figure.liquid loading="eager" path="assets/img/experiment_llms/ssh_key.png" class="img-fluid mx-auto d-block" width="70%" %}**Fig. 3:** Adding ssh key to vast.ai

Next, we navigate to our instance. I'm using the following for this example:

{% include figure.liquid loading="eager" path="assets/img/experiment_llms/instance.png" class="img-fluid mx-auto d-block" width="70%" %}**Fig. 4:** Instance on vast.ai

If we click on the small key symbol (our ssh key should appear, if not we add the ssh key here again) and on `ADD SSH KEY` then the section `direct ssh connect` should appear. vast.ai will show us which port we can use to connect to the instance.

Copy the provided SSH command, add your key name, and run it in your terminal:

```bash
ssh -i ~/.ssh/vast_ai -p 10532 root@194.26.196.132 -L 8080:localhost:8080
```

Enter the password you set for the key, and you’re in! vast.ai will automatically start a `tmux` session for our instance.

## Check GPU availability

At this point, we can check if the GPU is accessible with

```bash
nvidia-smi
```

We should see something like the following:

{% include figure.liquid loading="eager" path="assets/img/experiment_llms/nvidia.png" class="img-fluid mx-auto d-block" width="70%" %}**Fig. 5:** nvidia-smi showing the GPU

If not, for example, if there's an issue with the driver, the easiest way might be just to delete the instance and look for another one.

## Activate mamba environment

Luckily, the instance comes with `conda`and `mamba` pre-installed. For this reason, we can simply run the following to create a mamba environment which we can call "test_env":

```bash
mamba create -n test_env python=3.12
```

After the installation, initialize the shell for the environment:

```bash
eval "$(mamba shell hook --shell bash)"
```

will do it and then we can activate it:

```bash
mamba activate test_env
```

## Connect to your instance with VS Code

To make life much easier, we can also connect to our instance via VS Code. To do this, open your local terminal, create an `.ssh/config` file if it doesn’t already exist, and set the correct permissions.

```bash
cd ~/.ssh
touch config
chmod 700 config
```

We'll have to add (e.g., with `nano`) the following information, depending on the instance respectively:

````
Host vast_ai_instance
    HostName <IP-of-instance>
    IdentityFile ~/.ssh/vast_ai    
    Port <Port-of-instance>
    User root
    LocalForward 8080 localhost:8080
````

Next, we can open VS Code and try to connect to the remote host. The name from the config should appear as an option, when we click `Connect to Host...`.

{% include figure.liquid loading="eager" path="assets/img/experiment_llms/vs_code.png" class="img-fluid mx-auto d-block" width="70%" %}**Fig. 6:** ssh to remote host with VS Code

## Run python scripts

At this point, we can use VS Code to create new python scripts on our instance. If we go back to our instance terminal, we can install the following packages:

```bash
pip install --upgrade unsloth transformers accelerate bitsandbytes
```

With these libraries installed, we can write a simple Python script (e.g., `test.py`) that automatically downloads the model and runs an inference on a prompt.

```python
from unsloth import FastLanguageModel

model_name = "Qwen/Qwen2.5-7B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    load_in_4bit = True, # Load in 4-bit precision to save memory (optional)
    device_map = "auto", # Auto place model on GPU if available
)

prompt = "Explain what a high performance computer is."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


print("Generating response...")
outputs = model.generate(
    **inputs,
    max_new_tokens = 200,
    temperature = 0.7,
    do_sample = True,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== Model Response ===")
print(response)
```

Simply, run the script in your mamba environment with:

```bash
python3 test.py
```

## Fine-tune model

From here on, you can fine-tune your model with unsloth, using one of their many notebooks as an inspiration: [https://docs.unsloth.ai/get-started/unsloth-notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks)

You may also connect to your git repository to save code or data for later use.