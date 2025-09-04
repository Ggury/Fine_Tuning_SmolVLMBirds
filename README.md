# Fine_Tuning_SmolVLMBirds

# Fine-tuning SmolVLM on CUB-200-2011 (Bird Classification)

*SmolVLM-256* training on **[CUB-200-2011](http://www.vision.caltech.edu/datasets/cub_200_2011/)** dataset to generate textual descriptions of birds.

## Features
- Loads SmolVLM in **4-bit mode** (QLoRA).  
- Fine-tunes the model on the CUB-200-2011 dataset.  
- Generates bird species descriptions from images.  
- Uses Hugging Face `Trainer` for training management.  

## Requirments
- [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
- [CUB_200_2011](https://www.vision.caltech.edu/datasets/cub_200_2011/?utm_source=chatgpt.com)

## Installation
First install
```bash
pip install transformers torch bitsandbytes Pillow datasets peft torchvision
```
Download/ clone Model and Dataset to the repository directory