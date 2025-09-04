from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from datasets import load_dataset
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments, Trainer
import os
import bitsandbytes as bnb

MODEL_PATH = "./SmolVLM-256M-Instruct/"
OUTPUT_PATH = "./SmolVLMBirds"

NUM_EPOCHS = 3
LR = 2e-5
device = "cuda"


USE_LORA, USE_QLORA = True, True

if USE_LORA or USE_QLORA:
    lora_config = LoraConfig(
        r=8,
        lora_alpha = 8,
        lora_dropout = 0.1,
        target_modules = ['o_proj','k_proj','q_proj', 'v_proj' ],
        use_dora = False if USE_QLORA else True,
        init_lora_weights = "gaussian"
    )

if USE_QLORA:
    bnb_conf = BitsAndBytesConfig(
                        load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
    )



#optimizer = AdamW(model.parameters(), lr=LR)

model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, quantization_config = bnb_conf if USE_QLORA else None, device_map = "auto")
#model.add_adapter(lora_config)
#model.enable_adapters()
model = get_peft_model(model, lora_config)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

for name, param in model.named_parameters():
    print(name, param.dtype, param.device)
    break


for name, param in model.named_parameters():
    if isinstance(param, bnb.nn.Params4bit):
        print("QLoRA активно! Параметр:", name)
        found_4bit = True
        break


class DatasetCUBClass(Dataset):
    def __init__(self, root, split = "train", transform = None):
        self.root = root
        self.transform = transform

        images_file = root + "images.txt"
        labels_file = root + "image_class_labels.txt"
        classes_file = root + "classes.txt"
        split_file = root+ "train_test_split.txt"
        
        
        with open(images_file) as f:
            id2img = {int(line.split()[0]): line.split()[1] for line in f}
        with open(labels_file) as f:
            id2label = {int(line.split()[0]): int(line.split()[1]) for line in f}
        with open(split_file) as f:
            id2split = {int(line.split()[0]): int(line.split()[1]) for line in f}
        with open(classes_file) as f:
            self.classnames = {int(line.split()[0]): line.split()[1] for line in f}

        self.samples = []
        for img_id, img_name in id2img.items():
            is_train = id2split[img_id] == 1
            if (split == "train" and is_train) or (split == "test" and not is_train):
                self.samples.append({
                    "path": os.path.join(root, "images", img_name),
                    "label": id2label[img_id],
                    "classname": self.classnames[id2label[img_id]]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Вернем картинку и текст (для smolVLM)
        text = f"<image>A photo of a {sample['classname'].replace('_', ' ')}"
        return image, text, sample["label"]
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.ToTensor(),
])


def collate_fn(batch):
    images, texts, labels = zip(*batch)
    
    # Преобразуем батч через processor
    encodings = processor(
        images=list(images),
        text=list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    # Text-to-Text: используем input_ids как labels
    encodings["labels"] = encodings["input_ids"].clone()
    
    # Переносим всё на GPU
    encodings = {k: v.to(device) for k, v in encodings.items()}
    
    return encodings

train_dataset = DatasetCUBClass("CUB_200_2011/", split="train", transform=transform)
test_dataset = DatasetCUBClass("CUB_200_2011/", split="test", transform=transform)

print(len(train_dataset), len(test_dataset))
img, text, label = train_dataset[0]
print(text, label)

#train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
#test_loader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)


model.to("cuda")
# for batch in train_loader:
#     print (batch)
#     images, texts, labels = batch
#     images = list(images)
#     texts = list(texts)
#     torch.cuda.empty_cache()
#     print("_________________________")
#     print(texts)
#     print(images)

#     combined_text = " ".join(texts)

#     encodings = processor(
#         images=images,
#         text=combined_text,
#         return_tensors="pt",
#         padding=True
#     )

#     encodings = {k: v.to("cuda") for k,v in encodings.items()}

#     outputs = model(**encodings,labels = encodings["input_ids"])

#     loss = outputs.loss
#     logits = outputs.logits

#     print("Loss:", loss.item())
#     break

training_args = TrainingArguments(num_train_epochs = NUM_EPOCHS,
                                  per_device_train_batch_size = 1,
                                  gradient_accumulation_steps = 2,
                                  warmup_steps = 50,
                                  learning_rate = LR,
                                  weight_decay = 0.1,
                                  logging_steps = 25,
                                  save_strategy = "steps",
                                  save_steps = 25,
                                  eval_strategy = "steps",
                                  eval_steps = 25,
                                  bf16 = True,
                                  output_dir = OUTPUT_PATH,
                                  optim = "adamw_torch",
                                  dataloader_pin_memory=False)


trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset = test_dataset
    )

trainer.train()