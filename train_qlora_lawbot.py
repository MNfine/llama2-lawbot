import os
import torch
from datasets import load_dataset
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# ------------------------------ 
# 1. THÔNG SỐ CẦN CHỈNH
# ------------------------------
BASE_MODEL_ID = "meta-llama/Llama-2-7b-hf"     # đường dẫn Hugging Face của LLaMA2-7B
DATA_PATH        = "law_qa_dataset.jsonl"      # file JSONL QA
OUTPUT_DIR       = "qlora_lawbot_output"       # nơi lưu LoRA adapter & state
NUM_EPOCHS       = 3                           # số epoch training
MICRO_BATCH_SIZE = 1                           # mỗi GPU batch size (nên ≤1 trên 6 GB VRAM)
BATCH_SIZE       = 2                           # effective batch size (kết hợp gradient accumulation)
LEARNING_RATE    = 3e-4                        # LR cho LoRA
LORA_R           = 8                           # rank LoRA
MAX_SEQ_LENGTH   = 1024                        # độ dài token tối đa (prompt+completion)
# ------------------------------

def main():
    # 1) Tải tokenizer & config bitsandbytes cho 4-bit quant
    print("→ Load tokenizer và cấu hình 4-bit quant …")
    tokenizer = LlamaTokenizer.from_pretrained(
        BASE_MODEL_ID,
        token=True,             # dùng token đã login (thay cho use_auth_token)
        trust_remote_code=True  # cho phép tải code custom từ repo meta-llama
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4-bit bitsandbytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # 2) Load model gốc LLaMA2-7B ở 4-bit
    print("→ Load model LLaMA2-7B (4-bit) …")
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        token=True,             # dùng token đã login
        trust_remote_code=True, # cho phép tải code custom từ repo meta-llama
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # 3) Thiết lập LoRA (chỉ train phần adapter)
    print("→ Áp LoRA lên model (QLoRA) …")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,                         # rank
        lora_alpha=16,                    # scaling factor
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # fine-tune chỉ 2 module projection
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # kiểm tra chỉ có thêm weights LoRA

    # 4) Load dataset JSONL chứa các cặp QA
    print("→ Load dataset QA từ JSONL …")
    raw_datasets = load_dataset("json", data_files=DATA_PATH)

    # 5) Tokenize và tạo label: bỏ loss cho phần “prompt”
    def tokenize_and_mask(examples):
        prompts = examples["prompt"]
        completions = examples["completion"]
        inputs = [p + c for p, c in zip(prompts, completions)]
        tokenized = tokenizer(
            inputs,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length"
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        labels_list = []
        for prompt, seq in zip(prompts, input_ids):
            # Tính số token của prompt
            prompt_ids = tokenizer(
                prompt, truncation=True, max_length=MAX_SEQ_LENGTH
            )["input_ids"]
            plen = len(prompt_ids)
            # Gán -100 (ignore) cho phần prompt, phần còn lại giữ nguyên
            labels = [-100] * plen + seq[plen:]
            # Cắt hoặc padding labels để bằng độ dài input_ids
            labels = labels[: len(seq)]
            if len(labels) < len(seq):
                labels += [-100] * (len(seq) - len(labels))
            labels_list.append(labels)

        tokenized["labels"] = labels_list
        return tokenized

    print("→ Tokenize và tạo label cho dataset …")
    tokenized_datasets = raw_datasets.map(
        tokenize_and_mask,
        batched=True,
        remove_columns=["prompt", "completion"],
    )

    # 6) Data collator (cho causal LM, không mask LM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  
    )

    # 7) Cấu hình TrainingArguments & Trainer
    print("→ Cấu hình TrainingArguments và Trainer …")
    training_args = TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=BATCH_SIZE // MICRO_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        fp16=True,                 # bật FP16 để giảm VRAM và tăng tốc
        logging_steps=20,
        output_dir=OUTPUT_DIR,
        save_total_limit=2,
        save_steps=200,
        do_eval=False,             # không cần evaluation
        report_to="none",          # không gửi log lên WandB hay HF
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if "train" in tokenized_datasets else tokenized_datasets,
        data_collator=data_collator,
    )

    # 8) Chạy fine-tune
    print("→ Bắt đầu fine-tune QLoRA (4-bit LoRA) trên LLaMA2-7B …")
    trainer.train()

    # 9) Lưu LoRA checkpoint (adapter weights)
    print(f"→ Lưu LoRA adapter vào thư mục `{OUTPUT_DIR}` …")
    model.save_pretrained(OUTPUT_DIR)
    print("✅ Fine-tune hoàn tất.")
    return

if __name__ == "__main__":
    main()
