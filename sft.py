
import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm  # 引入进度条
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

wandb.init(
    project="rlhf-ppo-demo",
    name="run-001",
    config={"lr":2e-5},
    # 实时性微调：控制刷盘频率
    settings=wandb.Settings(
        # 可选：调小上传间隔，实时性更高（带宽消耗变大）
        # upload_buffer_size=10,
        # upload_interval=1
    )
)

# --- 超参数设置 ---
LEARNING_RATE = 5e-5  # 微调通常使用更小的学习率
BATCH_SIZE = 8
EPOCHS = 3
MAX_LENGTH = 256  # 增加长度以容纳人设和对话
MODEL_NAME = "./local/gpt2"
SAVE_PATH = "./models/customer_service_sft"
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

global_step = 0

def prepare_and_tokenize_data(tokenizer):
    # 读取CSV文件
    dataset = load_dataset("csv", data_dir="./data", data_files="customer_support_data_samples.csv")
    # 获取训练数据集
    raw_dataset = dataset['train']

    def process_function(examples):
        inputs = []
        targets = []

        conv_id_list = examples['conv_id']
        role_list = examples['role']
        text_list = examples['text']

        conversations = {}
        for i in range(len(conv_id_list)):
            conv_id = conv_id_list[i]
            role = role_list[i]
            text = text_list[i]

            if conv_id not in conversations:
                conversations[conv_id] = []
            conversations[conv_id].append({'role': role, 'text': text})

        for conv_id, turns in conversations.items():
            if(len(turns)) >= 2:
                customer_text = None
                agent_text = None

                for turn in turns:
                    if turn['role'] == 'customer' and customer_text is None:
                        customer_text = turn['text']
                    elif turn['role'] == 'agent' and customer_text is not None and agent_text is None:
                        agent_text = turn['text']
                        break;

                if customer_text is not None and agent_text is not None:
                    input_text = f"Customer: {customer_text}\n Agent:"
                    target_text = f" {agent_text}"

                    inputs.append(input_text)
                    targets.append(target_text)


        model_input = tokenizer(
            [p + t + tokenizer.eos_token for p, t in zip(inputs, targets)],
            maxlength = MAX_LENGTH,
            truncation = True,
            padding="max_length",
            return_tensors = "pt"
        )

        labels = model_input['input_ids'].clone()
        attention_mask = model_input['attention_mask']


        for i in range(len(inputs)):
            prompt_ids = tokenizer.encode(inputs[i], add_special_tokens = False)
            prompt_len = len(prompt_ids)

            if prompt_len < MAX_LENGTH:
                labels[i, :prompt_len] =-100
            # 布尔索引
            padding_mask = attention_mask[i] == 0
            labels[i, padding_mask] =-100

            model_input['labels'] = labels
            return model_input

    tokenized_train = raw_dataset.map(
        process_function,
        batched = True,
        remove_columns= ['conv_id', 'role', 'text']
    )

    split_dataset = tokenized_train.train_test_split(test_size = 0.1)
    train_ds = split_dataset['train']
    val_ds = split_dataset['test']

    return train_ds, val_ds

def inference():
    # 7.简单的推理测试
    print("Running inference test...")
    test_input = "Hello, my SSO is not working as expected."
    prompt = f"Customer: {test_input}\nAgent:"

    tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
    model = AutoModelForCausalLM.from_pretrained(SAVE_PATH).to(DEVICE) # type: ignore[arg-type]

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output_ids = model.generate(
        **inputs,  # tokenizer处理后的输入张量
        max_new_tokens=100, # 限制模型最多生成 50 个新 token
        pad_token_id=tokenizer.eos_token_id, # 指定填充 token 的 ID 为结束符 token ID
        do_sample=True, # 启用采样生成模式，引入随机性，使生成结果更加多样化
        top_p=0.9 # 启用 nucleus sampling，只从累积概率达到 90% 的词汇中采样
    ) # type: ignore[arg-type]

    print(f"Input: {prompt}")
    print(f"Generated: {tokenizer.decode(output_ids[0], skip_special_tokens=True)}")

def train():
    # 1. 模型与分词器加载
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # GPT2 没有 pad token，指定为 eos token
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE) # type: ignore[arg-type]

    # 2. 数据准备
    print("Processing datasets...")
    train_ds, val_ds = prepare_and_tokenize_data(tokenizer)

    #print(train_ds[0])
    #print(val_ds[0])

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)

    train_loader = DataLoader(
        train_ds, # type: ignore[arg-type]
        batch_size = BATCH_SIZE,
        collate_fn = data_collator
    )

    val_loader = DataLoader(
        val_ds, # type: ignore[arg-type]
        batch_size = BATCH_SIZE,
        collate_fn = data_collator
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)

    print('开始循坏')

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch +1} / {EPOCHS}")

        for batch in progress_bar:
            batch = {k: v.to(DEVICE) for k, v in batch.items() }

        outputs = model(**batch) # type: ignore[arg-type]
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss =  total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Avg Train Loss: {avg_train_loss:.4f}")

        wandb.log({
                    "rm/train_loss": total_train_loss,
                    "rm/train_acc": avg_train_loss,
                }, step=global_step)


        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                bth = {k: v.to(DEVICE) for k, v in batch.items() }
                outputs = model(**bth)
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}")
        print("-" * 50)
        wandb.log({
                "rm/val_loss": total_val_loss,
                "rm/val_acc": avg_val_loss
            }, step=global_step)

 #  6. 保存模型
    print(f"Saving model to {SAVE_PATH}")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    wandb.finish()


if __name__ == '__main__':
    train()
