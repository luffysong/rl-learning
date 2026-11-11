"""
模型结构变化：SFT模型是生成文本（CausalLM），RM是输出一个分值（输出维度为1）。
数据格式要求：RM训练通常需要成对数据（Pairwise Data），即 (Prompt, Chosen_Response, Rejected_Response)。
Chosen: 人类倾向的高质量回答（通常是数据集里的原始Ground Truth）。
Rejected: 质量较差的回答。
负采样（Negative Sampling）：由于本数据集（customer_support_data_samples.csv）只有正确的对话，
       没有“错误的回答”，代码中将使用负采样策略——把“其他对话中的Agent回复”作为当前问题的“错误回答”来构建训练数据。
"""
import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# --- 超参数设置 ---
# 使用监督学习微调后的模型，基于一个已经初步具备此方面知识的模型进行训练，收敛更快，效果通常更好
SFT_MODEL_PATH = "./models/customer_service_sft"
SAVE_PATH = "./models/customer_service_rm"

LEARNING_RATE = 2e-5  # 奖励模型（RM）通常使用极小的学习率
BATCH_SIZE = 4  # RM需要同时处理两个句子(chosen/rejected)，显存占用较大，适当调小Batch
EPOCHS = 1  # 奖励模型很容易过拟合，通常1-2轮即可
MAX_LENGTH = 256
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class PairDateset(Dataset):

    def __init__(self, data, tokenizer, max_length) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = []

        valid_dialogues = []
        all_agent_responses=[]

        conversations = {}
        for i in range(len(data['conv_id'])):
            conv_id = data['conv_id'][i]
            role = data['role'][i]
            text = data['text'][i]

            if conv_id not in conversations:
                conversations[conv_id] = []
            conversations[conv_id].append({'role': role, 'text': text})

            if role == 'agent' and text:
                all_agent_responses.append(text)

        for conv_id, turns in conversations.items():
            if(len(turns)) >=2:
                customer_text = None
                agent_text_chosen = None

                for turn in turns:
                    if turn['role'] == 'customer' and customer_text is None:
                        customer_text = turn['text']
                    elif turn['role'] == 'agent' and customer_text is not None and agent_text_chosen is None:
                        agent_text_chosen = turn['text']
                        break

                if customer_text  and agent_text_chosen:
                    # 随机选择一个回答作为坏回答
                    agent_text_rejected = random.choice(all_agent_responses)

                    while agent_text_rejected == agent_text_chosen and len(all_agent_responses) > 1:
                            agent_text_rejected = random.choice(all_agent_responses)


                    self.pairs.append({
                        'prompt': customer_text,
                        'chosen': agent_text_chosen,
                        'rejected': agent_text_rejected
                    })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        item = self.pairs[index]
        prompt = f"Customer: {item['prompt']}\nAgent:"

        #构造好的句子
        text_chosen = f"{prompt} {item['chosen']}{self.tokenizer.eos_token}"
        # 构造坏的句子
        text_rejected = f"{prompt} {item['rejected']}{self.tokenizer.eos_token}"

        en_chosen = self.tokenizer(
            text_chosen,
            max_length = self.max_length,
            truncation = True,
            padding = 'max_length',
            return_tensors = 'pt'
        )

        enc_reject = self.tokenizer(
            text_rejected,
            max_length = self.max_length,
            truncation = True,
            padding = 'max_length',
            return_tensors = 'pt'
        )


        return {
            "input_ids_chosen": en_chosen["input_ids"].squeeze(0),
            "attention_mask_chosen": en_chosen["attention_mask"].squeeze(0),
            "input_ids_rejected": enc_reject["input_ids"].squeeze(0),
            "attention_mask_rejected": enc_reject["attention_mask"].squeeze(0),
        }


def train():
    # 1. 加载 SFT 模型，但修改为分类头 (num_labels=1)
    print(f"Loading SFT model from {SFT_MODEL_PATH} for RM training...")
    # 注意：使用 AutoModelForSequenceClassification，而不是 AutoModelForCausalLM
    # AutoModelForCausalLM 是因果语言模型，AutoModelForSequenceClassification 是序列分类模型
    # 奖励模型需要输出一个分数，SequenceClassification输出层会换上一个简单的线性层（Score Head），
    # 将隐藏层状态映射为 1 个数值。

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            SFT_MODEL_PATH,
            num_labels = 1,
            problem_type = "regression"
        ).to(DEVICE)
    except OSError:
        print('未找到模型')
        return

    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    # 配置 pad_token_id 能够避免某些警告
    model.config.pad_token_id = tokenizer.pad_token_id

    #准备数据
    dataset = load_dataset("csv", data_dir="./data", data_files="customer_samples.csv")['train']

    split_dataset = dataset.train_test_split(test_size = 0.1, seed = 42)

    train_ds = PairDateset(split_dataset['train'], tokenizer, MAX_LENGTH)
    val_ds = PairDateset(split_dataset['test'], tokenizer, MAX_LENGTH)


    train_loader = DataLoader(train_ds,batch_size = BATCH_SIZE, shuffle= True)
    test_loader = DataLoader(val_ds,batch_size = BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss= 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch in progress_bar:
            batch={ k: v.to(DEVICE) for k, v in batch.items()}

            optimizer.zero_grad()

            # 前向传播，好的方面
            outputs_chosen = model(
                input_ids  = batch['input_ids_chosen'],
                attention_mask = batch['attention_mask_chosen']
            )

            rewards_chosen = outputs_chosen.logits

            # 前向传播，坏的方面
            outputs_rejected = model(
                input_ids  = batch['input_ids_rejected'],
                attention_mask = batch['attention_mask_rejected']
            )

            outputs_rejected = outputs_rejected.logits

            # --- 计算 Pairwise Ranking Loss ---
            # 目标：maximize (reward_chosen - reward_rejected)
            # Loss = -log(sigmoid(chosen - rejected))

            loss = -torch.log(torch.sigmoid(rewards_chosen - outputs_rejected)).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        print(f"Epoch {epoch + 1} - Avg Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}

                r_chosen = model(
                    input_ids = batch['input_ids_chosen'],
                    attention_mask = batch['attention_mask_chosen']
                ).logits

                r_rejected = model(
                    input_ids = batch['input_ids_chosen'],
                    attention_mask = batch['attention_mask_chosen']
                ).logits

                correct +=(r_chosen > r_rejected).sum().item()
                total += r_chosen.size(0)

        acc = correct / total if total > 0 else 0
        print(f"Validation Accuracy: {acc:.2%}")
        print("-" * 50)

         # 4. 保存模型
    print(f"Saving Reward Model to {SAVE_PATH}")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)


def run_inference(model, tokenizer):
   # model.eval()
    prompt = "Customer: I cannnot access my account.\n Agent: "

    # 两个假设的回答
    good_response = " Please check if your caps lock is on and try resetting your password."
    bad_response = " I like pizza and the weather is nice today."  # 完全不相关的回答

    text_good = prompt + good_response
    text_bad = prompt + bad_response

    #with torch.no_grad():
    inputs_good = tokenizer(text_good, return_tensors="pt").to(DEVICE)
    score_good = model(**inputs_good).logits.item()

    inputs_bad = tokenizer(text_bad, return_tensors="pt").to(DEVICE)
    score_bad = model(**inputs_bad).logits.item()

    print(f"Prompt: {prompt.strip()}")
    print(f"Option A (Relevant): '{good_response.strip()}' -> Score: {score_good:.4f}")
    print(f"Option B (Random):   '{bad_response.strip()}' -> Score: {score_bad:.4f}")

    if score_good > score_bad:
        print("Result: Model correctly preferred Option A.")
    else:
        print("Result: Model failed to distinguish.")


if __name__ == "__main__":
    #train()
    #model = AutoModel.from_pretrained(SAVE_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        SAVE_PATH,
        num_labels = 1,
    ).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
    run_inference(model, tokenizer)
