import numpy as np
import pandas as pd


def is_mostly_english(text):
    if not text: return False
    hinglish_keywords = ["abhi", "kar", "rahi", "kar", "raha", "ho", "gaya", "karna", "nahi"]
    for word in hinglish_keywords:
        if word.lower() in text.lower():
            return False
    return True

def clean_customer_text(text):
    if not text or not isinstance(text, str):
        return ""
    text= text.strip();
    valid_endings = ['.', '?', '!', '。', '？', '！']

    last_idx = -1
    for char in valid_endings:
        idx = text.rfind(char)
        last_idx = max(last_idx, idx)

    if last_idx != -1:
        cleaned = text[:last_idx + 1]
        return cleaned.strip()


if __name__ == '__main__':
    # 获取数据源
    data = pd.read_csv('./data/customer_support_data.csv')
    print(data.info())
    data['text'] = data['text'].apply(clean_customer_text)
    data['is_english'] = data['text'].apply(is_mostly_english)
    non_english_conv_ids = data[data['is_english'] == False]['conv_id']
    non_english_conv_ids = np.unique(non_english_conv_ids)
    data = data[~data['conv_id'].isin(non_english_conv_ids.tolist())]
    conv_ids = np.random.choice(np.unique(data['conv_id']), 2000, replace=False)
    data = data[data['conv_id'].isin(conv_ids.tolist())][['conv_id', 'role', 'text']] # type: ignore[arg-type]
    print(data.shape)
    data.to_csv('./data/customer_samples.csv', index=False, ) # type: ignore[arg-type]
