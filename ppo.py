"""
数据源：customer_support_data_samples.csv客服数据，仅提取 Customer 的提问作为 Prompt。
模型加载：
    Actor (策略模型)：SFT微调后的模型 customer_service_sft。
    Critic (价值模型)：在 SFT 模型基础上加一个 Value Head。
    Ref Model (参考模型)：加载 customer_service_sft 并冻结，用于计算 KL 散度。
    Reward Model (奖励模型)：使用 AutoModelForSequenceClassification 加载奖励模型customer_service_rm。
奖励计算：调用 RM 模型进行预测。
"""
