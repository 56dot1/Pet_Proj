import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
import torch
import time
import pandas as pd
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)

# -------------------------
# 配置参数（动态调整）
# -------------------------
MODEL_PATH = "./Lora"   #./Lora D:\\qwen3b
LORA_ADAPTER_PATH = MODEL_PATH  # 替换为你的LoRA适配器路径
ADAPTER_NAME = "default_lora"  # 动态适配器名称
# LoRA配置
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # 指定目标模块

)

# -------------------------
# 加载模型和分词器
# -------------------------
print(f"加载{MODEL_PATH}模型中...")


base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
print(f"实际加载的基础模型路径: {base_model.config._name_or_path}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 应用LoRA配置并加载适配器
model = get_peft_model(base_model, LORA_CONFIG)
model.unload()
model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH, adapter_name=ADAPTER_NAME)

# 确保模型处于评估模式
model.eval()

# -------------------------
# 其他初始化
# -------------------------
history_file = "conversation_history.json"
def load_history():
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            print("加载历史对话...")
            return json.load(f)
    except FileNotFoundError:
        return []


def summarize_history(history):
    # 初始化 HuggingFacePipeline
    hf_pipeline = pipeline(  # 使用重命名后的函数名
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    pipeline_instance = HuggingFacePipeline(pipeline=hf_pipeline)  # 避免命名冲突

    if not history:
        return ""

    # 初始化 ConversationSummaryBufferMemory
    conversation_memory = ConversationSummaryBufferMemory(llm=pipeline_instance, max_token_limit=1024)

    # 将历史对话转换为 HumanMessage 和 AIMessage 对象
    messages = []
    for entry in history:
        if entry["role"] == "user":
            messages.append(HumanMessage(content=entry["content"]))
        elif entry["role"] == "assistant":
            messages.append(AIMessage(content=entry["content"]))

    # 将消息添加到 conversation_memory
    conversation_memory.chat_memory.add_messages(messages)

    # 使用模型生成总结
    summary_prompt = "请总结我的历史对话内容，总结应包括每一轮用户输入和模型输出：\n" + conversation_memory.buffer + "\n总结："
    summary_inputs = tokenizer(summary_prompt, return_tensors="pt").to(model.device)
    summary_ids = model.generate(
        **summary_inputs,
    )

    # 提取模型输出
    summary_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(summary_inputs.input_ids, summary_ids)
    ]
    summary_response = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]

    return summary_response

def save_history(history):
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)


# 读取Excel文件
file = 'symptoms_data.csv'  # 文件路径
df = pd.read_csv(file)
df=df.head(20)
# 初始化JSON文件
json_file = 'new.json'
try:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    data = []


def automate_chat():
    global data  # 使用全局变量来存储所有的问题和答案
    # 用户自定义输入
    instruction = """
    请用思维树方法，为我输入的宠物疾病和疾病对应的症状，发挥想象力模拟一个思维树推理确诊的过程；
    思维树的结构不限，但总体上要满足‘分解子问题-为每个子问题生成分支-详细探讨每个子问题分支-对每个分支进行评估-总结归纳评估得出最终结果’的思想，
    思维树子问题的拆分应充分运用分治算法思想，通过递归的方式逐步深入每个问题、子问题与分支节点；
    思维树算法在每个分支节点上使用启发式方法评估分支的可能性、矛盾与自洽性从而决定是否继续探索该分支或回溯。
    思维树探索过程中可通过剪枝去除那些明显无关、不可能、不自洽或矛盾的分支从而提高推理效率。
    思维树的构建应充分运用发散性思维、穷举思想构建思维树结点，且最终结果指向必须是对应疾病；
    思维树算法在探索完所有分支后通过综合归纳、评估各个分支的结果，得出最终结论；
    思维树算法在推理过程中可以根据新信息动态调整推理路径，灵活应对复杂问题；
    思维树的层数、每层节点个数<5，文本长度不要太长,适当控制输出长度 ；
    """

    instruction2 = """
    请仔细阅读并批判以下初步回复，从症状与疾病关联的合理性、症状疾病推理的完整性和准确性、推理过程中的假设和前提、模型的自洽性和矛盾等角度，一一找出并指明初步回复的问题。
    输出长度不要太长，简明扼要为主；
    """

    for index, row in df.iterrows():
        symptoms = [str(value) for value in row[1:] if pd.notna(value)]
        disease = '疾病：'+row[0]
        symptoms_input = "我家狗出现了" + ",".join(symptoms)+"的症状；"
        user_input = symptoms_input + disease + "；"

        # 第一次调用模型生成初步回复
        messages1 = [
            {"role": "system", "content": f"Instruction:{instruction} "},
            {"role": "user", "content": user_input}
        ]
        text1 = tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
        model_inputs1 = tokenizer([text1], return_tensors="pt").to(model.device)
        start_time = time.time()
        generated_ids1 = model.generate(
            **model_inputs1,
            max_new_tokens=1536,
            # temperature=1.2,
            # top_k=50,
            # top_p=0.95,
            # do_sample=True
        )

        generated_ids1 = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs1.input_ids, generated_ids1)
        ]
        response1 = tokenizer.batch_decode(generated_ids1, skip_special_tokens=True)[0]

        print(f"一、系统(初步回复): {response1}")

        # 第二次调用模型进行批判与修正
        messages2 = [
            {"role": "system", "content": f"Instruction:{instruction2} "},
            {"role": "user", "content": f"初步回复: {response1}"}
        ]
        text2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
        model_inputs2 = tokenizer([text2], return_tensors="pt").to(model.device)

        generated_ids2 = model.generate(
            **model_inputs2,
            max_new_tokens=1024,
            # temperature=1.2,
            # top_k=50,
            # top_p=0.95,
            # do_sample=True
        )

        generated_ids2 = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs2.input_ids, generated_ids2)
        ]
        response2 = tokenizer.batch_decode(generated_ids2, skip_special_tokens=True)[0]

        print(f"二、系统(批判与修正): {response2}")

        # 记录生成响应的时间戳
        end_time = time.time()
        generation_time = end_time - start_time
        print(f"生成响应时间: {generation_time:.2f} 秒")

        # 将问题、初步回复和批判后的结果保存到JSON文件
        data.append({
            "NO": index + 1,
            "question": symptoms_input,
            "initial_response": response1,
            "critiqued_response": response2
        })
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    torch.cuda.empty_cache()
    automate_chat()
