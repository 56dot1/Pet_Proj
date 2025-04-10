import json
import time
import os
import pandas as pd
import sys
from google import genai

# 配置谷歌Gemini API密钥
api_key = 'AIzaSyC4PkMSQscr8EBVkSC5yN8MbP3oTJUs44o'
client = genai.Client(api_key=api_key)
MODEL = "gemini-2.0-flash"

# 添加API调用频率限制
_last_call_times = []
_max_calls_per_minute = 15
_last_call_time = 0  # 记录上次调用的时间

# 评估提示模板 - 数学推理链式思考评估
EVALUATION_PROMPT = """
请扮演一个公正的裁判，评估以下用户问题中两个AI助手提供的回答质量。您将获得助手A的回答和助手B的回答。您的任务是评估哪个助手的回答更好。您应该首先独立地逐步解决用户问题。然后比较两个助手的回答与您的答案。识别并纠正任何错误。避免任何位置偏见，确保回答的呈现顺序不会影响您的决定。不要让回答的长度影响您的评估。不要偏向某些助手的名字。尽可能客观。在提供解释前，先严格按照以下格式输出最终裁决：
“[[A]]”如果助手A更好，“[[B]]”如果助手B更好，“[[C]]”如果平局。一定要把最后的结果作为回答的开头。
[User Question]
{question}

[The Start of Assistant A’s Answer]
{response_a}
[The End of Assistant A’s Answer]

[The Start of Assistant B’s Answer]
{response_b}
[The End of Assistant B’s Answer]
严格按照以下格式输出最终裁决：“[[A]]”如果助手A更好，“[[B]]”如果助手B更好，“[[C]]”如果平局。 一定要把最后的结果作为回答的开头。
"""

def generate_response(prompt, model_name, max_retries=3, retry_delay=2):
    """生成响应，添加重试机制和频率限制
    
    Args:
        prompt: 提示文本
        model_name: 模型名称
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
    
    Returns:
        生成的文本响应
    """
    global _last_call_times, _last_call_time
    current_time = time.time()
    
    # 清理超过1分钟的调用记录
    _last_call_times = [t for t in _last_call_times if current_time - t < 60]
    
    # 检查是否超过频率限制
    if len(_last_call_times) >= _max_calls_per_minute:
        # 计算需要等待的时间
        wait_time = 60 - (current_time - _last_call_times[0])
        if wait_time > 0:
            print(f"达到每分钟调用限制，等待 {wait_time:.1f} 秒...")
            time.sleep(wait_time)
            # 重新获取当前时间
            current_time = time.time()
            _last_call_times = [t for t in _last_call_times if current_time - t < 60]
    
    retries = 0
    while retries < max_retries:
        try:
            # 确保与上次调用至少间隔4秒
            time_since_last_call = current_time - _last_call_time
            if time_since_last_call < 4:
                sleep_time = 4 - time_since_last_call
                print(f"等待API冷却时间: {sleep_time:.1f}秒...")
                time.sleep(sleep_time)
                current_time = time.time()
            
            # 记录本次调用时间
            _last_call_times.append(current_time)
            _last_call_time = current_time
            
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            retries += 1
            error_msg = f"API调用失败 (尝试 {retries}/{max_retries}): {str(e)}"
            print(error_msg)
            
            if retries >= max_retries:
                print("达到最大重试次数，跳过当前项")
                return f"[API调用错误: {str(e)}]"
            
            print(f"等待 {retry_delay} 秒后重试...")
            time.sleep(retry_delay)
            # 每次重试增加延迟时间，避免频繁请求
            retry_delay *= 1.5

def evaluate_responses(question, response_a, response_b):
    """评估两个回复的推理质量
    
    Args:
        question: 问题
        response_a: 微调后模型的回复
        response_b: 原始模型的回复
    
    Returns:
        评估结果，如果API调用失败则返回None和错误信息
    """
    prompt = EVALUATION_PROMPT.format(
        question=question,
        response_a=response_a,
        response_b=response_b
    )
    
    response = generate_response(prompt, MODEL)
    # 检查是否为API错误响应
    if response.startswith('[API调用错误'):
        error_msg = response[response.find('[')+1:response.find(']')]
        return None, error_msg
    return response, None

def main():
    # 设置批量保存的频率
    SAVE_FREQUENCY = 5  # 每处理5条数据保存一次
    
    # 读取两个JSON文件
    with open("results_lora.json", "r", encoding="utf-8") as f1:
        data1 = json.load(f1)
    
    with open("results3b.json", "r", encoding="utf-8") as f2:
        data2 = json.load(f2)
    
    # 创建NO到条目的映射（快速查找）
    no_to_data1 = {item["NO"]: item for item in data1}
    no_to_data2 = {item["NO"]: item for item in data2}
    
    # 结果文件
    results_file = "evaluation_reasoning_results.json"
    
    # 检查是否有现有结果文件，用于断点续传
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
            if results:  # 如果文件不为空
                # 获取已经处理的最大NO
                processed_nos = [item["NO"] for item in results]
                start_no = max(processed_nos) + 1
            else:
                start_no = 1
    except (FileNotFoundError, json.JSONDecodeError):
        results = []
        start_no = 1
    
    print(f"从NO={start_no}开始处理...")
    
    # 处理每个问题
    for no in range(start_no, 3489):  # 处理所有3489个问题
        if no not in no_to_data1 or no not in no_to_data2:
            print(f"跳过NO={no}，数据不完整")
            continue
        
        item1 = no_to_data1[no]
        item2 = no_to_data2[no]
        
        question = item1.get("question", "")
        response_a = item1.get("lora_response", "")
        response_b = item2.get("qwen3b_response", "")
        
        print(f"\n处理NO={no}...")
        start_time = time.time()
        
        # 评估回复
        evaluation, error_msg = evaluate_responses(question, response_a, response_b)
        
        # 如果API调用失败，终止程序
        if evaluation is None:
            print(f"程序在处理第{no}条数据时终止")
            print(f"错误信息: {error_msg}")
            # 保存当前进度
            if results:
                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"已保存{len(results)}条结果到{results_file}")
            sys.exit(1)
            
        # 记录结果
        result = {
            "NO": no,
            "question": question,
            "lora_response": response_a,
            "qwen3b_response": response_b,
            "evaluation": evaluation,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results.append(result)
        
        # 计算并显示处理时间
        processing_time = time.time() - start_time
        print(f"处理完成，耗时: {processing_time:.2f}秒")
        print(f"评估结果: {evaluation[:100]}...")
        
        # 定期保存结果
        if len(results) % SAVE_FREQUENCY == 0:
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"已保存{len(results)}条结果到{results_file}")
    
    # 最后保存一次结果
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估完成，共处理{len(results)}条数据，结果已保存到{results_file}")

if __name__ == "__main__":
    main()
