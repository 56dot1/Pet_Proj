import json
import time
import os
import pandas as pd
import sys
from google import genai

# 配置谷歌Gemini API密钥
api_key = 'AIzaSyDh9Bi0yEZm1kmAwWPavqpJGClEEbgKEh8'
client = genai.Client(api_key=api_key)
MODEL = "gemini-2.0-flash"

# 添加API调用频率限制
_last_call_times = []
_max_calls_per_minute = 15
_last_call_time = 0  # 记录上次调用的时间

# 评估提示模板 - 单一回答评分
EVALUATION_PROMPT = """
请扮演一个公正的裁判，评估以下用户问题中AI助手提供的回答质量。您的评估应考虑因素如帮助性、相关性、准确性、深度、创意以及回答的详细程度。开始评估时，请提供简短的解释。尽可能客观。在提供解释后，请严格遵循以下格式对回答进行评分：1到10分，“[[rating]]”，例如：“Rating: [[5]]"
[Question]
{question}
[The Start of Assistant’s Answer]
{response_b}
[The End of Assistant’s Answer]
在提供解释前先给出打分结论，请严格遵循以下格式对回答进行评分：1到10分，“[[rating]]”，例如：“Rating: [[5]],原因：......"
"""

def generate_response(prompt, model_name, max_retries=3, retry_delay=2):
    """生成响应，添加重试机制和频率限制
    
    Args:AIzaSyDh9Bi0yEZm1kmAwWPavqpJGClEEbgKEh8
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

def evaluate_responses(question, response):
    """评估回复的质量
    
    Args:
        question: 问题
        response: 模型的回复
    
    Returns:
        评估结果，如果API调用失败则返回None和错误信息
    """
    prompt = EVALUATION_PROMPT.format(
        question=question,
        response_b=response
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
    results_file = "results_indi_b.json"
    
    # 检查是否有现有结果文件，用于断点续传
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
            if results:  # 确保文件不为空
                # 获取已经处理的最大NO
                processed_nos = [item["NO"] for item in results]
                start_no = max(processed_nos) + 1
            else:
                print(f"警告：{results_file}为空文件")
                start_no = 1
    except FileNotFoundError:
        print(f"未找到结果文件：{results_file}，将创建新文件")
        results = []
        start_no = 1
    except json.JSONDecodeError:
        print(f"错误：{results_file}不是有效的JSON文件，请检查文件格式")
        sys.exit(1)
    
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
        evaluation, error_msg = evaluate_responses(question, response_b)
        
        # 如果API调用失败，记录错误并终止程序
        if evaluation is None:
            error_record = {
                "error_no": no,
                "error_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error_message": error_msg
            }
            print(f"\n错误发生在处理第{no}条数据时：")
            print(f"时间：{error_record['error_time']}")
            print(f"错误信息：{error_msg}")
            
            # 保存当前进度和错误信息
            if results:
                # 创建错误日志文件
                error_log = f"error_log_{time.strftime('%Y%m%d_%H%M%S')}.json"
                try:
                    with open(error_log, "w", encoding="utf-8") as f:
                        json.dump(error_record, f, ensure_ascii=False, indent=2)
                    print(f"错误信息已保存到：{error_log}")
                except Exception as e:
                    print(f"保存错误日志失败：{str(e)}")
                
                # 保存已处理的结果
                try:
                    with open(results_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    print(f"已保存{len(results)}条结果到{results_file}")
                except Exception as e:
                    print(f"保存结果文件失败：{str(e)}")
                    backup_file = f"{results_file}.backup"
                    try:
                        with open(backup_file, "w", encoding="utf-8") as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
                        print(f"结果已备份到：{backup_file}")
                    except Exception as e:
                        print(f"备份结果失败：{str(e)}")
            sys.exit(1)
            
        # 记录结果
        result = {
            "NO": no,
            "question": question,
            "original_response": response_b,
            "evaluation": evaluation
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