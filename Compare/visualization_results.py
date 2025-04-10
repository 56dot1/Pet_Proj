import json
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np

# 函数：从评估结果中提取[[]]标记的内容
def extract_evaluation_result(evaluation):
    if not evaluation:
        return None
    
    # 只匹配[[和]]之间的内容
    pattern = r"\[\[([^\]]+)\]\]"
    match = re.search(pattern, str(evaluation))
    if match:
        result = match.group(1)
        # 如果结果是A、B或C，则返回
        if result in ['A', 'B', 'C']:
            return result
    
    # 输出调试信息
    print(f"无法从评估中提取结果: {evaluation[:100]}...")
    return None

# 函数：从评分中提取数值
def extract_rating(rating):
    if not rating:
        return None
    
    # 只匹配[[和]]之间的数字
    pattern = r"\[\[(\d+(?:\.\d+)?)\]\]"
    match = re.search(pattern, str(rating))
    if match:
        return float(match.group(1))
    
    # 输出调试信息
    print(f"无法从评分中提取数值: {rating[:100]}...")
    return None

# 加载数据
def load_data():
    # 加载评估结果
    try:
        with open('evaluation_reasoning_results.json', 'r', encoding='utf-8') as f:
            evaluation_data = json.load(f)
    except Exception as e:
        print(f"加载evaluation_reasoning_results.json时出错: {e}")
        evaluation_data = []
    
    # 加载default结果
    try:
        with open('results_default.json', 'r', encoding='utf-8') as f:
            default_data = json.load(f)
    except Exception as e:
        print(f"加载results_default.json时出错: {e}")
        default_data = []
    
    # 加载indi_a结果
    try:
        with open('results_indi_a.json', 'r', encoding='utf-8') as f:
            indi_a_data = json.load(f)
    except Exception as e:
        print(f"加载results_indi_a.json时出错: {e}")
        indi_a_data = []
    
    # 加载indi_b结果
    try:
        with open('results_indi_b.json', 'r', encoding='utf-8') as f:
            indi_b_data = json.load(f)
    except Exception as e:
        print(f"加载results_indi_b.json时出错: {e}")
        indi_b_data = []
    
    return evaluation_data, default_data, indi_a_data, indi_b_data

# 分析评估结果
def analyze_evaluation_results(evaluation_data):
    results = {'A': 0, 'B': 0, 'C': 0, 'missing': 0}
    missing_entries = []
    
    for entry in evaluation_data:
        result = extract_evaluation_result(entry.get('evaluation', ''))
        if result in ['A', 'B', 'C']:
            results[result] += 1
        else:
            results['missing'] += 1
            missing_entries.append(entry.get('NO'))
    
    return results, missing_entries

# 分析default结果
def analyze_default_results(default_data):
    results = {'A': 0, 'B': 0, 'missing': 0}
    missing_entries = []
    
    for entry in default_data:
        result = extract_evaluation_result(entry.get('evaluation', ''))
        if result in ['A', 'B']:
            results[result] += 1
        else:
            results['missing'] += 1
            missing_entries.append(entry.get('NO'))
    
    return results, missing_entries

# 分析indi结果
def analyze_indi_results(indi_data):
    ratings = []
    missing_entries = []
    
    for entry in indi_data:
        rating = extract_rating(entry.get('evaluation'))
        if rating is not None:
            ratings.append(rating)
        else:
            missing_entries.append(entry.get('NO'))
    
    return ratings, missing_entries

# 可视化函数
def visualize_results(evaluation_results, default_results, indi_a_ratings, indi_b_ratings, 
                     evaluation_missing, default_missing, indi_a_missing, indi_b_missing):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建一个2x2的子图布局
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Default A/B选择总次数
    labels = ['A', 'B']
    values = [default_results['A'], default_results['B']]
    axs[0, 0].bar(labels, values, color=['blue', 'orange'])
    axs[0, 0].set_title('Default A/B选择总次数')
    axs[0, 0].set_ylabel('次数')
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(values):
        axs[0, 0].text(i, v + 0.1, str(v), ha='center')
    
    # 2. 两个模型的评分分布
    axs[0, 1].hist([indi_a_ratings, indi_b_ratings], bins=10, alpha=0.7, 
                  label=['Model A', 'Model B'])
    axs[0, 1].set_title('两个模型的评分分布')
    axs[0, 1].set_xlabel('评分')
    axs[0, 1].set_ylabel('频率')
    axs[0, 1].legend()
    
    # 计算并显示平均分
    avg_a = np.mean(indi_a_ratings) if indi_a_ratings else 0
    avg_b = np.mean(indi_b_ratings) if indi_b_ratings else 0
    axs[0, 1].text(0.05, 0.95, f'Model A 平均分: {avg_a:.2f}', transform=axs[0, 1].transAxes)
    axs[0, 1].text(0.05, 0.90, f'Model B 平均分: {avg_b:.2f}', transform=axs[0, 1].transAxes)
    
    # 3. Evaluation A,B,C的次数
    labels = ['A', 'B', 'C']
    values = [evaluation_results['A'], evaluation_results['B'], evaluation_results['C']]
    axs[1, 0].bar(labels, values, color=['blue', 'orange', 'green'])
    axs[1, 0].set_title('Evaluation A,B,C的次数')
    axs[1, 0].set_ylabel('次数')
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(values):
        axs[1, 0].text(i, v + 0.1, str(v), ha='center')
    
    # 4. 缺失数据统计
    labels = ['Evaluation', 'Default', 'Indi A', 'Indi B']
    values = [evaluation_results['missing'], default_results['missing'], 
              len(indi_a_missing), len(indi_b_missing)]
    axs[1, 1].bar(labels, values, color=['red', 'purple', 'brown', 'pink'])
    axs[1, 1].set_title('缺失数据统计')
    axs[1, 1].set_ylabel('缺失数量')
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(values):
        axs[1, 1].text(i, v + 0.1, str(v), ha='center')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('visualization_results.png')
    
    # 显示图表
    plt.show()
    
    # 打印缺失数据的详细信息
    print("\n缺失数据详情:")
    if evaluation_missing:
        print(f"Evaluation 缺失条目 NO: {evaluation_missing}")
    if default_missing:
        print(f"Default 缺失条目 NO: {default_missing}")
    if indi_a_missing:
        print(f"Indi A 缺失条目 NO: {indi_a_missing}")
    if indi_b_missing:
        print(f"Indi B 缺失条目 NO: {indi_b_missing}")

# 主函数
def main():
    print("开始加载数据...")
    evaluation_data, default_data, indi_a_data, indi_b_data = load_data()
    
    print("分析评估结果...")
    evaluation_results, evaluation_missing = analyze_evaluation_results(evaluation_data)
    default_results, default_missing = analyze_default_results(default_data)
    indi_a_ratings, indi_a_missing = analyze_indi_results(indi_a_data)
    indi_b_ratings, indi_b_missing = analyze_indi_results(indi_b_data)
    
    print("生成可视化图表...")
    visualize_results(evaluation_results, default_results, indi_a_ratings, indi_b_ratings,
                     evaluation_missing, default_missing, indi_a_missing, indi_b_missing)
    
    print("\n结果统计:")
    print(f"Evaluation 结果: A={evaluation_results['A']}, B={evaluation_results['B']}, C={evaluation_results['C']}, 缺失={evaluation_results['missing']}")
    print(f"Default 结果: A={default_results['A']}, B={default_results['B']}, 缺失={default_results['missing']}")
    
    if indi_a_ratings:
        print(f"Indi A 平均评分: {np.mean(indi_a_ratings):.2f}, 评分数量: {len(indi_a_ratings)}, 缺失: {len(indi_a_missing)}")
    else:
        print("Indi A 没有有效评分")
        
    if indi_b_ratings:
        print(f"Indi B 平均评分: {np.mean(indi_b_ratings):.2f}, 评分数量: {len(indi_b_ratings)}, 缺失: {len(indi_b_missing)}")
    else:
        print("Indi B 没有有效评分")

if __name__ == "__main__":
    main()
