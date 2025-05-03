from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image, display
import os

# 替换原有llm初始化代码
model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 定义状态
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 创建 graph
graph_builder = StateGraph(State)

def chatbot(state: State, system_message: str):
    """
    Chatbot函数，处理状态并生成响应。
    参数:
    state (State): 当前状态，包含消息列表。
    system_message (str): 系统消息，指示AI如何回答。
    返回:
    dict: 包含AI消息的字典。
    """
    # 获取用户输入
    user_input = state["messages"][-1].content

    # 构建对话历史（包含 system/user/assistant）
    conversation = [{"role": "system", "content": system_message}]
    for msg in state["messages"]:
        if msg.type == "system":
            conversation.append({"role": "system", "content": msg.content})
        elif msg.type == "human":
            conversation.append({"role": "user", "content": msg.content})
        elif msg.type == "ai":
            conversation.append({"role": "assistant", "content": msg.content})

    # 应用 Qwen 的 chat 模板
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    # 生成响应
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        num_beams=5,#范围在1-10之间
        temperature=1.6,#范围在0.1-2之间，默认值1.0
        top_k=100,#范围在1-100之间,
        top_p=0.9,#范围在0.1-1之间
        repetition_penalty=1.5,#范围在1.0-2.0之间


    )

    # 解码响应
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    # 返回 AIMessage 实例
    return {"messages": [AIMessage(content=response)]}

# 定义节点以及边
graph_builder.add_node("1", lambda state: chatbot(state, "B1：我会输入体征，请从呼吸系统角度根据狗生命体征进行约200tokens的简短推理，输出一个最可能症状，症状必须来源于狗的常见症状而非生命体征，输出格式如下：'B1:...'"))
graph_builder.add_node("2", lambda state: chatbot(state, "B2：我会输入体征，请从体温代谢角度根据狗生命体征进行约200tokens的简短推理，输出一个最可能症状，症状必须来源于狗的常见症状而非生命体征，输出格式如下：'B2:...'"))
graph_builder.add_node("3", lambda state: chatbot(state, "B3：我会输入体征，请从血液循环免疫系统角度根据狗生命体征进行约200tokens的简短推理，输出一个最可能症状，症状必须来源于狗的常见症状而非生命体征，输出格式如下：'B3:...'"))
graph_builder.add_node("4", lambda state: chatbot(state, "C1：为B1 B2 B3分别进行总结摘要，简短输出三段摘要，输出格式如下：'1. B1:...2. B2:...3. B3...'"))
# 定义边
graph_builder.add_edge(START, "1")
graph_builder.add_edge(START, "2")
graph_builder.add_edge(START, "3")
graph_builder.add_edge("1", "4")
graph_builder.add_edge("2", "4")
graph_builder.add_edge("3", "4")
graph_builder.add_edge("4", END)


graph = graph_builder.compile()

# 可视化展示并保存工作流图
try:
    # 获取 PNG 数据
    graph_image_data = graph.get_graph().draw_mermaid_png()

    # 展示图像
    display(Image(data=graph_image_data))

    # 保存图像到文件
    with open("workflow_diagram.png", "wb") as f:
        f.write(graph_image_data)
    print("工作流图已保存为 workflow_diagram.png")
except Exception as e:
    print(f"发生错误：{e}")

# 初始化系统指令


while True:
    user_input = input("请输入您的问题：")
    # 初始化历史,添加用户消息
    initial_messages = [SystemMessage(content="我会输入一组狗的生命体征，请根据以下步骤生成不同的、多样化的回答，考虑每一种可能性；"),
                        HumanMessage(content=user_input),
    ]
    # 向 graph传入含system+user+ai的历史
    for message in graph.stream({"messages": initial_messages}):
        for value in message.values():
            content = value["messages"][-1].content # 提取最新的消息内容
            print(f"AI: \n{content}\n","-"*200)
            # 将 AI 回复也加入历史，实现上下文连续
            initial_messages.append(AIMessage(content=content))
