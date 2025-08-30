import json
import pandas as pd
import openai
from math_verify import parse, verify
from dateutil import parser
def has_digit(s):
    for ch in s:
        if ch.isdigit():  # 检查字符是否是数字
            return True
    return False
def is_date(s):
    try:
        parser.parse(s, fuzzy=False)   # 严格解析
        return True
    except (ValueError, OverflowError):
        return False
parquet_files = [
    # "/share/home/sxjiang/myproject/Tool-Star-OCT/Tool_Star_RL/mix_grpo/grpo_mix_train_shuffle.parquet"
    "/home/sxjiang/myproject/agent/ARPO-OCT/ARPO/rl_datasets/train_10k.parquet"
]
dataframes = []
for parquet_file in parquet_files:
    # read parquet files and cache
    dataframe = pd.read_parquet(parquet_file)
    dataframes.append(dataframe)
rl_data = pd.concat(dataframes)
math_item=0
qa_item=0
new_rl_data=[]
nooxed_num=0
re_search_template_with_budget_sys = """You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool and python interpreter tool. \
Given a question, you need to first think about the reasoning process in the mind and then provide the answer. \
During thinking, you can invoke the wikipedia search tool to search and python interpreter tool to calculate the math problem for fact information about specific topics if needed. \
You should make every tool call count and obtain useful results, considering that each Python interpreter call costs [python_cost] and each search call costs [search_cost]. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, \
and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. \
For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> \
<think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> \
<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."""

re_search_template = """You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool and python interpreter tool. \
Given a question, you need to first think about the reasoning process in the mind and then provide the answer. \
During thinking, you can invoke the wikipedia search tool to search and python interpreter tool to calculate the math problem for fact information about specific topics if needed. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, \
and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. \
For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> \
<think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> \
<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."""


for idx in range(len(rl_data)):
    data_item = rl_data.iloc[idx].to_dict()
    # print(data_item.keys())
    answer = data_item["reward_model"]["ground_truth"]
    new_prompt = []
    old_prompt = data_item["prompt"]
    for line in old_prompt:
        if "\x08oxed" in line["content"]:
            nooxed_num+=1
            print("No \boxed")
        if line["role"]=="system":
            new_prompt.append({"role": "system", "content": re_search_template_with_budget_sys})
        else:
            new_prompt.append({"role": line["role"], "content": line["content"].replace("\x08oxed","\\boxed")})
    data_item["prompt"] = new_prompt
    new_rl_data.append(data_item)
    # if len(answer)==1:
    #     print(answer)
    # if data_item["ability"]=="math":
    if has_digit(answer):
        if is_date(answer):
            qa_item+=1
        else:
            math_item+=1
            if data_item["ability"]!="math":
                pass
    else:
        qa_item+=1
        if data_item["ability"]!="qa":
            pass
    # if data_item["ability"]=="qa":
        # qa_item+=1
    # if "A." in question:
    #     print(question)
    #     print(answer)
new_rl_data = pd.DataFrame(new_rl_data)
new_rl_data.to_parquet("train_10k_boxed_budget.parquet")
print(f"math_item:{math_item}")
print(f"qa_item:{qa_item}")
print(f"nooxed_num:{nooxed_num}")