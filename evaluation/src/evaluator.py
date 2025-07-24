import sys
import os
sys.path.append(os.getcwd())
import json
import asyncio
import datetime
import numpy as np
import re
from typing import List, Dict, Any, Union, Optional, Tuple
from collections import defaultdict
from tqdm.asyncio import tqdm as async_tqdm

from .metrics import (
    evaluate_math_prediction,
    evaluate_qa_prediction
)
from .llm_evaluator_sds import LLMEvaluator
# from .llm_evaluator import LLMEvaluator

class Evaluator:
    def __init__(
        self,
        task_type: str,
        output_path: str,
        use_llm: bool = False,
        api_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: str = "EMPTY",
        concurrent_limit: int = 50,
        sigma: float = 0.1
    ):
        """
        初始化评测器

        Args:
            task_type: 任务类型 ('math', 'qa')
            output_path: 模型输出JSON文件路径
            use_llm: 是否使用LLM评估
            api_base_url: LLM API基础URL
            model_name: LLM模型名称
            api_key: API密钥
            concurrent_limit: 并发限制
            sigma: 平滑因子
        """
        self.task_type = task_type
        self.output_path = output_path
        self.use_llm = use_llm
        self.concurrent_limit = concurrent_limit
        self.sigma = sigma

        # 输出路径
        base_path, ext = os.path.splitext(output_path)
        self.output_metrics_path = f"{base_path}_metrics.json"
        self.output_metrics_overall_path = f"{base_path}_metrics_overall.json"

        # 如果需要LLM评估，则初始化LLM评估器
        self.llm_evaluator = None
        if self.use_llm:
            self.llm_evaluator = LLMEvaluator(
                api_base_url=api_base_url,
                model_name=model_name,
                api_key=api_key,
                concurrent_limit=concurrent_limit
            )

    async def evaluate_sample(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个样本

        Args:
            item: 样本数据

        Returns:
            指标
        """
        question = item.get('input', '')
        answer = item.get('answer', '')
        prediction = item.get('prediction', '')
        output = item.get('output', '')
        # question = item.get('query', '')
        # answer = item.get('answer', '')
        # prediction = item.get('predict', '')
        # output = item.get('response', '')

        # 如果提取的答案为空，使用输出的最后几行
        if not prediction:
            if output:
                prediction = '\n'.join(output.replace(
                    "\n\n", "\n").strip().split('\n')[-5:])
            else:
                prediction = ''
        
        if not prediction:
            # 全部的指标都为0
            metrics = {
                "is_valid_answer": False,
                "em": 0,
                "acc": 0,
                "f1": 0,
                "math_equal": 0,
                "llm_equal": 0,
                "python_calls": 0,
                "search_calls": 0,
                "output_length": 0
            }
            
            # 设置工具使用指标
            python_calls = metrics["python_calls"]
            search_calls = metrics["search_calls"]
            metrics["tools_used"] = ("both" if python_calls and search_calls else
                             "python" if python_calls else
                             "search" if search_calls else "none")
            metrics["tool_counts"] = python_calls + search_calls
            
            return metrics

        # 初始化指标字典
        metrics = {
            "is_valid_answer": prediction != ''
        }

        # 工具调用统计分析
        python_calls = count_valid_tags(output, "python")
        search_calls = count_valid_tags(output, "search")

        metrics.update({
            "python_calls": python_calls,
            "search_calls": search_calls,
            "tools_used": ("both" if python_calls and search_calls else
                           "python" if python_calls else
                           "search" if search_calls else "none"),
            "tool_counts": python_calls + search_calls
        })

        # 响应长度统计分析
        metrics["output_length"] = len(remove_result_tags(output))

        # 根据任务类型进行评估
        if self.task_type == 'math':
            # 数学类评估
            math_metrics = evaluate_math_prediction(prediction, answer)
            metrics.update(math_metrics)

        elif self.task_type == 'qa':
            # QA类评估
            qa_metrics = evaluate_qa_prediction(prediction, answer)
            metrics.update(qa_metrics)
        else:   
            raise ValueError(f"Unsupported task type: {self.task_type}")

        # LLM评估
        if self.use_llm and self.llm_evaluator:
            semaphore = asyncio.Semaphore(self.concurrent_limit)
            is_correct, llm_reason_answer = await self.llm_evaluator.evaluate(
                question=question,  
                labeled_answer=answer,
                pred_answer=prediction,
                semaphore=semaphore
            )
            metrics["llm_equal"] = int(is_correct)
            metrics["llm_response"] = llm_reason_answer

        return metrics

    async def evaluate_batch(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量评估样本

        Args:
            data: 样本数据列表

        Returns:
            更新后的样本数据列表，包含评估指标
        """
        # 使用信号量控制并发
        semaphore = asyncio.Semaphore(self.concurrent_limit)
        
        async def _evaluate_with_semaphore(item):
            async with semaphore:
                metrics = await self.evaluate_sample(item)
                item_copy = item.copy()
                item_copy['metrics'] = metrics
                return item_copy
        
        # 显示进度
        tasks = [_evaluate_with_semaphore(item) for item in data]
        results = await async_tqdm.gather(*tasks, desc="评估样本")
        
        return results

    async def run(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        运行评估流程

        Args:
            data: 样本数据列表

        Returns:
            整体评估指标
        """
        print(f"开始评估 {len(data)} 个样本，任务类型: {self.task_type}")

        # 批量评估样本
        updated_data = await self.evaluate_batch(data)

        # 计算整体指标
        self.overall_metrics = self.calculate_overall_metrics(updated_data)

        # 添加元信息
        self.overall_metrics['datetime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 保存结果
        self.save_results(updated_data)

        return self.overall_metrics

    def calculate_overall_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, Union[float, str]]:
        """
        计算整体评估指标
        
        Args:
            data: 包含评估指标的样本数据列表
            
        Returns:
            整体评估指标字典
        """
        # 提取各项指标
        num_valid_answer = sum(item['metrics']['is_valid_answer'] for item in data)
        
        # 收集各指标的列表
        avg_em = [item['metrics'].get('em', 0) for item in data if 'em' in item['metrics']]
        avg_acc = [item['metrics'].get('acc', 0) for item in data if 'acc' in item['metrics']]
        avg_f1 = [item['metrics'].get('f1', 0) for item in data if 'f1' in item['metrics']]
        avg_math = [item['metrics'].get('math_equal', 0) for item in data if 'math_equal' in item['metrics']]
        avg_llm = [item['metrics'].get('llm_equal', 0) for item in data if 'llm_equal' in item['metrics']]
        
        # 工具使用统计
        avg_tool_counts = [item['metrics'].get('tool_counts', 0) for item in data]
        avg_python_calls = [item['metrics'].get('python_calls', 0) for item in data]
        avg_search_calls = [item['metrics'].get('search_calls', 0) for item in data]
        
        tool_usage_rate = sum(1 for count in avg_tool_counts if count > 0) / len(data) if data else 0
        avg_tool_count = np.mean(avg_tool_counts) if avg_tool_counts else 0
        
        # 根据任务类型计算相关指标
        if self.task_type == 'math':
            accuracy = np.mean(avg_math) if avg_math else 0.0
        elif self.task_type == 'qa':
            accuracy = np.mean(avg_f1) if avg_f1 else 0.0
        else:
            accuracy = 0.0
            
        # 计算工具生产力 (准确率 * 工具使用率 / (工具使用率 + sigma))
        if self.use_llm and avg_llm:
            final_accuracy = np.mean(avg_llm)
        else:
            final_accuracy = accuracy
            
        final_tool_productivity = final_accuracy * avg_tool_count / (avg_tool_count + self.sigma) * 100.0
        
        overall_metrics = {
            'em': np.mean(avg_em) if avg_em else 0.0,
            'acc': np.mean(avg_acc) if avg_acc else 0.0,
            'f1': np.mean(avg_f1) if avg_f1 else 0.0,
            'math_equal': np.mean(avg_math) if avg_math else 0.0,
            'num_valid_answer': f'{num_valid_answer} of {len(data)}',
            'tool_productivity': final_tool_productivity,
            'average_datas_used_tool_number': tool_usage_rate,
            'tool_call': avg_tool_count,
            'average_python_calls': np.mean(avg_python_calls) if avg_python_calls else 0.0,
            'average_search_calls': np.mean(avg_search_calls) if avg_search_calls else 0.0,
            'llm_equal': np.mean(avg_llm) if avg_llm else 0.0,
            'm1m2': final_tool_productivity,
        }
        return overall_metrics

    def save_results(self, data: List[Dict[str, Any]]):
        """
        保存评估结果

        Args:
            data: 更新后的样本数据列表
        """
        # 保存详细指标
        with open(self.output_metrics_path, mode='w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # 保存整体指标
        with open(self.output_metrics_overall_path, mode='w', encoding='utf-8') as f:
            json.dump(self.overall_metrics, f, indent=4, ensure_ascii=False)

        print(f"评估完成，结果已保存")
        print(f"详细指标: {self.output_metrics_path}")
        print(f"整体指标: {self.output_metrics_overall_path}")

        # 打印主要指标
        print(f"EM: {self.overall_metrics.get('em', 0):.4f}")
        print(f"F1: {self.overall_metrics.get('f1', 0):.4f}")
        print(f"有效回答数: {self.overall_metrics.get('num_valid_answer', '0')}")
        
        if self.task_type == 'math':
            print(f"Math Equal: {self.overall_metrics.get('math_equal', 0):.4f}")
        
        if 'llm_equal' in self.overall_metrics:
            print(f"LLM Equal: {self.overall_metrics['llm_equal']:.4f}")
        
        # 打印工具使用相关指标
        print(f"平均工具调用次数: {self.overall_metrics.get('tool_call', 0):.2f}")
        print(f"Python调用次数: {self.overall_metrics.get('average_python_calls', 0):.2f}")
        print(f"搜索调用次数: {self.overall_metrics.get('average_search_calls', 0):.2f}")
        print(f"使用工具的样本比例: {self.overall_metrics.get('average_datas_used_tool_number', 0):.2f}")
        print(f"工具生产力指标(M1*M2): {self.overall_metrics.get('tool_productivity', 0):.2f}")


def count_valid_tags(text: str, tag: str) -> int:
    """统计有效成对标签数量"""
    count = 0
    current_pos = 0

    while True:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"

        # 查找起始标签
        start_pos = text.find(start_tag, current_pos)
        if start_pos == -1:
            break

        # 查找对应的结束标签
        end_pos = text.find(end_tag, start_pos + len(start_tag))
        if end_pos == -1:
            break

        count += 1
        current_pos = end_pos + len(end_tag)

    return count


def remove_result_tags(text: str) -> str:
    """移除结果标签内的内容"""
    if not text:
        return ""
    # 移除<r>标签
    cleaned_text = re.sub(r'<r>.*?</r>', '', text, flags=re.DOTALL)
    # 移除<result>标签
    cleaned_text = re.sub(r'<result>.*?</result>', '', cleaned_text, flags=re.DOTALL)
    return cleaned_text.strip()

# 为兼容性保留旧函数，但使用新的实现
def remove_r_tags(text: str) -> str:
    """兼容性函数"""
    return remove_result_tags(text)
