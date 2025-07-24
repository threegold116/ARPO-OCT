import sys
import os
sys.path.append(os.getcwd())

import re
import string
from typing import Dict, Any, List, Union, Optional, Tuple, Set
from collections import Counter

from .math_equivalence import is_equiv


def normalize_answer(text: str, remove_articles: bool = False, remove_punctuations: bool = False) -> str:
    """
    标准化答案文本

    Args:
        text: 原始答案文本
        remove_articles: 是否移除冠词 (a/an/the)
        remove_punctuations: 是否移除标点符号

    Returns:
        标准化后的答案
    """
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower().strip()
    
    if remove_articles:
        text = re.sub(r"\b(a|an|the)\b", " ", text)
    
    # 移除标点符号
    if remove_punctuations:
        exclude = set(string.punctuation)
        text = "".join(ch for ch in text if ch not in exclude)
    
    # 修复空格
    return " ".join(text.split())


def compute_token_overlap(prediction: str, reference: str) -> Tuple[int, int, int]:
    """
    计算预测和参考答案之间的token重叠

    Args:
        prediction: 预测答案
        reference: 参考答案

    Returns:
        (重叠token数, 预测token数, 参考token数)的元组
    """
    prediction_tokens = prediction.split()
    reference_tokens = reference.split()
    
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    num_same = sum(common.values())
    
    return num_same, len(prediction_tokens), len(reference_tokens)


def compute_f1_score(num_same: int, pred_len: int, ref_len: int) -> float:
    """
    计算F1分数

    Args:
        num_same: 重叠token数
        pred_len: 预测token数
        ref_len: 参考token数

    Returns:
        F1分数 (0-1)
    """
    # 如果没有共同token，F1为0
    if num_same == 0:
        return 0.0

    precision = num_same / pred_len if pred_len > 0 else 0
    recall = num_same / ref_len if ref_len > 0 else 0

    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def evaluate_math_prediction(
    prediction: str,
    reference: str
) -> Dict[str, Union[int, float]]:
    """
    评估数学预测结果

    Args:
        prediction: 预测答案
        reference: 参考答案

    Returns:
        包含评估指标的字典
    """
    # 标准化答案
    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference)

    # 计算精确匹配和准确率
    em = int(normalized_prediction == normalized_reference)
    acc = int(normalized_reference in normalized_prediction)
    
    # 计算F1分数
    num_same, pred_len, ref_len = compute_token_overlap(normalized_prediction, normalized_reference)
    f1 = compute_f1_score(num_same, pred_len, ref_len)

    math_equal = int(is_equiv(normalized_prediction, normalized_reference))

    return {
        "em": em,
        "acc": acc,
        "f1": f1,
        "math_equal": math_equal
    }


def evaluate_qa_prediction(
    prediction: str,
    references: Union[str, List[str]]
) -> Dict[str, Union[int, float]]:
    """
    评估QA预测结果

    Args:
        prediction: 预测答案
        references: 参考答案或参考答案列表

    Returns:
        包含评估指标的字典
    """
    # 确保references是列表
    if isinstance(references, str):
        if not references.startswith("["):
            references = [references]
        else:
            # 尝试解析字符串列表
            references = [e.strip() for e in re.split(r",\s*", references.strip('[]'))]

    # 初始化结果
    result = {"em": 0, "acc": 0, "f1": 0, "math_equal": 0}
    
    # 标准化预测答案 (对QA使用更严格的标准化)
    normalized_prediction = normalize_answer(prediction, remove_articles=True)

    # 对每个参考答案计算指标，取最高分
    for reference in references:
        normalized_reference = normalize_answer(reference, remove_articles=True)

        # 计算精确匹配和准确率
        em = int(normalized_prediction == normalized_reference)
        acc = int(normalized_reference in normalized_prediction)
        
        # 计算F1分数
        num_same, pred_len, ref_len = compute_token_overlap(normalized_prediction, normalized_reference)
        f1 = compute_f1_score(num_same, pred_len, ref_len)

        # 更新结果，取最高分
        result["em"] = max(result["em"], em)
        result["acc"] = max(result["acc"], acc)
        result["f1"] = max(result["f1"], f1)

        math_equal = int(is_equiv(normalized_prediction, normalized_reference))
        result["math_equal"] = max(result["math_equal"], math_equal)


    return result
