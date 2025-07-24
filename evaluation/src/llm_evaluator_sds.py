import sys
import os
sys.path.append(os.getcwd())
import asyncio
import time
from typing import List, Tuple, Dict, Any, Optional

from openai import OpenAI, AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

from .math_equivalence import is_equiv


EVALUATION_PROMPT = """Given a Question and its Golden Answer, verify whether the Predicted Answer is correct. The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer. Respond with "Correct" if the prediction is correct and "Incorrect" otherwise.
Golden Answer may have multiple options, and matching any one of them is considered correct.

Question: {question}
Golden Answer: {labeled_answer}
Predicted Answer: {pred_answer}
"""


class LLMEvaluator:
    """使用LLM评估答案等价性的类"""

    def __init__(
        self,
        api_base_url: str = None,
        model_name: str = None,
        api_key: str = "empty",
        concurrent_limit: int = 50,
        retry_limit: int = 3
    ):
        """
        初始化LLM评估器

        Args:
            api_base_url: API基础URL，默认为本地服务
            model_name: 模型名称
            api_key: API密钥
            concurrent_limit: 并发限制
            retry_limit: 重试次数限制
        """
        if api_base_url is None:
            api_base_url = "http://localhost:8000/v1"
        if model_name is None:
            model_name = "/mmu_nlp_ssd/makai05/open_models/Qwen2.5-7B-Instruct"

        self.api_base_url = api_base_url
        self.model_name = model_name
        self.api_key = api_key
        self.concurrent_limit = concurrent_limit
        self.retry_limit = retry_limit

        # 创建OpenAI客户端
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base_url
        )

    async def evaluate(
        self,
        question: str,
        labeled_answer: str,
        pred_answer: str,
        semaphore: asyncio.Semaphore
    ) -> Tuple[bool, str]:
        """
        评估单个答案对

        Args:
            question: 问题
            labeled_answer: 标记答案
            pred_answer: 预测答案
            semaphore: 控制并发的信号量

        Returns:
            (是否正确, 评估原因)
        """
        global EVALUATION_PROMPT
        prompt = EVALUATION_PROMPT.format(
            question=question,
            labeled_answer=labeled_answer,
            pred_answer=pred_answer
        )

        for attempt in range(self.retry_limit):
            try:
                async with semaphore:
                    chat_response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}]
                    )

                    reason_answer = chat_response.choices[0].message.content.strip()

                    # 尝试从回复中提取判断结果
                    try:
                        start = reason_answer.index("<judgment>") + len("<judgment>")
                        end = reason_answer.index("</judgment>")
                        response_text = reason_answer[start:end].strip()
                    except:
                        response_text = reason_answer.strip()

                    # 分析判断结果
                    is_correct = is_equiv(pred_answer, labeled_answer) or \
                        "correct" in response_text.lower() and \
                        not ("incorrect" in response_text.lower() or
                             "wrong" in response_text.lower() or
                             "not correct" in response_text.lower())

                    return is_correct, reason_answer
            except Exception as e:
                if attempt == self.retry_limit - 1:
                    print(f"Error in LLM evaluation: {e}")
                    print(
                        f"-------------------pred_answer: {pred_answer}----------------------")
                    print(
                        f"-------------------labeled_answer: {labeled_answer}----------------------")
                    return is_equiv(pred_answer, labeled_answer), "Error"
                await asyncio.sleep(1 * (attempt + 1))

        return is_equiv(pred_answer, labeled_answer), "Error"
