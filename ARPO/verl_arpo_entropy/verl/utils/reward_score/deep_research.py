import re
import sys
import string
from typing import Union, List, Dict, Any, Optional, Tuple
from collections import Counter
from transformers import AutoTokenizer
from math_verify import parse, verify


SUBSTITUTIONS = [
    ("an ", ""),
    # ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
    ("\\left", ""),
    ("\\right", ""),
    ("∶", ":"),
    ("，", ","),
    ("$",  ""),
    ("\\approx", "="),
    ("\\simeq", "="),
    ("\\sim", "="),
    ("^\\prime", "'"),
    ("^{\\prime}", "'"),
    ("\\dfrac", "\\frac"),
    ("\\tfrac", "\\frac"),
    ("^\\circ", ""),
    ("%", ""),
    ("\u221a", "\\sqrt"),
    ("\u221e", "\\infty"),
    ("\u222a", "\\cup"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    # "ft", #this is dangerous, infty, left will be damaged!
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]
def validate_format(text: str) -> Tuple[bool, str]:
    """
    Validate if the text follows the required format with paired tags.
    
    Args:
        text: The text to validate
        
    Returns:
        tuple: (is_valid, reason)
    """
    # Check if <think></think>, <answer></answer> is paired
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> not paired"

    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "<think> or </think> not found"

    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> not found"

    # Check the order of search/result
    current_pos = 0
    while True:
        search_pos = text.find('<search>', current_pos)
        if search_pos == -1:
            break

        result_pos = text.find('<result>', search_pos)
        search_end_pos = text.find('</search>', search_pos)
        result_end_pos = text.find('</result>', result_pos)

        if -1 in (result_pos, search_end_pos, result_end_pos):
            return False, "search/result tags are incomplete"

        if not (search_pos < search_end_pos < result_pos < result_end_pos):
            return False, "search/result tags are nested in the wrong order"

        current_pos = result_end_pos

    # Check the order of python/result
    current_pos = 0
    while True:
        python_pos = text.find('<python>', current_pos)
        if python_pos == -1:
            break

        result_pos = text.find('<result>', python_pos)
        python_end_pos = text.find('</python>', python_pos)
        result_end_pos = text.find('</result>', result_pos)

        if -1 in (result_pos, python_end_pos, result_end_pos):
            return False, "python/result tags are incomplete"

        if not (python_pos < python_end_pos < result_pos < result_end_pos):
            return False, "python/result tags are nested in the wrong order"

        current_pos = result_end_pos

    # Check if \boxed{} is in the answer
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\boxed{} format"

    return True, "format is correct"



def validate_format_v2(text: str) -> Tuple[bool, str]:
    """
    Validate if the text follows the required format with paired tags.
    
    Args:
        text: The text to validate
        
    Returns:
        tuple: (is_valid, reason)
    """
    # Check if <think></think>, <answer></answer> is paired
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> not paired"

    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "<think> or </think> not found"

    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> not found"

    # Check the order of search/result
    current_pos = 0
    while True:
        search_pos = text.find('<search>', current_pos)
        if search_pos == -1:
            break

        result_pos = text.find('<result>', search_pos)
        search_end_pos = text.find('</search>', search_pos)
        result_end_pos = text.find('</result>', result_pos)

        if -1 in (result_pos, search_end_pos, result_end_pos):
            return False, "search/result tags are incomplete"

        if not (search_pos < search_end_pos < result_pos < result_end_pos):
            return False, "search/result tags are nested in the wrong order"

        current_pos = result_end_pos

    # Check the order of python/result
    current_pos = 0
    while True:
        python_pos = text.find('<python>', current_pos)
        if python_pos == -1:
            break

        result_pos = text.find('<result>', python_pos)
        python_end_pos = text.find('</python>', python_pos)
        result_end_pos = text.find('</result>', result_pos)

        if -1 in (result_pos, python_end_pos, result_end_pos):
            return False, "python/result tags are incomplete"

        if not (python_pos < python_end_pos < result_pos < result_end_pos):
            return False, "python/result tags are nested in the wrong order"

        current_pos = result_end_pos
    #check the order of think/search
    think_end = 0
    while True:
        think_end = text.find('</think>', think_end)
        if think_end == -1:
            break
        
        # 从 </think> 之后开始，跳过空白字符
        next_pos = think_end + len('</think>')
        while next_pos < len(text) and text[next_pos].isspace():
            next_pos += 1
        
        # 检查后续标签是否合法
        if not (text.startswith('<python>', next_pos) or 
                text.startswith('<search>', next_pos) or 
                text.startswith('<answer>', next_pos)):
            return False, "Invalid tag after </think>; must be <python>, <search>, or <answer>"
        
        think_end = next_pos
        
    # Check if \boxed{} is in the answer
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\boxed{} format"

    return True, "format is correct"




def validate_format_python(text: str) -> Tuple[bool, str]:
    """
    Validate if the text follows the required format for Python with paired tags.
    
    Args:
        text: The text to validate
        
    Returns:
        tuple: (is_valid, reason)
    """
    # Check if <think></think>, <answer></answer> is paired
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> not paired"

    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "<think> or </think> not found"

    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> not found"

    # Check the order of search/result
    current_pos = 0
    while True:
        search_pos = text.find('<python>', current_pos)
        if search_pos == -1:
            break

        result_pos = text.find('<result>', search_pos)
        search_end_pos = text.find('</python>', search_pos)
        result_end_pos = text.find('</result>', result_pos)

        if -1 in (result_pos, search_end_pos, result_end_pos):
            return False, "python/result tags are incomplete"

        if not (search_pos < search_end_pos < result_pos < result_end_pos):
            return False, "python/result tags are nested in the wrong order"

        current_pos = result_end_pos

    # Check if \boxed{} is in the answer
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\boxed{} format"

    return True, "format is correct"


def extract_answer(text: str) -> Optional[str]:
    """
    Extract answer content from the text within <answer> tags.
    
    Args:
        text: The text to extract answer from
        
    Returns:
        Optional[str]: The extracted answer or None if no match
    """
    text = text.strip()

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    return match.group(1)


def remove_boxed(s: str) -> str:
    """
    Remove the LaTeX \boxed{} wrapper from the string.
    
    Args:
        s: String potentially containing \boxed{}
        
    Returns:
        str: String with \boxed{} removed
    """
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string: str) -> Optional[str]:
    """
    Extract the last \boxed{} content from the string.
    
    Args:
        string: String to extract \boxed{} from
        
    Returns:
        Optional[str]: The extracted \boxed{} content or None if not found
    """
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def normalize_answer(s: str) -> str:
    """
    Normalize the answer string by removing articles, white spaces, punctuation and converting to lowercase.
    
    Args:
        s: String to normalize
        
    Returns:
        str: Normalized string
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score

def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.
    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    # final_answer = final_answer.split("=")[-1]
    final_answer=final_answer.strip()
    if final_answer[:2]=="\\(" or final_answer[:2]=='\\[':
        final_answer=final_answer[2:]
    if final_answer[-2:]=='\\)' or final_answer[-2:]=='\\]':
        final_answer=final_answer[:-2]
    
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    if final_answer[:2]=="\\(" or final_answer[:2]=='\\[':
        final_answer=final_answer[2:]
    if final_answer[-2:]=='\\)' or final_answer[-2:]=='\\]':
        final_answer=final_answer[:-2]
    return final_answer.strip()
# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False
    if str1.strip().lower() == str2.strip().lower(): return True
    try:
        str1=normalize_final_answer(str1)
        str2=normalize_final_answer(str2)
        str1=parse(str1)
        str2=parse(str2)
        return verify(str1, str2)
    except:
        pass

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1==ss2
    except Exception:
        return str1 == str2

def get_f1_score(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    """
    Calculate F1 score between prediction and ground truths.
    
    Args:
        prediction: The predicted answer
        ground_truths: The ground truth answer(s)
        
    Returns:
        float: F1 score
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    final_metric = {"f1": 0, "precision": 0, "recall": 0}

    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])

    return final_metric['f1']


def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compute reward score for a solution based on the ground truth.
    
    Args:
        data_source: The data source identifier
        solution_str: The solution string to evaluate
        ground_truth: The ground truth answer(s)
        extra_info: Optional additional information
        
    Returns:
        Dict[str, Any]: A dictionary containing the score and additional information
    """
    
    # 初始化统一的返回结构
    result = {
        "score": 0,
        "reason": "",
        "answer": "",
        "f1_score": 0
    }
    
    response = solution_str
    valid_template, reason = validate_format(response)
    
    if not valid_template:
        print(f"--------------------------------bad format: {reason}--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1
        result["reason"] = f"bad format: {reason}"
        return result
    
    # Remove EOS token if present
    if extra_info is not None and "tokenizer" in extra_info and extra_info["tokenizer"].eos_token and response.endswith(extra_info["tokenizer"].eos_token):
        response = response[:-len(extra_info["tokenizer"].eos_token)]
    
    answer_part = extract_answer(response)
    if answer_part is None:
        print(f"--------------------------------cannot extract answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1
        result["reason"] = "cannot extract answer"
        return result
    
    try:
        answer = remove_boxed(last_boxed_only_string(answer_part))
        result["answer"] = answer
    except Exception as e:
        print(f"--------------------------------find box error: {e}--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1
        result["reason"] = f"find box error: {e}"
        return result
    
    f1_score = get_f1_score(answer, ground_truth)
    result["f1_score"] = f1_score
    print(f"f1_score: {f1_score}, answer: {answer}, ground_truth: {ground_truth}")
    
    # Bonus for using multiple tools correctly
    if f1_score > 0 and "</search>" in response and "</python>" in response:
        print(f"--------------------------------correct answer with multi tool call--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = f1_score + 0.1
        result["reason"] = f"correct answer and calling search and python at the same time, get score: {f1_score + 0.1}"
    elif f1_score > 0:
        print(f"--------------------------------correct answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = f1_score
        result["reason"] = f"correct answer, get f1 score: {f1_score}"
    else:
        print(f"--------------------------------wrong answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = 0
        result["reason"] = f"wrong answer but good format: {answer}"
    
    return result



def compute_score_binary_mix(data_source: str, solution_str: str, ground_truth: Any, extra_info: Optional[Dict[str, Any]] = None, abality: str = None, mix_rules: bool = False, qa_rule: str = "f1_score",math_rule: str = "em_score", is_multi_tool: bool = False, binary_f1_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute reward score for a solution based on the ground truth.
    
    Args:
        data_source: The data source identifier
        solution_str: The solution string to evaluate
        ground_truth: The ground truth answer(s)
        extra_info: Optional additional information
        
    Returns:
        Dict[str, Any]: A dictionary containing the score and additional information
    """
    
    # 初始化统一的返回结构
    result = {
        "score": 0,
        "reason": "",
        "answer": "",
        "score_rule": "f1_score"
    }
    
    response = solution_str
    valid_template, reason = validate_format_v2(response)
    
    if not valid_template:
        print(f"--------------------------------bad format: {reason}--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1
        result["reason"] = f"bad format: {reason}"
        return result
    
    # Remove EOS token if present
    if extra_info is not None and "tokenizer" in extra_info and extra_info["tokenizer"].eos_token and response.endswith(extra_info["tokenizer"].eos_token):
        response = response[:-len(extra_info["tokenizer"].eos_token)]
    
    answer_part = extract_answer(response)
    if answer_part is None:
        print(f"--------------------------------cannot extract answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1
        result["reason"] = "cannot extract answer"
        return result
    
    try:
        answer = remove_boxed(last_boxed_only_string(answer_part))
        result["answer"] = answer
    except Exception as e:
        print(f"--------------------------------find box error: {e}--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1
        result["reason"] = f"find box error: {e}"
        return result
    # THREEGOLDCHANGE: 增加exact match score
    acc_score = 0
    if qa_rule=="f1_score":
        f1_score = get_f1_score(answer, ground_truth)
        acc_score = f1_score
    elif qa_rule=="em_score":
        em_score = em_check(answer, ground_truth)
        acc_score = em_score
    elif qa_rule=="binary_f1":
        f1_score = get_f1_score(answer, ground_truth)
        em_score = em_check(answer, ground_truth)
        if f1_score >= binary_f1_threshold or em_score == 1:
            acc_score = 1
        else:
            acc_score = 0
    else:
        raise ValueError(f"Invalid qa rule mode: {qa_rule}")
    
    if mix_rules:
        if abality == "math":
            qa_rule = f"math_{math_rule}"
            if math_rule=="em_score":
                em_score = is_equiv(answer, ground_truth)
                acc_score = em_score
            elif math_rule=="f1_score":
                f1_score = get_f1_score(answer, ground_truth)
                acc_score = f1_score
            elif math_rule=="binary_f1":
                f1_score = get_f1_score(answer, ground_truth)
                em_score = is_equiv(answer, ground_truth)
                if f1_score >= binary_f1_threshold or em_score == 1:
                    acc_score = 1
                else:
                    acc_score = 0
        else:
            acc_score = acc_score
    
    result["score_rule"] = qa_rule
    print(f"{qa_rule} score: {acc_score}, answer: {answer}, ground_truth: {ground_truth}")
    
    # Bonus for using multiple tools correctly
    if acc_score > 0 and "</search>" in response and "</python>" in response and is_multi_tool:
        print(f"--------------------------------correct answer with multi tool call--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = acc_score + 0.1
        result["reason"] = f"correct answer and calling search and python at the same time, get score: {f1_score + 0.1}"
    elif acc_score > 0:
        print(f"--------------------------------correct answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = acc_score
        result["reason"] = f"correct answer, get {qa_rule} score: {acc_score}"
    else:
        print(f"--------------------------------wrong answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = 0
        result["reason"] = f"wrong answer but good format: {answer}"
    
    return result



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("{path_of_your_model}/Qwen2.5-1.5B-Instruct")
    response = "<|im_start|>system\nYou are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool and python interpreter tool. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, you can invoke the wikipedia search tool to search and python interpreter tool to calculate the math problem for fact information about specific topics if needed. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> <think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format.<|im_end|>\n<|im_start|>user\nIn each cell, a strip of length $100$ is worth a chip. You can change any $2$ neighboring chips and pay $1$ rouble, and you can also swap any $2$ chips for free, between which there are exactly $4$ chips. What is the smallest amount of rubles you can spend to rearrange the chips in reverse order?<|im_end|>\n<|im_start|>assistant\n<think> To solve the given problem, we will use both natural language reasoning and Python code. Let's break down the problem step-by-step.\n\n1. **Given Equations:**\n   \\[\n   \\log_{10}(\\sin x) + \\log_{10}(\\cos x) = -1\n   \\]\n   Using the properties of logarithms, we can combine the logs:\n   \\[\n   \\log_{10}(\\sin x \\cos x) = -1 \\implies \\sin x \\cos x = 10^{-1} = \\frac{1}{10}\n   \\]\n   Recall the double-angle identity for sine: \\(\\sin(2x) = 2 \\sin x \\cos x\\). So,\n   \\[\n   \\sin(2x) = 2 \\left(\\frac{1}{10}\\right) = \\frac{1}{5}\n   \\]\n\n2. **Second Given Equation:**\n   \\[\n   \\log_{10}(\\sin x + \\cos x) = \\frac{1}{2}(\\log_{10}(n) - 1)\n   \\]\n   We can rewrite the right-hand side using properties of logarithms:\n   \\[\n   \\log_{10}(\\sin x + \\cos x) = \\frac{1}{2}\\log_{10}(n) - \\frac{1}{2} = \\log_{10}(n^{1\/2}) - \\log_{10}(10^{1\/2}) = \\log_{10}\\left(\\frac{\\sqrt{n}}{\\sqrt{10}}\\right)\n   \\]\n   This implies:\n   \\[\n   \\sin x + \\cos x = \\frac{\\sqrt{n}}{\\sqrt{10}}\n   \\]\n   Squaring both sides, we get:\n   \\[\n   (\\sin x + \\cos x)^2 = \\frac{n}{10}\n   \\]\n   Expanding the left side:\n   \\[\n   \\sin^2 x + \\cos^2 x + 2 \\sin x \\cos x = \\frac{n}{10}\n   \\]\n   Using the Pythagorean identity \\(\\sin^2 x + \\cos^2 x = 1\\) and the earlier result \\(\\sin x \\cos x = \\frac{1}{10}\\):\n   \\[\n   1 + 2 \\left(\\frac{1}{10}\\right) = \\frac{n}{10} \\implies 1 + \\frac{1}{5} = \\frac{n}{10} \\implies \\frac{6}{5} = \\frac{n}{10} \\implies n = 12\n   \\]\n\nNow, let's verify this with Python code to ensure the result is accurate.\n </think><python>\nimport math\r\nfrom sympy import symbols, Eq, solve, log, sin, cos\r\n\r\n# Define the variable\r\nx = symbols('x')\r\n\r\n# First equation: log10(sin(x)) + log10(cos(x)) = -1\r\neq1 = Eq(log(sin(x), 10) + log(cos(x), 10), -1)\r\n\r\n# Solve for sin(x) * cos(x)\r\nsin_x_cos_x = solve(eq1, sin(x) * cos(x))[0]\r\nsin_x_cos_x = sin_x_cos_x.simplify()\r\n\r\n# We know sin(x) * cos(x) = 1\/10\r\nsin_x_cos_x_value = 1 \/ 10\r\n\r\n# Second equation: log10(sin(x) + cos(x)) = 1\/2 * (log10(n) - 1)\r\nn = symbols('n')\r\neq2 = Eq(log(sin(x) + cos(x), 10), (1\/2) * (log(n, 10) - 1))\r\n\r\n# We know (sin(x) + cos(x))^2 = 1 + 2 * sin(x) * cos(x) = 1 + 2 * (1\/10) = 6\/5\r\nsin_x_plus_cos_x_squared = 6 \/ 5\r\nsin_x_plus_cos_x = math.sqrt(sin_x_plus_cos_x_squared)\r\n\r\n# Substitute sin(x) + cos(x) into the second equation\r\nlhs = log(sin_x_plus_cos_x, 10)\r\nrhs = (1\/2) * (log(n, 10) - 1)\r\n\r\n# Solve for n\r\nn_value = solve(Eq(lhs, rhs), n)[0]\r\nprint(n_value)\n</python><result>\n\n12.0000000000000\n</result><search> query </search><result> result </result><answer>The value of \\( n \\) is \\(\\boxed{12}\\). The calculations and the verification using Python both confirm that the final answer is indeed \\( n = 12 \\).</answer><|im_end|>"

    answer = "12"
    extra_info = {"tokenizer": tokenizer}
    res = compute_score("test_data_source", response, answer, extra_info)
    print(res)
