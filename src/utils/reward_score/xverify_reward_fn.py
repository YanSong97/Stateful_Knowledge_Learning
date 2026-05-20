import re
import random
import os
from typing import Dict, Any, Optional, Union, List

from openai import OpenAI

from examples.reward.test_xverify import call_xverify_model

XVERIFY_CONFIG={
    'model_name': "xverify",
    "url": [],
    "api_key": None,
    "temperature": 0.,
    "max_tokens": 1024,
}


def _load_xverify_config_from_env() -> Dict[str, Any]:
    cfg = dict(XVERIFY_CONFIG)
    urls = os.environ.get("XVERIFY_URLS") or os.environ.get("XVERIFY_URL")
    if urls:
        cfg["url"] = [url.strip() for url in urls.split(",") if url.strip()]
    cfg["api_key"] = os.environ.get("XVERIFY_API_KEY")
    cfg["model_name"] = os.environ.get("XVERIFY_MODEL_NAME", cfg["model_name"])
    return cfg

evaluation_prompt=\
'''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
-
Special considerations:

1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.

4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
-

Question: """{query}"""

Output sentence: """{generated_output}"""

Correct answer: {ground_truth}

Judgement:
'''

def extract_solution(solution_str):
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if answer_matches:
        solution_str = answer_matches[-1].group(1).strip()
        valid_answer_format = 1.0
    else:
        valid_answer_format = 0.0

        # if <search> tag is detected, remove search content
        if "</information>" in solution_str:
            solution_str = solution_str.split("</information>")[-1].strip()


        # if no <answer> tag is detected, remove <think> tag content
        if "</think>" in solution_str:
            solution_str = solution_str.split('</think>')[-1].strip()
        # else:
        #     solution_str = solution_str[-100:]      # if no format is detected, get the last 100

    return solution_str, valid_answer_format


def extract_solution_version2(solution_str):
    """
    https://github.com/SJTU-IAAR/verl/blob/99137c1fc8b7d9d9e734f52313e7533e7207182b/debug_reward_issues.py
    :param solution_str:
    :return:
    """
    original_solution_str = solution_str

    # look for answer tag
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if answer_matches:
        # use the last tag
        extracted_content = answer_matches[-1].group(1).strip()
        if extracted_content:
            return extracted_content

    # look for box tag
    boxed_pattern = r'\\?boxed\{'
    boxed_matches = []

    # search all boxed
    start_pos = 0
    while True:
        match = re.search(boxed_pattern, solution_str[start_pos:])
        if not match:
            break

        # starting position
        abs_start = start_pos + match.end()

        # manually match bracket to handle nesting
        brace_count = 1
        current_pos = abs_start

        while current_pos < len(solution_str) and brace_count > 0:
            if solution_str[current_pos] == '{':
                brace_count += 1
            elif solution_str[current_pos] == '}':
                brace_count -= 1
            current_pos += 1

        if brace_count == 0:
            # match complete box content
            content = solution_str[abs_start:current_pos - 1]
            if content.strip():  # not empty
                boxed_matches.append(content)

        start_pos = abs_start

    if boxed_matches:
        # for nested bracket
        final_answer = boxed_matches[-1]

        inner_boxed = extract_solution_version2(final_answer)
        if inner_boxed is not None:
            return inner_boxed
        else:
            return final_answer.strip()

    # fixed
    lines = solution_str.strip().split('\n')

    # look from back to front
    for line in reversed(lines):
        line = line.strip()
        # skip space
        if (line and
                not line.startswith('<') and
                not line.endswith('>') and
                not line.startswith('```') and
                line not in ['</execution_results>', '</search>', '</code>', '</answer>']):
            return line

    return None


def xverify_inference(prompt: str, inference_cfg):
    if not inference_cfg.get("url"):
        raise ValueError("XVerify URL is required. Pass inference_cfg['url'] or set XVERIFY_URLS.")
    if not inference_cfg.get("api_key"):
        raise ValueError("XVerify API key is required. Pass inference_cfg['api_key'] or set XVERIFY_API_KEY.")

    chosen_url = random.choice(inference_cfg['url'])
    client = OpenAI(base_url=f"{chosen_url}", api_key=inference_cfg['api_key'])

    response = client.chat.completions.create(
        model=inference_cfg["model_name"],
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        stop=["<|eot_id|>"],
        temperature=inference_cfg['temperature'],
        max_tokens=inference_cfg['max_tokens'],
    )

    return response.choices[0].message.content

def xverify_parse_response(response):
    """
    score 2 for correct, 0 for incorrect, None for invalid
    """
    cleaned_response = response.strip().lower()

    if "correct" in cleaned_response and "incorrect" not in cleaned_response:
        return 2, response.strip()
    elif "incorrect" in cleaned_response:
        return 0, response.strip()
    else:
        return None, response.strip()


def xverify_evaluate(question: str,
                     answer: str,
                     groundtruth: str,
                     inference_cfg: Dict = None):
    if inference_cfg is None:
        inference_cfg = _load_xverify_config_from_env()

    extracted_response, valid_answer_format = extract_solution(answer)

    do_print = random.random() < 0.1
    if do_print:
        print(f"======== REWARD CALCULATION DEBUG (10% sample) ========")
        if isinstance(groundtruth, dict) and 'target' in groundtruth:
            print(f"Golden answers: {groundtruth['target']}")
        else:
            print(f"Golden answers: {groundtruth}")
        print(f"Extracted answer: {extracted_response}")
        print(f"Solution string (first 500 chars): {answer[:500]}...")
        if len(answer) > 500:
            print(f"Solution string (last 200 chars): ...{answer[-200:]}")
        else:
            print(f"Full solution string: {answer}")


    xverify_input_prompt = evaluation_prompt.format(
        query=question,
        generated_output=extracted_response,
        ground_truth=groundtruth
    )

    xverify_response = xverify_inference(xverify_input_prompt, inference_cfg)
    score, cleaned_response = xverify_parse_response(xverify_response)

    if do_print:
        print(f"Final score: {score}")
        print(f"xverify response: {cleaned_response}")
        print(f"========================================================")

    return score, cleaned_response, valid_answer_format


def xverify_compute_score(data_source,
                          solution_str,
                          ground_truth,
                          question,
                          model_config,
                          extra_info=None,
                          return_dict=False,
                          sandbox_fusion_url=None,
                          concurrent_semaphore=None,
                          memory_limit_mb=None,):

    if not question and extra_info and "question" in extra_info:
        question = extra_info['question']

    search_pattern = r'<search>(.*?)</search>'
    search_matches = list(re.finditer(search_pattern, solution_str, re.DOTALL))
    if search_matches:
        do_search = 1.0
    else:
        do_search = 0.0

    if not question:
        raise NotImplementedError(f"no question for xverify evaluation")
    else:
        score, evaluation_response, valid_answer_format = xverify_evaluate(
            question, solution_str, ground_truth, model_config,
        )

        if score is None:
            is_correct = 0.
            return_score = 0.
        elif score == 2:
            is_correct = 1.
            return_score = 1.0
        else:
            is_correct = 0.
            return_score = 0.

    if return_dict:
        return {
            "score": return_score,
            "is_correct": is_correct,
            # "evaluation_response": evaluation_response,
            # "extracted_answer": extract_solution(solution_str),
            "valid_answer_format": valid_answer_format,
            "do_search": do_search
        }

    return return_score
