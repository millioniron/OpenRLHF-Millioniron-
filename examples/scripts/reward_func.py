import re
from typing import Dict
import os
import torch
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

def reward_func(queries, prompts, labels,responses_lengths,**kwargs):
    
    responses = []
    
    for query in queries:
        # 查找 "assistant\n" 在 query 中的位置
        assistant_index = query.find("assistant\n")
        if assistant_index != -1:
            # 计算 "assistant\n" 后面内容的起始位置
            response_start_index = assistant_index + len("assistant\n")
            # 提取从 response_start_index 开始到结尾的字符串，并去除前后空白字符
            response = query[response_start_index:].strip()
            # 将提取出的 response 添加到 responses 列表中
            responses.append(response)
        else:
            # 如果没有找到 "assistant\n"，可以选择抛出错误、警告或者跳过
            # 这里我们选择跳过该 query 或者可以给一个默认值
            responses.append("")  # 或者使用 pass 跳过

    
    format_rewards = format_reward(responses)
    accuracy_rewards = accuracy_reward(responses, labels)
    
    
    ##add overlong filtering
    
    overlong_rewards=[(5096 - 1000- response_length) / 1000 if response_length > 5096 - 1000 else 0 for response_length in responses_lengths]
    # overlong_rewards=[(4096 - 1000- len(response)) / 1000  if len(response) > 4096 - 1000 else 0 for response in responses]

    
    # print(responses_length)
    # print([len(response) for response in responses])
    
    # print(overlong_rewards)
    
    # 计算最终奖励（假设 format_rewards 和 accuracy_rewards 已存在）
    rewards = [
        format_r + accuracy_r + overlong_r
        for format_r, accuracy_r, overlong_r in zip(
            format_rewards,
            accuracy_rewards,
            overlong_rewards
        )
    ]

    return torch.tensor([rewards,format_rewards,accuracy_rewards,overlong_rewards])

# def reward_func(queries, prompts, labels):
    
#     responses = []
    
#     for query in queries:
#         # 查找 "assistant\n" 在 query 中的位置
#         assistant_index = query.find("assistant\n")
#         if assistant_index != -1:
#             # 计算 "assistant\n" 后面内容的起始位置
#             response_start_index = assistant_index + len("assistant\n")
#             # 提取从 response_start_index 开始到结尾的字符串，并去除前后空白字符
#             response = query[response_start_index:].strip()
#             # 将提取出的 response 添加到 responses 列表中
#             responses.append(response)
#         else:
#             # 如果没有找到 "assistant\n"，可以选择抛出错误、警告或者跳过
#             # 这里我们选择跳过该 query 或者可以给一个默认值
#             responses.append("")  # 或者使用 pass 跳过

    
#     format_rewards = format_reward(responses)
#     accuracy_rewards = accuracy_reward(responses, labels)
#     rewards = [accuracy_reward+format_reward for format_reward, accuracy_reward in zip(format_rewards, accuracy_rewards)]
#     # rewards = [accuracy_reward for format_reward, accuracy_reward in zip(format_rewards, accuracy_rewards)]

#     # print('-'*100)
#     # print('\n ALL rewards:',rewards)
#     # print('-'*100)
#     # print(responses[0])
    
#     return torch.tensor([rewards,format_rewards,accuracy_rewards])




def extract_answer(text):
    """Extract content between <answer> tags."""
    if text is None:
        return ""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def format_reward(responses):
    """Reward function that checks if the completion has a specific format."""
    
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    # pattern = r"^<think>.*?</think><answer>.*?</answer><|im_end|>$"
    # pattern = r".*?<think>.*?</think>\n<answer>.*?</answer>.*?"
    
    # pattern = r"<think>.*?</think><answer>.*?</answer>"
    
    matches = [re.match(pattern, response,re.DOTALL) for response in responses]

    rewards = [1.0 if match else 0.0 for match in matches]
    
    # print(responses[:2])
    # print(matches[:2])
    
    # print('-'*100)
    # print('\nformat rewards:', rewards)
    return rewards

# def format_reward(responses):
#     """Reward function that checks if the completion has a specific format and assigns partial rewards."""
    
#     rewards = []
#     for response in responses:
#         reward = 0.0
        
#         # Check <think> tag
#         think_start_index = response.find('<think>')
#         think_end_index = response.find('</think>')
#         if (think_start_index != -1 and 
#             response.count('<think>') == 1 and 
#             think_start_index < think_end_index and 
#             response[:think_start_index].find('</think>') == -1 and 
#             response[:think_start_index].find('<answer>') == -1):
#             reward += 0.25
            
#         # Check </think> tag
#         if (think_end_index != -1 and 
#             response.count('</think>') == 1 and 
#             response[:think_end_index].count('<think>') == 1 and 
#             response[:think_end_index].find('<answer>') == -1):
#             reward += 0.25
            
#         # Check <answer> tag
#         answer_start_index = response.find('<answer>')
#         answer_end_index = response.find('</answer>')
#         if (answer_start_index != -1 and 
#             response.count('<answer>') == 1 and 
#             answer_start_index > think_end_index and  # Ensure <answer> comes after </think>
#             answer_start_index < answer_end_index and 
#             response[:answer_start_index].find('</answer>') == -1):
#             reward += 0.25
            
#         # Check </answer> tag
#         if (answer_end_index != -1 and 
#             response.count('</answer>') == 1 and 
#             response[:answer_end_index].count('<answer>') == 1 and 
#             response[:answer_end_index].find('<think>') == 1 and  # This condition might need adjustment based on your requirement
#             response[:answer_end_index].find('</think>') != -1):  # Ensure </think> is before <answer>
#             reward += 0.25

#         rewards.append(reward)
    
#     print('-' * 100)
#     print('\nformat rewards:', rewards)
#     return rewards



def accuracy_reward(contents, labels, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    rewards = []
    for content, sol in zip(contents, labels):
        content=extract_answer(content)
        
        # First try latex parsing
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        # print('-'*100)
        # print('\ncontent:', content)
        # print('-'*100)
        # print('\ngold_parsed:', gold_parsed)
        if len(gold_parsed) != 0:
            # print('latex gold parsed')
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            reward = float(verify(answer_parsed, gold_parsed))
            # print('\nprompt:', prompt)
            # print('-'*100)
            # print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward)
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            # print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    print('\naccuracy rewards:', rewards)

    return rewards


def accuracy_answer_reward(completion, answer, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    '''
    input is completion string, answer is extracted gold answer.
    '''
    gold_parsed = answer
    if len(gold_parsed) != 0:
        answer_parsed = parse(
            completion,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        reward = float(verify(answer_parsed, gold_parsed))
        print('-'*100)
        print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward)
    return reward
