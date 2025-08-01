import random
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from numpy import mean

import torch
import torch.nn.functional as F

from torch.distributed import all_gather_object
from .experience_maker import Experience


@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    base_action_log_probs: (A)
    values: (1)
    returns: (1)
    advantages: (1)
    attention_mask: (S)
    action_mask: (A)
    r_format:(1)
    r_accuracy:(1)
    r_std:(1)
    r_mean:(1)
    entropy_old: (1)
    entropy_old_sft: (1)
    entropy_old_rl: (1)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    r_format: torch.Tensor
    r_accuracy: torch.Tensor
    r_std: torch.Tensor
    r_mean: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    entropy_old: Optional[torch.Tensor]
    entropy_old_sft: Optional[torch.Tensor]
    entropy_old_rl: Optional[torch.Tensor]
    info: Optional[dict]


def split_experience_batch(experience: Experience) -> List[BufferItem]:
    batch_size = len(experience.sequences) 
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "r_format",
        "r_accuracy",
        "r_std",
        "r_mean",
        "entropy_old",
        "entropy_old_sft",
        "entropy_old_rl",
    )
    for key in keys:
        value = getattr(experience, key)
        if value is None:
            for i in range(batch_size):
                batch_kwargs[i][key] = None
            continue
        vals = value
        if isinstance(vals, torch.Tensor):
            vals = torch.unbind(vals)
        # print('key:',key)
        # print('vals:',vals)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v

    for i in range(batch_size):
        batch_kwargs[i]["info"] = {}
    for k, v in experience.info.items():
        vals = torch.unbind(v)
        assert batch_size == len(vals)
        for i, vv in enumerate(vals):
            if isinstance(vv, torch.Tensor):
                assert vv.numel() == 1, f"info[{k}] must be a scalar tensor, but got {vv.shape}"
                vv = vv.item()
            batch_kwargs[i]["info"][k] = vv

    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left") -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def make_experience_batch(items: List[BufferItem], packing_samples=False) -> Experience:
    kwargs = {}
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "r_format",
        "r_accuracy",
        "r_std",
        "r_mean",
        "entropy_old",
        "entropy_old_sft",
        "entropy_old_rl",
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if not packing_samples:
            batch_data = zero_pad_sequences(vals, "left") if vals[0] is not None else None
        else:
            batch_data = vals if vals[0] is not None else None
        kwargs[key] = batch_data

    kwargs["info"] = {}
    for key in items[0].info.keys():
        vals = torch.tensor([item.info[key] for item in items])
        kwargs["info"][key] = vals
    return Experience(**kwargs)


def remove_padding_in_sequences(items):
    for item in items:
        seq, act_log_prob, base_act_log_prob, value, ret, adv, att_mask, act_mask ,r_format,r_accuracy,r_std,r_mean,entropy_old,entropy_old_sft,entropy_old_rl= (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
            item.r_format,
            item.r_accuracy,
            item.r_std,
            item.r_mean,
            item.entropy_old,
            item.entropy_old_sft,
            item.entropy_old_rl,
        )
        right_pad = (1 - act_mask.long()).sum()
        right_pad = None if right_pad == 0 else -right_pad

        # left_pad for seq and att_mask
        left_pad = att_mask.long().argmax()
        (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
            item.r_format,
            item.r_accuracy,
            item.r_std,
            item.r_mean,
            item.entropy_old,
            item.entropy_old_sft,
            item.entropy_old_rl,
            
        ) = (
            seq[left_pad:right_pad],
            act_log_prob[:right_pad] if item.action_log_probs is not None else None,
            base_act_log_prob[:right_pad] if item.base_action_log_probs is not None else None,
            value[:right_pad] if item.values is not None else None,
            ret[:right_pad],
            adv[:right_pad],
            att_mask[left_pad:right_pad],
            act_mask[:right_pad],
            r_format[:right_pad],
            r_accuracy[:right_pad],
            r_std[:right_pad],
            r_mean[:right_pad],
            entropy_old[:right_pad] ,
            entropy_old_sft[:right_pad] ,
            entropy_old_rl[:right_pad],
        )
    return items


class NaiveReplayBuffer(ABC):
    """Naive replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(
        self, sample_batch_size: int, limit: int = 0, cpu_offload: bool = True, packing_samples: bool = False
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.packing_samples = packing_samples
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.items: List[BufferItem] = []
        self.entropy_sft=None
        self.entropy_rl=None

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = split_experience_batch(experience)
        # the packed samples comes with no padding
        if not self.packing_samples:
            items = remove_padding_in_sequences(items)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()


    def cal_all_tokens(self) -> None:
        if self.items[0].action_mask is None:
            action_log_probs_list = [item.action_log_probs for item in self.items]
            return torch.cat(action_log_probs_list, dim=0).unsqueeze(0).shape[-1]
        else:
            action_mask_list = [item.action_mask for item in self.items]
            return torch.cat(action_mask_list, dim=0).unsqueeze(0).sum[-1]
    
    @torch.no_grad()
    def sample(self) -> Experience:
        items = random.sample(self.items, self.sample_batch_size)
        experience = make_experience_batch(items, self.packing_samples)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        return self.items[idx]
    
    # def filter(self,strategy) -> None:
        
    #     # filtered_count = sum(1 for item in self.items if item.r_std <= 0) 
        
    #     print("****************"*5)
    #     print('items_count:',len(self.items))

        
    #     self.items = [item for item in self.items if item.r_std > 0]
        
    #     # self.items = [
    #     #                 item for item in self.items 
    #     #                 if (item.r_mean > 0.5 and item.r_accuracy == 0) or
    #     #                 (item.r_mean == 0.5) or 
    #     #                 (item.r_mean < 0.5 and item.r_accuracy == 1)
    #     #             ]
        
    #     print("****************"*5)
    #     print('items_count:',len(self.items))
        
    #     # return filtered_count
        
    def filter(self,strategy) -> None:

		# 收集所有GPU上的self.items
        gathered_items = [[] for _ in range(strategy.world_size)]
        all_gather_object(gathered_items, self.items)
        
        # 合并列表
        aggregated_items = []
        for sublist in gathered_items:
            aggregated_items.extend(sublist)
        
        self.items = aggregated_items
        
        print("****************"*5)
        print('items_count:',len(self.items))
        
        filtered_std_count = sum(1 for item in self.items if item.r_std <= 0)
        
        filtered_0_count = sum(1 for item in self.items if item.r_mean == 0)
        
        filtered_low_count = sum(1 for item in self.items if item.r_mean == 1/strategy.args.n_samples_per_prompt)
        
        # self.items = [item for item in self.items if (item.r_std > 0) and (item.r_mean > 1/strategy.args.n_samples_per_prompt)]
        
        self.items = [item for item in self.items if item.r_std > 0]

        

        print("****************"*5)
        print('filtered_std_count:',filtered_std_count)
        
        print("****************"*5)
        print('filtered_0_count:',filtered_0_count)
        
        print("****************"*5)
        print('filtered_low_count:',filtered_low_count)
        
        
        chunk_size = len(self.items) // strategy.world_size
        ###### 回来添加保证是8的就行
        self.items = self.items[strategy.get_rank() * chunk_size : (strategy.get_rank() + 1) * chunk_size]
        print("****************"*5)
        print('filtered_items_count:',len(self.items))
        
        
        
    def collate_fn(self, batch) -> Experience:
        experience = make_experience_batch(batch, self.packing_samples)
        return experience

    def sample_weights(self, k=0) -> List[float]:
        """
        Calculate and return sample weights based on the mean reward.

        This method computes weights for each item in the buffer by comparing
        the mean reward of all items (`r_mean`) with the reward of each individual
        item. The weights are calculated using different methods based on the value of k.

        Args:
            k (int): Mode selector (0 or 1).

        Returns:
            List[float]: A list of weights for each item in the buffer.
        """
        p_l = [item.r_mean.item() for item in self.items]
        p = torch.mean(torch.tensor(p_l)).item()

        if k == 0:
            weights = 1 - torch.abs(torch.tensor(p_l) - p)
            # Ensure non-negative weights (handle cases where |difference| >1)
            weights = torch.clamp(weights, min=0.0)
            
            
            N = len(weights)  
            
            sum_weights = weights.sum()  

            # 计算缩放因子
            scale_factor = N / sum_weights  

            # 缩放权重
            weights = weights * scale_factor  
            
        elif k == 1:
            # Protect against division by zero by clamping p between epsilon and 1-epsilon
            epsilon = 1e-8
            p = max(min(p, 1 - epsilon), epsilon)  # Ensure p ∈ (0, 1)

            p_l_tensor = torch.tensor(p_l)

            def piecewise_linear(p_l_tensor, p_val):
                left_slope = 1.0 / p_val
                right_slope = -1.0 / (1 - p_val)
                condition = p_l_tensor <= p_val
                left_val = left_slope * p_l_tensor
                right_val = 1 + right_slope * (p_l_tensor - p_val)
                return torch.where(condition, left_val, right_val)

            weights = piecewise_linear(p_l_tensor, p)
            # Ensure weights stay within [0, 1]
            weights = torch.clamp(weights, min=0.0, max=1.0)
        else:
            raise ValueError("k must be 0 or 1")

        return weights.tolist()

    def difficulty_weighting(self,k=0) -> None:
        sample_weights = self.sample_weights(k)
        for i, item in enumerate(self):
            item.advantages = getattr(item, "advantages") * sample_weights[i]
            
    def entropy_weighting(self,strategy) -> float:
        # 收集所有GPU上的self.items
        gathered_items = [[] for _ in range(strategy.world_size)]
        all_gather_object(gathered_items, self.items)
        
        # 合并列表#####注意8的倍数
        aggregated_items = []
        for sublist in gathered_items:
            aggregated_items.extend(sublist)
            
            
        entropy_old_list= [item.entropy_old for item in aggregated_items]
        
        r_mean_list= [item.r_mean for item in aggregated_items]
        
        
        entropy_sft_list = [entropy for i, entropy in enumerate(entropy_old_list) if i % 8 == 0]
        entropy_rl_list = [entropy for i, entropy in enumerate(entropy_old_list) if i % 8 != 0]
        
        
        
        r_mean=sum(r_mean_list)/len(r_mean_list)
        
        entropy_sft=sum(entropy_sft_list)/len(entropy_sft_list)
        
        entropy_rl=sum(entropy_rl_list)/len(entropy_rl_list)
        
        
        
        print("entropy_old_sft is ",entropy_sft)
        print("entropy_old_rl is ",entropy_rl)
        
        if self.entropy_sft==None and self.entropy_rl==None:
            print("entropy is Done")
            ratio=1
        else:
            print("self.entropy_sft is ",self.entropy_sft)
            print("self.entropy_rl is ", self.entropy_rl)
            ratio=(1-entropy_sft/self.entropy_sft)/(1-entropy_rl/self.entropy_rl)
        self.entropy_sft=entropy_sft
        self.entropy_rl=entropy_rl
        
        
        ####version 1
        
        
        
        

            
        ########version 5
        if ratio > 0:
            ratio = min(max(ratio, 1), 7)
        else:
            ratio=-ratio
            ratio = min(max(ratio, 1), 7)
        
            
        
        
        
        
    
        
        print("ratio is ",ratio)
        
        for i, item in enumerate(self):
            if i%strategy.args.n_samples_per_prompt ==0 :
                item.advantages = getattr(item, "advantages") * ratio
        
        return ratio
        
    
    def normalize(self, attribute: str, strategy) -> None:
        assert attribute == "advantages"
        items = []
        action_masks = []
        for item in self:
            items.append(getattr(item, attribute))
            action_masks.append(item.action_mask)

        items_vector = torch.cat(items).float().flatten()

        if action_masks[0] is None:
            # packing samples has no action mask
            action_masks_vector = 1
            num_actions = items_vector.numel()
        else:
            action_masks_vector = torch.cat(action_masks).flatten()
            num_actions = action_masks_vector.sum()

        # for DP
        # mean
        sum_and_count = torch.tensor([items_vector.sum(), num_actions], device=items_vector.device)
        all_sum, all_count = strategy.all_reduce(sum_and_count, "sum")
        mean = all_sum / all_count
        # std
        std = ((items_vector - mean).pow(2) * action_masks_vector).sum()
        all_std = strategy.all_reduce(std, "sum")
        rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()

        for i, item in enumerate(self):
            setattr(item, attribute, (items[i] - mean) * rstd)
