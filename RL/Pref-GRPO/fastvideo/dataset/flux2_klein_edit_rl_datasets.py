# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0]
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.

import json
import os
import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def _extract_instruction(data_item: dict) -> str:
    for key in ("instruction", "prompt", "caption", "text"):
        if key in data_item and data_item[key] is not None:
            return str(data_item[key])
    return ""


def _resolve_path(base_dir: str, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    primary = os.path.normpath(os.path.join(base_dir, path))
    if os.path.exists(primary):
        return primary
    fallback = os.path.normpath(os.path.join(base_dir, "images", path))
    if os.path.exists(fallback):
        return fallback
    return primary


class EditLatentDataset(Dataset):
    def __init__(self, json_path: str, cfg_rate: float, num_sample: Optional[int] = None, load_qa_list: bool = False, embed_dir: Optional[str] = None,) -> None:
        self.json_path = json_path
        self.cfg_rate = float(cfg_rate)
        self.dataset_dir_path = os.path.dirname(json_path)
        self.prompt_embed_dir = os.path.join(self.dataset_dir_path, "prompt_embed")
        self.text_ids_dir = os.path.join(self.dataset_dir_path, "text_ids")
        self.load_qa_list = load_qa_list
        self.image_base = "/mnt/shared-storage-user/mllm/zhangbeichen/share_data/interleave_data/ThinkMorph/Jigsaw_Assembly/"
        
        if embed_dir is not None:
            base_dir = embed_dir
        else:
            base_dir = os.path.dirname(json_path)

        with open(self.json_path, "r") as f:
            if json_path.endswith(".jsonl"):
                self.data_anno = [json.loads(line) for line in f if line.strip()]
            else:
                self.data_anno = json.load(f)

        if num_sample:
            self.data_anno = self.data_anno[:num_sample]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str, Optional[str]]:
        item = self.data_anno[idx]
        prompt_embed_file = item["prompt_embed_path"]
        text_ids_file = item["text_ids"]
        prompt_embeds = torch.load(
            os.path.join(self.prompt_embed_dir, prompt_embed_file),
            map_location="cpu",
            weights_only=True,
        )
        text_ids = torch.load(
            os.path.join(self.text_ids_dir, text_ids_file),
            map_location="cpu",
            weights_only=True,
        )

        if random.random() < self.cfg_rate:
            prompt_embeds = torch.zeros_like(prompt_embeds)
        
        instruction = _extract_instruction(item)
        source_image = item.get("source_image") or item.get("image")
        target_image = item.get("target_image")
        dataset_root = item.get("dataset_root") or self.dataset_dir_path
        source_image = _resolve_path(dataset_root, source_image)
        target_image = _resolve_path(dataset_root, target_image)
        qa_list = []

        if self.load_qa_list:
            if 'qa_list' not in item:
                raise KeyError(f"Sample {idx} missing 'qa_list' key")
            qa_list = item['qa_list']

        prompt_id = item.get('id', idx)

        image_path = os.path.join(self.image_base, f"{idx}_input.jpg")

        return prompt_embeds, text_ids, instruction, source_image, target_image, qa_list, prompt_id

    def __len__(self) -> int:
        return len(self.data_anno)



def edit_latent_collate_function(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str, str, Optional[str]]]
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str], List[Optional[str]]]:
    prompt_embeds, text_ids, instructions, source_images, target_images, qa_list, prompt_id = zip(*batch)
    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    text_ids = torch.stack(text_ids, dim=0)
    return prompt_embeds, text_ids, list(instructions), list(source_images), list(target_images), list(qa_list), list(prompt_id)
