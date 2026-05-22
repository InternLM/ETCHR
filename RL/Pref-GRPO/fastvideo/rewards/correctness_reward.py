import re
import random
import base64
import torch
import torch.distributed as dist
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

PROMPT_OPEN_END_BOX = '''
Please judge whether the information relevant to solving this problem is marked with a red box in the second image. 
If the red box contains valid information, reply with 1.
If there is no red box in the second image, or the area enclosed by the red box does not contain valid information, Or there may be incorrect red boxes in the figure (i.e., failing to frame the valid information needed to solve the problem, reply with 0.
Question: {}
Only output "1" or "0", nothing else.
'''

PROMPT_OPEN_END_MAZE = '''
Please determine whether the following image contains a valid maze path: the middle path (blue) connects the starting point (green) to the destination (red), is continuous in four directions (up, down, left, right) with no diagonal movement allowed.
If the blue path is valid, reply with 1.
Otherwise, reply with 0.
Only output "1" or "0", nothing else.
'''


PROMPT_OPEN_END_JIGSAW = '''
Please determine whether the second image is a correct restoration of the jigsaw puzzle task in the first image.
If the the second image is a correct restoration, reply with 1.
Otherwise, reply with 0.
Only output "1" or "0", nothing else.
'''

PROMPT_OPEN_END_THREED = '''
Please judge whether the second image provides a novel perspective of the scene shown in the first image so that it can help better solve the question.
If yes, reply with 1.
Otherwise, reply with 0.
Question: {}
Only output "1" or "0", nothing else.
'''


class VLLMClient:
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        base_port: int = 10000,
        num_servers: int = 8,
        model_name: str = "qwen3",
    ):
        self.host = host
        self.base_port = base_port
        self.num_servers = num_servers
        self.model_name = model_name
        self.ports = [base_port + i for i in range(num_servers)]
        
        self.clients = {
            port: OpenAI(
                base_url=f"http://{host}:{port}/v1",
                api_key="EMPTY"
            )
            for port in self.ports
        }
        self._counter = 0
    
    def _get_client(self, strategy: str = "round_robin") -> tuple:
        if strategy == "random":
            port = random.choice(self.ports)
        else:
            port = self.ports[self._counter % self.num_servers]
            self._counter += 1
        return self.clients[port], port
    
    def get_client_by_idx(self, idx: int) -> tuple:
        port = self.ports[idx % self.num_servers]
        return self.clients[port], port
    
    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _get_image_mime_type(self, image_path: str) -> str:
        suffix = Path(image_path).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_types.get(suffix, "image/jpeg")
    
    def query(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        edit_image_path: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        max_tokens: int = 4,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        strategy: str = "round_robin",
        use_local_path: bool = False,
        client_idx: Optional[int] = None,
    ) -> str:
        if client_idx is not None:
            client, port = self.get_client_by_idx(client_idx)
        else:
            client, port = self._get_client(strategy)
        
        if image_path and edit_image_path:
            if use_local_path:
                abs_path = str(Path(image_path).resolve())
                abs_path_edit = str(Path(edit_image_path).resolve())
                content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"file://{abs_path_edit}"}
                    },
                    {"type": "text", "text": prompt}
                ]
            else:
                image_base64 = self._encode_image(image_path)
                mime_type = self._get_image_mime_type(image_path)
                edit_image_base64 = self._encode_image(edit_image_path)
                edit_mime_type = self._get_image_mime_type(edit_image_path)
                content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{edit_mime_type};base64,{edit_image_base64}"}
                    },
                    {"type": "text", "text": prompt}
                ]
        else:
            content = prompt
        
        try:
            extra_body = {
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if temperature == 0:
                extra_body["do_sample"] = False
                extra_body["top_k"] = 1
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[VLLMClient] Failure {port} {e}")
            raise


_client: Optional[VLLMClient] = None


def get_client(
    host: str = "127.0.0.1",
    base_port: int = 10000,
    num_servers: int = 8,
    model_name: str = "qwen3",
) -> VLLMClient:
    global _client
    if _client is None:
        _client = VLLMClient(host, base_port, num_servers, model_name)
    return _client


def _query_single_qa_open_end(
    client: VLLMClient,
    image_path: str,
    edit_image_path: str,
    qa: Dict[str, str],
    temperature: float = 0.7,
    client_idx: Optional[int] = None,
    prompt: str = None
) -> Dict[str, Any]:

    question = qa.get("question", "")
    gt_answer = qa.get("answer", "")
    random_seed = random.randint(0, 10000000)
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    
    try:
        if "Draw a red box" in prompt:
            prompt_judge = PROMPT_OPEN_END_BOX.format(question)
            task_type = "BOX"
        elif "shortest path" in prompt:
            prompt_judge = PROMPT_OPEN_END_MAZE
            task_type = "MAZE"
        elif "jigsaw" in prompt:
            prompt_judge = PROMPT_OPEN_END_JIGSAW
            task_type = "JIGSAW"
        elif "Imagine a new perspective" in prompt:
            prompt_judge = PROMPT_OPEN_END_THREED.format(question)
            task_type = "THREE D"
        else:
            print("not supported correctness reward type")
            exit()
        response_round1 = client.query(
            prompt=prompt_judge,
            image_path=image_path,
            edit_image_path=edit_image_path,
            max_tokens=1024,
            temperature=temperature,
            client_idx=client_idx,
        )
        
        judge_result = response_round1.strip()
        is_correct = judge_result == "1"
        
        return {
            "question": question,
            "gt_answer": gt_answer,
            "response": response_round1,
            "judge_response": "",
            "is_correct": is_correct,
            "error": None,
            "task_type": task_type
        }
    except Exception as e:
        return {
            "question": question,
            "gt_answer": gt_answer,
            "response": "",
            "judge_response": "",
            "is_correct": False,
            "error": str(e),
        }

def compute_qa_score_batch_open_end(
    image_paths: List[str],
    edit_image_paths: List[str],
    qa_list: List[Dict[str, str]],
    client: Optional[VLLMClient] = None,
    num_threads: int = 8,
    temperature: float = 0.7,
    prompts: List[str] = None,
) -> Tuple[List[float], List[List[Dict]]]:

    if client is None:
        client = get_client()
    
    if not qa_list:
        raise ValueError("[Correctness Reward] qa_list is empty")

    
    tasks = []
    for img_idx, (image_path, edit_image_path, prompt) in enumerate(zip(image_paths, edit_image_paths, prompts)):
        client_idx = img_idx % client.num_servers
        for qa_idx, qa in enumerate(qa_list):
            tasks.append({
                "img_idx": img_idx,
                "qa_idx": qa_idx,
                "image_path": image_path,
                "edit_image_path": edit_image_path,
                "qa": qa,
                "client_idx": client_idx,
                "prompt": prompt,
            })
    
    results = [None] * len(tasks)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_idx = {
            executor.submit(
                _query_single_qa_open_end,
                client,
                task["image_path"],
                task["edit_image_path"],
                task["qa"],
                temperature,
                task["client_idx"],  
                task['prompt']
            ): i
            for i, task in enumerate(tasks)
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[Correctness Reward] Task {idx} Fail: {e}")
                task = tasks[idx]
                results[idx] = {
                    "question": task["qa"].get("question", ""),
                    "gt_answer": task["qa"].get("answer", ""),
                    "response": "",
                    "judge_response": "",
                    "is_correct": False,
                    "error": str(e),
                }
    
    num_images = len(image_paths)
    num_qa = len(qa_list)
    accuracies = []
    all_details = []
    
    for img_idx in range(num_images):
        img_details = []
        correct_count = 0
        
        for qa_idx in range(num_qa):
            task_idx = img_idx * num_qa + qa_idx
            result = results[task_idx]
            img_details.append(result)
            if result["is_correct"]:
                correct_count += 1
        
        accuracy = correct_count / num_qa if num_qa > 0 else 0.0
        accuracies.append(accuracy)
        all_details.append(img_details)
    
    return accuracies, all_details


class CorrectnessRewardCalculator:    
    def __init__(
        self,
        host: str = "127.0.0.1",
        base_port: int = 10000,
        num_servers: int = 8,
        model_name: str = "qwen3",
        num_threads: int = 8,
        all_qa: bool = True,
        qa_num: int = 3,
        temperature: float = 0.7,
        open_end: bool = False,
        eplus_option: bool = False,
    ):
        self.client = VLLMClient(host, base_port, num_servers, model_name)
        self.num_threads = num_threads
        self.all_qa = all_qa
        self.qa_num = qa_num
        self.temperature = temperature
        self.open_end = open_end
        self.eplus_option = eplus_option
    
    def compute_rewards_for_group(
        self,
        image_paths: List[str],
        edit_image_paths: List[str],
        qa_list: List[Dict[str, str]],
        return_details: bool = False,
        prompt: List[str] = None,
    ) -> Tuple[torch.Tensor, Optional[List[List[Dict]]]]:
        
        if not qa_list:
            print(f"[Warning] Correctness Reward open_end: qa_list is empty, returning zero rewards for {len(image_paths)} images")
            rewards = torch.zeros(len(image_paths), dtype=torch.float32)
            details = [[] for _ in image_paths] if return_details else None
            return rewards, details
        
        if self.all_qa:
            selected_qa = qa_list
        else:
            if len(qa_list) <= self.qa_num:
                selected_qa = qa_list
            else:
                selected_qa = random.sample(qa_list, self.qa_num)
        
        accuracies, all_details = compute_qa_score_batch_open_end(
            image_paths,
            edit_image_paths,
            selected_qa,
            self.client,
            self.num_threads,
            self.temperature,
            prompt
        )
        
        if dist.is_initialized() and dist.get_rank() == 0 and all_details and len(all_details) > 0:
            print(f"[Correctness Reward OpenEnd] First image details:")
            for i, detail in enumerate(all_details[0]):
                print(f"  Q{i+1}: {detail['question']}")
                print(f"      GT: {detail['gt_answer']}")
                print(f"      TASK_TYPE: {detail['task_type']}")
                print(f"      Response: {detail['response']}")
                print(f"      Judge: {detail['judge_response']} | Correct: {detail['is_correct']}")
        
        rewards = torch.tensor(accuracies, dtype=torch.float32)
        
        if return_details:
            return rewards, all_details
        return rewards, None
        
    
    def compute_rewards_batch(
        self,
        reward_inputs: List[Dict[str, Any]],
        num_generations: int = 1,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[List[Dict]]]]:
        n = len(reward_inputs)
        
        if num_generations > 1 and n % num_generations == 0:
            num_prompts = n // num_generations
            all_rewards = []
            all_details = [] if return_details else None
            
            for i in range(num_prompts):
                start_idx = i * num_generations
                end_idx = (i + 1) * num_generations
                
                group_inputs = reward_inputs[start_idx:end_idx]
                image_paths = [inp["path"] for inp in group_inputs]
                edit_image_paths = [inp["edit_path"] for inp in group_inputs]
                qa_list = group_inputs[0].get("qa_list", [])
                prompt = [inp["prompt"] for inp in group_inputs]
                
                group_rewards, group_details = self.compute_rewards_for_group(
                    image_paths,
                    edit_image_paths,
                    qa_list,
                    return_details,
                    prompt
                )
                
                all_rewards.append(group_rewards)
                if return_details:
                    all_details.extend(group_details)
            
            rewards = torch.cat(all_rewards, dim=0)
            return rewards, all_details
        else:
            print(f"[Warning] Correctness Reward: num_generations={num_generations}, n={n}, falling back to sequential processing (slower)")
            all_rewards = []
            all_details = [] if return_details else None
            
            for inp in reward_inputs:
                image_path = inp["path"]
                edit_image_path = inp["edit_path"]
                prompt = inp['prompt']
                qa_list = inp.get("qa_list", [])

                if not isinstance(qa_list, list):
                    qa_list = [qa_list]
                
                if not qa_list:
                    all_rewards.append(torch.tensor([0.0]))
                    if return_details:
                        all_details.append([])
                    continue
                
                group_rewards, group_details = self.compute_rewards_for_group(
                    [image_path],
                    [edit_image_path],
                    qa_list,
                    return_details,
                    [prompt]
                )
                
                all_rewards.append(group_rewards)
                if return_details:
                    all_details.extend(group_details)
            
            rewards = torch.cat(all_rewards, dim=0)
            return rewards, all_details



