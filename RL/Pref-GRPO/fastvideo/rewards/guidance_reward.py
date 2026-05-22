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

EPLUS_PROMPT = '''You are a Visual Question Answering (VQA) expert.
You will be given an image and a question.  

Instructions:
1. Your task is to answer the multiple-choice question **solely based on the visual content** of the provided image, even if the answer may seem obvious from prior knowledge or question wording.
2. **Ignore all external knowledge.** Do not use prior knowledge to answer if the evidence is not visible in the image.
3. **Do not make assumptions.** Only consider what is explicitly depicted in the image.
4. If the visual evidence is insufficient to determine the answer, or if none of the options are correct, select option "E".

Question: {}

Answer the question with only the correct letter.'''

DEFAULT_PROMPT = '''You are a Visual Question Answering (VQA) expert.

You will be given two images and a question. The first picture is the original image of the question, and the second one may contain content helpful for solving the question for your reference. 
Your task is to answer the multiple-choice question. The first picture is the original image of the question, and the second one may contain content helpful for solving the question for your reference.

Question: {}

Answer the question with only the correct letter.'''


PROMPT_OPEN_END = '''You are a Visual Question Answering (VQA) expert.

You will be given two images and a question. The first picture is the original image of the question, and the second one may contain content helpful for solving the question for your reference. 
Question: {}

Answer the question shortly with only the answers.
'''


E_OPTION_DEFAULT = "Can not answer based on the image"
E_OPTION_EPLUS = "None of the above, or insufficient information to determine the answer."

PROMPT_OPEN_END_JUDGE = '''You are an expert at evaluating correctness.
You are given a question, a ground truth answer, and a model's response.
Your task is to judge whether the response correctly answers the question based on the ground truth.

If the response is semantically correct (matches the ground truth meaning), output "1".
If the response is incorrect, output "0".
Only output "1" or "0", nothing else.

### **Examples**

**Example 1**
Question: What color is the dog?
Ground Truth Answer: Black
Model's Response: According to the picture description, the dog is black.
Your Output: 1

**Example 2**
Question: How many cups are on the table?
Ground Truth Answer: 2
Model's Response: There are two stemmed glasses next to the plate, and another glass on the edge of the table. I'm not sure if there are any other glasses; there might be three.
Your Output: 0

**Example 3**
Question: {}
Ground Truth Answer: {}
Model's Response: {}
Your Output: '''


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
                        "image_url": {"url": f"file://{abs_path}"}
                    },
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
                        "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
                    },
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
            print(f"[VLLMClient] 请求端口 {port} 失败: {e}")
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
) -> Dict[str, Any]:
    
    question = qa.get("question", "")
    gt_answer = qa.get("answer", "")
    random_seed = random.randint(0, 10000000)
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    
    try:
        prompt_round1 = PROMPT_OPEN_END.format(question)
        response_round1 = client.query(
            prompt=prompt_round1,
            image_path=image_path,
            edit_image_path=edit_image_path,
            max_tokens=1024,
            temperature=temperature,
            client_idx=client_idx,
        )
        
        prompt_round2 = PROMPT_OPEN_END_JUDGE.format(question, gt_answer, response_round1)
        response_round2 = client.query(
            prompt=prompt_round2,
            image_path=None,
            edit_image_path=None,
            max_tokens=4,
            temperature=0,
            client_idx=client_idx,
        )
        
        judge_result = response_round2.strip()
        is_correct = judge_result == "1"
        
        return {
            "question": question,
            "gt_answer": gt_answer,
            "response": response_round1,
            "judge_response": response_round2,
            "is_correct": is_correct,
            "error": None,
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
) -> Tuple[List[float], List[List[Dict]]]:
    
    if client is None:
        client = get_client()
    
    if not qa_list:
        raise ValueError("[Guidance Reward] qa_list is empty")

    
    tasks = []
    for img_idx, (image_path, edit_image_path) in enumerate(zip(image_paths, edit_image_paths)):
        client_idx = img_idx % client.num_servers
        for qa_idx, qa in enumerate(qa_list):
            tasks.append({
                "img_idx": img_idx,
                "qa_idx": qa_idx,
                "image_path": image_path,
                "edit_image_path": edit_image_path,
                "qa": qa,
                "client_idx": client_idx,
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
            ): i
            for i, task in enumerate(tasks)
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[Guidance Reward] task {idx} Fail: {e}")
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


class GuidanceRewardCalculator:
    
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
    ) -> Tuple[torch.Tensor, Optional[List[List[Dict]]]]:
        
        if not qa_list:
            print(f"[Warning] Guidance Reward open_end: qa_list is empty, returning zero rewards for {len(image_paths)} images")
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
        )
        
        if dist.is_initialized() and dist.get_rank() == 0 and all_details and len(all_details) > 0:
            print(f"[Guidance Reward OpenEnd] First image details:")
            for i, detail in enumerate(all_details[0]):
                print(f"  Q{i+1}: {detail['question']}")
                print(f"      GT: {detail['gt_answer']}")
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
                
                group_rewards, group_details = self.compute_rewards_for_group(
                    image_paths,
                    edit_image_paths,
                    qa_list,
                    return_details,
                )
                
                all_rewards.append(group_rewards)
                if return_details:
                    all_details.extend(group_details)
            
            rewards = torch.cat(all_rewards, dim=0)
            return rewards, all_details
        else:
            print(f"[Warning] Guidance Reward: num_generations={num_generations}, n={n}, falling back to sequential processing (slower)")
            all_rewards = []
            all_details = [] if return_details else None
            
            for inp in reward_inputs:
                image_path = inp["path"]
                edit_image_path = inp["edit_path"]
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
                )
                
                all_rewards.append(group_rewards)
                if return_details:
                    all_details.extend(group_details)
            
            rewards = torch.cat(all_rewards, dim=0)
            return rewards, all_details

