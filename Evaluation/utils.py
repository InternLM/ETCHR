from datasets import load_dataset
from PIL import Image
import json
import io
import torch
import os
import pandas as pd
import jsonlines

from openai import OpenAI

import base64

from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_client(api_key, base_url):
    client = OpenAI(
        api_key=api_key, 
        base_url=base_url
    )
    return client

def extract_answer(response):
    try:
        position = response.rfind('\\boxed{')
        if position == -1:
            return None
        num = 1
        position += 7
        start = position
        while num > 0:
            if response[position] == '{':
                num += 1
            elif response[position] == '}':
                num -= 1
            position += 1
        answer_extracted = response[start:position-1]
        return answer_extracted
    except Exception as e:
        print(f"Error: {e}")
        return None

def prepare_message(prompt, question, image_path_list):
    content = []

    for image_path in image_path_list:
        content.append({"type": "image_url", "image_url":{"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}})
    
    content.append({"type": "text", "text": prompt})
    content.append({"type": "text", "text": question})
    
    messages = [
        {
            "role": "user",
            "content":content,
        }
    ]
    return messages

def prepare_dataset(parquet_path):
    ds = pd.read_parquet(parquet_path)
    eval_dataset = []
    for i in range(len(ds)):
        item = ds.iloc[i]
        question = "Question: " + item["question"] + "Candidates: " + item["candidates"]
        image_bytes = item["images"][0]["bytes"]
        image = Image.open(io.BytesIO(image_bytes))
        answer = item["answer"]
        eval_dataset.append((question, answer, image))

    return eval_dataset



def load_flux2_klein_pipeline(pipe_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = Flux2KleinPipeline.from_pretrained(
        pipe_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)
    return pipe
