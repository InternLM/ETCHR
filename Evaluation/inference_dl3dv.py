from PIL import Image
import json
import io
import torch
import os
import pandas as pd
from openai import OpenAI
import base64
import utils
import argparse

def generate_response(client, model, messages):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4096,
        temperature=0
    )
    output_text = response.choices[0].message.content
    return output_text
    
def edit_image(pipe, question, image, save_path, save_path_input):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    edit_prompt = "Imagine a new perspective of the original image that helps answer the question. " + question
    image.save(save_path_input)
    width, height = image.size
    width = min(width, 1024)
    height = min(height, 1024)
    with torch.no_grad():
        image_edit = pipe(
            prompt=edit_prompt,
            image=image,
            num_inference_steps=40,
            width=width,
            height=height,
            guidance_scale=3.5, 
            generator=torch.Generator(device=DEVICE).manual_seed(126),
        ).images[0]
    image_edit.save(save_path)

def judge(client, model, question, image_path, edit_image_path):
    prompt_judge = "Please judge whether you can see the objects mentioned in the question from the perspective of the second image. Output the final judgment in the form of \\boxed{0} or \\boxed{1}."
    message = utils.prepare_message(prompt_judge, question, [image_path, edit_image_path])
    output_judge = generate_response(client, model, message)
    result_judge = utils.extract_answer(output_judge)
    if result_judge is None:
        result_judge = "0"
    return output_judge, result_judge

def reason(client, model, question, image_list):
    image_num = len(image_list)
    if image_num == 1:
        prompt = "Answer the question based on the image above. Output the final answer within \\boxed{}."
    else:
        prompt = "Answer the question based on the two images above. The first image is the original one corresponding to the question, while the second one provides a novel perspective to help you solve the problem. Please focus on the second image to answer the question. Output the final answer within \\boxed{}."

    message = utils.prepare_message(prompt, question, image_list)
    output_reason = generate_response(client, model, message)
    result_reason = utils.extract_answer(output_reason)
    return output_reason, result_reason

def infer_single(client, model, pipe, question, image, save_path):
    image_path = os.path.join(os.path.dirname(save_path), "input.jpg")
    edit_image(pipe, question, image, save_path, image_path)
    output_judge, result_judge = judge(client, model, question, image_path, save_path)
    if result_judge == "1":
        image_list = [image_path, save_path]
    else:
        image_list = [image_path]
    output_reason, result_reason = reason(client, model, question, image_list)
    return output_reason, result_reason, output_judge, result_judge

def infer_task(client, model, pipe, eval_data, result_file):
    for i, (question, answer, image) in enumerate(eval_data):
        save_path = f"./output/dl3dv/image/{i}/output.jpg"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        output_reason, result_reason, output_judge, result_judge = infer_single(client, model, pipe, question, image, save_path)
        output_json = {
            "id": i,
            "result": result_reason,
            "gt":answer,
            "judge":result_judge,
            "question": question,
            "response": output_reason,
            "response_judge": output_judge,
        }
        with open(result_file, 'a', encoding='utf-8') as f:
            json_str = json.dumps(output_json, ensure_ascii=False)
            f.write(json_str + '\n')
            f.close()


    
def main(args):
    output_dir = "output/dl3dv/image"
    os.makedirs(output_dir, exist_ok=True)
    client = utils.create_client(args.vllm_api_key, args.vllm_baseurl)
    model = args.model

    result_file = args.result_file

    dl3dv_data = utils.prepare_dataset(args.data_parquet_path)
    pipe = utils.load_flux2_klein_pipeline(args.pipe_path)

    infer_task(client, model, pipe, dl3dv_data, result_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vllm_baseurl', default="http://10.102.246.31:10000/v1", type=str, help='vllm base url')
    parser.add_argument('--vllm_api_key', default="EMPTY", type=str, help='vllm api key')
    parser.add_argument('--model', default="qwen3", type=str, help='vllm inference model')
    parser.add_argument('--result_file', default="dl3dv.json", type=str, help='result path')
    parser.add_argument('--pipe_path', default="YOUR_PIPE_PATH", help='image edit model path')
    parser.add_argument('--data_parquet_path', default="DL3DV.parquet", help='DL3DV data parquet path')
    args = parser.parse_args()
    main(args)
