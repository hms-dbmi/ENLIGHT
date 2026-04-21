import argparse
import os
from functools import partial
import json
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.builder import load_enlight_model
from datasets.lmm_dataset import TestDataset
from utils.constants import CACHE_DIR


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, qs, gt_ans, qs_index = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes, None, qs, gt_ans, qs_index

def collate_fn_padding(batch, pad_token_id=0):
    input_ids, image_tensors, image_sizes, qs, gt_ans, qs_index = zip(*batch)
    image_tensors = torch.stack(image_tensors, dim=0)
    ##padding for batch processing
    input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_token_id)
    attention_mask=input_ids.ne(pad_token_id)
    return input_ids, image_tensors, image_sizes, attention_mask, qs, gt_ans, qs_index


def create_mm_data_loader(args, questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=1):
    dataset = TestDataset(questions, image_folder, tokenizer, image_processor, model_config, args)
    if batch_size>1:
        collate_func = partial(collate_fn_padding, pad_token_id=tokenizer.pad_token_id)
    else:
        collate_func = collate_fn
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_func)
    return data_loader

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

    
def infer_model_batch(args):
    # Load model
    disable_torch_init()
    
    tokenizer, model, _, _ = load_enlight_model(args.model_gen, cache_dir = args.cache_dir)
    model.config.tokenizer_padding_side = "left" 
    if args.bf16:
        model=model.to(torch.bfloat16)
    dtype = next(model.get_model().mm_projector.parameters()).dtype

    # Preapre questions
    question_file = os.path.expanduser(args.question_file)
    if not os.path.exists(question_file):
        print(f"No question file found in {question_file}. Exiting.")
        return 
    
    questions = [json.loads(q) for q in open(question_file, "r")]
    num_questions = len(questions)
    if num_questions == 0:
        print(f"No questions found in {args.question_file}. Exiting.")
        return
    if args.num_chunks > 1:
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    print(f"Loaded {len(questions)}/{num_questions} questions from {args.question_file} for chunk {args.chunk_idx}/{args.num_chunks}.")
    
    # Prepare answer file
    if args.num_chunks > 1:
        answers_file = args.answers_file.replace('.jsonl', f'/chunk{args.chunk_idx}of{args.num_chunks}.jsonl')
    else:
        answers_file = os.path.expanduser(args.answers_file)
    if os.path.dirname(answers_file):
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    if os.path.exists(answers_file):
        num_ans = len([json.loads(q) for q in open(answers_file)])
        if num_ans == len(questions):
            print(f"Answers {answers_file} already exists with #questions.")
            return
    
    ans_file = open(answers_file, "w")
    data_loader = create_mm_data_loader(args, questions, args.image_folder, tokenizer, image_processor, model.config,
                                        batch_size=args.batch_size, num_workers=args.num_workers)
    
    try:
        is_feat = data_loader.dataset.is_feat
    except:
        is_feat = False

    for (input_ids, image_tensor, image_sizes, attention_mask, qs_text, gt_text, qs_ids) in tqdm(data_loader):
        if is_feat:
            img_feats = image_tensor.clone()
            image_tensor = None
            image_sizes = None
        else:
            img_feats = None
        
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image_tensor.to(device='cuda', non_blocking=True) if image_tensor is not None else None,
                image_sizes=image_sizes,
                img_feats=img_feats.to(dtype=dtype, device='cuda', non_blocking=True) if img_feats is not None else None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True) 

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # torch.cuda.empty_cache()
        for qs_id, output, qs, gt in zip(qs_ids, outputs, qs_text, gt_text):
            ans_file.write(json.dumps({"question_id": qs_id, 
                                        "qs": qs, 
                                        "text": output, 
                                        "gt": gt }) + "\n")

    ans_file.close()
    print(f"Answers saved to {answers_file}.")

def get_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--model-gen", type=str, default='./ckpts/enlight-fm')
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR)
    # Data
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--use-image", type=int, default=1)
    parser.add_argument("--question-file", type=str, default="question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--question_suffix", type=str, default="")
    # Setting
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bf16", type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    infer_model_batch(args)
