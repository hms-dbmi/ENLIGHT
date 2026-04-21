import os
import random
from dataclasses import dataclass, field
import json
from typing import Dict, Optional, Sequence, List, Union
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

import transformers 


from utils.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from utils.conversation import conv_templates
from utils.mm_utils import tokenizer_image_token


def format_conversation(question, use_image, conv_mode = 'vicuna_v1', question_suffix=''):
    if use_image:
        if DEFAULT_IMAGE_TOKEN not in question:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
    conv = conv_templates[conv_mode].copy()
    if question_suffix:
        question = ' '.join([question, question_suffix])

    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt

class EvalDataset(Dataset):
    """Dataset for online evaluation. Rewrite from CustomDataset."""

    def __init__(self, data_path: str,
                 image_folder: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args ): #DataArguments
        super(EvalDataset, self).__init__()
        self.questions = [json.loads(q) for q in open(os.path.expanduser(data_path), "r")]
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = data_args.image_processor
        self.conv_mode = 'vicuna_v1'
        self.use_image = True
        self.data_args = data_args
        self.is_feat = 'feat' in self.questions[0].keys()
        
    def __getitem__(self, index):
        line = self.questions[index]
        qs_idx = line["question_id"]
        
        qs = line["text"]

        prompt = format_conversation(qs, 
                                    self.use_image, 
                                    conv_mode = self.conv_mode, 
                                    question_suffix=getattr(self.data_args, 'question_suffix', ''))
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        try:
            answer = line["answer"]
        except:
            answer = ''

        if 'image' in line:
            image_file = os.path.join(self.image_folder,line["image"])
            image = Image.open(image_file)#.convert('RGB') 
            processor = self.image_processor
            try:
                image_tensor = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            except:
                image_tensor = processor(image)
            return input_ids, image_tensor, image.size, qs, answer, qs_idx
        elif 'feat' in line: #TBD
            if isinstance(line["feat"], list):
                feat = []
                for f in line["feat"]:
                    feat_file = os.path.join(self.image_folder, f)
                    feat.append(torch.tensor(np.load(feat_file)))
                feat = torch.stack(feat)
            else:    
                feat_file = os.path.join(self.image_folder, line["feat"])
                feat = torch.tensor(np.load(feat_file))
            return input_ids, feat, feat.size, qs, answer, qs_idx
        else:
            return input_ids, answer
    
    def __len__(self):
        return len(self.questions)
    

class TestDataset(EvalDataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, args):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_mode = args.conv_mode
        self.use_image = args.use_image
        assert model_config.mm_use_im_start_end == False
        self.data_args = args
        self.is_feat = 'feat' in self.questions[0].keys()