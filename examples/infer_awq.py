#!/usr/bin/env python3
#
# download a model or dataset from the Huggingface Hub to $TRANSFORMERS_CACHE
# https://huggingface.co/docs/huggingface_hub/guides/download#download-an-entire-repository
#
import os
import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from awq.quantize.quantizer import real_quantize_model_weight


# parse command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='', required=True, help="name or path of the huggingface model")
parser.add_argument('--quant', type=str, default='', required=True, help="path to the real AWQ quantized model checkpoint")
parser.add_argument('--prompt', type=str, default='California is in which country?')

parser.add_argument('--w_bit', type=int, default=4)
parser.add_argument('--q_group_size', type=int, default=128)
parser.add_argument('--no_zero_point', action='store_true',help="disable zero_point")
                    
args = parser.parse_args()

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization
}

print("Quantization config:", q_config)

# select compute device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Running on device {device}")
print(f"Loading model {args.model} with quantized weights from {args.quant}")

# load huggingface model, without the weights (just need the model structure)
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    
# prepare model to apply quantized weights
real_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config, init_only=True)

# load quantized weights
model = load_checkpoint_and_dispatch(
    model, args.quant, device_map='balanced', 
    no_split_module_classes=["OPTDecoderLayer", "LlamaDecoderLayer"]
)
                  
# create tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)

# run inference
generated_ids = model.generate(input_ids) #, do_sample=False, min_length=num_tokens, max_length=num_tokens)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))