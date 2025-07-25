"""
Inference script for Focus Decoding with Siren-DFD
Demonstrates the difference between vanilla generation and focus decoding.
"""

import argparse
import sys

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList, StopStringCriteria


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inference script for Focus Decoding demonstration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument(
        "--model", 
        type=str, 
        default="meta-llama/Llama-3.1-8B",
        help="Path to the model directory"
    )
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling parameter")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--typical_p", type=float, default=1.0, help="Typical sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    
    # Focus decoding parameters
    parser.add_argument(
        "--focus_decoding", 
        action="store_true", 
        default=False,
        help="Enable focus decoding"
    )
    parser.add_argument(
        "--focus_decoding_layers", 
        type=str, 
        default=None,
        help="Comma-separated list of layer indices for focus decoding (e.g., '20,21,22'), None for all layers"
    )
    parser.add_argument(
        "--focus_relative_top", 
        type=float, 
        default=0.1,
        help="Relative top parameter for focus decoding"
    )
    parser.add_argument(
        "--base_temperature", 
        type=float, 
        default=1.1,
        help="Base temperature for focus decoding"
    )
    parser.add_argument(
        "--focus_sigma", 
        type=float, 
        default=4.0,
        help="Focus sigma parameter"
    )
    parser.add_argument(
        "--focus_converter", 
        choices=["exponential_decay"], 
        default="exponential_decay",
        help="Focus converter method"
    )
    
    # Input parameters
    parser.add_argument(
        "--input_text",
        type=str,
        default="Question: Who formulated the laws of motion?\nAnswer:",
        help="Input text for generation"
    )
    
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer."""
    print(f"Loading model from: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map='auto', 
            torch_dtype=torch.bfloat16
        )
        print("✓ Model and tokenizer loaded successfully")
        return tokenizer, model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)


def generate_text(
    model, 
    tokenizer, 
    input_text: str, 
    generation_kwargs: dict
) -> str:
    """Generate text using the model."""
    inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    
    output_str = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:], 
        skip_special_tokens=True
    ).strip()
    
    return output_str


def main():
    """Main inference function."""
    args = parse_arguments()
    
    print("=" * 60)
    print("🔥 Siren-DFD Focus Decoding Inference Demo")
    print("=" * 60)
    
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(args.model)
    
    # Setup stopping criteria
    stopping_criteria = StoppingCriteriaList()
    stopping_criteria.append(StopStringCriteria(tokenizer, "\n"))
    
    # Parse focus decoding layers
    focus_decoding_layers = None
    if args.focus_decoding_layers:
        try:
            focus_decoding_layers = [int(x.strip()) for x in args.focus_decoding_layers.split(',')]
            print(f"Focus decoding layers: {focus_decoding_layers}")
        except ValueError:
            print("✗ Invalid focus_decoding_layers format. Use comma-separated integers.")
            sys.exit(1)
    
    print(f"\nInput: {args.input_text}")
    print("\n" + "─" * 60)
    
    # Vanilla generation
    print("🔹 Vanilla Generation:")
    vanilla_kwargs = {
        'max_new_tokens': args.max_new_tokens,
        'do_sample': True,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'temperature': args.temperature,
        'typical_p': args.typical_p,
        'repetition_penalty': args.repetition_penalty,
        'stopping_criteria': stopping_criteria,
        'focus_decoding': False,
        'focus_decoding_layers': None,
        'focus_relative_top': args.focus_relative_top,
        'base_temperature': args.base_temperature,
        'focus_sigma': args.focus_sigma,
        'focus_converter': args.focus_converter
    }
    
    vanilla_output = generate_text(model, tokenizer, args.input_text, vanilla_kwargs)
    print(f"Output: {vanilla_output}")
    
    print("\n" + "─" * 60)
    
    # Focus decoding generation
    print("🎯 Focus Decoding Generation:")
    focus_kwargs = vanilla_kwargs.copy()
    focus_kwargs.update({
        'focus_decoding': True,
        'focus_decoding_layers': focus_decoding_layers
    })
    
    focus_output = generate_text(model, tokenizer, args.input_text, focus_kwargs)
    print(f"Output: {focus_output}")

if __name__ == "__main__":
    main()