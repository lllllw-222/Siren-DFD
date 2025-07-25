# 🔥 Siren-DFD: Dynamic Focus Decoding for Factual yet Diverse Generation

This repository contains the code for the ACL 25 paper "[Odysseus Navigates the Sirens’ Song: Dynamic Focus Decoding for Factual and Diverse Open-Ended Text Generation](https://aclanthology.org/2025.acl-long.1320)."

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/lllllw-222/Siren-DFD.git
cd Siren-DFD

# Install dependencies
pip install torch
pip install -e transformers-4.45.1
pip install accelerate
```

### Usage

```bash
# Run inference with default settings
python demo.py --model path/to/your/model

# Enable focus decoding on specific layers
python demo.py \
    --model path/to/your/model \
    --focus_decoding \
    --focus_decoding_layers "20,21,22,23"

# Customize generation parameters
python demo.py \
    --model path/to/your/model \
    --focus_decoding \
    --focus_decoding_layers "20,21,22" \
    --max_new_tokens 512 \
    --temperature 0.8 \
    --focus_sigma 1.5
```

## 📋 Example Comparison

**Input**: `"Question: Who formulated the laws of motion?\nAnswer:"`

**Vanilla Generation**:
```
Sir Isaac Newton
```

**Focus Decoding**:
```
Sir Isaac Newton was responsible for laying down the laws of motion in 1680s.
```

## 📄 Citation

If you use Siren-DFD in your research, please consider citing our paper:

```bibtex
@inproceedings{luo-etal-2025-odysseus,
    title = "Odysseus Navigates the Sirens' Song: Dynamic Focus Decoding for Factual and Diverse Open-Ended Text Generation",
    author = "Luo, Wen  and
      Song, Feifan  and
      Li, Wei  and
      Peng, Guangyue  and
      Wei, Shaohang  and
      Wang, Houfeng",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1320/",
    pages = "27200--27218",
    ISBN = "979-8-89176-251-0",
    abstract = "Large Language Models (LLMs) are increasingly required to generate text that is both factually accurate and diverse across various open-ended applications. However, current stochastic decoding methods struggle to balance such objectives. We introduce Dynamic Focus Decoding (DFD), a novel plug-and-play stochastic approach that resolves this trade-off without requiring additional data, knowledge, or models. DFD adaptively adjusts the decoding focus based on distributional differences across layers, leveraging the modular and hierarchical nature of factual knowledge within LLMs. This dynamic adjustment improves factuality in knowledge-intensive decoding steps and promotes diversity in less knowledge-reliant steps. DFD can be easily integrated with existing decoding methods, enhancing both factuality and diversity with minimal computational overhead. Extensive experiments across seven datasets demonstrate that DFD significantly improves performance, providing a scalable and efficient solution for open-ended text generation."
}
```
