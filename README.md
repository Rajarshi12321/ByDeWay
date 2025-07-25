# ByDeWay: Depth-based Captioning for Multimodal Large Language Models

## Overview

**ByDeWay** is a training-free framework designed to improve the spatial reasoning and grounding abilities of Multimodal Large Language Models (MLLMs) through **Layered-Depth-based Prompting (LDP)**. This Python package uses monocular depth estimation along with automatic region-aware captioning to enrich model prompts with structured spatial context, boosting performance on tasks prone to hallucination and those needing detailed visual reasoning.[1]

## Features

- **Training-Free Enhancement:** No fine-tuning or model parameter updates required.
- **Plug-and-Play:** Compatible with any black-box MLLM that accepts image and text inputs.
- **Layered Spatial Context:** Utilizes monocular depth estimation to segment images into closest, mid-range, and farthest regions.
- **Region-Specific Captioning:** Automatically generates captions for each depth layer.
- **Improved Performance:** Consistently enhances hallucination resistance and spatial reasoning—as shown on POPE and GQA tasks.[1]
- **Modular & Scalable:** Add LDP as an input prompt component for diverse use cases.

## Workflow

1. **Image Input:** Provide an RGB image for analysis.
2. **Depth Estimation:** Segment the image into three depth-based layers (closest, mid-range, farthest) using an off-the-shelf monocular depth estimator (e.g., Depth Anything V2).
3. **Region Masking & Captioning:** Apply spatial masks to extract regions and generate captions using a grounded vision-language model (e.g., KOSMOS-2).
4. **Prompt Construction:** Concatenate the layer-wise region captions into a structured, spatially-aware prompt.
5. **MLLM Inference:** Feed the image, question/instruction, and enhanced prompt into any supported MLLM for improved, hallucination-resistant outputs.

## Repository Structure

```
ByDeWay-Depth-Captioning/
│
├── data/                   # Sample images and data (optional)
├── notebooks/              # Example and demo Jupyter notebooks
├── src/
│   └── depth_captioning/   # Core depth and captioning modules
├── test/                   # Unit and functional tests
├── install_depth_anything.sh # Script to install Depth Anything model
├── requirements.txt        # List of Python dependencies
├── .gitignore
└── README.md               # (This file)
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rajarshi12321/ByDeWay-Depth-Captioning.git
   cd ByDeWay-Depth-Captioning
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Depth Anything dependencies:**
   ```bash
   bash install_depth_anything.sh
   ```

*Note: Instructions assume a Python 3.8+ environment.*

## Quick Start Example

Open a demo notebook from the `notebooks/` directory or run a Python script from the `src/depth_captioning/` module to process an image. The standard workflow is as follows:

```python
from src.depth_captioning.pipeline import DepthCaptioningPipeline

# Initialize pipeline
pipeline = DepthCaptioningPipeline()

# Process image and generate LDP prompt
ldp_prompt = pipeline.generate_ldp_prompt(
    image_path='path/to/image.jpg',
    question='Is the person in the front wearing a white robe?'
)

print(ldp_prompt)
```

## Usage Details

- **Depth Estimation:** Uses `Depth Anything V2` for zero-shot monocular depth prediction.
- **Region Segmentation:** Depth values split into top 30% (closest), middle 40% (mid-range), bottom 30% (farthest).
- **Captioning:** Each masked region is described using a vision-language model such as KOSMOS-2.
- **Prompting:** Region captions are concatenated with the original question/task to instruct the MLLM.

## Benchmark Results

| Model          | Task  | Baseline Accuracy | LDP Accuracy | Δ  |
|----------------|-------|-------------------|--------------|----|
| GPT-4o         | POPE  | 0.860             | 0.873        | +0.013 |
| Qwen2.5-VL     | POPE  | 0.7267            | 0.9000       | +0.1733 |
| ViLT           | POPE  | 0.8533            | 0.9267       | +0.0734 |
| BLIP           | POPE  | 0.8733            | 0.9533       | +0.08 |
| Qwen2.5-VL     | GQA   | 0.5007            | 0.6592       | +0.1585 |
| ViLT           | GQA   | 0.527             | 0.627        | +0.1 |
| BLIP           | GQA   | 0.5552            | 0.6704       | +0.1152 |

*LDP = With Layered-Depth-based Prompting. See full paper for F1, Precision, Recall, and qualitative examples.*[1]

## License

This project is released for research and academic use. See the repository for details.

## Citation

If you use **ByDeWay** or its depth captioning workflow in your research, please cite:

> Roy, R., Das, D., Banerjee, A., Bhattacharjee, A., Dasgupta, K., & Tripathi, S. ByDeWay: Boost Your multimodal LLM with DEpth prompting in a training-free Way. Kalyani Government Engineering College & Intel Labs, 2024.[1]

## Acknowledgments

- Depth Anything V2 and KOSMOS-2 authors.
- Inspired by work on multimodal large language models and depth-aware captioning.[1]

## Links

- [Project Paper: ByDeWay — Boost Your multimodal LLM with DEpth prompting in a training-free Way][1]
- [GitHub Repository][2]

: ByDeWay_depth_prompting_for_MLLM.pdf  [1]
: https://github.com/Rajarshi12321/ByDeWay-Depth-Captioning[2]

[1] https://arxiv.org/pdf/2507.08679
[2] https://github.com/Rajarshi12321/ByDeWay-Depth-Captioning
