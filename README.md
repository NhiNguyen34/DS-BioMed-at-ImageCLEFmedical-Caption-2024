# DS@BioMed at ImageCLEFmedical Caption 2024

[![ImageCLEFmedical Caption 2024](https://img.shields.io/badge/ImageCLEFmedical-Caption%202024-blue)](https://www.imageclef.org/2024/medical/caption) [![Concept Detection Top 3](https://img.shields.io/badge/Concept%20Detection-Top%203-green)](https://www.imageclef.org/2024/medical/caption) 

This repository contains the code and models developed by DS@BioMed for the ImageCLEFmedical Caption 2024 challenge. Our approach focuses on enhancing medical image captioning by integrating concept detection into attention mechanisms, resulting in improved image understanding and caption generation.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Achievements](#key-achievements)
- [Methodology](#methodology)
  - [Concept Detection](#concept-detection)
  - [Caption Generation](#caption-generation)
- [Models](#models)
- [Results](#results)
- [Installation and Usage](#installation-and-usage)
- [Citation](#citation)
- [License](#license)

## Project Overview

Medical image captioning is a critical task for automated medical image interpretation and documentation. This project explores innovative ways to improve caption quality by explicitly incorporating medical concepts detected within the images.

## Key Achievements

- **Top 3 in Concept Detection Task:** Achieved an F1 score of 0.61998 on the private test set using the Swin-V2 model.
- **Concept-Aware Captioning:** Successfully integrated detected concepts into the caption generation process, leading to more accurate and contextually relevant captions.
- **State-of-the-Art Model:** Developed a competitive medical image captioning model that leverages cutting-edge techniques in concept detection and attention mechanisms.

## Methodology

### Concept Detection

We utilized the Swin-V2 model for concept detection. This model was trained on a large medical image dataset to identify relevant medical concepts within images. The detected concepts were then used to guide the attention mechanism in the caption generation model.

### Caption Generation

Our caption generation model is based on the BEiT+BioBart architecture. This model takes the image and detected concepts as input and generates a natural language description of the image. We also employed various post-processing techniques to further refine the captions.

## Models

- **Concept Detection:** Swin-V2
- **Caption Generation:** BEiT+BioBart

## Results

| Task                  | Model               | Validation Set | Private Test Set |
| ----------------------- | --------------------- | ------------- | ---------------- |
| Concept Detection | Swin-V2             | 0.58944       | 0.61998          |
| Caption Generation    | BEiT+BioBart (w/ concepts) | 0.60589       | 0.5794           |

## Installation and Usage

1. **Clone the repository:** `git clone https://github.com/your-username/ImageCLEFmedical-Caption-2024.git`
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Download pretrained models:** (Provide instructions and links)
4. **Run inference:** Follow the instructions in the `inference.ipynb` notebook.

## Citation

If you find this work helpful, please cite our paper:

@inproceedings{DSBioMed_ImageCLEF2024,
title={DS@BioMed at ImageCLEFmedical Caption 2024: Enhanced Attention Mechanisms in Medical Caption Generation through Concept Detection Integration},
author={DS@BioMed Team},
booktitle={ImageCLEF 2024},
year={2024}
}

## License

This project is licensed under the MIT License.

