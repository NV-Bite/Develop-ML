# Model Development

This repository is dedicated to the development of machine learning models for NV Bite, focusing on innovative approaches like transfer learning and advanced generative models to achieve optimal results.

## Tasks

Our project involves two key tasks:

- **Image Classification:** The objective is to accurately classify images into 13 predefined categories.
- **Text Generation:** Using the Vertex AI Gemini 1.5 Flash model, we aim to produce coherent and contextually appropriate text based on input prompts.

## Approach

To accomplish these tasks, we employ transfer learning to adapt pre-trained models for image classification, fine-tuning them on our dataset. For text generation, we utilize the cutting-edge capabilities of Vertex AI Gemini 1.5 Flash, leveraging its generative strengths to produce high-quality text outputs efficiently.

## Final Model

- [Image Classification Model](task_comp-vis/e-waste_xception_v1.ipynb)
- [Text Generation Model](task_nlp/vertex_gemini_v1.5_flash.ipynb)

## Preview

Explore our implementation through this [web app](https://ml-preview-6b9daowwxjoa4iqvpagyrp.streamlit.app/)!

## Todos

- [ ] Add support for text classification (optional)
- [ ] Extend image analysis with an object detection model (optional)

## Model Architecture

### Image Classification

- **Base Model**  
  ![base](docs/img/img-base.png)

- **Fine-Tuned Model**  
  ![fine-tuned](docs/img/img-fine-tuned.png)

### Text Generation

- Vertex AI Gemini 1.5 Flash  
  ![gemini](docs/img/gemini.png)

---

This version aligns with your requirements for both tasks while maintaining a fresh and professional tone.
