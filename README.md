# Aduknow / Adullama3.1

A custom-tuned, locally deployable Language Model designed to act as an intelligent assistant for Adamson University student handbook. 

### Purpose
This project provides a localized, privacy-focused solution for processing complex institutional regulations. By generating a high-quality synthetic dataset from the Adamson University student handbook, the model was fine-tuned to interpret, retrieve, and explain campus policies, academic guidelines, and student rights through an accessible web interface to answer common questions about the handbook and reduce load on relevant departments.

### Tech Stack
* **Dataset Generation**: [augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit) paired with Ollama (Llama 3.1 and Qwen 2.5 7B) for automated, high-quality synthetic data pipelines*
* **Model Training**: [Unsloth Notebooks](https://github.com/unslothai/notebooks) for optimized, low-memory fine-tuning via Google Colab cloud compute
* **Inference Engine & Backend**: `llama-cpp-python` integrated with a Flask web framework

### Model Weights
The fine-tuned model weights are hosted on Hugging Face:
👉 **[Adullama3.1](https://huggingface.co/lucid-gunner/adullama3.1-v2-GGUF)**

---

## Installation

1. **Prerequisites**: Install Python (Make sure to tick *Add python.exe to PATH* while installing), `cmake`, `git`, and **MSVC Build Tools** (Tick *Desktop Development with C++* then install).
2. **Clone the Repository**:
   ```bash
   git clone [https://github.com/reimo22/aduknow.git](https://github.com/reimo22/aduknow.git)
   cd aduknow
3. Install Dependencies:
   ```bash
    pip install -r requirements.txt
   ```
4. Run Application:
   ```bash
    python app.py
   ```

## Hardware Requirements

### Minimum Requirements
* **CPU**: 64-bit architecture
* **RAM**: 8GB RAM
* **Storage**: At least 10GB Free Disk Space

### Recommended
* **GPU**: Dedicated graphics card with 4GB VRAM or more

## Lessons Learned

If I were to build this again today, I would probably use **RAG (Retrieval-Augmented Generation)** and smart prompting instead of fine-tuning. 

Fine-tuning taught me a lot, but it showed me how tough the process can be:
* **Mixed Results & Hallucinations**: Despite the dataset generation, the model still had mixed results and would occasionally hallucinate or misinterpret specific handbook rules. 
* **High Effort**: Fine-tuning takes a lot of time, and tweaking hyperparameters to get it right is tedious.
* **Data is Hard**: Generating and cleaning up high-quality synthetic data is a massive challenge.
* **Hard to Upgrade**: A fine-tuned model is stuck with what it learned during training. With newer models coming out so fast, it is a pain to re-train the whole thing from scratch every time a better model drops.

With how fast modern models are improving, using a RAG setup would be way easier to maintain. It would lock the AI's answers directly to the actual text of the handbook to stop hallucinations, and let you swap in a newer, better AI model easily without touching any training code.
