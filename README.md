# Starting Point for the Final Project of the "From Beginner to Advanced LLM Developer" course

## Overview

This repository contains the code of the final "Part 4; Building Your Own advanced LLM + RAG Project to receive certification" lesson of the "From Beginner to Advanced LLM Developer" course.

This project is a version of the project number 49. Academic research assistant with literature search and summarization. It consists 

Here are the necessary constraints that you must follow in your project:

    It must be a RAG project written in Python. We suggest forking or cloning this repository as a starting point for your project if you plan to use Gradio as the UI, but it’s not strictly required.
    The project must use at least one LLM (either local or managed via API key).
    It must be deployed on a public Hugging Face Space.
    If you write code for data collection and curation, include the scripts in the project repository.
    It must contain a README file with a textual explanation of the project.
    You must NOT put your API keys in the project folder. Add a UI element where the user can paste an API key for one of the following LLM providers: OpenAI, Google Gemini, and Claude.
    Do NOT add costly pipelines in your code that will be run with the user API key. For example, it’s fine to use the API key to write the RAG answers in the order of 10k tokens, but it’s not fine to use the API key to process 100 images on the fly for example.
    In the README, add a quick cost estimation that the user will incur when trying your project with their API keys. You should show that the user can try all the functionalities with $0.50 or less.
    The README file must list all the API keys (e.g., “OpenAI API key,” not the key itself) that the user has to input to use the app.


Uses dynamic few-shot prompting, where the best examples are selected according to the user query.
There’s code for RAG evaluation in the folder, and the README contains the evaluation results. The folder must also contain the evaluation dataset and the evaluation scripts.
The app is designed for a specific goal/domain that is not a tutor about AI. For example, it could be about finance, healthcare, etc.
Use a reranker in your RAG pipeline. It can be a fine-tuned version (your choice).
Use hybrid search in your RAG pipeline.
Use a fine-tuned LLM in your app.
Use a fine-tuned embedding model in your app. 
Your query pipeline includes function calling.

## Setup

1. Create a `.env` file and add there your OpenAI API key. Its content should be something like:

```bash
OPENAI_API_KEY="sk-..."
```

2. Create a local virtual environment, for example using the `venv` module. Then, activate it.

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install the dependencies.

```bash
pip install -r requirements.txt
```

4. Launch the Gradio app.

```bash
python app.py
```
Evaluation results obtained with T=0.7 and number of tokens 250.

| Model             | BLEU         | ROUGE-L      | ModelBasedScore |
|-------------------|--------------|--------------|--------------|
| Base              | 0.0764       | 0.2156       | 4.4834       |
| Intermediate      | 0.2878       | 0.4675       | 4.1107       |
| FineTune          | 0.3003       | 0.4819       | 4.0775       |


Results obtained with T=0.2 and no of tokens 500
| Model             | BLEU         | ROUGE-L      | ModelBasedScore |
|-------------------|--------------|--------------|--------------|
| Base              | 0.0688       | 0.1972       | 4.5812       |
| Intermediate      | 0.3208       | 0.5025       | 4.1956       |
| FineTune          | 0.3218       | 0.5070       | 4.1125       |

