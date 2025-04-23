# Standard Library Imports
import logging
import os
import re

# Third-party Imports
from dotenv import load_dotenv
import chromadb
import gradio as gr
from huggingface_hub import snapshot_download

# LlamaIndex (Formerly GPT Index) Imports
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llms import MessageRole
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.tools import RetrieverTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from config import OPENAI_API_KEY
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

import arxiv
import config
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

import tiktoken

async def agenerate_answer_from_context(question: str, context: str, llm) -> str | None:
    prompt = f"""Context:
{context}

Question: {question}

Based *only* on the provided context, answer the question concisely."""
    try:
        # USE THE ASYNC METHOD (e.g., acomplete or achat)
        response = await llm.acomplete(prompt)
        # Or if using a chat model:
        # messages = [ChatMessage(role="system", content="Answer based only on context."), ChatMessage(role="user", content=prompt)]
        # response = await llm.achat(messages=messages)
        # answer = response.message.content.strip()

        answer = response.text.strip() # Adjust based on actual response object

        # Basic validation
        if answer and "cannot answer" not in answer.lower() and "don't know" not in answer.lower():
            return answer
        else:
            logging.warning(f"LLM couldn't generate valid answer for question: {question[:50]}...")
            return None
    except Exception as e:
        logging.error(f"LLM call failed for Q: {question[:50]}... Error: {e}", exc_info=True)
        return None


async def generate_finetuning_data_async(nodes: list, llm, output_file: str, min_examples_needed: int = 50):
    """Processes nodes asynchronously to generate Q&A pairs."""
    all_examples = []
    total_questions_processed = 0
    total_answers_generated = 0
    print(f"Starting async processing for {len(nodes)} nodes...")
    start_time = time.time()

    # Process nodes sequentially, but questions within each node concurrently
    for i, node in enumerate(nodes):
        node_start_time = time.time()
        node_text = node.get_content()
        print("NODE TEXT: ", node_text)
        if not node_text.strip():
            continue

        questions = node.metadata.get("questions_this_excerpt_can_answer", [])
        print("Questions: ", questions)
        if not questions:
            continue

        tasks = []
        original_questions_map = {} # Map task to original question text
        pattern = r'\d+\.\s+'

        # Use re.split() to split the string based on the pattern
        # This will separate the text *between* the patterns
        split_parts = re.split(pattern, questions)

        # Process the results:
        # - The first element might be empty if the string starts with "1. ".
        # - Filter out any empty strings resulting from the split.
        # - Strip leading/trailing whitespace from each valid part.
        individual_questions = [part.strip() for part in split_parts if part and part.strip()]
        for q in individual_questions:
            if q and isinstance(q, str):
                task = asyncio.create_task(agenerate_answer_from_context(q.strip(), node_text, llm))
                tasks.append(task)
                original_questions_map[task] = q.strip() # Store mapping

        if not tasks:
            continue

        # Wait for all questions for the current node to be processed concurrently
        generated_answers_results = await asyncio.gather(*tasks, return_exceptions=True)

        node_answers_count = 0
        for task, result in zip(tasks, generated_answers_results):
            total_questions_processed += 1
            if isinstance(result, Exception):
                logging.error(f"Task for Q: '{original_questions_map[task][:50]}...' resulted in error: {result}")
            elif result: # If answer is not None
                asyncio.sleep(0.5)
                answer = result
                question = original_questions_map[task]
                print(question)
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
                all_examples.append({"messages": messages})
                total_answers_generated += 1
                node_answers_count +=1

        node_end_time = time.time()
        print(f"Node {i+1}/{len(nodes)} processed {len(tasks)} questions, generated {node_answers_count} answers in {node_end_time - node_start_time:.2f}s. Total Q&A pairs: {total_answers_generated}")

        await asyncio.sleep(0.5) # Example: wait 0.5 seconds between processing nodes

    # --- Write to File ---
    end_time = time.time()
    print(f"\nFinished processing {total_questions_processed} questions across {len(nodes)} nodes in {end_time - start_time:.2f}s.")
    print(f"Generated {total_answers_generated} valid Q&A pairs.")

    if total_answers_generated < min_examples_needed:
        print(f"WARNING: Only generated {total_answers_generated} examples, which might be too few.")
    ################################
    import json
    import logging
    from sklearn.model_selection import train_test_split

    if len(all_examples) < 10: # OpenAI recommends at least 10 examples *total* for a job to run
        logging.error(f"Dataset too small ({len(all_examples)} examples). Cannot create meaningful train/validation split.")
        
    validation_size = 0.1 
    train_size = 1.0 - validation_size

    try:
        train_set, validation_set = train_test_split(
            all_examples,
            test_size=validation_size,
            random_state=42 # Use any integer for reproducibility
        )
        logging.info(f"Data split complete:")
        logging.info(f"  Training set size: {len(train_set)}")
        logging.info(f"  Validation set size: {len(validation_set)}")
################################
        train_file = "train_data.jsonl"
        validation_file = "validation_data.jsonl"

        # Write training set
        with open(train_file, 'w') as f_train:
            for example in train_set:
                f_train.write(json.dumps(example) + "\n")
        logging.info(f"Training data saved to {train_file}")

        # Write validation set
        with open(validation_file, 'w') as f_val:
            for example in validation_set:
                f_val.write(json.dumps(example) + "\n")
        logging.info(f"Validation data saved to {validation_file}")

    except ImportError:
        logging.error("scikit-learn is not installed. Please install it using: pip install scikit-learn")
        logging.info("Alternatively, use the manual shuffling method.")
    except Exception as e:
        logging.error(f"An error occurred during splitting or saving: {e}", exc_info=True)

    ###############################
    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")

    print(f"Generated data written to {output_file}")

def counting_no_tokens(output_data):

  tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

  total_tokens = sum(len(tokenizer.encode(" ".join(message['content'] for message in entry['messages']))) for entry in output_data)

  print(f"Total number of tokens in the Dataset: {total_tokens} \n")


def search_arxiv(keywords, max_results=15):
    """
    Searches for research papers on arXiv based on keywords.

    Args:
        keywords (str): Keywords to search for.
        max_results (int): Maximum number of results to return.

    Returns:
        list: A list of dictionaries, where each dictionary represents a paper
              and contains 'title', 'summary', and 'pdf_url'.
    """
    search = arxiv.Search(
        query=keywords,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "summary": result.summary,
            "pdf_url": result.pdf_url
        })
    return papers

# --- 2. Paper Summarization Function ---
def summarize_papers(papers):
    """
    Summarizes research papers.

    Args:
        papers (list): A list of pdf's.

    Returns:
        str: A string containing the summaries of all papers.
    """
    #summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # You can choose a different summarization model
    all_summaries = []
    for paper in papers:
        #summary = paper["summary"] #summarizer(paper["summary"], max_length=250, min_length=100)[0]["summary_text"]
        summary = summarize_pdf(paper, config.OPENAI_API_KEY, max_summary_length=1000, model="gpt-3.5-turbo")
        all_summaries.append([f"Title: {paper}\nSummary: {summary}\n\n"]) 
    return all_summaries

import io
import PyPDF2
import openai
import requests

def save_pdf(content, filename):
    with open(filename, 'wb') as outfile:
        outfile.write(content.read())

def get_text_pdf(pdf_title):
    pdf_path = "papers/"+pdf_title
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def summarize_pdf(pdf, openai_api_key, max_summary_length=1000, model="gpt-3.5-turbo"):
    """
    Summarizes a PDF file from a URL using OpenAI's GPT models.

    Args:
        pdf_url (str): The URL of the PDF file.
        openai_api_key (str): Your OpenAI API key.
        max_summary_length (int, optional): The maximum length of the summary. Defaults to 1000 characters.
        model (str, optional): The OpenAI model to use. Defaults to "gpt-3.5-turbo".

    Returns:
        str: The summary of the PDF content, or an error message.
    """
    openai.api_key = openai_api_key
    pdf_path = "papers/"+pdf
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            return "Error: Could not extract text from the PDF."

        chunks = _split_text_into_chunks(text, 2048)

        summaries = []
        for chunk in chunks:
            prompt = f"Summarize the following text in less than 400 tokens:\n\n{chunk}\n\nSummary:"
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_summary_length,
            )
            summary = response.choices[0].message.content.strip()
            summaries.append(summary)

        full_summary = " ".join(summaries)

        return full_summary

def _split_text_into_chunks(text, chunk_size):
    """Splits text into chunks of a given size."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks
from typing import List
from llama_index.core import Document

def create_docs_from_list(data_list):
    documents = []
    i = 0
    for data in data_list:
        documents.append(
            Document(
                doc_id=str(i),
                text=data[0],
            )
        )
        i = i + 1
    return documents

def load_documents_from_json_lines(file_path):
    documents = []
    logging.info(f"Attempting to load documents from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_content = f.read()
            logging.debug("Parsing outer JSON array...")
            list_of_json_strings = json.loads(full_content)
            logging.info(f"Successfully parsed outer array. Found {len(list_of_json_strings)} items.")

        if not isinstance(list_of_json_strings, list):
            logging.error("Error: File content did not parse into a JSON array (list).")
            return []

        for item_num, json_string_content in enumerate(list_of_json_strings):
            if not isinstance(json_string_content, str):
                logging.warning(f"Item {item_num + 1} in the array is not a string. Type: {type(json_string_content)}. Skipping.")
                continue

            try:
                inner_data = json.loads(json_string_content)

                if isinstance(inner_data, dict): # Ensure inner JSON parsed to a dictionary
                    doc_text = inner_data.get("text")
                    if doc_text:
                        # Extract metadata, create Document
                        doc_metadata = inner_data.get("metadata", {})
                        if "mimetype" in inner_data and inner_data["mimetype"]:
                            doc_metadata["mimetype"] = inner_data["mimetype"]

                        doc_id = inner_data.get("id_") # Use the ID from inner JSON

                        llama_doc = Document(
                            text=doc_text,
                            metadata=doc_metadata,
                            doc_id=doc_id
                        )
                        documents.append(llama_doc)
                    else:
                        logging.warning(f"No 'text' field found in inner JSON object {item_num + 1}. Skipping.")
                else:
                    logging.warning(f"Inner JSON in item {item_num + 1} did not parse to a dictionary. Type: {type(inner_data)}. Skipping.")

            except json.JSONDecodeError as e:
                logging.error(f"Error decoding inner JSON string for item {item_num + 1}: {e}")
            except Exception as e:
                logging.error(f"Error processing inner data for item {item_num + 1}: {e}")

    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}")
    except json.JSONDecodeError as e:
        # This catches errors parsing the OUTER array
        logging.error(f"Error decoding outer JSON array from file {file_path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred reading or parsing the file {file_path}: {e}", exc_info=True)

    logging.info(f"Finished loading. Created {len(documents)} LlamaIndex Documents.")
    return documents

if __name__ == "__main__":
    import json
    
    import os
    import logging
    import sys
    import chromadb
    from llama_index.core import (
        VectorStoreIndex,
        StorageContext,
        Document,  # Make sure Document is imported
        Settings,  # Use Settings for global configuration
        SimpleDirectoryReader # Example loader, replace if needed
    )
    from llama_index.core.node_parser import TokenTextSplitter
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core.extractors import (
        SummaryExtractor,
        QuestionsAnsweredExtractor,
        KeywordExtractor,
        # TitleExtractor # Example: You could add more extractors
    )
    from llama_index.core.ingestion import IngestionPipeline

    # --- Configuration ---
    # Ensure OPENAI_API_KEY is set as an environment variable
    # from dotenv import load_dotenv
    # load_dotenv()
    # if not os.getenv("OPENAI_API_KEY"):
    #     raise ValueError("OPENAI_API_KEY environment variable not set.")

    # Optional: Configure logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO) # Use INFO for less verbose output, DEBUG for more
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    # --- Global Settings (Recommended Practice) ---
    # Define LLM for metadata extraction and potentially for query synthesis
    Settings.llm = OpenAI(temperature=0, model="gpt-3.5-turbo") 

    # Define embedding model
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # Define chunk size and overlap (used by the splitter)
    Settings.chunk_size = 512
    Settings.chunk_overlap = 128

    keywords = "non terrestrial networks 5G networks 6G networks"
    documents=[]
    papers = search_arxiv(keywords, 100)
    for i, paper in enumerate(papers):
        pdf_url = paper["pdf_url"]
        title = paper["title"]
        try:
            response = requests.get(pdf_url)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            
            with io.BytesIO(response.content) as file:
                save_pdf(file, "papers/"+title)
            t = get_text_pdf(title)
            doc_metadata = {
                "title": title
                # Example: Add another fixed key if desired
                # "source_type": "json_import"
            }
            llama_doc = Document(
                text=t,
                metadata=doc_metadata, 
                doc_id=str(i)
            )
            documents.append(llama_doc)
            print(f"Document {i}/100: {title}")
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to download PDF from URL: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    db_path = "./Ntn6G5G_tutor_knowledge_db" # Store in a sub-directory
    # Use PersistentClient for data to be saved to disk
    db = chromadb.PersistentClient(path=db_path)
    # Get or create the collection
    print("CHROMA CREATE")
    chroma_collection = db.get_or_create_collection("Ntn6G5G_tutor_knowledge")
    print(Ntn6G5G_tutor_knowledge)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    print("Defining ingestion pipeline...")
    text_splitter = TokenTextSplitter(separator=" ")

    extractors = [
        QuestionsAnsweredExtractor(questions=3), 
        SummaryExtractor(summaries=["prev", "self"]),
        KeywordExtractor(keywords=10),
        # TitleExtractor(),
    ]

    pipeline = IngestionPipeline(
        transformations=[text_splitter] + extractors + [Settings.embed_model],
        vector_store=vector_store,
    )
    nodes = pipeline.run(documents=documents, show_progress=True)
    if nodes: # Check if nodes were actually generated
        print("\n--- Sample Processed Node ---")
        print("Node ID:", nodes[-1].node_id)
        print("Node Text Snippet:", nodes[-1].get_content(metadata_mode="all")[:200] + "...") # Show text + metadata
        print("Node Metadata:", nodes[-1].metadata)
        print("Node Embedding Length:", len(nodes[-1].get_embedding()))

    import asyncio
    import json
    import logging
    import time
    from llama_index.core.llms import ChatMessage, MessageRole


    output_filename = "finetuning_data_async.jsonl"

    try:
        asyncio.run(generate_finetuning_data_async(nodes, Settings.llm, output_filename))
    except Exception as e:
        logging.error(f"An error occurred during async processing: {e}", exc_info=True)
