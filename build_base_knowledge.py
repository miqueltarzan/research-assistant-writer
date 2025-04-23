# Standard Library Imports
import logging
import os

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

def counting_no_tokens(output_data):

  tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

  total_tokens = sum(len(tokenizer.encode(" ".join(message['content'] for message in entry['messages']))) for entry in output_data)

  print(f"Total number of tokens in the Dataset: {total_tokens} \n")

def dataset_preparation(file_name):
    file_path = hf_hub_download(
        repo_id="jaiganesan/GPT_4o_mini_Fine_tune",
        filename=file_name,
        repo_type="dataset",
        local_dir="/content"
    )

    with open(file_path, "r") as file:
        data = [json.loads(line) for line in file]

    print("Total entries in the dataset:", len(data))
    print("-_"*30)
    print(data[4])

    output_data = []

    for entry in data:
        formatted_entry = {
            "messages": [
                {"role": "system", "content": "As AI Tutor, answer questions related to AI topics in an in-depth and factual manner."},
                {"role": "user", "content": entry['question']},
                {"role": "assistant", "content": entry['answer']}
            ]
        }
        output_data.append(formatted_entry)

    # Validate and analyze the output data
    validate_dataset(output_data)
    counting_no_tokens(output_data)

    print("-_"*30)
    print(output_data[4])

    base_file_name = os.path.splitext(file_name)[0]
    output_file_path = f'formatted_{base_file_name}.jsonl'

    with jsonlines.open(output_file_path, mode='w') as writer:
        writer.write_all(output_data)

    print(f"\nFormatted dataset has been saved to {output_file_path}.")


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
            # Read the entire file content as one string
            full_content = f.read()
            # Parse the outer JSON array
            logging.debug("Parsing outer JSON array...")
            list_of_json_strings = json.loads(full_content)
            logging.info(f"Successfully parsed outer array. Found {len(list_of_json_strings)} items.")

        if not isinstance(list_of_json_strings, list):
            logging.error("Error: File content did not parse into a JSON array (list).")
            return []

        # Iterate through the list of strings obtained from the outer array
        for item_num, json_string_content in enumerate(list_of_json_strings):
            if not isinstance(json_string_content, str):
                logging.warning(f"Item {item_num + 1} in the array is not a string. Type: {type(json_string_content)}. Skipping.")
                continue

            try:
                # Parse the inner (escaped) JSON string into a Python dict
                # logging.debug(f"Parsing inner JSON string for item {item_num + 1}...")
                inner_data = json.loads(json_string_content)

                if isinstance(inner_data, dict): # Ensure inner JSON parsed to a dictionary
                    doc_text = inner_data.get("text")
                    if doc_text:
                        # Extract metadata, create Document
                        doc_metadata = inner_data.get("metadata", {})
                        if "mimetype" in inner_data and inner_data["mimetype"]:
                            doc_metadata["mimetype"] = inner_data["mimetype"]
                        # Add any other metadata processing you need

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
    '''
    keywords = "non terrestrial networks 5G networks 6G networks"
    papers = search_arxiv(keywords, 15)
    for paper in papers:
        pdf_url = paper["pdf_url"]
        title = paper["title"]
        try:
            response = requests.get(pdf_url)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            with io.BytesIO(response.content) as file:
                save_pdf(file, "papers/"+title)
        
        except requests.exceptions.RequestException as e:
            print(f"Error: Failed to download PDF from URL: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    papers_dir = os.path.join(script_dir, "papers")
    items = os.listdir(papers_dir)
    summaries = summarize_papers(items)
    doc = create_docs_from_list(summaries)
    # Saving to JSON
    import json
    documents_data = [doc.to_json() for doc in doc]
    with open("documents.json", "w") as f:
        json.dump(documents_data, f, indent=4)
    print(doc[-1])
    doc = load_documents_from_json_lines("documents.json")
    from llama_index.core.node_parser import TokenTextSplitter
    # Define the splitter object that split the text into segments with 1536 tokens,
    # with a 128 overlap between the segments.
    text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.llms.openai import OpenAI
    from llama_index.core.extractors import (
        SummaryExtractor,
        QuestionsAnsweredExtractor,
        KeywordExtractor,
    )
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core.ingestion import IngestionPipeline

    # set up ChromaVectorStore and load in data
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("ai_tutor_knowledge")

    # save to disk
    db = chromadb.PersistentClient(path="ai_tutor_knowledge")
    chroma_collection = db.get_or_create_collection("ai_tutor_knowledge")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    llm = OpenAI(temperature=0, model="gpt-4o-mini")

    pipeline = IngestionPipeline(
        transformations=[
            text_splitter,
            QuestionsAnsweredExtractor(questions=2, llm=llm),
            SummaryExtractor(summaries=["prev", "self"], llm=llm),
            KeywordExtractor(keywords=10, llm=llm),
            OpenAIEmbedding(model = "text-embedding-3-small"),
        ],
        vector_store=vector_store,
    )

    # Run the transformation pipeline.
    nodes = pipeline.run(documents=doc, show_progress=True)
    print("Nodes: ", nodes[-1])
    print("Nodes embedding: ", len(nodes[-1].embedding))
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    results = chroma_collection.query(
        query_texts=["Explain Non terrestrial networks and 5G and 6G?"],
        n_results=10  # Or your desired top_k
    )
    print(results)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    from llama_index.core import VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    for i in [2, 4, 6, 8, 10, 15, 20, 25, 30]:
        query_engine = index.as_query_engine(similarity_top_k=i)

        res = query_engine.query("Explain Non terrrestrial networks and 5G and 6G?")

        print(f"top_{i} results:")
        print("\t", res.response)
        print("-_" * 20)
'''
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
Settings.llm = OpenAI(temperature=0.1, model="gpt-4o-mini") # Using a slightly higher temp for variety if needed, 0 is fine too.

# Define embedding model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Define chunk size and overlap (used by the splitter)
Settings.chunk_size = 512
Settings.chunk_overlap = 128

# --- 1. Load Documents ---
# Replace this with your actual loading logic for 'documents.json'
# Assuming 'documents.json' is a JSON Lines file where each line is a JSON object
# with a 'text' field (or similar) representing the document content.
# If using a standard LlamaIndex loader, adapt as needed.
# Example using SimpleDirectoryReader if docs were text files in a directory:
# reader = SimpleDirectoryReader(input_dir="./data_folder")
# documents = reader.load_data()

# Placeholder for your custom loader function:


# If your load_documents_from_json_lines isn't defined, you need to provide it.
# For demonstration, let's create dummy documents if the function isn't available
try:
    # Attempt to load using the placeholder function name
    docs = load_documents_from_json_lines("documents.json")
    if not docs: # If loading failed or returned empty
        raise ImportError # Treat as if the function wasn't found
except (ImportError, NameError, FileNotFoundError):
    print("Warning: 'load_documents_from_json_lines' not found or 'documents.json' missing/invalid. Using dummy documents.")
    docs = [
        Document(text="Non-terrestrial networks (NTNs) extend connectivity beyond Earth's surface using satellites or airborne platforms. They are crucial for 5G Advanced and 6G, aiming for global coverage, especially in remote or underserved areas. NTNs integrate with terrestrial networks to provide seamless service."),
        Document(text="5G incorporates initial support for NTNs, primarily focusing on specific use cases like IoT backhaul and enhancing coverage. Standardization efforts in 3GPP Release 17 and 18 laid the groundwork for satellite integration into the 5G ecosystem."),
        Document(text="6G envisions a deeper integration of terrestrial and non-terrestrial networks, creating a unified, multi-layered infrastructure. This 'network of networks' will leverage satellites (LEO, MEO, GEO) and HAPS for ubiquitous, high-resilience, low-latency connectivity, supporting advanced applications like holographic communication and the tactile internet."),
        Document(text="Challenges for NTN integration include latency (especially for GEO satellites), Doppler shift, handover management between terrestrial and non-terrestrial cells, and regulatory aspects for spectrum usage and orbital slots.")
    ]
    # You would typically load your actual documents here.
    # docs = load_documents_from_json_lines("documents.json")

# --- 2. Setup ChromaDB Vector Store ---
print("Setting up ChromaDB...")
# Define the path for the persistent database
db_path = "./ai_tutor_knowledge_db" # Store in a sub-directory
# Use PersistentClient for data to be saved to disk
db = chromadb.PersistentClient(path=db_path)
# Get or create the collection
chroma_collection = db.get_or_create_collection("ai_tutor_knowledge")
# Create the LlamaIndex vector store wrapper
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# --- 3. Define Ingestion Pipeline ---
print("Defining ingestion pipeline...")
# Define the splitter - uses Settings.chunk_size and Settings.chunk_overlap
text_splitter = TokenTextSplitter(separator=" ")

# Define metadata extractors (using the LLM from Settings)
extractors = [
    QuestionsAnsweredExtractor(questions=3), 
    SummaryExtractor(summaries=["prev", "self"]),
    KeywordExtractor(keywords=10),
    # TitleExtractor(), # Example: Add title extractor if desired
]

# Define the ingestion pipeline
pipeline = IngestionPipeline(
    transformations=[text_splitter] + extractors + [Settings.embed_model], # Add embed_model last
    vector_store=vector_store,
)
nodes = []
# --- 4. Run Ingestion Pipeline ---
# Check if the vector store is empty before running ingestion
# This avoids re-processing if the script is run multiple times
if chroma_collection.count() == 0:
    print(f"Vector store is empty. Running ingestion pipeline for {len(docs)} documents...")
    # Run the pipeline (this splits, extracts metadata, embeds, and stores)
    nodes = pipeline.run(documents=docs, show_progress=True)
    print(f"Ingestion complete. Processed {len(nodes)} nodes.")
    if nodes: # Check if nodes were actually generated
      print("\n--- Sample Processed Node ---")
      print("Node ID:", nodes[-1].node_id)
      print("Node Text Snippet:", nodes[-1].get_content(metadata_mode="all")[:200] + "...") # Show text + metadata
      print("Node Metadata:", nodes[-1].metadata)
      print("Node Embedding Length:", len(nodes[-1].get_embedding()))
else:
    nodes = pipeline.run(documents=docs, show_progress=True)
    print(f"Ingestion complete. Processed {len(nodes)} nodes.")
    if nodes: # Check if nodes were actually generated
      print("\n--- Sample Processed Node ---")
      print("Node ID:", nodes[-1].node_id)
      print("Node Text Snippet:", nodes[-1].get_content(metadata_mode="all")[:200] + "...") # Show text + metadata
      print("Node Metadata:", nodes[-1].metadata)
      print("Node Embedding Length:", len(nodes[-1].get_embedding()))
    print(f"Vector store already contains {chroma_collection.count()} nodes. Skipping ingestion.")
    # If ingestion is skipped, pipeline.run() doesn't return nodes from the store
    # You might load them if needed, but usually you just proceed to querying

# --- 5. Direct ChromaDB Query (Verification) ---
# --- 5. Direct LlamaIndex Vector Store Query (Verification) ---
print("\n--- Direct LlamaIndex Vector Store Query (Verification) ---")
from llama_index.core.vector_stores import VectorStoreQuery 

query_text = "Explain Non terrestrial networks and 5G and 6G?"
try:
    print(f"Query: '{query_text}'")
    # 1. Manually embed the query text using the correct model from Settings
    query_embedding = Settings.embed_model.get_query_embedding(query_text)
    print(f"Generated query embedding with dimension: {len(query_embedding)}") # Should be 1536

    # 2. Create a VectorStoreQuery object
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=5, # Fetch top 5 similar nodes
        filters=None # Explicitly None, though this might still hit the previous 'where={}' bug
    )

    # 3. Query using the LlamaIndex vector_store wrapper object
    # This ensures consistent embedding usage and query construction
    verification_results = vector_store.query(vector_store_query)

    # 4. Process and print results
    if verification_results and verification_results.nodes:
        print(f"Found {len(verification_results.nodes)} results via vector_store.query:")
        for i, node in enumerate(verification_results.nodes):
            print(f"  Result {i+1}:")
            print(f"    ID: {node.node_id}")
            print(f"    Similarity Score: {verification_results.similarities[i]:.4f}")
            # Accessing metadata stored within the Node object
            # print(f"    Metadata: {node.metadata}") # Can be verbose
            print(f"    Document Snippet: {node.get_content()[:150]}...") # Show text snippet
    else:
        print("No results found via vector_store.query.")

except Exception as e:
    print(f"Error during direct vector_store.query verification: {e}")
    import traceback
    traceback.print_exc() # Print full traceback if an error occurs


# --- 6. Setup LlamaIndex Query Engine ---
print("\n--- Setting up LlamaIndex Query Engine ---")
# Create the index object from the existing vector store
# It automatically uses the embed_model from Settings to embed the query
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
)

# --- 7. Query using LlamaIndex with varying top_k ---
print("\n--- Querying with LlamaIndex ---")
query_text = "Explain Non terrestrial networks and 5G and 6G?"

# Loop through different values of similarity_top_k
top_k_values = [2, 4, 6] # Reduced list for quicker testing

for k in top_k_values:
    print(f"\n--- Running Query with similarity_top_k = {k} ---")
    # Create the query engine
    # It automatically uses the llm from Settings to synthesize the answer
    query_engine = index.as_query_engine(
        similarity_top_k=k,
        # You can add other configurations like response_mode here if needed
        # response_mode="tree_summarize"
    )

    # Execute the query
    response = query_engine.query(query_text)

    # Print the response
    print(f"Query: '{query_text}'")
    print(f"Response (k={k}):\n{response}")

    # Optional: Print source nodes used for the response
    print("\nSource Nodes:")
    for node in response.source_nodes:
        print(f"  Node ID: {node.node_id}, Score: {node.score:.4f}")
        print(f"  Content: {node.text[:100]}...")
        print("-" * 20)

nodes = response.source
print("\nScript finished for nodes, now starts fine-tuning pre-processing.")

import json
import logging
from llama_index.core.llms import ChatMessage, MessageRole # Assuming these might be needed if converting dicts

# Assume 'nodes' is your list from the pipeline run
# Assume 'Settings.llm' is configured for generating answers
# Assume the metadata key is 'questions_this_excerpt_can_answer'

output_file = "finetuning_data_from_extractor.jsonl"
examples = []
min_examples_needed = 50

def generate_answer_from_context(question, context):
    """Placeholder: Uses LLM to generate answer based ONLY on context."""
    try:
        # Construct a prompt for the LLM
        prompt = f"""Context:
{context}

Question: {question}

Based *only* on the provided context, answer the question concisely."""

        # Make the LLM call (ensure Settings.llm is configured)
        # Adjust based on your specific LLM setup (e.g., .complete vs .chat)
        response = Settings.llm.complete(prompt)
        answer = response.text.strip()

        # Basic check if answer is reasonable (not empty, not refusal)
        if answer and "cannot answer" not in answer.lower() and "don't know" not in answer.lower():
            return answer
        else:
            logging.warning(f"LLM couldn't generate valid answer for question: {question}")
            return None
    except Exception as e:
        logging.error(f"LLM call failed for Q: {question}. Error: {e}")
        return None

# --- Main Loop ---
processed_questions = 0
for node in nodes:
    node_text = node.get_content()
    if not node_text.strip():
        continue

    # Adjust key if necessary based on inspecting node.metadata
    questions = node.metadata.get("questions_this_excerpt_can_answer", [])

    if not questions:
        continue

    for q in questions:
        if not q or not isinstance(q, str): # Basic validation
             continue

        # Generate the answer using the node's text as context
        answer = generate_answer_from_context(q, node_text)

        if answer:
            # Format for OpenAI fine-tuning
            messages = [
                {"role": "user", "content": q.strip()},
                {"role": "assistant", "content": answer} # Use the generated answer
            ]
            examples.append({"messages": messages})
            processed_questions += 1

# --- Write to File ---
if len(examples) < min_examples_needed:
    print(f"WARNING: Only generated {len(examples)} Q&A pairs, which might be too few.")

with open(output_file, 'w') as f:
    for example in examples:
        f.write(json.dumps(example) + "\n")

print(f"Generated {len(examples)} Q&A pairs in {output_file} from {processed_questions} questions.")

# Now upload output_file to OpenAI and create fine-