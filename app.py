# Standard Library Imports
import logging
import os
import config

# Third-party Imports
from dotenv import load_dotenv
import chromadb
import gradio as gr
from huggingface_hub import snapshot_download
from llama_index.core.llms import ChatMessage, MessageRole

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
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

PROMPT_SYSTEM_MESSAGE = """You are a telecommunications teacher, answering questions from students of a curse of non terrestrial networks, 5G, 6G networks. Questions should be understood in this context. Your answers are aimed to teach 
students, so they should be complete, clear, and easy to understand. Use the available tools to gather insights pertinent to the field of non terrestrial networks, 5G and 6G networks.
To find relevant information for answering student questions, always use the "AI_Information_related_resources" tool.

Only some information returned by the tool might be relevant to the question, so ignore the irrelevant part and answer the question with what you have. Your responses are exclusively based on the output provided 
by the tools. Refrain from incorporating information not directly obtained from the tool's responses.
If a user requests further elaboration on a specific aspect of a previously discussed topic, you should reformulate your input to the tool to capture this new angle or more profound layer of inquiry. Provide 
comprehensive answers, ideally structured in multiple paragraphs, drawing from the tool's variety of relevant details. The depth and breadth of your responses should align with the scope and specificity of the information retrieved. 
Should the tool response lack information on the queried topic, politely inform the user that the question transcends the bounds of your current knowledge base, citing the absence of relevant content in the tool's documentation. 
At the end of your answers, always invite the students to ask deeper questions about the topic if they have any.
Do not refer to the documentation directly, but use the information provided within it to answer questions. If code is provided in the information, share it with the students. It's important to provide complete code blocks so 
they can execute the code when they copy and paste them. Make sure to format your answers in Markdown format, including code blocks and snippets.
"""

PDF_READER_LIB="PyPDF2"
import gradio as gr
import arxiv
from transformers import pipeline  # For summarization
from llama_index.core.tools import FunctionTool

def extract_text_from_pdf(pdf_path: str) -> str | None:
    """
    Extracts text content from a given PDF file path using PyPDF2.

    Args:
        pdf_path: The full path to the PDF file.

    Returns:
        A string containing the extracted text, or None if an error occurs
        or no text is extracted.
    """
    # 1. Validate the input path
    if not pdf_path or not isinstance(pdf_path, str):
        print(f"Error: Invalid pdf_path provided: {pdf_path}")
        return None
    if not os.path.exists(pdf_path):
        print(f"Error: File does not exist at path: {pdf_path}")
        return None

    extracted_text = ""
    file_basename = os.path.basename(pdf_path) # For clearer log messages

    try:
        # 2. Open the PDF file in binary read mode ('rb')
        with open(pdf_path, 'rb') as pdf_file_object:
            # 3. Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file_object)

            # 4. Check if the PDF is encrypted
            if pdf_reader.is_encrypted:
                try:
                    # Attempt to decrypt with an empty password (common for unlocked PDFs)
                    pdf_reader.decrypt('')
                except Exception as decrypt_error:
                    # If decryption fails, we likely can't read it
                    print(f"Warning: Skipping password-protected PDF: '{file_basename}'. Decryption failed: {decrypt_error}")
                    return None # Cannot process encrypted file

            # 5. Iterate through each page and extract text
            num_pages = len(pdf_reader.pages)
            # print(f"Reading {num_pages} pages from '{file_basename}'...") # Optional progress
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text: # Check if text was actually extracted
                        extracted_text += page_text + "\n" # Add extracted text
                    # else: # Optional debug for empty pages
                    #     print(f"Debug: No text found on page {page_num + 1} of '{file_basename}'")
                except Exception as page_error:
                    # Log error for specific page but try to continue
                    print(f"Warning: Error extracting text from page {page_num + 1} of '{file_basename}': {page_error}")
                    # traceback.print_exc() # Uncomment for full traceback during debugging

        # 6. Return the concatenated text, or None if nothing was extracted
        return extracted_text if extracted_text.strip() else None

    except FileNotFoundError:
        print(f"Error: File suddenly not found during extraction at {pdf_path}")
        return None
    except PyPDF2.errors.PdfReadError as pdf_error:
        print(f"Error: PyPDF2 failed to read PDF '{file_basename}'. File might be corrupted or not a valid PDF. Error: {pdf_error}")
        return None
    except Exception as e:
        # Catch any other unexpected errors during file handling or reading
        print(f"Error: An unexpected error occurred while processing PDF '{file_basename}': {e}")
        # traceback.print_exc() # Uncomment for full traceback during debugging
        return None

# --- 1. ArXiv Paper Search Function ---
def search_arxiv(keywords, max_results=5):
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
    Summarizes the abstracts of research papers.

    Args:
        papers (list): A list of dictionaries, where each dictionary represents a paper
                      and contains 'title', 'summary', and 'pdf_url'.

    Returns:
        str: A string containing the summaries of all papers.
    """
    print(papers)
    all_summaries = ""
    for paper in papers:
        title = os.path.basename(paper.name)
        #summary = #summarizer(paper["summary"], max_length=250, min_length=100)[0]["summary_text"]
        summary = summarize_single_pdf(paper.name, OPENAI_API_KEY, max_summary_length=1000, model=config.intermediate_model_id)
        all_summaries += f"Title: {title}\nSummary: {summary}\n\n"
    return all_summaries

import os
import io
import PyPDF2
import openai
import requests
from llama_index.core.memory import ChatMemoryBuffer

def summarize_single_pdf(pdf_path, openai_api_key, max_summary_length=1000, model="gpt-3.5-turbo"):
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

    text = extract_text_from_pdf(pdf_path)

    chunks = _split_text_into_chunks(text, 10000)

    summaries = []
    for chunk in chunks:
        prompt = f"Summarize the following text in less than 1000 tokens:\n\n{chunk}\n\nSummary:"
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

# --- 3. Article Writing Function (Conceptual - Requires LLM Integration) ---
def write_article(summaries, arxiv_summaries):
    """
    Generates an article based on the provided summaries and an article prompt.

    Args:
        summaries (str): Summaries of the research papers.
        article_prompt (str): Prompt guiding the article generation.

    Returns:
        str: The generated article.
    """
    prompt = f"""You are an expert researcher and writer specializing in the fields of non terrestrial networks (NTN) and 5G networks. Your task is to analyze a given input text (a research paper excerpt, a technical report, or a description of a project) and a summary of current research papers and generate two outputs:

1.  **Abstract:** A concise summary of the input text, highlighting the key contributions, methodologies, and findings. Limit the abstract to 150 words.
2.  **Article Skeleton:** A structured outline for a text based on the input text, and the summary of current research including:
    * **Introduction:** Briefly introduce the topic and its relevance to NTN and/or 5G Networks. Take into account the summaries to explain why this paper goes beyond state of the art.
    * **Methodology/Approach:** Describe the methods or approaches used in the input text.
    * **Results/Findings:** Summarize the key results or findings.
    * **Discussion/Implications:** Discuss the implications of the findings and potential applications in AI and/or 5G/6G Networks.
    * **Conclusion:** Conclude with a summary of the main points and potential future directions.

    **Input Text:**

    [{arxiv_summaries}]

    **Input summaries:**
    [{summaries}]

    **Output:**

    **Abstract:**

    [Generated Abstract Here]

    **Article Skeleton:**

    **Introduction:**

    [Generated Introduction Here]

    **Methodology/Approach:**

    [Generated Methodology/Approach Here]

    **Results/Findings:**

    [Generated Results/Findings Here]

    **Discussion/Implications:**

    [Generated Discussion/Implications Here]

    **Conclusion:**

    [Generated Conclusion Here]
    """
    # --- LLM Integration Here ---
    # This is where you would integrate with a Large Language Model (LLM) like:
    # - OpenAI's GPT models
    # - Google's Gemini models
    # - A locally hosted LLM
    # You would need to:
    # 1. Choose an LLM and an API or library to interact with it.
    # 2. Construct a prompt that includes the summaries and the article_prompt.
    # 3. Send the prompt to the LLM and get the generated article.
    # --- Placeholder ---
    tools = get_tools(db_collection="Ntn6G5G_tutor_knowledge_db")
    agent = OpenAIAgent.from_tools(
        llm=Settings.llm,
        tools=tools,
        system_prompt=prompt,
    )
    completion = agent.chat("Write an article based on the information provided en the system prompt. Do not add anything else and more specifically," 
    "do not write things like here goes that or similar stuff. Write at least 1000 tokens, be ver in each of the parts of the skeleton").response
    print("COMPLETION: ", completion)
    return completion

import numexpr
import logging
from llama_index.core.tools import FunctionTool

def evaluate_expression(expression: str) -> str:
    """
    Safely evaluates a mathematical expression string using numexpr.
    Handles arithmetic operations, exponentiation, parentheses, etc.
    Returns the result as a string or an error message.
    """
    allowed_chars = "0123456789.+-*/() **" # Basic arithmetic + exponentiation
    # Basic validation (can be enhanced)
    if not all(c in allowed_chars or c.isspace() for c in expression):
         logging.warning(f"Disallowed characters in expression: {expression}")
         return "Error: Expression contains invalid characters."
    try:
        result = numexpr.evaluate(expression)
        return str(result.item()) # .item() converts numpy type to Python scalar
    except SyntaxError:
        logging.error(f"Syntax error in expression: {expression}")
        return "Error: Invalid mathematical expression syntax."
    except Exception as e:
        logging.error(f"Error evaluating expression '{expression}': {e}")
        return f"Error: Could not evaluate expression ({type(e).__name__})."

# --- 4. Gradio Interface ---
def process_research(paper_uploads, keywords_input, max_results_slider):
    """
    Orchestrates the research paper search, summarization, and article writing.

    Args:
        keywords (str): Keywords to search for.
        article_prompt (str): Prompt guiding the article generation.

    Returns:
        tuple: A tuple containing the summaries and the generated article.
    """
    papers = search_arxiv(keywords_input, max_results_slider)
    print(papers)
    print(paper_uploads)
    summaries = summarize_papers(paper_uploads)
    arxiv_summaries = [x["summary"] for x in papers]
    article = write_article(summaries, arxiv_summaries)
    return article

def get_tools(db_collection="Ntn6G5G_tutor_knowledge_db"):
    db = chromadb.PersistentClient(path=f"{db_collection}")
    chroma_collection = db.get_or_create_collection(db_collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        show_progress=True,
        use_async=True,
        embed_model=Settings.embed_model
    )
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
        embed_model=Settings.embed_model,
        use_async=True,
    )
    tools = RetrieverTool(
            retriever=vector_retriever,
            metadata=ToolMetadata(
                name="NTN_5G_6G_Information_related_resources",
                description="Useful for info related to non terrestrial networks, 5G and 6G. It gathers the info from local data.",
            ),
        )
    return [tools]

from llama_index.core.memory import ChatSummaryMemoryBuffer # Ensure correct memory class
from llama_index.core.llms import ChatMessage, MessageRole
# --- Assuming your other imports ---
# from your_module import process_research, multiply, get_tools, PROMPT_SYSTEM_MESSAGE
# from llama_index.core.tools import FunctionTool
# from llama_index.agent.openai import OpenAIAgent # Or AgentRunner
# from llama_index.core.settings import Settings
# --- End assumed imports ---

logging.basicConfig(level=logging.INFO) # Configure logging

# --- Dummy Agent/Tool setup if needed for testing ---
# def get_tools(db_collection): return []
# class Settings: llm = None # Assign your LLM
# PROMPT_SYSTEM_MESSAGE="System prompt"
# class OpenAIAgent: # Dummy class
#     @classmethod
#     def from_tools(cls, llm, memory, tools, system_prompt, verbose=True):
#         return cls(memory)
#     def __init__(self, memory): self.memory = memory
#     def chat(self, query):
#         response_text = f"AI full response to '{query}' using context."
#         # Simulate agent updating memory
#         ai_message = ChatMessage(role=MessageRole.ASSISTANT, content=response_text)
#         self.memory.put(ai_message)
#         from llama_index.core.base.response.schema import Response
#         return Response(response=response_text)
# --- End Dummy Setup ---


# --- Chat Handler Function for Manual Blocks ---
def handle_chat_submit(query: str, chat_state_input: list | dict, summaries: str, evaluation:str) -> tuple[list, list]:
    """
    Handles chat submission for manual gr.Blocks setup.
    """
    logging.info("--- handle_chat_submit function started ---")
    logging.info(f"Received query: {query}")
    logging.info(f"Received chat state input - Type: {type(chat_state_input)}, Value: {chat_state_input}")

    actual_chat_list = []
    # --- Robustly handle receiving a dict or list from state ---
    if isinstance(chat_state_input, list):
        actual_chat_list = chat_state_input
        logging.info("Chat state input is a list, using directly.")
    elif isinstance(chat_state_input, dict):
        logging.warning("Chat state input is a dict! Attempting to extract history list.")
        # Check common keys where Gradio might store the actual list value
        if 'value' in chat_state_input and isinstance(chat_state_input['value'], list):
             actual_chat_list = chat_state_input['value']
             logging.info("Extracted 'value' list from dict state.")
        # Add other potential key checks based on inspecting the logged dict value
        else:
             logging.warning(f"Could not find known list key in dict state: {chat_state_input}. Defaulting to empty list.")
             actual_chat_list = []
    else:
        logging.error(f"Received chat state input of unexpected type: {type(chat_state_input)}. Defaulting to empty list.")
        actual_chat_list = []

    # --- Reconstruct history (ensure ChatMessage objects) ---
    reconstructed_history = []
    try:
        for item in actual_chat_list:
            if isinstance(item, ChatMessage):
                reconstructed_history.append(item)
            elif isinstance(item, dict):
                role_str = item.get("role", MessageRole.USER) # Default role if missing
                # Handle str or Enum representation of role
                try:
                    role = role_str if isinstance(role_str, MessageRole) else MessageRole(str(role_str).lower())
                except ValueError:
                    logging.warning(f"Invalid role value '{role_str}' in state dict, defaulting to USER.")
                    role = MessageRole.USER
                reconstructed_history.append(ChatMessage(role=role, content=item.get("content", "")))
        logging.info(f"Reconstructed history contains {len(reconstructed_history)} ChatMessage objects.")
    except Exception as e:
        logging.error(f"Error reconstructing history: {e}. Starting fresh.", exc_info=True)
        reconstructed_history = []

    # 1. Recreate the memory object
    try:
        # *** Make sure you are using the correct Memory class ***
        memory = ChatSummaryMemoryBuffer(chat_history=reconstructed_history, token_limit=120000)
        logging.info(f"Recreated {type(memory).__name__}. Initial length: {len(memory.get_all())}")
    except Exception as e:
        logging.error(f"Failed to recreate memory object: {e}. Using default.", exc_info=True)
        memory = ChatSummaryMemoryBuffer.from_defaults(token_limit=120000)

    # 2. Execute Agent Logic (Non-Streaming)
    response_text = ""
    try:
        logging.info("Configuring and running agent (non-streaming)...")

        # --- Add user message to memory BEFORE agent call ---
        user_message = ChatMessage(role=MessageRole.USER, content=query)
        memory.put(user_message)
        logging.info(f"Memory updated with user message. Length: {len(memory.get_all())}")

        # --- Replace with your ACTUAL agent setup and call ---
        # tools_list = get_tools(db_collection="ai_tutor_knowledge")
        # multiply_tool = FunctionTool.from_defaults(fn=multiply)
        # all_tools = tools_list + [multiply_tool]
        # agent = OpenAIAgent.from_tools(
        #     llm=Settings.llm, memory=memory, tools=all_tools,
        #     system_prompt=PROMPT_SYSTEM_MESSAGE, verbose=True
        # )
        # response = agent.chat(query) # NON-STREAMING CALL
        # response_text = response.response
        tools = get_tools(db_collection="Ntn6G5G_tutor_knowledge_db")
        calculator_tool = FunctionTool.from_defaults(
            fn=evaluate_expression,
            name="calculator",
            description="Useful for evaluating mathematical expressions. Input should be a string representing the calculation (e.g., '10 * (5.5 + 3)', '10**3')."
        )
        tools = tools + [calculator_tool]
        TEXT_QA_TEMPLATE = f"""
You must answer only related to non terrestrial networks, 5G and 6G networks and related concepts queries.
Always leverage the retrieved documents to answer the questions, don't answer them on your own.
If the query is not relevant to non terrestrial networks, 5G and 6G networks, say that you don't know the answer.
As context you can use the following summaries: {summaries}.
As context also you can use the following evaluation: {evaluation}
"""
        agent = OpenAIAgent.from_tools(
            llm=Settings.llm,
            memory=memory,
            tools=tools,
            system_prompt=TEXT_QA_TEMPLATE,
        )
        response = agent.chat(query) # NON-STREAMING CALL
        response_text = response.response
        # --- End Agent Setup ---

        # --- Simulation (Remove when using real agent) ---
        if not response_text:
            response_text = f"AI full response to '{query}' using context."
            ai_message = ChatMessage(role=MessageRole.ASSISTANT, content=response_text)
            memory.put(ai_message) # Add AI response to memory
            logging.info(f"Memory updated with AI message. Length: {len(memory.get_all())}")
        # --- End Simulation ---

        logging.info(f"Agent finished. Full response received.")

    except Exception as e:
        logging.error(f"Error during agent execution: {e}", exc_info=True)
        response_text = f"Sorry, an error occurred: {e}"
        # Add error message as AI response to memory if agent failed before doing so
        if not any(msg.role == MessageRole.ASSISTANT for msg in memory.get_all()[-2:]): # Check if last wasn't assistant
             memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=response_text))


    # 3. Get Updated History & Format for Chatbot UI
    updated_history_list = memory.get_all() # List of ChatMessage
    logging.info(f"Final history list length from memory object: {len(updated_history_list)}")

    gradio_chatbot_history = []
    user_msg_content = None
    for msg in updated_history_list:
        # Ensure content is string, handle None gracefully
        content = msg.content if msg.content is not None else ""
        if msg.role == MessageRole.USER:
            user_msg_content = content
        elif msg.role == MessageRole.ASSISTANT:
            # Pair with previous user message, handle potential None content
            gradio_chatbot_history.append([user_msg_content, content])
            user_msg_content = None # Reset after pairing

    logging.info(f"Formatted Gradio chatbot history for display: {len(gradio_chatbot_history)} pairs")

    # 4. Return formatted history for chatbot UI and raw list for state
    return gradio_chatbot_history, updated_history_list

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using the available library."""
    if PDF_READER_LIB == "PyPDF2":
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF with PyPDF2: {e}")
            return None
    elif PDF_READER_LIB == "pdfminer":
        try:
            return extract_text(pdf_path)
        except Exception as e:
            print(f"Error reading PDF with pdfminer.six: {e}")
            return None
    else:
        return "Error: No PDF text extraction library found."

def _perform_evaluation(paper_text, paper_name):
    """Placeholder: Actual LLM call/logic to evaluate paper text."""
    print(f"Simulating evaluation for: {paper_name}")

    # 1. Extracting specific claims or results.
    # 2. Comparing against criteria or other papers.
    # 3. Using another LLM to assess quality, novelty, methodology etc.
    # Example:
    # try:
    #    response = client.chat.completions.create(
    #        model="gpt-4-turbo", # Or another strong model for evaluation
    #        messages=[
    #             {"role": "system", "content": "Evaluate the following research paper abstract/text based on clarity, methodology soundness, and potential impact. Provide a brief evaluation."},
    #             {"role": "user", "content": paper_text[:4000]} # Provide relevant section
    #        ]
    #    )
    #    evaluation = response.choices[0].message.content
    # except Exception as e:
    #    evaluation = f"Error evaluating {paper_name}: {e}"
    try:
       response = openai.chat.completions.create(
           model="gpt-4-turbo", # Or another strong model for evaluation
           messages=[
                {"role": "system", "content": "Evaluate the following research paper abstract/text based on clarity, methodology soundness, and potential impact. Provide a brief evaluation."},
                {"role": "user", "content": paper_text[:4000]} # Provide relevant section
           ]
       )
       evaluation = response.choices[0].message.content
    except Exception as e:
       evaluation = f"Error evaluating {paper_name}: {e}"
    return evaluation


def evaluate_paper(file_obj):
    """Processes a single uploaded PDF for evaluation."""
    if file_obj is None:
        return "No file uploaded. Please upload a single PDF paper for evaluation."
    
    for file in file_obj:
        file_path = file.name
        file_name = os.path.basename(file_path)
        print(f"Evaluating file: {file_name} at {file_path}")

    extracted_text = extract_text_from_pdf(file_path)

    if extracted_text:
        evaluation_result = _perform_evaluation(extracted_text, file_name)
        return f"--- Evaluation for: {file_name} ---\n{evaluation_result}\n"
    else:
        return f"--- Could not extract text from: {file_name} for evaluation ---"


if __name__ == "__main__":
    '''
        with gr.Blocks() as iface:
            gr.Markdown("# ArXiv Research Assistant & NTN, 5G and 6G Tutor")

            # State holds the list of ChatMessage objects (or dicts representing them)
            chat_state_list = gr.State([])

            # --- Research Section ---
            with gr.Tab("ArXiv Search & Article"):
                with gr.Row():
                    keywords_input = gr.Textbox(label="Keywords for ArXiv Search")
                    max_results_slider = gr.Slider(minimum=1, maximum=15, value=5, step=1, label="Maximum Results")
                article_prompt_input = gr.Textbox(label="Article Writing Prompt (Optional)")
                search_button = gr.Button("Search and Generate")
                with gr.Row():
                    summaries_output = gr.Markdown(label="Paper Summaries")
                    article_output = gr.Textbox(label="Generated Article", lines=15)

                search_button.click(
                    fn=process_research,
                    inputs=[keywords_input, article_prompt_input, max_results_slider],
                    outputs=[summaries_output, article_output]
                )

            # --- Chat Section (Manual Blocks) ---
            with gr.Tab("NTN, 5G and 6G Tutor Chat"):
                chatbot_display = gr.Chatbot(
                    scale=1,
                    placeholder="Ask questions about non-terrestrial networks, 5G, 6G, or the researched papers.",
                    label="NTN, 5G and 6G Tutor Chat",
                    show_label=True,
                    show_copy_button=True,
                    height=600
                )
                with gr.Row(): # Use Row for better layout possibly
                    chat_input_textbox = gr.Textbox(
                        scale=4, # Make textbox wider
                        show_label=False,
                        placeholder="Type your message here and press Enter...",
                        container=False
                    )
                    # Optional: Add a submit button if needed, though submit on Enter works too
                    # submit_btn = gr.Button("Send", scale=1)

                # Define submit action for the TEXTBOX (pressing Enter)
                chat_input_textbox.submit(
                    fn=handle_chat_submit,
                    inputs=[chat_input_textbox, chat_state_list],
                    # Output 1 updates Chatbot display (expects list of pairs)
                    # Output 2 updates State (expects list of ChatMessage or dicts)
                    outputs=[chatbot_display, chat_state_list]
                )
                # Clear textbox after submit
                chat_input_textbox.submit(fn=lambda: gr.update(value=""), inputs=None, outputs=chat_input_textbox, queue=False)

                # If using a button, wire its click event too:
                # submit_btn.click(...) # Same fn, inputs, outputs as textbox.submit


            iface.queue(default_concurrency_limit=64)
            iface.launch(debug=True, share=False)
    '''
    MAX_SUMMARY_FILES = 10
    with gr.Blocks(title="Paper Processing Suite", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Research Paper Processing Suite
            Use the tabs below for different functions.
            *Note: Processing is currently based on extracted TEXT only.*
            """
        )
        chat_state_list = gr.State([])
        with gr.Tabs():

            # --- Tab 1: Summarization & Evaluation ---
            with gr.TabItem("Summarization"):
                gr.Markdown("### Summarize Multiple Papers")
                with gr.Row():
                    keywords_input = gr.Textbox(label="Keywords for ArXiv Search")
                    max_results_slider = gr.Slider(minimum=1, maximum=15, value=5, step=1, label="Maximum Results", interactive=True)
                with gr.Row():
                    paper_uploads = gr.File(
                        label=f"Upload Papers (PDFs, Max {MAX_SUMMARY_FILES})",
                        file_count="multiple",
                        file_types=[".pdf"],
                        scale=1 # Give slightly less space
                    )
                with gr.Row():
                    summary_output = gr.Textbox(
                        label="Generated Summaries",
                        lines=10,
                        interactive=True,
                        show_copy_button=True
                    )
                    summarize_button = gr.Button("Generate Summaries", variant="primary", scale=0) # Take minimum width
                summarize_button.click(
                    fn=process_research,
                    inputs=[paper_uploads, keywords_input, max_results_slider],
                    outputs=summary_output
                )
            # --- Tab 2: Evaluation ---
            with gr.TabItem("Evaluation"):
                gr.Markdown("## Evaluate Single Paper (using context from Summarization tab)")
                with gr.Row():
                    eval_paper_upload = gr.File(
                        label="Upload Single Paper for Evaluation (PDF)",
                        file_count="single",
                        file_types=[".pdf"],
                        scale=1
                    )
                    evaluate_button = gr.Button("Evaluate Paper", variant="primary", scale=0)

                evaluation_output = gr.Textbox(
                    label="Evaluation Result",
                    lines=150,
                    interactive=False,
                    show_copy_button=True
                )
                evaluate_button.click(
                    fn=evaluate_paper,
                    inputs=paper_uploads,
                    outputs=evaluation_output
                )
            # --- Tab 3: Chatbot (Placeholder) ---
            with gr.TabItem("Chatbot"):
                chatbot_display = gr.Chatbot(
                        scale=1,
                        placeholder="Ask questions about non-terrestrial networks, 5G, 6G, or the researched papers.",
                        label="NTN, 5G and 6G Tutor Chat",
                        show_label=True,
                        show_copy_button=True,
                        height=600
                )
                with gr.Row(): # Use Row for better layout possibly
                    chat_input_textbox = gr.Textbox(
                        scale=4, # Make textbox wider
                        show_label=False,
                        placeholder="Type your message here and press Enter...",
                        container=False
                    )
                    # Optional: Add a submit button if needed, though submit on Enter works too
                    # submit_btn = gr.Button("Send", scale=1)

                # Define submit action for the TEXTBOX (pressing Enter)
                chat_input_textbox.submit(
                    inputs=[chat_input_textbox, chat_state_list, evaluation_output, summary_output],
                    fn=handle_chat_submit,
                    # Output 1 updates Chatbot display (expects list of pairs)
                    # Output 2 updates State (expects list of ChatMessage or dicts)
                    outputs=[chatbot_display, chat_state_list]
                )
                # Clear textbox after submit
                chat_input_textbox.submit(fn=lambda: gr.update(value=""), inputs=None, outputs=chat_input_textbox, queue=False)

                # If using a button, wire its click event too:
                # submit_btn.click(...) # Same fn, inputs, outputs as textbox.submit

    demo.launch()