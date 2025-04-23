# --- Code Refinement ---
import json
import argparse
import os
from typing import List, Dict, Any # For type hints

def convert_openai_to_vertex_contents(openai_file_path: str, vertex_file_path: str):
    """
    Converts an OpenAI fine-tuning JSONL file (multi-turn chat format)
    to the Vertex AI fine-tuning JSONL format using the "contents" structure
    (suitable for models like Gemini).

    Input line format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    Output line format: {"contents": [{"role": "user", "parts": [{"text": "..."}]}, {"role": "model", "parts": [{"text": "..."}]}]}
    """
    print(f"Starting conversion from '{openai_file_path}' to Vertex AI 'contents' format at '{vertex_file_path}'...")
    processed_lines = 0
    skipped_lines = 0
    total_lines = 0

    try:
        # Count lines first
        with open(openai_file_path, 'r', encoding='utf-8') as infile:
            total_lines = sum(1 for _ in infile)
        print(f"Found {total_lines} lines in input file.")

        with open(openai_file_path, 'r', encoding='utf-8') as infile, \
             open(vertex_file_path, 'w', encoding='utf-8') as outfile:

            for i, line in enumerate(infile):
                line_num = i + 1
                try:
                    data = json.loads(line)
                    openai_messages = data.get("messages", [])

                    if not openai_messages:
                        print(f"Warning (Line {line_num}): Skipping line - no 'messages' found. Content: {line.strip()[:100]}...")
                        skipped_lines += 1
                        continue

                    vertex_contents = []
                    valid_line = True
                    for msg in openai_messages:
                        role = msg.get("role")
                        content = msg.get("content", "").strip()

                        if not role or not content:
                            print(f"Warning (Line {line_num}): Skipping line - message missing role or content. Content: {line.strip()[:100]}...")
                            skipped_lines += 1
                            valid_line = False
                            break # Skip this whole line

                        # Map roles: OpenAI's 'assistant' becomes Vertex AI's 'model'
                        if role == "assistant":
                            vertex_role = "model"
                        elif role == "user":
                            vertex_role = "user"
                        # Decide how to handle system prompts - skip, map to user, or error?
                        # Skipping system prompts for now, as their handling varies.
                        elif role == "system":
                            print(f"Info (Line {line_num}): Skipping system message. Content: {content[:100]}...")
                            continue # Skip this specific message but process rest of line
                        else:
                            print(f"Warning (Line {line_num}): Skipping line - unknown role '{role}'. Content: {line.strip()[:100]}...")
                            skipped_lines += 1
                            valid_line = False
                            break # Skip this whole line

                        # Create the Vertex AI message structure
                        vertex_msg = {
                            "role": vertex_role,
                            "parts": [{"text": content}]
                        }
                        vertex_contents.append(vertex_msg)

                    # Only write if the line was valid and resulted in contents
                    if valid_line and vertex_contents:
                        # Ensure conversation doesn't end with user turn if model expects alternation
                        # if vertex_contents[-1]["role"] == "user":
                        #      print(f"Warning (Line {line_num}): Skipping line - conversation ends with user role.")
                        #      skipped_lines += 1
                        #      continue

                        vertex_data = {"contents": vertex_contents}
                        json_line = json.dumps(vertex_data)
                        outfile.write(json_line + '\n')
                        processed_lines += 1
                    elif valid_line and not vertex_contents:
                        # Case where only system prompts were present and skipped
                        print(f"Warning (Line {line_num}): Skipping line - no user/model messages found after filtering. Content: {line.strip()[:100]}...")
                        skipped_lines += 1


                except json.JSONDecodeError:
                    print(f"Warning (Line {line_num}): Skipping invalid JSON line: {line.strip()[:100]}...")
                    skipped_lines += 1
                except Exception as e:
                    print(f"Warning (Line {line_num}): Unexpected error processing line: {e} - Content: {line.strip()[:100]}...")
                    skipped_lines += 1
                    # import traceback
                    # traceback.print_exc()

        print("\nConversion finished.")
        print(f"Successfully processed and wrote lines: {processed_lines}")
        print(f"Skipped lines due to format/errors: {skipped_lines}")
        print(f"Total lines read: {total_lines}")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{openai_file_path}'")
    except Exception as e:
        print(f"An error occurred during file operations: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OpenAI fine-tuning JSONL chat format to Vertex AI 'contents' format.")
    parser.add_argument("-i", "--openai_file", required=True, help="Path to the input OpenAI JSONL file.")
    parser.add_argument("-o", "--vertex_file", required=True, help="Path for the output Vertex AI JSONL file.")
    args = parser.parse_args()

    convert_openai_to_vertex_contents(args.openai_file, args.vertex_file)