import sys
import os
import argparse
from transformers import GPT2Tokenizer
import anthropic
import ebooklib
from ebooklib import epub
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Claude API client
claude = anthropic.Client(api_key="[insert API key]")

def split_text(input_file, output_dir, chunk_size=3000, overlap=100):
    # Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenize the entire text
    tokens = tokenizer.encode(text)

    # Split tokens into chunks
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[max(0, i - overlap):i + chunk_size]
        chunks.append(chunk)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write chunks to separate files
    for i, chunk in enumerate(chunks):
        chunk_text = tokenizer.decode(chunk)
        output_file = os.path.join(output_dir, f'chunk_{i+1}.txt')
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(chunk_text)

    logging.info(f"Split {len(tokens)} tokens into {len(chunks)} chunks.")
    return output_dir

def process_chunk(chunk):
    try:
        response = claude.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": f"This is a podcast transcript. Please edit the punctuation and format it into clear paragraphs. Ensure proper spacing between paragraphs. Make no other changes and add no other words to the output:\n\n{chunk}"
                }
            ]
        )
        
        if response and response.content:
            processed_text = response.content[0].text.strip()
            logging.info(f"Chunk processed successfully. Length: {len(processed_text)}")
            return processed_text
        else:
            logging.error("Error: Claude API did not return a valid response.")
            return ""
    
    except Exception as e:
        logging.error(f"An error occurred while processing the chunk: {e}")
        return ""

def update_chunk_files(chunks_dir):
    updated_chunks = []
    for filename in tqdm(sorted(os.listdir(chunks_dir)), desc="Processing chunks"):
        if filename.endswith('.txt'):
            filepath = os.path.join(chunks_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                chunk = f.read()
            logging.info(f"Processing chunk from file: {filename}")
            logging.info(f"Original chunk length: {len(chunk)}")
            processed_chunk = process_chunk(chunk)
            if processed_chunk:
                updated_chunks.append(processed_chunk)
                # Update the chunk file with processed content
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(processed_chunk)
                logging.info(f"Updated {filename} with processed content. Length: {len(processed_chunk)}")
            else:
                logging.warning(f"Empty processed chunk for file: {filename}")
    return updated_chunks

def combine_chunks(chunks_dir):
    combined = ""
    for filename in sorted(os.listdir(chunks_dir)):
        if filename.endswith('.txt'):
            filepath = os.path.join(chunks_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                chunk = f.read()
            combined += chunk + "\n\n"  # Add two newlines between chunks
    logging.info(f"Combined chunks. Total length: {len(combined)}")
    return combined.strip()

def create_epub(content, filename, author):
    logging.info(f"Starting ePub creation with content length: {len(content)}")
    
    book = epub.EpubBook()
    book.set_identifier('id123456')
    book.set_title(os.path.splitext(os.path.basename(filename))[0])
    book.set_language('en')
    book.add_author(author)

    html_content = ''.join(f'<p>{p}</p>' for p in content.split('\n\n') if p.strip())
    logging.info(f"Created HTML content for ePub. Length: {len(html_content)}")

    chapter = epub.EpubHtml(title='Chapter 1', file_name='chap_01.xhtml', lang='en')
    chapter.content = f'<html><body>{html_content}</body></html>'
    book.add_item(chapter)

    book.toc = (epub.Link('chap_01.xhtml', 'Chapter 1', 'chapter1'),)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    book.spine = ['nav', chapter]

    epub.write_epub(filename, book, {})
    logging.info(f"ePub file created: {filename}")
    logging.info(f"ePub file size: {os.path.getsize(filename)} bytes")

def main():
    parser = argparse.ArgumentParser(description="Split a large text file into smaller chunks, process them with Claude, combine them, and create an ePub.")
    parser.add_argument("input_file", help="Path to the input text file")
    parser.add_argument("output_dir", help="Directory to save the output chunks and ePub")
    parser.add_argument("--chunk_size", type=int, default=3000, help="Maximum number of tokens per chunk (default: 3000)")
    parser.add_argument("--overlap", type=int, default=100, help="Number of overlapping tokens between chunks (default: 100)")

    args = parser.parse_args()

    try:
        # Split the text into chunks
        chunks_dir = split_text(args.input_file, args.output_dir, args.chunk_size, args.overlap)

        # Process chunks with Claude API and update chunk files
        update_chunk_files(chunks_dir)

        # Combine processed chunks
        full_content = combine_chunks(chunks_dir)

        # Save combined content to a text file for verification
        text_filename = os.path.join(args.output_dir, "combined_content.txt")
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write(full_content)
        logging.info(f"Saved combined content to {text_filename}")
        logging.info(f"Combined content file size: {os.path.getsize(text_filename)} bytes")

        # Ask the user for the ePub filename and author
        epub_filename = input("Enter the name for the ePub file (without extension): ") + ".epub"
        epub_filename = os.path.join(args.output_dir, epub_filename)
        author = input("Enter the author of the ePub: ")

        # Create ePub with combined content
        create_epub(full_content, epub_filename, author)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()