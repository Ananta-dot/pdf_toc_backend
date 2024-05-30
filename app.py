import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  
import re
import os
import json
import torch
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app)  
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

model = SentenceTransformer('all-MiniLM-L6-v2')

def split_content_by_position(page):
    blocks = page.get_text("blocks")
    height = page.rect.height
    header_blocks = []
    footer_blocks = []
    text_blocks = []

    for block in blocks:
        x0, y0, x1, y1, text, _, _ = block
        if y0 < height * 0.075:
            header_blocks.append(text)
        elif y1 > height * 0.925:
            footer_blocks.append(text)
        else:
            text_blocks.append(text)
    
    header_text = "\n".join(header_blocks)
    footer_text = "\n".join(footer_blocks)
    main_text = "\n".join(text_blocks)
    
    return header_text, main_text, footer_text

def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele
    return str1

dic = {"cur_page": 0}

def search_for_first_occurrence(pdf_path, phrase, phrase_val=None, cur_page=0):
    phrase2 = "END OF "  
    document = fitz.open(pdf_path)
    
    if cur_page >= document.page_count:
        return (0, 0, document.page_count)
    
    for page_num in range(cur_page, document.page_count):
        page = document.load_page(page_num)
        _, text_to_check, _ = split_content_by_position(page)
        first_line = ""
        i = 0

        while first_line.strip() == "" and i < len(text_to_check.split('\n')):
            first_line = text_to_check.split('\n')[i].strip() if text_to_check else ""
            i += 1

        if phrase.lower().split(" ")[1] in first_line.lower():
            for pg in range(page_num + 1, document.page_count):
                page_eos = document.load_page(pg)
                if phrase2 in page_eos.get_text("text"):
                    return (page_num + 1, pg + 1, pg + 1)
                
        elif listToString(phrase_val.lower().split(" ")[:2]) in first_line.lower().split(" ")[:2] and listToString(phrase_val.lower().split(" ")[:2]) != '' and first_line.lower().split(" ")[:2] != ['']:
            for pg in range(page_num + 1, document.page_count):
                page_eos = document.load_page(pg)
                if phrase2 in page_eos.get_text("text"):
                    return (page_num + 1, pg + 1, pg + 1)

    return (0, 0, document.page_count)

def search_for_phrase_consecutive(pdf_path, phrase):
    document = fitz.open(pdf_path)
    consecutive_pages = []
    longest_list = []

    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text_to_check = page.get_text("text")
        if phrase.lower() in text_to_check.lower():
            consecutive_pages.append(page_num + 1) 
        else:
            if consecutive_pages:
                if len(consecutive_pages) > len(longest_list):
                    longest_list = consecutive_pages
                consecutive_pages = []

    if consecutive_pages:
        if len(consecutive_pages) > len(longest_list):
            longest_list = consecutive_pages
    
    return longest_list

def extract_toc_lines(pdf_path, pages):
    document = fitz.open(pdf_path)
    toc_lines = []

    for page_num in pages:
        page = document.load_page(page_num - 1)
        text_to_check = page.get_text("text")   
        lines = text_to_check.split('\n')
        for line in lines:
            toc_lines.append(line.strip())

    return toc_lines

def structure_toc(toc_lines):
    structured_data = {}
    current_division = None
    capture_section = False
    capture_document = False
    section_number = None
    document_number = None
    empty_line_count = 0
    section_pattern1 = re.compile(r'(SECTION)\s*(\d+)\s*[-–—]\s*(.*)', re.IGNORECASE)
    section_pattern2 = re.compile(r'(DOCUMENT)\s*(\d+)\s*[-–—]\s*(.*)', re.IGNORECASE)
    
    for line in toc_lines:
        if re.match(r'DIVISION \d+', line, re.IGNORECASE):
            current_division = line
            structured_data[current_division] = []
            capture_section = False
            empty_line_count = 0
        
        elif section_pattern1.match(line):
            match = section_pattern1.match(line)
            section_type, section_number, section_name = match.groups()
            section_identifier = f'SECTION {section_number.strip()}'
            if current_division not in structured_data:
                structured_data[current_division] = []
            structured_data[current_division].append({section_identifier: section_name.strip()})
            capture_section = False
            empty_line_count = 0
        
        elif section_pattern2.match(line):
            match = section_pattern2.match(line)
            document_type, document_number, document_name = match.groups()
            document_identifier = f'DOCUMENT {document_number.strip()}'
            if current_division not in structured_data:
                structured_data[current_division] = []
            structured_data[current_division].append({document_identifier: document_name.strip()})
            capture_document = False
            empty_line_count = 0
        
        elif re.match(r'SECTION:', line, re.IGNORECASE):
            capture_section = True
            empty_line_count = 0
        
        elif re.match(r'DOCUMENT:', line, re.IGNORECASE):
            capture_document = True
            empty_line_count = 0
        
        elif capture_section and re.match(r'^\d{6}$', line):
            section_number = line.strip()
            empty_line_count = 0
            continue
        
        elif capture_document and re.match(r'^\d{6}$', line):
            document_number = line.strip()
            empty_line_count = 0
            continue
        
        elif capture_section and line:
            section_name = line.strip()
            section_identifier = f'SECTION {section_number}'
            if current_division not in structured_data:
                structured_data[current_division] = []
            structured_data[current_division].append({section_identifier: section_name})
            empty_line_count = 0
        
        elif capture_section and not line:
            empty_line_count += 1
            if empty_line_count >= 2:
                capture_section = False
                section_number = None 
                empty_line_count = 0

        elif capture_document and line:
            document_name = line.strip()
            document_identifier = f'DOCUMENT {document_number}'
            if current_division not in structured_data:
                structured_data[current_division] = []
            structured_data[current_division].append({document_identifier: document_name})
            empty_line_count = 0
        
        elif capture_document and not line:
            empty_line_count += 1
            if empty_line_count >= 2:
                capture_document = False
                document_number = None 
                empty_line_count = 0

    return structured_data

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        phrase = "Table of Contents"
        longest_consecutive_pages_with_phrase = search_for_phrase_consecutive(file_path, phrase)
        toc_pages = longest_consecutive_pages_with_phrase
        toc_lines = extract_toc_lines(file_path, longest_consecutive_pages_with_phrase)
        structured_data = structure_toc(toc_lines)
        page_deets = {}
        sentences = []
        cur_page = 0

        print(f"Structured data: {json.dumps(structured_data, indent=2)}")  # Debugging

        for division, items in structured_data.items():
            for item in items:
                for key, val in item.items():
                    print(f"Processing {key} {val}")  # Debugging
                    page_start, page_end, cur_page = search_for_first_occurrence(file_path, key, val, cur_page)
                    page_deets[f"{key} {val}"] = (page_start, page_end)
                    print(f"cur_page after searching for {key} {val}: {cur_page}")  # Debugging
                    if page_start != 0 and page_end != 0:
                        sentences.append(f"{key} {val}")

        response = {
            "filename": file.filename,
            "TOC Pages": f"{toc_pages[0]} - {toc_pages[-1]}",
            "Sections": {key: f"Page {value[0]} - {value[1]}" for key, value in page_deets.items() if value[0] != 0 and value[1] != 0},
            "Sentences": sentences
        }
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{file.filename}.json")
        print(f"Final cur_page: {cur_page}")  # Debugging
        
        dic['cur_page'] = 0  # Explicitly reset cur_page after processing
        return jsonify(response)
        
    return jsonify({'error': 'File upload failed'})

@app.route('/query', methods=['POST'])
def process_query():
    data = request.get_json()
    query = data['query']
    sentences = data['sentences']

    def find_relevant_sentences(query, sentences, top_k=len(sentences)):
        query_embedding = model.encode(query, convert_to_tensor=True)
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        
        relevant_sentences = []
        for score, idx in zip(top_results[0], top_results[1]):
            relevant_sentences.append((sentences[idx], score.item()))
        return relevant_sentences

    relevant_sentences = find_relevant_sentences(query, sentences)
    result = []
    for sentence, score in relevant_sentences:
        if score >= 0.4:
            result.append({'sentence': sentence})
    return jsonify(result)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('results'):
        os.makedirs('results')
    app.run(debug=True)
