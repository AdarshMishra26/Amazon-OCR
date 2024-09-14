import cv2
import numpy as np
import easyocr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
import os
import pandas as pd
import requests
from io import BytesIO

def preprocess_image(image):
    # Apply Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Thresholding to binarize the image
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optional: apply dilation or erosion to improve text clarity
    kernel = np.ones((1, 1), np.uint8)
    processed_image = cv2.dilate(binary_image, kernel, iterations=1)
    processed_image = cv2.erode(processed_image, kernel, iterations=1)
    
    return processed_image

def extract_text(image):
    # Initialize EasyOCR Reader
    reader = easyocr.Reader(['en'])
    
    # Perform OCR on the image
    result = reader.readtext(image, detail=0)  # detail=0 for extracting only text
    
    # Join results into a single string
    extracted_text = " ".join(result)
    
    return extracted_text

def extract_entities(text):
    # Load pre-trained BERT model for NER
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

    # Set up NER pipeline
    nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer)
    
    ner_results = nlp_ner(text)
    entities = {}
    
    # Filter specific entities such as weight, volume, dimensions
    for entity in ner_results:
        if entity['entity'] == 'I-MISC':  # Assuming MISC includes measurement units
            entity_text = text[entity['start']:entity['end']]
            entities[entity_text] = entity['entity']
    
    return entities

def extract_values_from_text(text):
    # Define regex patterns for common entities like weight, dimensions, voltage, wattage
    patterns = {
        'weight': r'(\d+(\.\d+)?\s?(kg|g|lbs))',
        'volume': r'(\d+(\.\d+)?\s?(ml|liters|L))',
        'dimensions': r'(\d+(\.\d+)?\s?x\s?\d+(\.\d+)?\s?x\s?\d+(\.\d+)?\s?(cm|mm|inches))',
        'voltage': r'(\d+(\.\d+)?\s?V)',
        'wattage': r'(\d+(\.\d+)?\s?W)'
    }
    
    extracted_values = {}
    
    # Extract matching patterns
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            extracted_values[key] = match.group()
    
    return extracted_values

def process_image_from_url(image_url):
    try:
        # Download image from URL
        response = requests.get(image_url)
        image = np.array(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        
        # Step 1: Preprocess the image
        processed_image = preprocess_image(image)
        
        # Step 2: Extract text using OCR
        extracted_text = extract_text(processed_image)
        
        # Step 3: Extract entities using NER
        extracted_entities = extract_entities(extracted_text)
        
        # Step 4: Extract specific values using regex
        extracted_values = extract_values_from_text(extracted_text)
        
        return {
            'image_url': image_url,
            'text': extracted_text,
            'entities': extracted_entities,
            'values': extracted_values
        }
    except Exception as e:
        print(f"Error processing image {image_url}: {e}")
        return {
            'image_url': image_url,
            'text': '',
            'entities': {},
            'values': {}
        }

def process_images_from_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    image_urls = df['images_url'].head(30).tolist()
    
    results = []
    
    for url in image_urls:
        result = process_image_from_url(url)
        results.append(result)
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

# Example usage
input_csv = 'test.csv'
output_csv = 'ocr_results.csv'
process_images_from_csv(input_csv, output_csv)
print(f"Results saved to {output_csv}")
