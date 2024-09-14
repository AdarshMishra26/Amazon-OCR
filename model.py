import cv2
import numpy as np
import easyocr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
import os

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Thresholding to binarize the image
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optional: apply dilation or erosion to improve text clarity
    kernel = np.ones((1, 1), np.uint8)
    processed_image = cv2.dilate(binary_image, kernel, iterations=1)
    processed_image = cv2.erode(processed_image, kernel, iterations=1)
    
    return processed_image

def extract_text(image_path):
    # Initialize EasyOCR Reader
    reader = easyocr.Reader(['en'])
    
    # Perform OCR on the image
    result = reader.readtext(image_path, detail=0)  # detail=0 for extracting only text
    
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

def process_image_and_extract_entities(image_path):
    # Step 1: Preprocess the image
    processed_image = preprocess_image(image_path)
    cv2.imwrite('processed_image.jpg', processed_image)
    
    # Step 2: Extract text using OCR
    extracted_text = extract_text('processed_image.jpg')
    
    # Step 3: Extract entities using NER
    extracted_entities = extract_entities(extracted_text)
    
    # Step 4: Extract specific values using regex
    extracted_values = extract_values_from_text(extracted_text)
    
    # Create a unique output filename based on image name
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{base_filename}_output.txt"
    
    # Save the output to a unique text file
    with open(output_filename, 'w') as f:
        f.write("Extracted Text:\n")
        f.write(extracted_text + "\n\n")
        f.write("Extracted Entities:\n")
        for entity, entity_type in extracted_entities.items():
            f.write(f"{entity}: {entity_type}\n")
        f.write("\nExtracted Values:\n")
        for value_type, value in extracted_values.items():
            f.write(f"{value_type}: {value}\n")
    
    return {
        'text': extracted_text,
        'entities': extracted_entities,
        'values': extracted_values,
        'output_file': output_filename
    }

# Example usage:
image_path = 'image_85.jpg'
result = process_image_and_extract_entities(image_path)
print("Output saved to:", result['output_file'])
