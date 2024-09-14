import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import textwrap

# Initialize the OCR model
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

def process_image_from_url(image_url):
    try:
        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()  # Check for HTTP errors
        
        # Open the image using PIL
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Convert PIL image to DocumentFile
        docs = DocumentFile.from_images(image)
        
        # Perform OCR on the document
        result = model(docs)
        
        # Export OCR result to JSON format
        json_output = result.export()
        
        # Extract text values from JSON
        text_values = extract_text(json_output)
        
        # Join the text values into a single string
        formatted_text = ' '.join(text_values)
        
        # Wrap the text at the specified maximum width
        wrapped_text = textwrap.fill(formatted_text, width=80)
        
        return wrapped_text
    
    except Exception as e:
        print(f"Error processing image {image_url}: {e}")
        return ""

def extract_text(data):
    text_values = []
    if isinstance(data, dict):
        for key, value in data.items():
            text_values.extend(extract_text(value))
    elif isinstance(data, list):
        for item in data:
            text_values.extend(extract_text(item))
    elif isinstance(data, str):
        text_values.append(data)
    return text_values

def process_images_from_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    image_urls = df['images_url'].head(30).tolist()  # Process first 30 images for demonstration
    
    results = []
    
    for url in image_urls:
        print(f"Processing: {url}")
        ocr_result = process_image_from_url(url)
        results.append({'image_url': url, 'ocr_text': ocr_result})
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

# Example usage
input_csv = 'test.csv'
output_csv = 'ocr_results.csv'
process_images_from_csv(input_csv, output_csv)
print(f"Results saved to {output_csv}")
