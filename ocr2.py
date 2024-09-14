import pandas as pd
import aiohttp
import asyncio
import os
from io import BytesIO
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import textwrap
import tensorflow as tf
# Initialize the OCR model
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

# Create download directory if it does not exist
download_folder = 'downloaded_images'
os.makedirs(download_folder, exist_ok=True)

# Function to download an image from a URL
async def fetch_image(session, image_url, file_path):
    try:
        async with session.get(image_url) as response:
            response.raise_for_status()  # Check for HTTP errors
            image_data = await response.read()
            with open(file_path, 'wb') as f:
                f.write(image_data)
    except Exception as e:
        print(f"Error downloading image {image_url}: {e}")

# Function to process an image file and perform OCR
def process_image(image_path):
    try:
        # Read the image file
        docs = DocumentFile.from_images(image_path)
        
        # Perform OCR on the document
        result = model(docs)
        
        # Export OCR result to JSON format
        json_output = result.export()
        
        # Function to extract text values from a JSON object
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

        # Extract text values
        text_values = extract_text(json_output)

        # Join the text values into a single string with spaces as separators
        formatted_text = ' '.join(text_values)

        # Wrap the text at the specified maximum width
        wrapped_text = textwrap.fill(formatted_text, width=80)
        
        return wrapped_text

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return ""

# Asynchronous function to download and process images
async def process_images_from_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    image_urls = df['images_url'].head(5000).tolist()  # Process first 30 images for demonstration
    
    async with aiohttp.ClientSession() as session:
        download_tasks = []
        for i, url in enumerate(image_urls):
            file_path = os.path.join(download_folder, f'image_{i}.jpg')
            download_tasks.append(fetch_image(session, url, file_path))
        
        await asyncio.gather(*download_tasks)

    # Process images
    results = []
    for i in range(len(image_urls)):
        image_path = os.path.join(download_folder, f'image_{i}.jpg')
        ocr_result = process_image(image_path)
        results.append({'image_url': image_urls[i], 'ocr_text': ocr_result})
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

# Example usage
physical_devices = tf.config.list_physical_devices('GPU')

if physical_devices:
    print("GPUs detected:", physical_devices)
else:
    print("No GPUs detected.")
input_csv = 'test.csv'
output_csv = 'ocr_results.csv'
asyncio.run(process_images_from_csv(input_csv, output_csv))
print(f"Results saved to {output_csv}")
