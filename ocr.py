from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import textwrap

# Read the image file
docs = DocumentFile.from_images("image_80.jpg")

# Initialize the OCR model
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

# Perform OCR on the document
result = model(docs)

# Export OCR result to JSON format
json_output = result.export()

# Define the path to the output text file
output_file_path = 'output.txt'

max_line_width = 80

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
        # Add the recognized text as is without replacing characters
        text_values.append(data)
    return text_values

# Extract text values
text_values = extract_text(json_output)

# Join the text values into a single string with spaces as separators
formatted_text = ' '.join(text_values)

# Wrap the text at the specified maximum width
wrapped_text = textwrap.fill(formatted_text, width=max_line_width)

# Write the wrapped text to a text file
with open(output_file_path, 'w') as text_file:
    text_file.write(wrapped_text)
