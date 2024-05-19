import os

# Retrieve and print the Huggingface token from environment variable
hf_token = os.getenv('HUGGINGFACE_TOKEN')
print(f"Huggingface Token: {hf_token}")
