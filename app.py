import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts import PromptTemplate
from transformers import BitsAndBytesConfig
import torch
import os
from huggingface_hub import HfFolder
# Load data
reader = SimpleDirectoryReader(input_dir="documents")
documents = reader.load_data()

# Setup the LLM
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Authentication
token = os.getenv('HUGGINGFACE_TOKEN')
if token is not None:
    HfFolder.save_token(token)
else:
    st.error("Huggingface token not found. Please set the HUGGINGFACE_TOKEN environment variable.")
    st.stop()
llm = HuggingFaceLLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
    query_wrapper_prompt=PromptTemplate("<s>[INST] As a knowledgeable tour guide, respond to the tourist's query: {query_str}. Include local stories, best places to visit, recommended hotels, and budget tips.  [/INST] </s>\\n"),
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"quantization_config": quantization_config},
    device_map="auto",
)
# llm.to('cuda')
    
Settings.llm = llm
Settings.embed_model = "local:BAAI/bge-small-en-v1.5"


# Index data
vector_index = VectorStoreIndex.from_documents(documents)

# Streamlit app interface
st.title('MoroccoGuideBot - Your Tourist Guide')
query = st.text_input("Hi, I'm your guide. What would you like to know about Morocco?")

if st.button('Ask'):
    query_engine = vector_index.as_query_engine(response_mode="compact")
    response = query_engine.query(query)
    if response:
        st.write(response)
    else:
        st.write("I'm sorry, I couldn't find information on that topic. Please try asking something else about Morocco.")
