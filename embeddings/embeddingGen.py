from transformers import AutoTokenizer, AutoModel
import torch
import os

try:
    import streamlit as st
    token = st.secrets["HUGGINGFACE_HUB_TOKEN"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    
if not token:
    raise ValueError("HuggingFace token is missing or not loaded properly!")
torch.classes.__path__ = []  # to solve the error that happen between torch and streamlit


class EmbeddingGenerator:
    def __init__(self):
        if not token:
            raise ValueError("HuggingFace token is missing or not loaded properly!")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "nomic-ai/nomic-embed-text-v1", trust_remote_code=True,
                use_auth_token=token
            )
            self.model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1", trust_remote_code=True,
                                                   use_auth_token=token)
            print("Model and tokenizer loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.tokenizer = None
            self.model = None


    def generate(self, text, max_length=512):
        if not self.tokenizer or not self.model:
            raise ValueError("Tokenizer or model not loaded properly.")

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",  # Return PyTorch tensors
            max_length=max_length,
        )

        # Generate embeddings
        with torch.no_grad():
            output = self.model(**inputs)

        # Extract the last hidden state from the output dictionary
        last_hidden_state = output.last_hidden_state

        print(f"Shape of last_hidden_state: {last_hidden_state.shape}")

        # Mean pooling to get a single embedding vector
        embeddings = last_hidden_state.mean(dim=1).squeeze().detach().numpy()

        # Debug: Check the shape of embeddings
        print(f"Shape of embeddings: {embeddings.shape}")

        return embeddings
