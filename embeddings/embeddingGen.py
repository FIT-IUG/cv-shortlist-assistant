from transformers import AutoTokenizer, AutoModel
import torch
import os

torch.classes.__path__ = []  # to solve the error that happen between torch and streamlit


class EmbeddingGenerator:
    def __init__(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "nomic-ai/nomic-embed-text-v1", trust_remote_code=True,
                use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN")
            )
            self.model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1", trust_remote_code=True,
                                                   use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
            print("Model and tokenizer loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")

    def generate(self, text, max_length=512):
        # Tokenize the input text
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
