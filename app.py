import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import os

# ==============================================================================
#  STEP 1: RE-DEFINE THE MODEL ARCHITECTURE
# ==============================================================================

class NewsClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_hidden_state = bert_output.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token_hidden_state)
        return logits

# ==============================================================================
#  STEP 2: LOAD THE MODEL AND TOKENIZER (with Caching)
# ==============================================================================

# --- Configuration ---
NUM_CLASSES = 4
# THIS IS THE CRUCIAL FIX: Use a relative path, not an absolute one.
MODEL_PATH = "models/saved_models/distilbert_ag_news_classifier.pth"
TOKENIZER_NAME = "distilbert-base-uncased"
ID_TO_LABEL = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

@st.cache_resource
def load_model_and_tokenizer():
    device = torch.device("cpu") # Run on CPU for inference
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    model = NewsClassifier(num_classes=NUM_CLASSES)
    
    # We must check the existence of the file *before* trying to load it.
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at '{MODEL_PATH}'. Make sure it was correctly pushed to your GitHub repository with Git LFS.")
        return None, None

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# ==============================================================================
#  STEP 3: CREATE THE USER INTERFACE
# ==============================================================================

st.title("📰 AG News Category Classifier")
st.markdown("Enter a news headline below to classify it into one of four categories: World, Sports, Business, or Sci/Tech.")

user_input = st.text_area("News Headline:", "The US economy shows signs of recovery as the stock market rallies.")

if st.button("Classify"):
    if model is not None and tokenizer is not None and user_input:
        # --- Preprocess ---
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        
        # --- Predict ---
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        
        # --- Post-process ---
        predicted_id = torch.argmax(outputs, dim=1).item()
        predicted_label = ID_TO_LABEL[predicted_id]
        
        st.success(f"Predicted Category: **{predicted_label}**")
    elif not user_input:
        st.warning("Please enter a headline.")

