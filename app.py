import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# ==============================================================================
#  STEP 1: DEFINE THE MODEL ARCHITECTURE & CONFIGURATION
#  This must be the exact same class as the one used for training.
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

# --- Configuration ---
NUM_CLASSES = 4
MODEL_PATH = "/Users/tushar04master/Documents/news-classifier/models/saved_models/distilbert_ag_news_classifier.pth"
TOKENIZER_NAME = "distilbert-base-uncased"
ID_TO_LABEL = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ==============================================================================
#  STEP 2: LOAD THE TRAINED ASSETS (MODEL & TOKENIZER)
#  We use @st.cache_resource to load these only once, making the app fast.
# ==============================================================================

@st.cache_resource
def load_model_and_tokenizer():
    """Loads the saved model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    # Instantiate the model architecture
    model = NewsClassifier(num_classes=NUM_CLASSES)
    
    # Load the trained weights (the "state_dict")
    # Use map_location to ensure the model loads correctly onto the CPU if no GPU is available
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # Move the model to the correct device
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    return model, tokenizer

# Load the assets
model, tokenizer = load_model_and_tokenizer()

# ==============================================================================
#  STEP 3: CREATE THE USER INTERFACE (UI)
# ==============================================================================

st.title("📰 AG News Category Classifier")
st.markdown("Enter a news headline below to classify it into one of four categories: World, Sports, Business, or Sci/Tech.")

# Create a text area for user input
user_input = st.text_area("News Headline:", "Apple's stock hits a new high after the WWDC event.", height=100)

# Create a button to trigger the classification
if st.button("Classify"):
    if user_input:
        # ==============================================================================
        #  STEP 4: MAKE A PREDICTION WHEN THE BUTTON IS CLICKED
        # ==============================================================================
        
        # Preprocess the text
        inputs = tokenizer(
            user_input, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=256 # Use same max_length as training
        )
        
        # Move inputs to the correct device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Make a prediction
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Post-process the output
        predicted_id = torch.argmax(outputs, dim=1).item()
        predicted_label = ID_TO_LABEL[predicted_id]
        
        # Display the result
        st.success(f"Predicted Category: **{predicted_label}**")
    else:
        st.warning("Please enter a news headline.")