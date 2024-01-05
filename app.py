import streamlit as st
import pandas as pd
import re
import nltk
from urllib.parse import urlparse, parse_qs
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pyarabic.araby import strip_tashkeel, strip_tatweel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from datasets import load_dataset,Dataset,DatasetDict
from datetime import datetime
from langdetect import detect
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pyarabic.araby import strip_tashkeel, strip_tatweel

DEVELOPER_KEY = "Set your YouTube Data API key here "

YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Initialize the YouTube Data API client
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
# Streamlit setup
st.title("YouTube كاشف خطاب الكراهية على منصة ")
url = st.text_input("أدخل رابط الفيديو ")
if st.button("اكشف عن خطاب الكراهية"):
    # Create a function to scrape YouTube comments from the URL
    def extract_video_id(url):
        query = urlparse(url)
        if query.hostname == 'youtu.be':
            return query.path[1:]
        if query.hostname in ('www.youtube.com', 'youtube.com'):
            if query.path == '/watch':
                p = parse_qs(query.query)
                return p['v'][0]
            if query.path[:7] == '/embed/':
                return query.path.split('/')[2]
            if query.path[:3] == '/v/':
                return query.path.split('/')[2]
        return None

    def get_comments(video_id, max_comments=10):
        comments_data = []
        try:
            # Get comments for the specified video
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=max_comments
            ).execute()

            # Extract and store the comments, their upload dates, and usernames
            for comment in response.get("items", []):
                comment_text = comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comment_date = comment["snippet"]["topLevelComment"]["snippet"]["publishedAt"]
                username = comment["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]

                # Check if the comment is in Arabic
                clean_comment = re.sub(r'[^\w\s]', '', comment_text)
                if len(clean_comment) >= 3:
                    if detect(clean_comment) == "ar":
                        formatted_date = datetime.strptime(comment_date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
                        comments_data.append({"date": formatted_date, "text": comment_text, "username": username})

        except HttpError as e:
            print("An HTTP error occurred:", e)

        return comments_data

    def get_comments_from_url(url, max_comments=100):
        video_id = extract_video_id(url)
        if video_id:
            return get_comments(video_id, max_comments)
        else:
            print("Invalid YouTube URL")
            return []

    # Scrap comments

    # Initialize the YouTube Data API client
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

    def extract_video_id(url):
        query = urlparse(url)
        if query.hostname == 'youtu.be':
            return query.path[1:]
        if query.hostname in ('www.youtube.com', 'youtube.com'):
            if query.path == '/watch':
                p = parse_qs(query.query)
                return p['v'][0]
            if query.path[:7] == '/embed/':
                return query.path.split('/')[2]
            if query.path[:3] == '/v/':
                return query.path.split('/')[2]
        return None

    def get_comments(video_id, max_comments=10):
        comments_data = []
        try:
            # Get comments for the specified video
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=max_comments
            ).execute()

            # Extract and store the comments, their upload dates, and usernames
            for comment in response.get("items", []):
                comment_text = comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comment_date = comment["snippet"]["topLevelComment"]["snippet"]["publishedAt"]
                username = comment["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]

                # Check if the comment is in Arabic
                clean_comment = re.sub(r'[^\w\s]', '', comment_text)
                if len(clean_comment) >= 3:
                    if detect(clean_comment) == "ar":
                        formatted_date = datetime.strptime(comment_date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
                        comments_data.append({"date": formatted_date, "text": comment_text, "username": username})

        except HttpError as e:
            print("An HTTP error occurred:", e)

        return comments_data

    def get_comments_from_url(url, max_comments=100):
        video_id = extract_video_id(url)
        if video_id:
            return get_comments(video_id, max_comments)
        else:
            print("Invalid YouTube URL")
            return []

    # Call the function to get comments from the video URL
    video_comments_data = get_comments_from_url(url, max_comments=100)

    # Create a DataFrame from the collected comments data
    video_comments_df = pd.DataFrame(video_comments_data, columns=["date", "text", "username"])

    # Add a "source" column and save to Excel
    source_value = "YouTube"
    video_comments_df["source"] = source_value
    data = video_comments_df
    nltk.download('punkt')
    nltk.download('stopwords')
    def preprocess_arabic_text(text):
        text = strip_tashkeel(text)
        text = strip_tatweel(text)

        additional_symbols = r'[،؟]'  

        pattern = r'[' + re.escape(additional_symbols) + ']'
        text = re.sub(pattern, '', text)
        
        # Remove non-Arabic characters and numbers
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        
        words = word_tokenize(text)
        
        stop_words = set(stopwords.words('arabic'))
        words = [word for word in words if word not in stop_words]
        
        preprocessed_text = ' '.join(words)
        
        return preprocessed_text

        
    data['Text_pro'] = data['text'].apply(preprocess_arabic_text)

    dataset_hf = data[["date","Text_pro","username"]]

    new_column_names = {'Text_pro': 'text'}
    dataset_hf = dataset_hf.rename(columns=new_column_names)

    class MyDataset(Dataset):
        def __init__(self, dataframe, tokenizer, max_seq_length=256):
            self.data = dataframe
            self.tokenizer = tokenizer
            self.max_seq_length = max_seq_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            text = row["text"]

            username = row["username"]
            date = row["date"]

            # Tokenize the text and truncate/pad to the desired sequence length
            inputs = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors='pt'
            )

            input_ids = inputs["input_ids"].squeeze()
            attention_mask = inputs["attention_mask"].squeeze()

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,  
                "username": username,
                "date": date
            }

    
    checkpoint = "UBC-NLP/MARBERT"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

  
    max_seq_length = 256 



  
    test_dataset = MyDataset(dataset_hf, tokenizer, max_seq_length)
    test_batch_size = 32 
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    
    # Load pre-trained model
    class MyTopicPredictionModel(nn.Module):
        def __init__(self, checkpoint, num_topics):
            super(MyTopicPredictionModel, self).__init__()
            self.num_topics = num_topics

            self.model = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint, output_hidden_states=True))
            self.dropout = nn.Dropout(0.1)
            self.lstm = nn.LSTM(768, 256, num_layers=1, dropout=0.1, bidirectional=False, batch_first=True)
            self.classifier = nn.Linear(256, num_topics)  # Number of topics as output labels

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            last_hidden_state = outputs.last_hidden_state
            sequence_outputs = self.dropout(last_hidden_state)
            lstm_out, _ = self.lstm(sequence_outputs)
            logits = F.softmax(self.classifier(lstm_out[:, -1, :]))

            return logits

    # Load the model
    model = MyTopicPredictionModel('UBC-NLP/MARBERT', 2)  # Replace with the actual model class and architecture
    model.load_state_dict(torch.load(r'C:\Users\HP\OneDrive\Bureau\hackathon isic\model.pth', map_location='cpu'))
    model.eval()


    # 5. Create a function to make predictions with the loaded model
    test_predictions = []  # To store predicted labels
    test_true_labels = []  # To store true labels
    test_texts = []  # To store the initial message text
    test_usernames = []  # To store usernames
    test_dates = []  # To store dates

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            usernames = batch["username"]
            dates = batch["date"]

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_labels = torch.argmax(logits, dim=1)

            # Convert the tensor values to Python lists
            predicted_labels = predicted_labels.cpu().numpy()
            text_batch = batch["input_ids"]

            test_predictions.extend(predicted_labels.tolist())
            test_texts.extend(text_batch)
            test_usernames.extend(usernames)
            test_dates.extend(dates)

    # You can convert the label IDs back to their original labels if needed.
    # For example, if label 0 corresponds to 'NV' and label 1 corresponds to 'V':
    label_mapping = {0: 'NV', 1: 'V'}
    test_predictions = [label_mapping[label] for label in test_predictions]
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('UBC-NLP/MARBERT')

    # Convert token IDs back to human-readable text
    decoded_texts = [tokenizer.decode(text, skip_special_tokens=True) for text in test_texts]
    # Make predictions for each comment


    # Create a dictionary with the lists
    data = {
        'text': decoded_texts,
        'predicted_Label': test_predictions,
        'username': test_usernames,
        'date': test_dates
    }

    # Create a DataFrame
    results = pd.DataFrame(data)

    violence = results[results['predicted_Label']=="V"]
    # 6. Display predictions
    st.dataframe(results[['date', 'text','username']])
