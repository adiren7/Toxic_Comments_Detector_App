
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoModel, AutoConfig, AutoTokenizer
from tqdm import tqdm
import torch.nn.functional as F

def create_model(checkpoint = "UBC-NLP/MARBERT" , NUM_LABELS = 2 ):
  class MyTopicPredictionModel(nn.Module):
      def __init__(self, checkpoint, num_topics):
          super(MyTopicPredictionModel, self).__init__()

          self.num_topics = num_topics

          self.bert = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint, output_hidden_states=True))
          self.dropout = nn.Dropout(0.1)
          self.lstm1 = nn.LSTM(self.bert.config.hidden_size, 512, num_layers=4, dropout=0.1, bidirectional=False, batch_first=True)
          self.classifier = nn.Linear(512, num_topics)

      def forward(self, input_ids=None, attention_mask=None, labels=None):

          outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
          last_hidden_state = outputs.last_hidden_state
          sequence_outputs = self.dropout(last_hidden_state)
          lstm_out1, _ = self.lstm1(sequence_outputs)
          logits = F.softmax(self.classifier(lstm_out1[:, -1, :]))

          return logits


  model = MyTopicPredictionModel(checkpoint=checkpoint, num_topics=NUM_LABELS)

  # Freeze weights
  for param in model.bert.parameters():
    param.requires_grad = False

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  return model

def create_tokenizer(checkpoint= "UBC-NLP/MARBERT"):
  tokenizer = AutoTokenizer.from_pretrained(checkpoint)
  return tokenizer
