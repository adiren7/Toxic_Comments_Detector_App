
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoModel, AutoConfig, AutoTokenizer


#predict labels for texts
def predict_labels(texts, tokenizer, model, max_seq_length=256, batch_size=32):
    # Tokenize texts
    inputs = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
        return_tensors='pt'
    )
    
    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(inputs["input_ids"], inputs["attention_mask"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    # Predict labels
    model.eval()
    predicted_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to("cpu")
            attention_mask = batch[1].to("cpu")
            outputs = model(input_ids, attention_mask=attention_mask)
            predicted_class = torch.argmax(outputs, dim=1)
            predicted_labels.extend(predicted_class.cpu().numpy())
    
    return predicted_labels

#add predicted labels to dataframe
def predictions_dataframe(df, tokenizer, model,  id2label = {1: 'OF', 0: 'NOF'} ,max_seq_length=256, batch_size=32):
    texts = df['text'].tolist()
    predicted_labels = predict_labels(texts, tokenizer, model, max_seq_length, batch_size)
    predicted_labels = [id2label[label] for label in predicted_labels]
    df['predicted_label'] = predicted_labels
    return df[df['predicted_label'] == "OF"]
