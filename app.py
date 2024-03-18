
def predict_from_ytb(youtube_url):
    import pandas as pd
    import torch
    from transformers import AutoTokenizer

    from model import create_model 
    from web_scraping import get_comments_from_url  , youtube_data_api
    from preprocessing import preprocess_arabic_text
    from predict import predictions_dataframe


    #import Model
    model = create_model()
    model.load_state_dict(torch.load("model.pth",map_location=torch.device('cpu')))

    # import tokenizer
    checkpoint= "UBC-NLP/MARBERT"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Data scraping and preprocessing
    DEVELOPER_KEY = "AIzaSyAjxeM_uYL3XtSnr5EZjXMeuVod__CV3fo"

    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"

    youtube = youtube_data_api(YOUTUBE_API_SERVICE_NAME , YOUTUBE_API_VERSION  , DEVELOPER_KEY)

    #get comments from the video URL
    video_comments_data = get_comments_from_url(youtube , url = youtube_url, max_comments=100)

    #df from the collected comments data
    video_comments_df = pd.DataFrame(video_comments_data, columns=["text"])

    video_comments_df['Text_pro'] = video_comments_df['text'].apply(preprocess_arabic_text)

    data = pd.DataFrame({"text" : video_comments_df['Text_pro']})


    # predict

    results = predictions_dataframe(data, tokenizer, model)

    return results

import gradio as gr
import pandas as pd

# Create the Gradio interface
gr.Interface(fn=predict_from_ytb, 
             inputs=gr.inputs.Textbox(lines=5, label="Enter youtube link"), 
             outputs= gr.outputs.Dataframe(label="results"), 
             title="Toxic comments detection").launch()
