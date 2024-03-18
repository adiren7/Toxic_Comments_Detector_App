# Toxic Comments Detection App


Welcome to the Toxic Comments Detection App! This project utilizes web scraping techniques to gather data from the YouTube platform, employs cutting-edge machine learning models based on BERT and LSTM models, and finally deploys it into an accessible application using Gradio.

## Overview

In this project, we aim to address the issue of toxicity in online discussions, particularly in comments sections. The proliferation of toxic comments can have detrimental effects on online communities, leading to hostility, harassment, and the stifling of healthy discourse. Our solution focuses on automatically detecting and filtering out toxic comments, thus fostering a more positive and inclusive online environment.

## How it Works

### Web Scraping

The initial step involves scraping data from YouTube comments sections. This process is detailed in the provided notebook [`data_collection_preprocessing.ipynb`](https://github.com/adiren7/Toxic_Comments_Detector_App/blob/main/data_collection_preprocessing.ipynb). By extracting comments from various YouTube videos, we create a diverse dataset for training our toxicity detection models.

### Toxic Comment Detection

Next, we delve into the heart of the project: toxic comment detection. We've developed sophisticated machine learning models utilizing state-of-the-art techniques such as BERT and LSTM. The notebook [`DarijaBert_toxic_comments.ipynb`](https://github.com/adiren7/Toxic_Comments_Detector_App/blob/main/DarijaBert_toxic_comments.ipynb) elaborates on the model training process, evaluation metrics, and performance analysis.

### Deployment with Gradio

To make our solution accessible and user-friendly, we've deployed the toxicity detection model into an interactive web application using Gradio. Users can easily input comments and receive real-time predictions on their toxicity levels. No prior knowledge of machine learning is required, making it accessible to a wide range of users.

## Getting Started

If you're eager to try out our Toxic Comments Detection App, follow these simple steps:

1. **Install Requirements**: Start by installing the necessary dependencies. You can achieve this by running:
    ```
    pip install -r requirements.txt
    ```

2. **Train the Model**: If you intend to run the app locally, you need to train the model on your data. Refer to the dataset provided in [`data/`](https://github.com/adiren7/Toxic_Comments_Detector_App/tree/main/data) and train the model accordingly.

3. **Set YouTube API Key**: In the [`app.py`](https://github.com/adiren7/Toxic_Comments_Detector_App/blob/main/app.py) file, set your YouTube API key to enable data scraping. This ensures that the app can retrieve comments from YouTube videos.

4. **Save Model Weights Locally**: Before running the app, ensure that you've saved the model weights locally. These weights will be utilized by the app for prediction purposes.

5. **Run the App**: Finally, execute the following command to run the Toxic Comments Detection App:
    ```
    python app.py
    ```

## Contributing

We welcome contributions from the community to further enhance and improve our Toxic Comments Detection App. Whether it's refining the models, adding new features to the app, or optimizing the scraping process, every contribution is valuable in our mission to create a safer online space.

---

*Your feedback and suggestions are highly appreciated. Together, let's combat toxicity and foster a more positive online environment!*
