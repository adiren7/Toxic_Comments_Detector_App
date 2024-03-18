

from langdetect import detect
from googleapiclient.errors import HttpError
import pandas as pd
import re
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build


# Initialize the YouTube Data API
def youtube_data_api(YOUTUBE_API_SERVICE_NAME , YOUTUBE_API_VERSION  , DEVELOPER_KEY):
  youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
  return youtube

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

def get_comments(youtube , video_id, max_comments=10):

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

def get_comments_from_url(youtube , url, max_comments=100):

    video_id = extract_video_id(url)
    if video_id:
        return get_comments(youtube , video_id, max_comments)
    else:
        print("Invalid YouTube URL")
        return []
