import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
if os.environ.get("STREAMLIT_SERVER_FILE_WATCHER_TYPE") != "none":
    raise RuntimeError("Failed to disable Streamlit file watcher")
print(os.environ.get("STREAMLIT_SERVER_FILE_WATCHER_TYPE"))


import time, csv, random, logging, requests, json, re, os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Set, Optional
import pandas as pd
import streamlit as st
import pyaudio
import wave
import folium
from streamlit_folium import st_folium
from configparser import ConfigParser
from pydub import AudioSegment
import speech_recognition as sr
from pynput import keyboard
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline, AutoModelForSequenceClassification, AutoTokenizer
import spacy
import dateparser
import unicodedata
import html
from emoji import demojize
from contractions import fix
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# Import RAG system
from rag_module import RAGSystem


# Constants
config = ConfigParser()
config.read('config.ini')
password = config['X']["password"]
mail = config['X']["email"]
usernameee = config['X']["username"]
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 600
WAVE_OUTPUT_PATH = "audio/output.wav"
frames = []
stop_recording = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True
)
logger = logging.getLogger(__name__)
logger.info("Imports and logging initialized")

# Cache the DisasterAnalyzer initialization
@st.cache_resource
def init_disaster_analyzer():
    logger.info("Initializing DisasterAnalyzer")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    try:
        model = AutoModelForSequenceClassification.from_pretrained("sladereaperr/BERTdisaster", num_labels=2).to(device)
        tokenizer = AutoTokenizer.from_pretrained("sladereaperr/BERTdisaster")
        logger.info("BERT model and tokenizer loaded")
    except Exception as e:
        logger.error(f"Failed to load BERT model: {e}")
        raise

    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "t5_model": t5_model,
        "t5_tokenizer": t5_tokenizer,
        "sentiment_pipeline": sentiment_pipeline,
        "device": device
    }

# Cache the TextPreprocessor initialization
@st.cache_resource
def init_text_preprocessor():
    logger.info("Initializing TextPreprocessor")
    nlp = spacy.load("en_core_web_sm")
    return {"nlp": nlp}

# Cache the DisasterReportOrchestrator initialization
@st.cache_resource
def init_orchestrator(api_key: str):
    logger.info("Initializing DisasterReportOrchestrator")
    preprocessor = TextPreprocessor()
    analyzer = DisasterAnalyzer()
    scraper = DataScraper(api_key)
    pdf_generator = PDFGenerator()
    summarizer = ArticleSummarizer(analyzer)
    return DisasterReportOrchestrator(api_key, preprocessor, analyzer, scraper, pdf_generator, summarizer)

# Cache the RAG system initialization
@st.cache_resource
def init_rag_system():
    logger.info("Initializing RAG system")
    return RAGSystem()

# TextPreprocessor Class
class TextPreprocessor:
    def __init__(self):
        self.nlp = init_text_preprocessor()["nlp"]

    def preprocess(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = unicodedata.normalize("NFKD", str(text))
        text = html.unescape(text.lower())
        text = fix(text)
        text = demojize(text)
        text = re.sub(r'http\S+|www\S+', "", text)
        text = re.sub(r"[@#]\w+", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def extract_time(self, text: str) -> Dict[str, List[str]]:
        text = self.preprocess(text)
        doc = self.nlp(text)
        parsed_dates = [
            dateparser.parse(ent.text).strftime("%Y-%m-%d %H:%M:%S")
            for ent in doc.ents if ent.label_ in ["DATE", "TIME"] and dateparser.parse(ent.text)
        ]
        return {"structured_dates": parsed_dates or ["Not specified"]}

    def extract_locations(self, text: str) -> Set[str]:
        text = self.preprocess(text)
        doc = self.nlp(text)
        return {ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]}

# DisasterAnalyzer Class
class DisasterAnalyzer:
    def __init__(self):
        self.analyzer_data = init_disaster_analyzer()
        self.model = self.analyzer_data["model"]
        self.tokenizer = self.analyzer_data["tokenizer"]
        self.t5_model = self.analyzer_data["t5_model"]
        self.t5_tokenizer = self.analyzer_data["t5_tokenizer"]
        self.sentiment_pipeline = self.analyzer_data["sentiment_pipeline"]
        self.device = self.analyzer_data["device"]
        self.class_labels = ['not_disaster', 'disaster']
        self.disaster_keywords = {
            "earthquake": "Earthquake", "flood": "Flood", "hurricane": "Hurricane",
            "wildfire": "Wildfire", "tornado": "Tornado", "tsunami": "Tsunami"
        }
        logger.info("DisasterAnalyzer initialized")

    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Summarize text using T5 model"""
        if not text or len(text) < 100:  # Don't summarize very short texts
            return text

        # Truncate very long texts to avoid token limits
        if len(text) > 10000:
            text = text[:10000]

        try:
            # Prepare input for T5 (it expects "summarize: " prefix)
            input_text = "summarize: " + text
            inputs = self.t5_tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)

            # Generate summary
            with torch.no_grad():
                summary_ids = self.t5_model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    min_length=30,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )

            # Decode the summary
            summary = self.t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            # Return a truncated version if summarization fails
            return text[:500] + "..."

    def classify_urgency(self, text: str) -> str:
        if not text.strip():
            return "Low Urgency"
        sentiment = self.sentiment_pipeline(text)[0]
        label, score = sentiment["label"], sentiment["score"]
        if label == "NEGATIVE":
            return "Critical" if score > 0.9 else "High"
        return "Low" if score > 0.9 else "Medium"

    def classify_disaster(self, text: str) -> Dict[str, str]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1)[0].item()
        confidence = torch.softmax(logits, dim=1)[0, predicted_class_id].item()
        return {
            "class_label": self.class_labels[predicted_class_id],
            "confidence": f"{confidence:.4f}"
        }

    def analyze_sentiment(self, text: str) -> str:
        sentiment_input = self.t5_tokenizer("sst2: " + text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            sentiment_output = self.t5_model.generate(**sentiment_input)
        return self.t5_tokenizer.decode(sentiment_output[0], skip_special_tokens=True)

    def detect_disaster_type(self, text: str) -> str:
        text = text.lower()
        return next((v for k, v in self.disaster_keywords.items() if k in text), "Other/Unknown")

    def analyze(self, text: str, locations: Set[str], time_info: Dict[str, List[str]]) -> Dict:
        disaster_classification = self.classify_disaster(text)
        return {
            "location": list(locations),
            "time": time_info,
            "disaster_type": self.detect_disaster_type(text),
            "urgency": self.classify_urgency(text),
            "sentiment": self.analyze_sentiment(text),
            **disaster_classification
        }

# DataScraper Class
class DataScraper:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"

    def fetch_articles(self, query: str, days: int = 5, max_results: int = 15) -> List[Dict]:
        payload = {
            "query": query,
            "topic": "news",
            "search_depth": "basic",
            "max_results": max_results,
            "days": days,
            "include_answer": True,
            "include_raw_content": False,
            "include_images": False,
            "include_image_descriptions": False,
            "include_domains": [],
            "exclude_domains": []
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            articles = response.json().get('results', [])
            logger.info(f"Fetched {len(articles)} articles for query: {query}")
            return articles
        except requests.RequestException as e:
            logger.error(f"Failed to fetch articles: {e}")
            return []


# ArticleSummarizer Class
class ArticleSummarizer:
    def __init__(self, analyzer=None):
        self.analyzer = analyzer or DisasterAnalyzer()

    def summarize_article(self, content: str, title: str) -> Dict[str, str]:
        """Summarize article content using the T5 model"""
        try:
            # Combine title and content for better context
            full_text = f"{title}\n\n{content}"

            # Generate summary
            summary = self.analyzer.summarize_text(full_text)

            return {
                "title": title,
                "summary": summary,
                "original_length": len(content)
            }
        except Exception as e:
            logger.error(f"Failed to summarize article {title}: {e}")
            return {
                "title": title,
                "summary": f"Error summarizing content: {str(e)}",
                "original_length": len(content)
            }

# PDFGenerator Class
class PDFGenerator:
    def save_to_pdf(self, content: str, title: str, index: int, output_dir: str = "pdfs") -> Optional[tuple[str, bytes]]:
        try:
            os.makedirs(output_dir, exist_ok=True)
            pdf_path = os.path.join(output_dir, f"{title[:45].replace(' ', '_')}_{index}.pdf")
            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            elements = [
                Paragraph(title, ParagraphStyle('Title', fontSize=16, alignment=TA_CENTER)),
                Spacer(1, 20)
            ]
            elements += [Paragraph(html.escape(p), getSampleStyleSheet()["Normal"]) for p in content.split('\n\n')]
            doc.build(elements)
            logger.info(f"Saved PDF: {pdf_path}")
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            return pdf_path, pdf_bytes
        except Exception as e:
            logger.error(f"Failed to save PDF for {title}: {e}")
            return None

# DisasterReportOrchestrator Class
class DisasterReportOrchestrator:
    def __init__(self, api_key: str, preprocessor=None, analyzer=None, scraper=None, pdf_generator=None, summarizer = None):
        self.preprocessor = preprocessor or TextPreprocessor()
        self.analyzer = analyzer or DisasterAnalyzer()
        self.scraper = scraper or DataScraper(api_key)
        self.pdf_generator = pdf_generator or PDFGenerator()
        self.summarizer = summarizer or ArticleSummarizer(self.analyzer)
        self.api_key = api_key
        logger.info("DisasterReportOrchestrator initialized")

    def process(self, prompt: str, max_articles: int = None) -> tuple[Dict, List[tuple[str, bytes]]]:
        logger.info(f"Processing prompt: {prompt}")
        locations = self.preprocessor.extract_locations(prompt)
        time_info = self.preprocessor.extract_time(prompt)
        logger.info(f"Locations: {locations}, Time: {time_info}")
        analysis = self.analyzer.analyze(prompt, locations, time_info)
        logger.info(f"Analysis result: {json.dumps(analysis, indent=2)}")
        pdf_files = []
        article_summaries = []
        if analysis["class_label"] == "disaster":
            query_parts = []
            query_parts.extend(analysis['location'])
            if analysis['disaster_type'] != "Other/Unknown":
                query_parts.append(analysis['disaster_type'])
            for date_str in analysis['time']['structured_dates']:
                if date_str.lower() != "not specified":
                    try:
                        date_obj = datetime.fromisoformat(date_str)
                        query_parts.append(date_obj.date().isoformat())
                    except ValueError:
                        pass
            query = " ".join(query_parts)
            print("This is the query: " + query)
            articles = self.scraper.fetch_articles(query)

            if not articles:
                logger.warning("No articles fetched, check API key or network.")
                return analysis, pdf_files

            max_articles = len(articles) if max_articles is None else min(max_articles, len(articles))
            if st.session_state.mode == "chat":
                for i, article in enumerate(articles[:max_articles]):
                    content = article.get('content', 'No content available')
                    title = article.get('title', f"Article_{i+1}")
                    result = self.pdf_generator.save_to_pdf(content, title, i)
                    if result:
                        pdf_files.append(result)
                    else:
                        logger.warning(f"PDF generation failed for {title}")
            else:
                for i, article in enumerate(articles[:max_articles]):
                    content = article.get('content', 'No content available')
                    title = article.get('title', f"Article_{i+1}")
                    url = article.get('url', '#')

                    # Summarize the article instead of creating a PDF
                    summary_result = self.summarizer.summarize_article(content, title)
                    summary_result['url'] = url  # Add the URL to the summary
                    article_summaries.append(summary_result)

        else:
            logger.info("No disaster detected, skipping article fetching and PDF generation.")

        if st.session_state.mode == "chat":
            return analysis, pdf_files
        else:
            return analysis, article_summaries


    def process_disaster_event(self, event_query: str, max_articles: int = 5) -> List[Dict]:
        """Process a specific disaster event query and return article summaries"""
        logger.info(f"Processing disaster event query: {event_query}")
        article_summaries = []

        # Fetch articles related to the disaster event
        articles = self.scraper.fetch_articles(event_query, max_results=max_articles)

        if not articles:
            logger.warning(f"No articles fetched for query: {event_query}")
            return article_summaries

        # Summarize each article
        for i, article in enumerate(articles):
            content = article.get('content', 'No content available')
            title = article.get('title', f"Article_{i+1}")
            url = article.get('url', '#')

            summary_result = self.summarizer.summarize_article(content, title)
            summary_result['url'] = url
            article_summaries.append(summary_result)

        return article_summaries

# AudioInput Class
class AudioInput:
    def record(self):
        global frames
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        st.info("🎙️ Recording... Press 'q' to stop recording from your keyboard terminal.")

        while not stop_recording:
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_PATH, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        print(f"✅ Saved recording to {WAVE_OUTPUT_PATH}")

    def on_press(self, key):
        global stop_recording
        try:
            if key.char == 'q':
                stop_recording = True
                return False
        except AttributeError:
            pass

    def transcribe_audio(self, file_path):
        try:
            st.info("⏳ Transcribing your voice...")
            sound = AudioSegment.from_file(file_path, format="wav")
            sound.export("audio/temp.wav", format="wav")

            r = sr.Recognizer()
            with sr.AudioFile("audio/temp.wav") as source:
                audio_data = r.record(source)
                user_input = r.recognize_google(audio_data)

            os.remove("audio/temp.wav")
            return user_input
        except Exception as e:
            print(f"⚠️ Error during transcription: {e}")
            return None



class TweetScraper:
    def __init__(self):
        self.analyser = DisasterAnalyzer()
        self.preprocessor = TextPreprocessor()
        # Setup browser with more realistic settings
        self.chrome_options = Options()
        # Uncomment the line below if you want to see the browser in action
        # self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--window-size=1920,1080")
        self.chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
        self.chrome_options.add_experimental_option("detach", True)


    def scrape_twitter_search(self, search_term, max_scroll_attempts=20):
        """
        Scrape Twitter search results, filtering for:
        1. Disaster-related tweets (using RoBERTa model)
        2. Tweets less than 1 month old
        Scrolls until no new tweets appear
        """
        # Initialize browser
        driver = webdriver.Chrome(options=self.chrome_options)

        # Navigate to Twitter search
        url = f"https://twitter.com/search?q={search_term}&src=typed_query"
        print(f"Opening URL: {url}")
        driver.get(url)

        # Wait longer for initial page load
        time.sleep(5)

        # Login process
        userfield = driver.find_element(By.CLASS_NAME, "r-30o5oe")
        userfield.send_keys(mail)
        time.sleep(1)
        userfield.send_keys(Keys.RETURN)

        time.sleep(5)

        usernamefield = driver.find_element(By.CSS_SELECTOR, '[data-testid="ocfEnterTextTextInput"]')
        usernamefield.send_keys(usernameee)
        time.sleep(1)
        usernamefield.send_keys(Keys.RETURN)

        time.sleep(5)

        passfield = driver.find_element(By.NAME, "password")
        passfield.send_keys(password)
        time.sleep(1)
        passfield.send_keys(Keys.RETURN)

        time.sleep(8)

        # Debug current URL
        print(f"Current URL: {driver.current_url}")

        # Calculate date threshold (1 month ago) with timezone awareness
        one_month_ago = datetime.now(timezone.utc) - timedelta(days=30)
        print(f"Only including tweets after: {one_month_ago.isoformat()}")

        tweets = []
        tweet_ids = set()  # To track unique tweets
        consecutive_no_new_tweets = 0
        scroll_count = 0

        while scroll_count < max_scroll_attempts and consecutive_no_new_tweets < 3:
            scroll_count += 1
            print(f"Scroll {scroll_count}, consecutive no new tweets: {consecutive_no_new_tweets}")

            # Wait for tweet elements to be present
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'article[data-testid="tweet"]'))
                )
            except TimeoutException:
                print("No tweets found. Page might be loading slowly or structure changed.")
                consecutive_no_new_tweets += 1
                if consecutive_no_new_tweets >= 3:
                    print("No new tweets found after 3 consecutive attempts. Stopping.")
                    break
                continue

            # Get all tweet elements with updated wait
            time.sleep(random.uniform(2, 4))  # Random wait to appear more human-like
            tweet_elements = driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')
            print(f"Found {len(tweet_elements)} tweets on current page")

            new_tweets_found = 0

            for tweet in tweet_elements:
                try:
                    # Extract tweet ID or unique identifier (from data attribute or URL)
                    try:
                        # Try to get a unique identifier for the tweet
                        tweet_link = tweet.find_element(By.CSS_SELECTOR, 'a[href*="/status/"]')
                        tweet_url = tweet_link.get_attribute('href')
                        tweet_id = tweet_url.split('/status/')[1].split('?')[0]
                    except:
                        # Fallback: use a combination of other data
                        username_elem = tweet.find_element(By.CSS_SELECTOR, '[data-testid="User-Name"]')
                        username = username_elem.text
                        timestamp = tweet.find_element(By.TAG_NAME, 'time').get_attribute('datetime')
                        tweet_id = f"{username}_{timestamp}"

                    # Skip if we've seen this tweet before
                    if tweet_id in tweet_ids:
                        continue

                    # Extract information with more robust selectors
                    username_elem = tweet.find_element(By.CSS_SELECTOR, '[data-testid="User-Name"]')
                    username = username_elem.text

                    try:
                        content_elem = tweet.find_element(By.CSS_SELECTOR, '[data-testid="tweetText"]')
                        content = content_elem.text
                    except:
                        content = "[No text content]"  # Some tweets might be images only
                        continue  # Skip tweets without text as we can't classify them

                    try:
                        timestamp_str = tweet.find_element(By.TAG_NAME, 'time').get_attribute('datetime')
                        # Parse the timestamp string with proper timezone handling
                        # Twitter timestamps are in ISO 8601 format with 'Z' indicating UTC
                        if timestamp_str.endswith('Z'):
                            # Replace Z with +00:00 for proper ISO format parsing
                            timestamp_str = timestamp_str[:-1] + '+00:00'
                        tweet_date = datetime.fromisoformat(timestamp_str)
                    except Exception as e:
                        print(f"Error parsing timestamp '{timestamp_str}': {e}")
                        continue

                    # Check if tweet is within the last month
                    if tweet_date < one_month_ago:
                        print(f"Skipping older tweet from {tweet_date.isoformat()}")
                        continue

                    # Check if tweet is disaster-related using RoBERTa model

                    result = self.analyser.classify_disaster(content)
                    is_disaster = result['class_label']
                    confidence = float(result['confidence'])

                    if is_disaster == "disaster":
                        tweet_data = {
                            'username': username,
                            'content': content,
                            'timestamp': tweet_date.isoformat(),
                            'confidence': f"{confidence:.4f}"
                        }

                        # Add the tweet and update the ID set
                        tweets.append(tweet_data)
                        tweet_ids.add(tweet_id)
                        new_tweets_found += 1
                        print(f"Added disaster tweet ({confidence:.4f}): {username} - {content[:30]}...")
                    else:
                        print(f"Skipping non-disaster tweet ({confidence:.4f}): {content[:30]}...")

                except Exception as e:
                    print(f"Error parsing tweet: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

            print(f"Added {new_tweets_found} new disaster-related tweets in this scroll")

            if new_tweets_found == 0:
                consecutive_no_new_tweets += 1
            else:
                consecutive_no_new_tweets = 0  # Reset counter if we found new tweets

            # Scroll down for more tweets
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(3, 5))  # Wait longer for scroll to complete

        print(f"Total unique disaster-related tweets collected: {len(tweets)}")
        driver.quit()
        return tweets


    def extract_disaster_descriptions(self, tweets, max_descriptions=5):
        """Extract concise disaster descriptions from tweets."""
        if not tweets:
            return []

        # Dictionary to store unique disaster descriptions
        disaster_events = {}

        for tweet in tweets:
            content = tweet.get('content', '')

            # Extract locations from the tweet
            locations = self.preprocessor.extract_locations(content)

            # Detect disaster type
            disaster_type = self.analyser.detect_disaster_type(content)

            # If we have both location and disaster type, create a description
            if locations and disaster_type != "Other/Unknown":
                for location in locations:
                    # Create a key combining location and disaster type
                    key = f"{location.title()} {disaster_type.lower()}"

                    # Store with confidence as value for later sorting
                    if key not in disaster_events or float(tweet.get('confidence', 0)) > float(disaster_events[key]):
                        disaster_events[key] = tweet.get('confidence', 0)

        # If no structured descriptions were found, try a more general approach
        if not disaster_events:
            for tweet in tweets:
                content = tweet.get('content', '')
                # Use NLP to extract noun phrases that might represent disasters
                doc = self.preprocessor.nlp(content)
                for chunk in doc.noun_chunks:
                    if any(keyword in chunk.text.lower() for keyword in self.analyser.disaster_keywords):
                        key = chunk.text.strip()
                        if key and len(key.split()) >= 2:
                            disaster_events[key] = tweet.get('confidence', 0)

        # Sort by confidence and take top max_descriptions
        sorted_events = sorted(disaster_events.items(), key=lambda x: float(x[1]), reverse=True)
        return [event[0] for event in sorted_events[:max_descriptions]]

    def process(self, selected_disaster_type:str):
        try:
            tweets = self.scrape_twitter_search(selected_disaster_type)

            csv_filename = 'disaster_tweets.csv'

            with open(csv_filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=['username', 'content', 'timestamp', 'confidence'])
                writer.writeheader()
                writer.writerows(tweets)
            print(f"Saved {len(tweets)} disaster-related tweets to {csv_filename}")

            # Extract disaster descriptions
            disaster_descriptions = self.extract_disaster_descriptions(tweets)

            # Save descriptions to a separate file for easy access
            descriptions_file = 'disaster_descriptions.json'
            with open(descriptions_file, 'w', encoding='utf-8') as f:
                json.dump(disaster_descriptions, f)

            return True, disaster_descriptions
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return False, []


# Modified handle_enter function
def handle_enter():
    current_text = st.session_state.text_prompt
    # Enter Tavily API 
    api_key = ""
    max_articles = 5

    # Get or initialize orchestrator from session state
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = init_orchestrator(api_key)

    # Get or initialize RAG system from session state
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = init_rag_system()

    # Store the analysis results in session state instead of displaying immediately
    with st.spinner("Processing your request..."):
        if st.session_state.mode == "chat":
            # Process with orchestrator to get analysis and PDFs
            analysis, pdf_files = st.session_state.orchestrator.process(current_text, max_articles)

            # Store results in session state for later display
            st.session_state.last_analysis = analysis
            st.session_state.last_pdf_files = pdf_files
            st.session_state.last_article_summaries = None

            # If PDFs were generated, add them to the RAG system and generate a response
            if pdf_files and analysis["class_label"] == "disaster":
                # Add PDFs to RAG system
                for pdf_path, pdf_bytes in pdf_files:
                    st.session_state.rag_system.add_pdf_to_vector_store(pdf_path, pdf_bytes)

                # Generate RAG response
                rag_response, sources = st.session_state.rag_system.answer_query(current_text)
                st.session_state.last_rag_response = rag_response
                st.session_state.last_rag_sources = sources
            else:
                st.session_state.last_rag_response = None
                st.session_state.last_rag_sources = None
        else:
            analysis, article_summaries = st.session_state.orchestrator.process(current_text, max_articles)
            # Store results in session state for later display
            st.session_state.last_analysis = analysis
            st.session_state.last_pdf_files = None
            st.session_state.last_article_summaries = article_summaries


    if current_text.strip():
        # Add to chat history
        st.session_state.chat_history.append(("user", current_text.strip()))

        # Add RAG response if available, otherwise use a default response
        if st.session_state.last_rag_response:
            bot_response = st.session_state.last_rag_response
        else:
            bot_response = f"I'm still learning. You said: '{current_text.strip()}'"

        st.session_state.chat_history.append(("bot", bot_response))

        # Set a flag to indicate we need to clear the input on next rerun
        st.session_state.clear_input = True

class Map:
    # Function to fetch ongoing disasters from ReliefWeb API
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def fetch_ongoing_disasters(_self=None):  # Add _self parameter to avoid hashing issues
        url = "https://api.reliefweb.int/v1/disasters"
        params = {
            "appname": "disaster-map",
            "filter[field]": "status",
            "filter[value]": "ongoing",
            "limit": 100,  # Adjust as needed
            "fields[include][]": ["name", "date.created", "country", "description", "primary_type.name"],
            "sort[]": "date.created:desc"
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json().get("data", [])

            # Process disasters
            disasters = []
            for item in data:
                fields = item.get("fields", {})
                disaster = {
                    "id": item.get("id", ""),
                    "name": fields.get("name", "Unknown"),
                    "type": fields.get("primary_type", {}).get("name", "Unknown"),
                    "description": fields.get("description", "No description available")[:200] + "...",
                    "date": fields.get("date", {}).get("created", ""),
                    "countries": []
                }
                # Extract country details
                for country in fields.get("country", []):
                    country_info = {
                        "name": country.get("name", "Unknown"),
                        "lat": country.get("location", {}).get("lat", None),
                        "lon": country.get("location", {}).get("lon", None)
                    }
                    disaster["countries"].append(country_info)
                disasters.append(disaster)
            return disasters
        except requests.RequestException as e:
            st.error(f"Error fetching data: {e}")
            return []

    # Function to create a Folium map with disaster markers
    def create_disaster_map(self, disasters):
        # Initialize map centered on the world
        m = folium.Map(location=[0, 0], zoom_start=2, tiles="OpenStreetMap")

        # Add markers for each disaster's countries
        for disaster in disasters:
            for country in disaster["countries"]:
                lat = country.get("lat")
                lon = country.get("lon")
                if lat is not None and lon is not None:
                    popup_text = f"<b>{disaster['name']}</b><br>Type: {disaster['type']}<br>Country: {country['name']}"
                    folium.Marker(
                        location=[lat, lon],
                        popup=popup_text,
                        icon=folium.Icon(color="red", icon="exclamation-circle")
                    ).add_to(m)

        return m

# Streamlit Frontend
def main():
    st.set_page_config(
        page_title="Disaster Chatbot",
        page_icon="🌍",
        layout="wide"
    )



    # Session states
    if "view" not in st.session_state:
        st.session_state.view = "chat"  # Options: "chat", "map"
    if "mode" not in st.session_state:
        st.session_state.mode = "chat"  # Options: "chat", "disaster"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "text_prompt" not in st.session_state:
        st.session_state.text_prompt = ""
    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False
    if "last_analysis" not in st.session_state:
        st.session_state.last_analysis = None
    if "last_pdf_files" not in st.session_state:
        st.session_state.last_pdf_files = None
    if "last_article_summaries" not in st.session_state:
        st.session_state.last_article_summaries = None
    if "last_rag_response" not in st.session_state:
        st.session_state.last_rag_response = None
    if "last_rag_sources" not in st.session_state:
        st.session_state.last_rag_sources = None

    # Header
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='color: #1f77b4;'>Disaster-Aware Chatbot</h1>
            <p style='font-size: 1.2rem;'>Empowering Relief Through Conversations.</p>
        </div>
    """, unsafe_allow_html=True)

    # Create a more prominent toggle button with better styling
    col1, col2 = st.columns([3, 1])
    with col2:
        toggle_label = "Show Disaster Map" if st.session_state.view == "chat" else "Return to Chat"
        if st.button(toggle_label, use_container_width=True, type="primary"):
            # Toggle between chat and map view
            st.session_state.view = "map" if st.session_state.view == "chat" else "chat"
            # Force a rerun to update the UI
            st.rerun()

    # Map View
    if st.session_state.view == "map":
        map = Map()
        # Fetch disaster data
        with st.spinner("Loading disaster data..."):
            disasters = map.fetch_ongoing_disasters()

        # Create two columns: map on the left, list on the right
        col1, col2 = st.columns([2, 1])

        # Display map in the first column
        with col1:
            st.subheader("Map of Ongoing Disasters")
            if disasters:
                folium_map = map.create_disaster_map(disasters)
                st_folium(folium_map, width=700, height=500)
            else:
                st.warning("No ongoing disasters found or unable to fetch data.")

        # Display list of disasters in the second column
        with col2:
            st.subheader("List of Ongoing Disasters")
            if disasters:
                # Convert to DataFrame for better display
                df = pd.DataFrame([
                    {
                        "Name": d["name"],
                        "Type": d["type"],
                        "Countries": ", ".join([c["name"] for c in d["countries"]]),
                        "Date": d["date"],
                        "Description": d["description"]
                    }
                    for d in disasters
                ])
                # Display table without index
                st.dataframe(
                    df[["Name", "Type", "Countries", "Date", "Description"]],
                    hide_index=True,
                    use_container_width=True
                )
                st.write(f"Total ongoing disasters: {len(disasters)}")
            else:
                st.info("No ongoing disasters to display.")


    # Only show chat interface if we're in chat view
    if st.session_state.view == "chat":
        # Mode toggle button
        _ , col2 = st.columns([8, 1])
        with col2:
            if st.button("🌍 Switch Chat Mode", use_container_width=True):
                st.session_state.mode = "disaster" if st.session_state.mode == "chat" else "chat"
                st.rerun()

        # Initialize input
        user_input = ""

        # Main Chat Mode
        if st.session_state.mode == "chat":
            st.markdown("### 🤖 General Chat Mode")

        # Microphone Input
        with st.expander("🎙️ Speak Instead of Typing", expanded=True):
            audio = AudioInput()
            if st.button("Start Recording"):
                global stop_recording, frames
                stop_recording = False
                frames = []

                listener = keyboard.Listener(on_press=audio.on_press)
                listener.start()

                # Start recording in current thread
                audio.record()

                # Transcribe the audio
                user_input = audio.transcribe_audio(WAVE_OUTPUT_PATH)
                if user_input:
                    existing_text = st.session_state.get("text_prompt", "")
                    # Update the text prompt without directly modifying it after widget creation
                    st.session_state.text_prompt = (existing_text + " " + user_input).strip()
                    # Use rerun to refresh the UI with the new text
                    st.rerun()

        st.markdown("💬 Type your message:")
        col1, col2 = st.columns([5, 1])

        # Check if we need to clear the input field
        if st.session_state.clear_input:
            st.session_state.text_prompt = ""
            st.session_state.clear_input = False

        with col1:
            st.text_input("Enter Input", placeholder="Ask something...", key="text_prompt",
                         label_visibility="collapsed", on_change=handle_enter)

        with col2:
            # Align the button better using HTML
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)  # Spacer to align
            send_clicked = st.button("Send", use_container_width=True)
            if send_clicked:
                handle_enter()
                st.rerun()  # Rerun the app to clear the input

        # Display chat history
        st.markdown("### Chat History")
        chat_pairs = list(zip(st.session_state.chat_history[::2], st.session_state.chat_history[1::2]))[::-1]

        for user_msg, bot_msg in chat_pairs:
            if user_msg[0] == "user":
                st.markdown(f"<div style='background-color: black; padding: 10px; border-radius: 8px;'> <strong>You:</strong> {user_msg[1]} </div>", unsafe_allow_html=True)
            if bot_msg[0] == "bot":
                st.markdown(f"<div style='background-color: grey; padding: 10px; border-radius: 8px;'> <strong>Bot:</strong> {bot_msg[1]} </div>", unsafe_allow_html=True)

        # Display analysis results and reports AFTER chat history
        if "last_analysis" in st.session_state and st.session_state.last_analysis:
            st.subheader("Analysis Results")
            with st.expander("View Details", expanded=False):
                col1, col2 = st.columns(2)
                analysis = st.session_state.last_analysis
                with col1:
                    st.markdown(f"**Disaster Type:** {analysis['disaster_type']}")
                    st.markdown(f"**Class Label:** {analysis['class_label']}")
                    st.markdown(f"**Confidence:** {analysis['confidence']}")
                with col2:
                    st.markdown(f"**Locations:** {', '.join(analysis['location']) or 'None'}")
                    st.markdown(f"**Time:** {', '.join(analysis['time']['structured_dates'])}")
                    st.markdown(f"**Urgency:** {analysis['urgency']}")
                    st.markdown(f"**Sentiment:** {analysis['sentiment']}")

            # Display RAG sources if available
            if "last_rag_sources" in st.session_state and st.session_state.last_rag_sources:
                st.subheader("Information Sources")
                st.markdown("The response was generated based on the following sources:")
                for source in st.session_state.last_rag_sources:
                    st.markdown(f"- {source}")

                # Still provide PDF downloads if available
                if "last_pdf_files" in st.session_state and st.session_state.last_pdf_files:
                    with st.expander("Download PDF Reports", expanded=False):
                        for pdf_path, pdf_bytes in st.session_state.last_pdf_files:
                            file_name = os.path.basename(pdf_path)
                            st.download_button(
                                label=f"Download {file_name}",
                                data=pdf_bytes,
                                file_name=file_name,
                                mime="application/pdf"
                            )
            # Display PDF files if available but no RAG sources
            elif "last_pdf_files" in st.session_state and st.session_state.last_pdf_files:
                st.subheader("Generated Reports")
                for pdf_path, pdf_bytes in st.session_state.last_pdf_files:
                    file_name = os.path.basename(pdf_path)
                    st.download_button(
                        label=f"Download {file_name}",
                        data=pdf_bytes,
                        file_name=file_name,
                        mime="application/pdf"
                    )
            # Display article summaries if available
            elif "last_article_summaries" in st.session_state and st.session_state.last_article_summaries:
                st.subheader("Article Summaries")
                for i, summary in enumerate(st.session_state.last_article_summaries, 1):
                    with st.expander(f"📰 {summary['title']}", expanded=i==1):
                        st.markdown(f"**Summary:** {summary['summary']}")
                        st.markdown(f"**Original Length:** {summary['original_length']} characters")
                        st.markdown(f"[Read Full Article]({summary['url']})")

            elif "last_pdf_files" in st.session_state and not st.session_state.last_pdf_files and not st.session_state.last_rag_sources:
                st.info("No information sources available. Either no disaster was detected or no articles were fetched.")

        # Disaster Info Mode
        else:
            st.markdown("### 🌍 Disaster Info Chat")

            # Initialize session state for disaster info mode
            if "disaster_fetch_completed" not in st.session_state:
                st.session_state.disaster_fetch_completed = False

            if "current_disaster_descriptions" not in st.session_state:
                st.session_state.current_disaster_descriptions = []

            st.markdown("Choose a disaster type:")
            disaster_types = ["Earthquakes", "Floods", "Hurricane", "Wildfires", "Tornado", "Tsunami"]
            selected_disaster_type = st.selectbox("Disaster Type", disaster_types)
            st.markdown(f"**You selected:** {selected_disaster_type}")

            if st.button("Fetch Disaster Tweets"):
                with st.spinner("Fetching disaster tweets... This may take a while."):
                    try:
                        twt = TweetScraper()
                        success, disaster_descriptions = twt.process(selected_disaster_type)

                        if success:
                            st.success("Data saved successfully!")
                            # Store the descriptions in session state
                            st.session_state.current_disaster_descriptions = disaster_descriptions
                            st.session_state.disaster_fetch_completed = True
                            # Force a rerun to update the UI
                            st.rerun()
                        else:
                            st.error("An error occurred while saving the data.")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        logger.error(f"Tweet scraping error: {e}")

            # Only show disaster events after fetch is completed
            if st.session_state.disaster_fetch_completed and st.session_state.current_disaster_descriptions:
                st.subheader("Identified Disaster Events")
                selected_disaster_event = st.selectbox("Select a disaster event to get more information",
                                                    st.session_state.current_disaster_descriptions)

                if st.button("Get Articles About This Event"):
                    # Get or initialize orchestrator from session state
                    # Enter tavily api key
                    api_key = ""
                    if "orchestrator" not in st.session_state:
                        st.session_state.orchestrator = init_orchestrator(api_key)

                    with st.spinner(f"Fetching and summarizing articles about {selected_disaster_event}..."):
                        # Process the selected disaster event
                        article_summaries = st.session_state.orchestrator.process_disaster_event(selected_disaster_event, max_articles=5)

                        if article_summaries:
                            st.subheader("Article Summaries")
                            for i, summary in enumerate(article_summaries, 1):
                                with st.expander(f"📰 {summary['title']}", expanded=i==1):
                                    st.markdown(f"**Summary:** {summary['summary']}")
                                    st.markdown(f"**Original Length:** {summary['original_length']} characters")
                                    st.markdown(f"[Read Full Article]({summary['url']})")
                        else:
                            st.info(f"No articles found about {selected_disaster_event}.")
            elif st.session_state.disaster_fetch_completed and not st.session_state.current_disaster_descriptions:
                st.info("No specific disaster events could be identified from the tweets.")


    # Footer
    st.markdown("""
        <hr style='border-top: 1px solid #ccc;' />
        <div style='text-align: center; color: gray; font-size: 0.9rem;'>
            All rights reserved.
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()