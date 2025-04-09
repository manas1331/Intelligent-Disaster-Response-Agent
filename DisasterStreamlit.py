import os
import re
import json
import requests
from datetime import datetime
import logging
from typing import Dict, List, Set, Optional
import pandas as pd
import streamlit as st
import time

# Disable Streamlit's file watcher explicitly before any imports that might trigger it
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
# Verify it’s set
if os.environ.get("STREAMLIT_SERVER_FILE_WATCHER_TYPE") != "none":
    raise RuntimeError("Failed to disable Streamlit file watcher")

# Delayed imports for heavy libraries (PyTorch, transformers, etc.) to avoid watcher issues
def lazy_import():
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
    return {
        "torch": torch,
        "T5ForConditionalGeneration": T5ForConditionalGeneration,
        "T5Tokenizer": T5Tokenizer,
        "pipeline": pipeline,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "spacy": spacy,
        "dateparser": dateparser,
        "unicodedata": unicodedata,
        "html": html,
        "demojize": demojize,
        "fix": fix,
        "letter": letter,
        "SimpleDocTemplate": SimpleDocTemplate,
        "Paragraph": Paragraph,
        "Spacer": Spacer,
        "getSampleStyleSheet": getSampleStyleSheet,
        "ParagraphStyle": ParagraphStyle,
        "TA_CENTER": TA_CENTER
    }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True
)
logger = logging.getLogger(__name__)
logger.info("Imports and logging initialized")

# TextPreprocessor Class
class TextPreprocessor:
    def __init__(self):
        libs = lazy_import()
        self.nlp = libs["spacy"].load("en_core_web_sm")

    def preprocess(self, text: str) -> str:
        libs = lazy_import()
        if pd.isna(text):
            return ""
        text = libs["unicodedata"].normalize("NFKD", str(text))
        text = libs["html"].unescape(text.lower())
        text = libs["fix"](text)
        text = libs["demojize"](text)
        text = re.sub(r'http\S+|www\S+', "", text)
        text = re.sub(r"[@#]\w+", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def extract_time(self, text: str) -> Dict[str, List[str]]:
        libs = lazy_import()
        text = self.preprocess(text)
        doc = self.nlp(text)
        parsed_dates = [
            libs["dateparser"].parse(ent.text).strftime("%Y-%m-%d %H:%M:%S")
            for ent in doc.ents if ent.label_ in ["DATE", "TIME"] and libs["dateparser"].parse(ent.text)
        ]
        return {"structured_dates": parsed_dates or ["Not specified"]}

    def extract_locations(self, text: str) -> Set[str]:
        text = self.preprocess(text)
        doc = self.nlp(text)
        return {ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]}

# DisasterAnalyzer Class
class DisasterAnalyzer:
    def __init__(self):
        logger.info("Initializing DisasterAnalyzer")
        libs = lazy_import()
        self.torch = libs["torch"]
        self.class_labels = ['not_disaster', 'disaster']
        self.device = "cuda" if self.torch.cuda.is_available() else "mps" if self.torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        try:
            self.model = libs["AutoModelForSequenceClassification"].from_pretrained("sladereaperr/BERTdisaster", num_labels=2).to(self.device)
            self.tokenizer = libs["AutoTokenizer"].from_pretrained("sladereaperr/BERTdisaster")
            logger.info("BERT model and tokenizer loaded")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise
        self.t5_model = libs["T5ForConditionalGeneration"].from_pretrained("t5-small").to(self.device)
        self.t5_tokenizer = libs["T5Tokenizer"].from_pretrained("t5-small")
        self.sentiment_pipeline = libs["pipeline"]("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.disaster_keywords = {
            "earthquake": "Earthquake", "flood": "Flood", "hurricane": "Hurricane",
            "wildfire": "Wildfire", "tornado": "Tornado", "tsunami": "Tsunami"
        }
        logger.info("DisasterAnalyzer initialized")

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
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class_id = self.torch.argmax(logits, dim=1)[0].item()
        confidence = self.torch.softmax(logits, dim=1)[0, predicted_class_id].item()
        return {
            "class_label": self.class_labels[predicted_class_id],
            "confidence": f"{confidence:.4f}"
        }

    def analyze_sentiment(self, text: str) -> str:
        sentiment_input = self.t5_tokenizer("sst2: " + text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
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

# PDFGenerator Class
class PDFGenerator:
    def save_to_pdf(self, content: str, title: str, index: int, output_dir: str = "pdfs") -> Optional[tuple[str, bytes]]:
        libs = lazy_import()
        try:
            os.makedirs(output_dir, exist_ok=True)
            pdf_path = os.path.join(output_dir, f"{title[:45].replace(' ', '_')}_{index}.pdf")
            doc = libs["SimpleDocTemplate"](pdf_path, pagesize=libs["letter"])
            elements = [
                libs["Paragraph"](title, libs["ParagraphStyle"]('Title', fontSize=16, alignment=libs["TA_CENTER"])),
                libs["Spacer"](1, 20)
            ]
            elements += [libs["Paragraph"](libs["html"].escape(p), libs["getSampleStyleSheet"]()["Normal"]) for p in content.split('\n\n')]
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
    def __init__(self, api_key: str):
        logger.info("Initializing DisasterReportOrchestrator")
        self.preprocessor = TextPreprocessor()
        self.analyzer = DisasterAnalyzer()
        self.scraper = DataScraper(api_key)
        self.pdf_generator = PDFGenerator()
        logger.info("DisasterReportOrchestrator initialized")

    def process(self, prompt: str, max_articles: int = None) -> tuple[Dict, List[tuple[str, bytes]]]:
        logger.info(f"Processing prompt: {prompt}")
        locations = self.preprocessor.extract_locations(prompt)
        time_info = self.preprocessor.extract_time(prompt)
        logger.info(f"Locations: {locations}, Time: {time_info}")
        analysis = self.analyzer.analyze(prompt, locations, time_info)
        logger.info(f"Analysis result: {json.dumps(analysis, indent=2)}")
        pdf_files = []
        if analysis["class_label"] == "disaster":
            query = " ".join(filter(None, [*analysis['location'], analysis['disaster_type'], *analysis['time']['structured_dates']]))
            articles = self.scraper.fetch_articles(query)
            if not articles:
                logger.warning("No articles fetched, check API key or network.")
                return analysis, pdf_files
            max_articles = len(articles) if max_articles is None else min(max_articles, len(articles))
            for i, article in enumerate(articles[:max_articles]):
                content = article.get('content', 'No content available')
                title = article.get('title', f"Article_{i+1}")
                result = self.pdf_generator.save_to_pdf(content, title, i)
                if result:
                    pdf_files.append(result)
                else:
                    logger.warning(f"PDF generation failed for {title}")
        else:
            logger.info("No disaster detected, skipping article fetching and PDF generation.")
        return analysis, pdf_files

# Streamlit Frontend
def main():
    st.set_page_config(
        page_title="Disaster Report Generator",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .main {background-color: #f5f5f5;}
        .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
        .stTextInput>div>input {border-radius: 5px;}
        .sidebar .sidebar-content {background-color: #ffffff;}
        h1 {color: #2c3e50;}
        .stExpander {background-color: #ffffff; border-radius: 5px;}
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("Settings")
    st.sidebar.markdown("Configure your disaster report generation.")
    api_key = "" #Enter your API Key
    max_articles = st.sidebar.slider("Max Articles to Process", 1, 20, 5)

    st.title("🌍 Disaster Report Generator")
    st.markdown("Enter a prompt to analyze a potential disaster and generate reports.")

    prompt = st.text_input("Enter your prompt (e.g., 'Earthquake in Myanmar')", "Give me information about Myanmar Earthquake")

    if st.button("Generate Report"):
        if not api_key:
            st.error("Please provide a valid Tavily API Key in the sidebar.")
            return

        with st.spinner("Processing your request..."):
            orchestrator = DisasterReportOrchestrator(api_key)
            analysis, pdf_files = orchestrator.process(prompt, max_articles)

        st.subheader("Analysis Results")
        with st.expander("View Details", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Disaster Type:** {analysis['disaster_type']}")
                st.markdown(f"**Class Label:** {analysis['class_label']}")
                st.markdown(f"**Confidence:** {analysis['confidence']}")
            with col2:
                st.markdown(f"**Locations:** {', '.join(analysis['location']) or 'None'}")
                st.markdown(f"**Time:** {', '.join(analysis['time']['structured_dates'])}")
                st.markdown(f"**Urgency:** {analysis['urgency']}")
                st.markdown(f"**Sentiment:** {analysis['sentiment']}")

        if pdf_files:
            st.subheader("Generated Reports")
            for pdf_path, pdf_bytes in pdf_files:
                file_name = os.path.basename(pdf_path)
                st.download_button(
                    label=f"Download {file_name}",
                    data=pdf_bytes,
                    file_name=file_name,
                    mime="application/pdf"
                )
        else:
            st.info("No PDFs generated. Either no disaster was detected or no articles were fetched.")

    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit | © 2025 Disaster Report Generator")

if __name__ == "__main__":
    main()