import os
import torch
from typing import List, Dict, Tuple, Optional, Set, Any, Callable, Union
import logging
import uuid
from datetime import datetime
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True
)
logger = logging.getLogger(__name__)

# Base Agent class for the agentic framework
class Agent:
    """Base Agent class for the agentic framework"""
    def __init__(self, name: str):
        self.name = name
        self.message_queue = []
        self.responses = {}
        self.logger = logging.getLogger(f"Agent:{name}")
        self.logger.info(f"Agent {name} initialized")

    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming message and return a response"""
        self.logger.info(f"Processing message: {message['message_type']}")
        handler_name = f"handle_{message['message_type']}"
        if hasattr(self, handler_name):
            handler = getattr(self, handler_name)
            return handler(message)
        else:
            self.logger.warning(f"No handler for message type: {message['message_type']}")
            return {"status": "error", "error": f"No handler for message type: {message['message_type']}"}

    def send_message(self, recipient: str, content: Dict[str, Any], message_type: str, request_id: str = None) -> Dict[str, Any]:
        """Create a message to be sent to another agent"""
        if request_id is None:
            request_id = str(uuid.uuid4())

        message = {
            "sender": self.name,
            "recipient": recipient,
            "content": content,
            "message_type": message_type,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }

        self.logger.info(f"Sending message to {recipient}: {message_type}")
        return message



class RAGSystem:
    """
    Retrieval-Augmented Generation system for processing PDFs and answering questions
    based on their content.
    """

    def __init__(self, initial_pdf_dir: str = "./RAG Pre-Info", chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the RAG system with PDFs from the specified directory.

        Args:
            initial_pdf_dir: Directory containing initial PDFs to load
            chunk_size: Size of text chunks for vectorization
            chunk_overlap: Overlap between chunks
        """
        logger.info(f"Initializing RAG system with PDFs from {initial_pdf_dir}")

        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Initialize vector store
        self.vector_store = self._initialize_vector_store(initial_pdf_dir)

        # Initialize LLM
        self._initialize_llm()

        logger.info("RAG system initialized successfully")

    def make_decision(self, query: str, context: List[Document]) -> Dict[str, Any]:
        """Make a decision about how to respond to a query based on the retrieved context"""
        # Analyze the query to determine intent
        query_lower = query.lower()

        # Check if query is about a specific disaster type
        disaster_types = ["earthquake", "flood", "hurricane", "wildfire", "tornado", "tsunami"]
        detected_disaster = next((dt for dt in disaster_types if dt in query_lower), None)

        # Check if query is asking for specific information
        is_asking_location = any(term in query_lower for term in ["where", "location", "place", "area", "region"])
        is_asking_time = any(term in query_lower for term in ["when", "time", "date", "day", "month", "year"])
        is_asking_impact = any(term in query_lower for term in ["impact", "effect", "damage", "casualties", "deaths"])
        is_asking_response = any(term in query_lower for term in ["response", "aid", "help", "rescue", "relief"])

        # Determine the focus of the response based on the query
        focus = []
        if detected_disaster:
            focus.append(f"information about {detected_disaster}")
        if is_asking_location:
            focus.append("location information")
        if is_asking_time:
            focus.append("timing information")
        if is_asking_impact:
            focus.append("impact assessment")
        if is_asking_response:
            focus.append("response efforts")

        # If no specific focus is detected, provide a general response
        if not focus:
            focus.append("general information")

        return {
            "detected_disaster": detected_disaster,
            "focus": focus,
            "is_asking_location": is_asking_location,
            "is_asking_time": is_asking_time,
            "is_asking_impact": is_asking_impact,
            "is_asking_response": is_asking_response
        }

    def _initialize_llm(self):
        """Initialize the local LLM for generating responses"""
        hf_token = ""
        print("Hugging face token : " + hf_token)
        try:
            # Try to use Llama-3-8B-Instruct if available
            model_id = "meta-llama/Llama-3-8B-Instruct"

            logger.info(f"Loading LLM: {model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token = hf_token)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                token = hf_token
            )
            logger.info("LLM loaded successfully")
        except Exception as e:
            # Fallback to a smaller model if Llama is not available
            logger.warning(f"Failed to load Llama model: {e}")
            logger.info("Falling back to smaller model")

            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16
            )
            logger.info("Fallback model loaded successfully")

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        try:
            logger.info(f"Extracting text from {pdf_path}")
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def _initialize_vector_store(self, pdf_dir: str) -> FAISS:
        """Initialize vector store with PDFs from the specified directory"""
        if not os.path.exists(pdf_dir):
            logger.warning(f"Directory {pdf_dir} does not exist. Creating empty vector store.")
            return FAISS.from_texts(["Placeholder text"], self.embeddings)

        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

        all_docs = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            text = self._extract_text_from_pdf(pdf_path)
            if text:
                chunks = self.text_splitter.split_text(text)
                for chunk in chunks:
                    all_docs.append(Document(
                        page_content=chunk,
                        metadata={"source": pdf_file}
                    ))

        if not all_docs:
            logger.warning("No valid text extracted from PDFs. Creating empty vector store.")
            return FAISS.from_texts(["Placeholder text"], self.embeddings)

        logger.info(f"Created {len(all_docs)} document chunks for vector store")
        return FAISS.from_documents(all_docs, self.embeddings)

    def add_pdf_to_vector_store(self, pdf_path: str, pdf_bytes: bytes = None) -> bool:
        """Add a PDF to the vector store"""
        try:
            if pdf_bytes:
                # Save PDF bytes to a temporary file
                temp_path = f"temp_{os.path.basename(pdf_path)}"
                with open(temp_path, "wb") as f:
                    f.write(pdf_bytes)
                pdf_path = temp_path

            text = self._extract_text_from_pdf(pdf_path)

            if pdf_bytes and os.path.exists(temp_path):
                os.remove(temp_path)

            if not text:
                logger.warning(f"No text extracted from {pdf_path}")
                return False

            chunks = self.text_splitter.split_text(text)
            docs = [Document(
                page_content=chunk,
                metadata={"source": os.path.basename(pdf_path)}
            ) for chunk in chunks]

            if not docs:
                logger.warning(f"No document chunks created for {pdf_path}")
                return False

            new_vector_store = FAISS.from_documents(docs, self.embeddings)
            self.vector_store.merge_from(new_vector_store)
            logger.info(f"Added {len(docs)} document chunks from {pdf_path} to vector store")
            return True
        except Exception as e:
            logger.error(f"Error adding PDF to vector store: {e}")
            return False

    def add_pdfs_from_directory(self, pdf_dir: str) -> int:
        """Add all PDFs from a directory to the vector store"""
        if not os.path.exists(pdf_dir):
            logger.warning(f"Directory {pdf_dir} does not exist")
            return 0

        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

        count = 0
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            if self.add_pdf_to_vector_store(pdf_path):
                count += 1

        logger.info(f"Added {count} PDFs from {pdf_dir} to vector store")
        return count

    def generate_prompt(self, query: str, context: str, custom_instructions: str = "") -> str:
        """Generate a prompt for the LLM with optional custom instructions"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant specialized in disaster management and emergency response.
Use the provided context to answer the user's question accurately and concisely.
If the information is not in the context, say that you don't have enough information.
{custom_instructions}
<|start_header_id|>user<|end_header_id|>
Context:
{context}

Question: {query}

<|start_header_id|>assistant<|end_header_id|>
"""

    def generate_answer(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate an answer using the LLM"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the assistant's response
            if "<|start_header_id|>assistant<|end_header_id|>" in response:
                response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()

            return response
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I'm sorry, I encountered an error while generating a response: {str(e)}"

    def answer_query(self, query: str, k: int = 5) -> Tuple[str, List[str]]:
        """
        Answer a query using the RAG system

        Args:
            query: The user's question
            k: Number of documents to retrieve

        Returns:
            Tuple of (answer, sources)
        """
        try:
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(query, k=k)

            if not docs:
                return "I don't have enough information to answer that question.", []

            # Make a decision about how to respond based on the query and context
            decision = self.make_decision(query, docs)

            # Customize the prompt based on the decision
            custom_instructions = ""
            if decision["focus"]:
                custom_instructions = f"Focus on providing {', '.join(decision['focus'])}.\n"

            # Extract unique content and sources
            context = "\n\n".join([doc.page_content for doc in docs])
            sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

            # Generate prompt with custom instructions and answer
            prompt = self.generate_prompt(query, context, custom_instructions)
            answer = self.generate_answer(prompt)

            # Log the decision and response
            logger.info(f"Query: {query}")
            logger.info(f"Decision: {decision}")
            logger.info(f"Generated response with focus on: {decision['focus']}")

            return answer, sources
        except Exception as e:
            logger.error(f"Error answering query: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}", []

# RAGSystemAgent Class
class RAGSystemAgent(Agent):
    """Agent wrapper for the RAG system"""
    def __init__(self, initial_pdf_dir: str = "./RAG Pre-Info"):
        super().__init__("RAGSystem")
        self.rag_system = RAGSystem(initial_pdf_dir)
        self.logger.info("RAGSystemAgent initialized")

    def handle_add_pdf(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request to add a PDF to the vector store"""
        pdf_path = message["content"].get("pdf_path", "")
        pdf_bytes = message["content"].get("pdf_bytes", None)

        success = self.rag_system.add_pdf_to_vector_store(pdf_path, pdf_bytes)

        return {
            "status": "success" if success else "error",
            "message": f"Added PDF {pdf_path} to vector store" if success else f"Failed to add PDF {pdf_path} to vector store",
            "request_id": message["request_id"]
        }

    def handle_query(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request to query the RAG system"""
        query = message["content"].get("query", "")
        k = message["content"].get("k", 5)

        answer, sources = self.rag_system.answer_query(query, k)

        return {
            "status": "success",
            "answer": answer,
            "sources": sources,
            "request_id": message["request_id"]
        }

    def handle_add_pdfs_from_directory(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request to add PDFs from a directory"""
        pdf_dir = message["content"].get("pdf_dir", "")

        count = self.rag_system.add_pdfs_from_directory(pdf_dir)

        return {
            "status": "success",
            "count": count,
            "message": f"Added {count} PDFs from {pdf_dir} to vector store",
            "request_id": message["request_id"]
        }