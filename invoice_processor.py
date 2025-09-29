import logging
from typing import Any, Dict, List

import fitz
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

class InvoiceProcessor:
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the Invoice Processor with the specified model.
        Args:
            model_name: The name of the OpenAI model to use.
        """
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.index = None

    def read_and_chunk_file(self, pdf_path: str) -> List[Document]:
        """
        Read a PDF file and chunk it into smaller documents using the fitz library.
        Args:
            pdf_path: Path to the PDF file.
        Returns:
            List of document chunks.
        """
        logger.info(f"Reading and chunking PDF: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = text_splitter.create_documents([text])
            logger.info(f"Created {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return []

    def create_index(self, chunks: List[Document]) -> FAISS:
        """
        Create a vector index from document chunks.
        Args:
            chunks: List of document chunks.
        Returns:
            FAISS index.
        """
        logger.info("Creating FAISS index...")
        if not chunks:
            raise ValueError("No chunks provided to create the index.")
        self.index = FAISS.from_documents(chunks, self.embeddings)
        logger.info("FAISS index created successfully.")
        return self.index

    def retrieve_top_chunks(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve the top k relevant document chunks for a given query.
        Args:
            query: The query to search for.
            k: Number of chunks to retrieve.
        Returns:
            List of relevant document chunks.
        Raises:
            ValueError: If the index does not exist.
        """
        if self.index is None:
            raise ValueError("Index does not exist. Please process an invoice first.")
        logger.info(f"Retrieving top {k} chunks for query: '{query}'")
        return self.index.similarity_search(query, k=k)

    def generate_answer(self, query: str) -> Dict[str, Any]:
        """
        Generate an answer to a query using the RAG system.
        Args:
            query: The query to answer.
        Returns:
            Dictionary containing:
                - "answer": The generated answer.
                - "source_chunks": The relevant document chunks used to generate the answer.
        Raises:
            ValueError: If the index does not exist.
        """
        if self.index is None:
            raise ValueError("Index does not exist. Please process an invoice first.")

        # Create a custom prompt template
        template = """You are an AI assistant specialized in extracting specific information from invoice documents. Use only the provided document snippets to answer the user's question. If the information is not present in the snippets, state that you cannot find the answer. Do not use any external knowledge.

        Invoice snippets:
        {context}

        Question: {question}

        Answer:"""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # Set up the retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.index.as_retriever(),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

        logger.info(f"Generating answer for query: '{query}'")
        result = qa_chain({"query": query})

        # Process and format the output
        source_chunks = result['source_documents']
        return {
            "answer": result['result'],
            "source_chunks": [doc.page_content for doc in source_chunks]
        }

    def process_invoice(self, pdf_path: str) -> bool:
        """
        Process an invoice PDF and prepare for querying.
        Args:
            pdf_path: Path to the PDF file.
        """
        logger.info(f"Starting to process invoice: {pdf_path}")
        try:
            chunks = self.read_and_chunk_file(pdf_path)
            self.create_index(chunks)
            logger.info("Invoice processing completed successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to process invoice: {e}")
            return False

    def answer_invoice_query(self, query: str) -> Dict[str, Any]:
        """
        Answer a query about the processed invoice.
        Args:
            query: The query to answer.
        Returns:
            Dictionary containing the answer and source chunks.
        """
        return self.generate_answer(query)

if __name__ == "__main__":
    from invoice_generator import generate_multiple_invoices

    invoice_files = generate_multiple_invoices(1)
    if not invoice_files:
        print("Failed to generate an invoice. Exiting.")
    else:
        pdf_path = invoice_files[0]
        processor = InvoiceProcessor()
        
        if processor.process_invoice(pdf_path):
            sample_queries = [
                "What is the invoice number?",
                "What is the payment term?",
                "What is the shipper line?",
                "What is shipment term?",
            ]

            for query in sample_queries:
                result = processor.answer_invoice_query(query)
                print("-" * 50)
                print(f"\nQuery: {query}")
                print(f"Answer: {result['answer']}")
                print(f"Source chunks: {result['source_chunks']}")