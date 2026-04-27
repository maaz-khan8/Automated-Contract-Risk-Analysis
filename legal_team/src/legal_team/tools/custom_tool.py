import os
from typing import Type # <--- ADDED IMPORT
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from pinecone import Pinecone
from llama_parse import LlamaParse
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# ==========================================
# TOOL 1: Pinecone RAG Search Tool
# Used by the Researcher Agent
# ==========================================
class PineconeSearchInput(BaseModel):
    query: str = Field(description="The specific contract risk or clause to search for in the precedent database.")

class PineconeSearchTool(BaseTool):
    name: str = "Pinecone Precedent Search"
    description: str = "Searches a vector database of fair, standard computer science contracts to verify if a flagged clause is predatory or standard industry practice."
    
    # THE FIX: Added ': Type[BaseModel]' type annotation
    args_schema: Type[BaseModel] = PineconeSearchInput 

    def _run(self, query: str) -> str:
        # 1. Connect to Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("contracts")

        # 2. Generate the embedding for the Researcher's query
        query_embedding = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=[query],
            parameters={"input_type": "query", "truncate": "END"}
        )

        # 3. Search the "contracts" index for the top 3 most relevant fair clauses
        results = index.query(
            vector=query_embedding.data[0].values,
            top_k=3,
            include_metadata=True
        )

        # 4. Format the results so the Researcher Agent can read them easily
        formatted_results = []
        for match in results['matches']:
            clause_type = match['metadata'].get('clause_type', 'Unknown')
            text = match['metadata'].get('text', '')
            source = match['metadata'].get('source_file', 'Unknown')
            
            formatted_results.append(
                f"--- Standard '{clause_type}' Precedent ---\n"
                f"Source Contract: {source}\n"
                f"Text: {text}"
            )

        return "\n\n".join(formatted_results) if formatted_results else "No relevant precedents found in the database."


# ==========================================
# TOOL 2: LlamaParse PDF Reader Tool
# Used by the Extractor Agent
# ==========================================
class PDFExtractionInput(BaseModel):
    file_path: str = Field(description="The absolute file path to the PDF contract to be parsed.")

class LlamaParseTool(BaseTool):
    name: str = "LlamaParse PDF Reader"
    description: str = "Extracts highly accurate text and structure from complex legal PDF contracts."
    
    # THE FIX: Added ': Type[BaseModel]' type annotation
    args_schema: Type[BaseModel] = PDFExtractionInput 

    def _run(self, file_path: str) -> str:
        # Check if file exists
        if not os.path.exists(file_path):
            return f"Error: File not found at path {file_path}"
            
        # Initialize LlamaParse
        parser = LlamaParse(
            api_key=os.getenv("LLAMAPARSE_API_KEY"),
            result_type="markdown", 
            verbose=False
        )
        
        # Parse the document
        documents = parser.load_data(file_path)
        
        # Combine the text from all pages into one large string
        full_text = "\n".join([doc.text for doc in documents])
        return full_text