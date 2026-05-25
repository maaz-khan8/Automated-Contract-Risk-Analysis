import os
import time
import uuid
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm

class VectorDatabasePipeline:
    def __init__(self, index_name="contracts", batch_size=50):
        """
        Initializes the Pinecone connection.
        """
        load_dotenv()
        
        self.api_key = os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY missing from environment variables.")
            
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = index_name
        self.batch_size = batch_size
        
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index_exists(self):
        """Checks if the Pinecone index exists, creates it if it doesn't."""
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=1024, # Dimension for llama-text-embed-v2
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print("Waiting for index to initialize...")
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            print("Index created and ready!")

    def ingest_dataframe(self, df: pd.DataFrame, text_column: str, metadata_columns: dict, id_column: str = None):
        """
        The core pipeline to embed and upload ANY dataset to Pinecone.
        
        Parameters:
        - df: The Pandas DataFrame containing your data.
        - text_column: The name of the column in the DataFrame that contains the text to be embedded.
        - metadata_columns: A dictionary mapping Pinecone metadata keys to DataFrame column names.
                            Example: {"source_file": "DocumentName", "clause_type": "Category"}
        - id_column: (Optional) The column to use as a unique ID. If None, UUIDs are generated.
        """
        # 1. Clean the data: Drop rows where the text column is empty or NaN
        df = df.dropna(subset=[text_column])
        df = df[df[text_column].str.strip() != ""]
        
        total_rows = len(df)
        print(f"\nStarting ingestion of {total_rows} rows into Pinecone...")

        # 2. Process in batches
        for i in tqdm(range(0, total_rows, self.batch_size)):
            batch = df.iloc[i:i + self.batch_size]
            texts = batch[text_column].astype(str).tolist()

            try:
                # 3. Generate Embeddings using Llama model
                embeddings_response = self.pc.inference.embed(
                    model="llama-text-embed-v2",
                    inputs=texts,
                    parameters={"input_type": "passage", "truncate": "END"}
                )

                # 4. Prepare vectors dynamically based on provided metadata mapping
                vectors = []
                for j, (_, row) in enumerate(batch.iterrows()):
                    # Determine ID
                    if id_column and id_column in row:
                        raw_id = str(row[id_column])
                    else:
                        raw_id = str(uuid.uuid4()) # Fallback to random unique ID
                    
                    # ASCII safety fix from your previous code
                    safe_id = raw_id.encode('ascii', 'ignore').decode('ascii')

                    # Build metadata dynamically based on the dictionary passed by the user
                    metadata = {"text": str(row[text_column])}
                    for standard_key, df_col_name in metadata_columns.items():
                        if df_col_name in row and pd.notna(row[df_col_name]):
                            metadata[standard_key] = str(row[df_col_name])

                    vectors.append({
                        "id": safe_id,
                        "values": embeddings_response.data[j].values,
                        "metadata": metadata
                    })

                # 5. Upsert to Pinecone
                self.index.upsert(vectors=vectors)
                
                # 6. Safety sleep for API rate limits
                time.sleep(2)

            except Exception as e:
                print(f"\nUpload stopped due to error at batch {i}. Error details: {e}")
                break
                
        print("\nData Ingestion Complete! Use dynamic resume next time if it failed halfway.")


# ==========================================
# EXAMPLE USAGE (You can put this in a separate test_pipeline.py file)
# ==========================================
if __name__ == "__main__":
    # Example: Let's say you downloaded a completely different Kaggle dataset
    # about NDA agreements, and it looks like this:
    mock_data = {
        "Doc_ID": ["NDA_001", "NDA_002"],
        "Clause_Text": ["The receiving party shall not disclose...", "Confidential info remains property of..."],
        "Agreement_Type": ["Non-Disclosure", "Non-Disclosure"],
        "Year": [2022, 2023]
    }
    new_dataset_df = pd.DataFrame(mock_data)

    # 1. Initialize the Pipeline
    pipeline = VectorDatabasePipeline(index_name="contracts", batch_size=50)

    # 2. Define the Mapping (This is the secret to flexibility!)
    # We tell the pipeline exactly which columns contain the data we care about.
    target_metadata = {
        "clause_type": "Agreement_Type", # Map Pinecone 'clause_type' to DF 'Agreement_Type'
        "source_file": "Doc_ID",         # Map Pinecone 'source_file' to DF 'Doc_ID'
        "year_signed": "Year"            # We can easily add NEW metadata fields!
    }

    # 3. Run the ingestion
    pipeline.ingest_dataframe(
        df=new_dataset_df,
        text_column="Clause_Text",       # The column containing the actual text to embed
        metadata_columns=target_metadata,
        id_column="Doc_ID"
    )