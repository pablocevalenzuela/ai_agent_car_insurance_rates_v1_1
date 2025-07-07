# Agent car insurance rates v 1.1
AI Agent can response your asks about car insurance and where this information, such as business rules, rates, and insurance coverage, is private and is typically managed using an Excel spreadsheet.

### Features:
🧠 **The Chat quickly understands your question:** Because it integrates a high-level and mature LLM model.  
🔒 **Easy access to private Excel in Google Drive**: Through RAG, you can access to info into Excel file with the key information of the sale operation.  
🔄 **The Agent always response with updated information**: From your private Excel in Google Drive.

### Prerequisites:
- A Hugging Face account API inference serverless.
- A Huggin Face Token for consume the API.
- A Google Colab account associated with a Gmail account.
- 


### Steps to implementation in Google Colab

- [Instalación](#instalación)




## Set up Google Colab environment(MVP)

1. Set up Google Drive
   ```bash
   from google.colab import drive
   drive.mount('/content/drive')

2. Set up dependencies
   ```bash
   !pip install --quiet langchain huggingface_hub faiss-cpu pandas python-dotenv
   !pip install --quiet datasets

3. Set up Inference serverless API
   ```bash
   from huggingface_hub import notebook_login
   notebook_login()

4. Set up dataset and dataframe
   ```bash
   import pandas as pd
   import datasets
   from datasets import Dataset
   from langchain.docstore.document import Document
   excel_path = '/content/drive/MyDrive/Colab Notebooks/base_acme_2025.xlsx'
   df = pd.read_excel(excel_path)
   print(f"Filas cargadas: {len(df)}")
   
   from datasets import Dataset
   hf_dataset = Dataset.from_pandas(df)
   print(hf_dataset)
   from langchain.docstore.document import Document
