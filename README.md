# Agent car insurance rates v 1.1
AI Agent can response your asks about car insurance and where this information, such as business rules, rates, and insurance coverage, is private and is typically managed using an Excel spreadsheet.

## Features:
🧠 **The Chat quickly understands your question:** Because it integrates a high-level and mature LLM model.  
🔒 **Easy access to private Excel in Google Drive**: Through RAG, you can access to info into Excel file with the key information of the sale operation.  
🔄 **The Agent always response with updated information**: From your private Excel in Google Drive.

## Prerequisites:
- A Hugging Face account API inference serverless.
- A Huggin Face Token for consume the API.
- A Google Colab account associated with a Gmail account.


## Step-by-step plan

- [Set up Google Drive](#Set up Google Drive) 
- [Asking to AI Agentic](#asking-to-ai-agentic)
- [Instalación](#instalación)



# Set up Google Colab environment(MVP)

1. ## Set up Google Drive
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

5. Create Document Object LangGraph
   ```bash
   from langchain.docstore.document import Document
   hf_dataset = Dataset.from_pandas(df)
   print(hf_dataset)
   
   documents = []
   for idx, example in enumerate(hf_dataset):
    aseguradora = example['aseguradora']
    coberturas = example['coberturas_incluidas']
    precio_anual = example['precio_aproximado_anual']
    precio_mensual = example['precio_aproximado_mensual']
    deducible_value = example['deducible_value']
    asistencia_en_viaje = example['asistencia_en_viaje']
    vehiculo_de_sustitucion = example['vehiculo_de_sustitucion']
    descuentos_y_bonificaciones = example['descuentos_y_bonificaciones']
    red_de_talleres = example['red_de_talleres']
    proceso_de_reclamaciones = example['proceso_de_reclamaciones']
    valoracion_de_clientes = example['valoracion_de_clientes']
    observaciones_adicionales = example['observaciones_adicionales']

    text = (
        f"Aseguradora: {aseguradora}\n"
        f"Coberturas incluidas: {coberturas}\n"
        f"Precio aproximado anual: {precio_anual}\n"
        f"Precio aproximado mensual: {precio_mensual}"
        f"Deducible: {deducible_value}"
        f"Asistencia En Viaje: {asistencia_en_viaje}"
        f"Vehículo De Sustitución: {vehiculo_de_sustitucion}"
        f"Descuentos Y Bonificaciones: {descuentos_y_bonificaciones}"
        f"Red De Talleres: {red_de_talleres}"
        f"Proceso De Rclamaciones: {proceso_de_reclamaciones}"
        f"Valoración De Clientes: {valoracion_de_clientes}"
        f"Observaciones Adicionales: {observaciones_adicionales}"
    )
    doc = Document(
        page_content=text,
        metadata={
            "row": idx,
            "aseguradora": aseguradora,
            "precio_anual": precio_anual,
            "precio_mensual": precio_mensual,
            "deducible_value": deducible_value,
            "asistencia_en_viaje": asistencia_en_viaje,
            "vehiculo_de_sustitucion": vehiculo_de_sustitucion,
            "descuentos_y_bonificaciones": descuentos_y_bonificaciones,
            "red_de_talleres": red_de_talleres,
            "proceso_de_reclamaciones": proceso_de_reclamaciones,
            "valoracion_de_clientes": valoracion_de_clientes,
            "observaciones_adicionales": observaciones_adicionales
        }
    )
    documents.append(doc)
    
    print(f"Document objects creados: {len(documents)}")

6. Set up Retriever tool(LangGraph)
   ```bash
   !pip install langchain_community
   from langchain_community.retrievers import BM25Retriever
   from langchain.tools import Tool
   !pip install rank_bm25
   
   bm25_retriever = BM25Retriever.from_documents(documents)
   
   def extract_text(query: str) -> str:
    """Retrieve detailed information about Acme's auto insurance policies."""
    results = bm25_retriever.invoke(query)
    if results:
        return "\n\n".join([doc.page_content for doc in results[:3]])
    else:
        return "No matching guest information found."
        
        
   guest_info_tool = Tool(
    name="service_info_retriever",
    func=extract_text,
    description="Retrieve detailed information about Acme's auto insurance policies."
    )

7. Set up Agent and setup RAG
   ```bash
   !pip install langgraph
   !pip install langchain_huggingface
   from typing import TypedDict, Annotated
   from langgraph.graph.message import add_messages
   from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
   from langgraph.prebuilt import ToolNode
   from langgraph.graph import START, StateGraph
   from langgraph.prebuilt import tools_condition
   from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

8. Set up Graph Agent, nodes, tools and etc
   ```bash
   # Generate the chat interface, including the tools
   from google.colab import userdata
   hf_token = userdata.get('HF_TOKEN')
   
   llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=hf_token,
    )
    
    chat = ChatHuggingFace(llm=llm, verbose=True)
    tools = [guest_info_tool]
    chat_with_tools = chat.bind_tools(tools)
    
    # Generate the AgentState and Agent graph
    class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    
    def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }
    
    ## The graph
    builder = StateGraph(AgentState)
    
    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    
    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
    )
    
    builder.add_edge("tools", "assistant")
    patrick = builder.compile()

## Asking to AI Agentic
   ```bash
   messages = [HumanMessage(content="¿Cuál es el precio aproximado anual de un plan?")]
   response = patrick.invoke({"messages": messages})
   
   print("🎩 Patrick's Response:")
   print(response['messages'][-1].content)


## AI Agent output:
🎩 Patrick's Response:
Según la información proporcionada por la aseguradora Acme, el precio aproximado anual de un plan de seguro de autos varía entre $500.000 y $1.103.000, dependiendo del modelo y del deducible elegido.

