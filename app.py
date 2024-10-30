import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import pandas as pd

# Configurações de página Streamlit
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_title="Alicia | 3246",
    page_icon="💁🏻‍♀️"
)

# Função para carregar documentos de um arquivo Excel
def load_documents_from_excel(file_path, column_name):
    df = pd.read_excel(file_path)
    return df[column_name].dropna().tolist()

# Definir a classe de recuperação de documentos
class DocumentRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.index = self.create_index(documents)
    
    def embed_texts(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        embeddings = self.model(**tokens).last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings

    def create_index(self, documents):
        embeddings = self.embed_texts(documents)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def retrieve(self, query, k=2):
        query_embedding = self.embed_texts([query])
        _, indices = self.index.search(query_embedding, k)
        retrieved_docs = " ".join([self.documents[i] for i in indices[0]])
        return retrieved_docs

# Definir a classe ChatLlama com RAG
class ChatLlama:
    def __init__(self, retriever):
        load_dotenv()
        self.retriever = retriever
        self.llm = ChatGroq(temperature=0.2, model_name="llama3-8b-8192")

    def processar_resposta(self, transcricao):
        """Processar a resposta da LLM com recuperação de documentos."""
        if not transcricao.strip():
            return "A entrada está vazia. Por favor, digite algo."
        
        try:
            # Recupera documentos relevantes
            contexto = self.retriever.retrieve(transcricao)
            prompt = f"""
            Contexto: {contexto}. Só use o contexto quando perguntado sobre.
            Você é um consultor de produtos e serviços da Credseguro. 
            Use o contexto a seguir para responder de maneira educada e ilustrativa, contudo, não seja prolixo:
             Pergunta: {transcricao}
            """
            resposta_llm = self.llm.invoke(prompt).content
            print("LLM:", resposta_llm, "\n")
            return resposta_llm
        except Exception as e:
            print(f"Erro ao processar a resposta: {e}")
            return "Erro ao processar a resposta. Tente novamente."

# Inicializar a aplicação Streamlit
def main():
    st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    pcol, pcol2, pcol3 = st.columns(3)
    
    col2.image('Images/Logo_side.png', width=390)
    with pcol2:
        st.markdown("<h1 style='text-align: center; color: #00AE9D; font-size: 40px'>Alicia | Chat com o LLM da Credseguro</h1>", unsafe_allow_html=True)
    
    # Carregar documentos do Excel
    documents = load_documents_from_excel("documentos_credseguro.xlsx", "texto_documento")
    
    # Inicializar o modelo de recuperação e ChatLlama
    retriever = DocumentRetriever(documents)
    chat_model = ChatLlama(retriever)
    
    # Manter um histórico de conversas
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Exibir o histórico de conversas
    if st.session_state.conversation_history:
        for i, message in enumerate(st.session_state.conversation_history):
            with st.chat_message("user" if i % 2 == 0 else "assistant"):
                st.write(message)
    
    # Campo de input para o chat
    user_input = st.chat_input("Digite sua mensagem:")
    
    if user_input:
        # Adicionar a mensagem do usuário ao histórico
        st.session_state.conversation_history.append(f"Usuário: {user_input}")
        
        # Processar a resposta do modelo Llama com recuperação
        resposta_llm = chat_model.processar_resposta(user_input)
        
        # Adicionar a resposta ao histórico
        st.session_state.conversation_history.append(f"Llama: {resposta_llm}")
        
        # Exibir a mensagem do usuário
        with st.chat_message("user"):
            st.write(user_input)
        
        # Exibir a resposta da Llama
        with st.chat_message("assistant"):
            st.write(resposta_llm)

# Executar a aplicação
if __name__ == "__main__":
    main()
