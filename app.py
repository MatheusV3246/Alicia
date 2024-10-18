import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

st.set_page_config(
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_title="Alicia | 3246",
    page_icon="💁🏻‍♀️"
)
# Definir a classe ChatLlama
class ChatLlama():
    def __init__(self):
        load_dotenv()  # Carregar variáveis de ambiente do .env
        
        # Configurar o modelo Llama (verifique se as variáveis do .env estão corretas)
        self.llm = ChatGroq(temperature=0.2, model_name="llama3-8b-8192") 

    def processar_resposta(self, transcricao):
        """Processar a resposta da LLM."""
        if not transcricao.strip():
            return "A entrada está vazia. Por favor, digite algo."
        
        try:
            # Chama o modelo Llama para gerar uma resposta
            contexto="""
            Responda como você fosse um consultor de produtos e serviços da Credseguro, 
            seja o mais ilustrativo e bem educado possível 
            """
            resposta_llm = self.llm.invoke(f"{contexto}: {transcricao}").content
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
    
    
    # Inicializar o modelo ChatLlama
    chat_model = ChatLlama()

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
        
        # Processar a resposta do modelo Llama
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
