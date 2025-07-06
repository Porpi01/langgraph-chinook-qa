import os
from dotenv import load_dotenv
import streamlit as st
from langchain.sql_database import SQLDatabase
from langchain.chains import SQLDatabaseChain
from langchain.llms import Gemini 

load_dotenv()  

DATABASE_URL = os.getenv("APi_URL")

db = SQLDatabase.from_uri(DATABASE_URL)

llm = Gemini(temperature=0)
db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

def main():
    st.title("Consulta a base Chinook con LangChain")
    
    pregunta = st.text_input("Pregunta sobre la base de datos Chinook:")

    if pregunta:
        respuesta = db_chain.run(pregunta)
        st.write("Respuesta:")
        st.write(respuesta)

if __name__ == "__main__":
    main()