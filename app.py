import os
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate
import json # Import json to handle potential JSON string results from SQL

# --- STATE DEFINITION ---
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]

# --- STREAMLIT PAGE CONFIG & CUSTOM CSS ---
st.set_page_config(page_title="SQL QA App", layout="wide", page_icon="📊") # Changed icon to something more neutral

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="st-emotion"] {
        font-family: 'Inter', sans-serif;
        color: #333;
    }

    .main {
        background-color: #f8f9fa; /* Lighter background */
        padding: 2rem 3rem;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05); /* Subtle shadow */
    }

    h1 {
        color: #2c3e50; /* Darker heading color */
        font-weight: 700;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .stButton>button {
        background-color: #007bff; /* Primary blue */
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 12px 28px;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 4px 8px rgba(0, 123, 255, 0.2);
    }
    .stButton>button:hover {
        background-color: #0056b3; /* Darker blue on hover */
        transform: translateY(-2px); /* Slight lift effect */
        box-shadow: 0 6px 12px rgba(0, 123, 255, 0.3);
    }

    .stTextInput>div>div>input {
        padding: 12px 15px;
        border-radius: 8px;
        border: 1.5px solid #ced4da; /* Light grey border */
        font-size: 1rem;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: border-color 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #80bdff; /* Blue focus border */
        box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
        outline: none;
    }

    .stTextArea>div>div>textarea {
        padding: 12px 15px;
        border-radius: 8px;
        border: 1.5px solid #ced4da;
        font-size: 1rem;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: border-color 0.3s ease;
    }
    .stTextArea>div>div>textarea:focus {
        border-color: #80bdff;
        box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
        outline: none;
    }

    .stCode {
        background: #e9ecef; /* Light grey for code blocks */
        border-radius: 8px;
        padding: 1rem 1.5rem;
        border: 1px solid #dee2e6;
        font-family: 'Fira Code', monospace; /* A common monospaced font for code */
        font-size: 0.95rem;
        line-height: 1.4;
        overflow-x: auto;
    }

    .stMarkdown h3 {
        color: #34495e; /* Darker grey for section titles */
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0; /* Subtle line under titles */
        padding-bottom: 0.5rem;
    }

    .stSuccess {
        background-color: #d4edda; /* Light green for success messages */
        color: #155724; /* Dark green text */
        border-color: #c3e6cb;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
    }

    .stWarning {
        background-color: #fff3cd; /* Light yellow for warnings */
        color: #856404; /* Dark yellow text */
        border-color: #ffeeba;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
    }

    .stError {
        background-color: #f8d7da; /* Light red for errors */
        color: #721c24; /* Dark red text */
        border-color: #f5c6cb;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
    }

    .stInfo {
        background-color: #d1ecf1; /* Light blue for info messages */
        color: #0c5460; /* Dark blue text */
        border-color: #bee5eb;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
    }

    .stSpinner > div > div {
        color: #007bff; /* Spinner color */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Pregúntale a tu Base de Datos")
st.write(
    "Escribe tu pregunta en lenguaje natural y obtén la respuesta directa, sin preocuparte por SQL."
)

# --- API KEY INPUT ---
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    api_key = st.text_input("Ingresa tu API Key de Gemini:", type="password", help="Necesitas una API Key de Gemini para usar este servicio.")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        st.stop()

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# --- DATABASE URL INPUT ---
DATABASE_URL = os.getenv("API_URL") or st.secrets.get("API_URL")
if not DATABASE_URL:
    DATABASE_URL = st.text_input("Ingresa tu DATABASE_URL:", help="Ejemplo: sqlite:///./chinook.db")
    if DATABASE_URL:
        os.environ["API_URL"] = DATABASE_URL
    else:
        st.stop()

# Initialize DB connection (only if URL is provided)
db = None
if DATABASE_URL:
    try:
        db = SQLDatabase.from_uri(DATABASE_URL)
    except Exception as e:
        st.error(f"Error al conectar con la base de datos: {e}. Por favor, verifica la URL.")
        st.stop()
else:
    st.info("Por favor, ingresa la URL de tu base de datos para continuar.")
    st.stop()


# --- LANGCHAIN & LANGGRAPH COMPONENTS ---
system_message = """
Dado una pregunta de entrada, crea una consulta SQL sintácticamente correcta en dialecto {dialect}
para ejecutar y ayudar a encontrar la respuesta. A menos que el usuario especifique en su pregunta un
número específico de ejemplos que desea obtener, siempre limita tu consulta a
un máximo de {top_k} resultados. Puedes ordenar los resultados por una columna relevante para
devolver los ejemplos más interesantes en la base de datos.

Nunca consultes todas las columnas de una tabla específica, solo solicita las
pocas columnas relevantes dada la pregunta.

Presta atención a usar solo los nombres de columna que puedes ver en la descripción del esquema.
Ten cuidado de no consultar columnas que no existen. Además,
presta atención a qué columna está en qué tabla.

Solo usa las siguientes tablas:
{table_info}
"""

user_prompt = "Pregunta: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

def write_query(state: State):
    """Generates an SQL query based on the user's question and database schema."""
    try:
        prompt = query_prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 10,
                "table_info": db.get_table_info(),
                "input": state["question"],
            }
        )
        structured_llm = llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        return {"query": result["query"]}
    except Exception as e:
        st.error(f"Error al generar la consulta SQL: {e}")
        return {"query": None}


def execute_query(state: State):
    """Executes the generated SQL query against the database."""
    if not state["query"]:
        return {"result": "No SQL query to execute."}
    try:
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        # The tool returns a string, often a representation of the query result.
        # If it's a JSON string, we might want to parse it for better display.
        raw_result = execute_query_tool.invoke(state["query"])
        return {"result": raw_result}
    except Exception as e:
        st.error(f"Error al ejecutar la consulta SQL: {e}")
        return {"result": f"Error executing query: {e}"}

def generate_answer(state: State):
    """Generates a natural language answer based on the question, query, and result."""
    if not state["result"] or state["result"] == "No SQL query to execute.":
        return {"answer": "No se pudo obtener un resultado para generar la respuesta."}
    try:
        prompt = (
            "Dada la siguiente pregunta del usuario, la consulta SQL correspondiente "
            "y el resultado de SQL, responde a la pregunta del usuario de forma clara y concisa.\n\n"
            f'Pregunta: {state["question"]}\n'
            f'Consulta SQL: {state["query"]}\n'
            f'Resultado SQL: {state["result"]}'
        )
        response = llm.invoke(prompt)
        return {"answer": response.content}
    except Exception as e:
        st.error(f"Error al generar la respuesta: {e}")
        return {"answer": None}

# --- LANGGRAPH SETUP ---
graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()

# --- STREAMLIT UI ---
st.markdown("---") # Separator

with st.form("sql_qa_form", clear_on_submit=True):
    question = st.text_input("Escribe tu pregunta sobre la base de datos:", placeholder="Ej: ¿Cuántos clientes hay en la tabla de clientes?", key="user_question")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2: # Center the button
        submitted = st.form_submit_button("Obtener respuesta")

if submitted and question:
    st.markdown("### Procesando tu solicitud...")
    with st.spinner("Generando consulta y obteniendo respuesta..."):
        config = {"configurable": {"thread_id": "user_thread_1"}}
        state = {"question": question}
        query = None
        result = None
        answer = None

        try:
            # Stream the steps and capture the final state
            final_state = None
            for step in graph.stream(state, config, stream_mode="updates"):
                if "write_query" in step:
                    query = step["write_query"].get("query")
                if "execute_query" in step:
                    result = step["execute_query"].get("result")
                if "generate_answer" in step:
                    answer = step["generate_answer"].get("answer")
                final_state = step # Keep track of the last state for full output

            # Display results
            if query:
                st.markdown("### Consulta SQL generada:")
                st.code(query, language="sql")
            else:
                st.error("❌ No se pudo generar una consulta SQL válida.")

            if result:
                st.markdown("### Resultado de la consulta:")
                try:
                    # Attempt to parse result as JSON for better display if it's a JSON string
                    parsed_result = json.loads(result)
                    st.json(parsed_result)
                except json.JSONDecodeError:
                    # If not JSON, display as plain text
                    st.text(result)
            else:
                st.error("❌ No se obtuvo resultado de la base de datos.")

            if answer:
                st.markdown("### Respuesta final:")
                st.success(answer)
            else:
                st.warning("⚠️ No se pudo generar una respuesta final.")

        except Exception as e:
            st.error(f"Ocurrió un error inesperado durante el procesamiento: {e}")
            st.info("Por favor, intenta con otra pregunta o verifica tu configuración (API Key, Database URL).")
elif submitted and not question:
    st.warning("Por favor, escribe una pregunta antes de enviar.")
else:
    st.info("Escribe una pregunta y presiona 'Obtener respuesta' para comenzar a interactuar con tu base de datos.")
