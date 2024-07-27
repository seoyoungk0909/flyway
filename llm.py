from llama_parse import LlamaParse

from chromadb.api.types import EmbeddingFunction, Documents
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_groq import ChatGroq

import joblib
import os
import nest_asyncio  # noqa: E402
import nltk


# Hide warning message
os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.download("averaged_perceptron_tagger")
nest_asyncio.apply()


# Convert the parsed data into a format that can be used by the vectorstore
class ChromaEmbeddingsAdapter(Embeddings):
    def __init__(self, ef: EmbeddingFunction):
        self.ef = ef

    def embed_documents(self, texts):
        return self.ef(texts)

    def embed_query(self, query):
        return self.ef([query])[0]


class LangChainEmbeddingAdapter(EmbeddingFunction[Documents]):
    def __init__(self, ef: Embeddings):
        self.ef = ef

    def __call__(self, input: Documents) -> Embeddings:
        # LC EFs also have embed_query but Chroma doesn't support that so we just use embed_documents
        # TODO: better type checking
        return self.ef.embed_documents(input)


def load_or_parse_data(llamaparse_api_key):
    """
    This function loads the parsed data from a file if it exists, otherwise it performs the parsing step and stores the result in a file.
    It utilises llamaparse to parse data from a pdf file.
    """
    data_file = "data/parsed_data.pkl"

    if os.path.exists(data_file):
        try:
            # Load the parsed data from the file
            parsed_data = joblib.load(data_file)
            print("Data loaded successfully from file.")
        except (
            EOFError,
            FileNotFoundError,
            joblib.externals.loky.process_executor._RemoteTraceback,
        ) as e:
            print(f"Error loading data: {e}")
            parsed_data = None
    else:
        parsed_data = None

    pdf_files = []
    for root, _, files in os.walk("data/travel_guides"):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    if parsed_data is None:
        # Perform the parsing step and store the result in llama_parse_documents
        parsed_data = []
        parsingInstruction = """
        The provided document is a tour guide of cities and countries around the world. These document provides details of what activities a tourists can do.
        They include information about the places to visit, things to do, food to eat, and the culture of Seoul.
        They contain lots of images, maps, and text. Try to be precise while answering the questions.
        """

        parser = LlamaParse(
            api_key=llamaparse_api_key,
            result_type="markdown",
            parsing_instruction=parsingInstruction,
            max_timeout=5000,
        )
        # llama_parse_documents = parser.load_data("data/travel_guides/South_Korea.pdf")
        for pdf_file in pdf_files:
            print(f"Parsing {pdf_file}...")
            llama_parse_documents = parser.load_data(pdf_file)
            if not llama_parse_documents:
                print(f"Parsing returned an empty result for {pdf_file}.")
                continue
            parsed_data.append(llama_parse_documents)

        # if not llama_parse_documents:
        #     raise ValueError("Parsing returned an empty result.")

        # Save the parsed data to a file
        print("Saving the parse results in .pkl format ..........")
        try:
            joblib.dump(parsed_data, data_file)
            print("Data saved successfully.")
        except Exception as e:
            print(f"Error saving data: {e}")

        # Set the parsed data to the variable
        # parsed_data = llama_parse_documents

    return parsed_data


def initialise_vectorstore(llamaparse_api_key):
    """
    Initialise llm by loading or creating a vector database and feeding that to llm model as a retriever.

    For vector database, it loads urls,
    splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    and persists the embeddings into a Chroma vector database.

    For llm model, it uses ChatGroq with a custom prompt template and a retrieval question answering chain.
    """

    persist_directory = "chroma_db_llamaparse"
    collection_name = "rag"

    # Initialize Embeddings
    embed_model = ChromaEmbeddingsAdapter(
        SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    )

    # Load a chroma vector db if it already exists
    if os.path.exists(persist_directory):
        print("Loading existing vector database.")
        vs = Chroma(
            persist_directory=persist_directory,
            embedding_function=embed_model,
            collection_name=collection_name,
        )

    # Create a new vector database if it doesn't exist
    else:
        print("Creating new vector database.")
        # Call the function to either load or parse the data
        llama_parse_documents = load_or_parse_data(llamaparse_api_key)
        # print(llama_parse_documents[0].text[:100])

        with open("data/output.md", "a") as f:  # Open the file in append mode ('a')
            for pdf in llama_parse_documents:
                print(f"writing to file...")
                for doc in pdf:
                    f.write(doc.text + "\n")

        markdown_path = "data/output.md"
        loader = UnstructuredMarkdownLoader(markdown_path)

        documents = loader.load()
        # Split loaded documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)

        print(f"length of documents loaded: {len(documents)}")
        print(f"total number of document chunks generated :{len(docs)}")

        # Create and persist a Chroma vector database from the chunked documents
        vs = Chroma.from_documents(
            documents=docs,
            embedding=embed_model,
            persist_directory=persist_directory,  # Local mode with in-memory storage only
            collection_name=collection_name,
        )

        print("Vector DB created successfully !")

    # Create a retriever from the vector database
    retriever = vs.as_retriever()

    return retriever


# Retrieve response by invoking the QA Chain
def get_ai_response(user_input, retriever, groq_api_key, store, session_id):
    # Create custom prompt template
    chat_model = ChatGroq(
        temperature=0,
        model_name="mixtral-8x7b-32768",
        api_key=groq_api_key,
    )

    # Retrieval chain with history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        chat_model, retriever, contextualize_q_prompt
    )

    # Custom prompt for question answering chain
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        Use the following pieces of information to answer the question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Only return the helpful answer below and nothing else.
        """,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)

    # Combine QA chain and retrieval chain into rag_chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Helper function to retrieve session history
    def get_session_history(session_id):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )["answer"]

    return response


# ______________________ For debugging LLM ______________________
# from dotenv import load_dotenv
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
# llamaparse_api_key = os.getenv("LLAMAPARSE_API_KEY")
# if __name__ == "__main__":
#     store = {}
#     retriever = initialise_vectorstore(llamaparse_api_key)

#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             break
#         response = get_ai_response(user_input, retriever, groq_api_key, store)
#         print("Assistant:", response)
