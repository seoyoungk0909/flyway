from llama_parse import LlamaParse

from chromadb.api.types import EmbeddingFunction, Documents
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

import joblib
import os
import nest_asyncio  # noqa: E402
import nltk

# Hide API Key
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llamaparse_api_key = os.getenv("LLAMAPARSE_API_KEY")

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


def load_or_parse_data():
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

    if parsed_data is None:
        # Perform the parsing step and store the result in llama_parse_documents
        parsingInstruction = """
        The provided document is a tour guide of South Korea. This document provides details of what activities a tuorists can do in Seoul.
        It includes information about the places to visit, things to do, food to eat, and the culture of Seoul.
        It contains lots of images, maps, and text. Try to be precise while answering the questions.
        """

        parser = LlamaParse(
            api_key=llamaparse_api_key,
            result_type="markdown",
            parsing_instruction=parsingInstruction,
            max_timeout=5000,
        )
        llama_parse_documents = parser.load_data("data/travel_guides/South_Korea.pdf")

        if not llama_parse_documents:
            raise ValueError("Parsing returned an empty result.")

        # Save the parsed data to a file
        print("Saving the parse results in .pkl format ..........")
        try:
            joblib.dump(llama_parse_documents, data_file)
            print("Data saved successfully.")
        except Exception as e:
            print(f"Error saving data: {e}")

        # Set the parsed data to the variable
        parsed_data = llama_parse_documents

    return parsed_data


def initialise_llm():
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
        llama_parse_documents = load_or_parse_data()
        # print(llama_parse_documents[0].text[:100])

        with open("data/output.md", "a") as f:  # Open the file in append mode ('a')
            for doc in llama_parse_documents:
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

    # Initialize llm model
    chat_model = ChatGroq(
        temperature=0,
        model_name="mixtral-8x7b-32768",
        api_key=groq_api_key,
    )

    # Create a retriever from the vector database
    retriever = vs.as_retriever()

    # Create custom prompt template
    custom_prompt_template = """Use the following pieces of information to answer the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """

    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )

    # Instantiate retrieval question answering chain
    qa = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return qa


# Retrieve response by invoking the QA Chain
def get_ai_response(user_input):
    qa = initialise_llm()

    # Invoke the QA Chain
    response = qa.invoke({"query": user_input})
    return response["result"]
