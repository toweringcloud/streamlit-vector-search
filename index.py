import streamlit as st
from dotenv import dotenv_values
from elasticsearch import Elasticsearch
from langchain.callbacks.base import BaseCallbackHandler
from langchain.embeddings import CacheBackedEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import ElasticsearchStore
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_unstructured import UnstructuredLoader
from opensearchpy import OpenSearch
from pathlib import Path


st.set_page_config(
    page_title="::: Vector Search Chatbot :::",
    page_icon="📜",
)
st.title("Vector Search 😎")

st.markdown(
    """         
        Use this chatbot to ask questions about your document.

        1. Input your OpenAI API Key on the sidebar.
        2. Choose an AI model (gpt-4o-mini, ...).
        3. Upload a document file (txt | doc | pdf).
        4. Ask questions related to the document.
    """
)
st.divider()

with st.sidebar:
    # Input LLM API Key
    openai_api_key = st.text_input("Input your OpenAI API Key", type="password")

    # Select AI Model
    selected_model = st.selectbox(
        "Choose your AI Model",
        ("gpt-4.1-nano", "gpt-4o-mini"),
    )

    # Select Vector Storage
    selected_storage = st.selectbox(
        "Choose your Vector Storage",
        ("LocalFileStore", "OpenSearch", "Elasticsearch"),
    )

    # Upload Document File
    file = st.file_uploader(
        "Upload a text file",
        # type=["docx", "pdf", "txt"],
        type=["txt"],
    )

    # Link to Github Repo
    st.markdown("---")
    github_link = (
        "https://github.com/toweringcloud/streamlit-vector-search/blob/main/index.py"
    )
    badge_link = "https://badgen.net/badge/icon/GitHub?icon=github&label"
    st.write(f"[![Repo]({badge_link})]({github_link})")


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


# Load Configuration
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    OPENSEARCH_HOST = st.secrets["OPENSEARCH_HOST"]
    OPENSEARCH_PORT = st.secrets["OPENSEARCH_PORT"]
    OPENSEARCH_USERNAME = st.secrets["OPENSEARCH_USERNAME"]
    OPENSEARCH_PASSWORD = st.secrets["OPENSEARCH_PASSWORD"]
    OPENSEARCH_INDEX_NAME = st.secrets["OPENSEARCH_INDEX_NAME"]
    OPENSEARCH_CACHE_INDEX_NAME = st.secrets["OPENSEARCH_CACHE_INDEX_NAME"]
    ELASTICSEARCH_HOST = st.secrets["ELASTICSEARCH_HOST"]
    ELASTICSEARCH_PORT = st.secrets["ELASTICSEARCH_PORT"]
    ELASTICSEARCH_USERNAME = st.secrets["ELASTICSEARCH_USERNAME"]
    ELASTICSEARCH_PASSWORD = st.secrets["ELASTICSEARCH_PASSWORD"]
    ELASTICSEARCH_INDEX_NAME = st.secrets["ELASTICSEARCH_INDEX_NAME"]
    ELASTICSEARCH_CACHE_INDEX_NAME = st.secrets["ELASTICSEARCH_CACHE_INDEX_NAME"]
else:
    config = dotenv_values(".env")
    OPENAI_API_KEY = config["OPENAI_API_KEY"]
    OPENSEARCH_HOST = config["OPENSEARCH_HOST"]
    OPENSEARCH_PORT = config["OPENSEARCH_PORT"]
    OPENSEARCH_USERNAME = config["OPENSEARCH_USERNAME"]
    OPENSEARCH_PASSWORD = config["OPENSEARCH_PASSWORD"]
    OPENSEARCH_INDEX_NAME = config["OPENSEARCH_INDEX_NAME"]
    OPENSEARCH_CACHE_INDEX_NAME = config["OPENSEARCH_CACHE_INDEX_NAME"]
    ELASTICSEARCH_HOST = config["ELASTICSEARCH_HOST"]
    ELASTICSEARCH_PORT = config["ELASTICSEARCH_PORT"]
    ELASTICSEARCH_USERNAME = config["ELASTICSEARCH_USERNAME"]
    ELASTICSEARCH_PASSWORD = config["ELASTICSEARCH_PASSWORD"]
    ELASTICSEARCH_INDEX_NAME = config["ELASTICSEARCH_INDEX_NAME"]
    ELASTICSEARCH_CACHE_INDEX_NAME = config["ELASTICSEARCH_CACHE_INDEX_NAME"]

# Initiate Session Data
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# OpenSearch 클라이언트 초기화 함수
@st.cache_resource
def get_opensearch_client():
    try:
        client = OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
            http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
            use_ssl=True,  # HTTPS 사용 시 True
            verify_certs=False,  # 개발/테스트 시 False, 프로덕션에서는 True 및 인증서 설정 필요
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        # 클라이언트가 제대로 연결되었는지 확인
        client.info()
        return client
    except Exception as e:
        st.error(f"OpenSearch 클라이언트 초기화 중 오류 발생: {e}")
        st.stop()


# Elasticsearch 클라이언트 초기화 함수
@st.cache_resource
def get_elasticsearch_client():
    try:
        client = Elasticsearch(
            hosts=[{"host": ELASTICSEARCH_HOST, "port": ELASTICSEARCH_PORT}],
            http_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD),
            use_ssl=True,  # HTTPS 사용 시 True
            verify_certs=False,  # 개발/테스트 시 False, 프로덕션에서는 True 및 인증서 설정 필요
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        # 클라이언트가 제대로 연결되었는지 확인
        client.info()
        return client
    except Exception as e:
        st.error(f"Elasticsearch 클라이언트 초기화 중 오류 발생: {e}")
        st.stop()


@st.cache_resource(show_spinner=f"Embedding file and storing in {selected_storage}...")
def embed_file(file):
    # 1. 파일 내용 로컬에 저장
    # 주의: .files 폴더는 Streamlit 앱 재시작 시 사라질 수 있으므로,
    # 장기적인 파일 저장이 필요하다면 다른 스토리지 솔루션(S3 등)을 고려해야 합니다.
    file_content = file.read()
    file_path = f"./.files/{file.name}"
    Path("./.files").mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb+") as f:
        f.write(file_content)

    # 2. 텍스트 분할
    loader = UnstructuredLoader(file_path=file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(text_splitter=splitter)

    # 3. 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # 4. 임베딩 캐시(ByteStore) 설정
    # OpenSearch 자체를 Key-Value 스토어로 활용하는 커스텀 ByteStore를 구현하면 임베딩 캐시로 설정 가능
    if selected_storage == "LocalFileStore" or selected_storage == "OpenSearch":
        cache_path = "./.cache"
        embedding_path = f"{cache_path}/local"
        Path(embedding_path).mkdir(parents=True, exist_ok=True)
        embedding_cache_dir = LocalFileStore(embedding_path)
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, embedding_cache_dir
        )
    else:
        client = get_elasticsearch_client()
        embedding_cache_store = ElasticsearchStore(
            client=client,  # Elasticsearch 클라이언트
            es_url=f"http://{ELASTICSEARCH_HOST}:{ELASTICSEARCH_PORT}",
            index_name=ELASTICSEARCH_CACHE_INDEX_NAME,  # 임베딩 캐시 전용 인덱스
        )
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, embedding_cache_store
        )

    # 5. 벡터 저장소 설정
    if selected_storage == "LocalFileStore":
        vectorstore = FAISS.from_documents(
            docs,
            cached_embeddings,
        )
    elif selected_storage == "OpenSearch":
        vectorstore = OpenSearchVectorSearch.from_documents(
            docs,
            cached_embeddings,
            opensearch_url=f"https://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}",
            http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),  # 인증 정보
            index_name=OPENSEARCH_INDEX_NAME,  # 문서 벡터를 저장할 인덱스
            use_ssl=True,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            # bulk_size=1000 # 대량 인덱싱 시 최적화 옵션 (선택 사항)
        )
    elif selected_storage == "Elasticsearch":
        vectorstore = ElasticsearchStore.from_documents(
            docs,
            cached_embeddings,
            es_url=f"http://{ELASTICSEARCH_HOST}:{ELASTICSEARCH_PORT}",
            index_name=ELASTICSEARCH_INDEX_NAME,
        )
    else:
        st.error("Invalid storage option selected.")
        st.stop()

    # 6. Retriever 생성
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. 
            If you don't know the answer just say you don't know. 
            DON't make anything up.

            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


def main():
    if not openai_api_key:
        return

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model=selected_model,
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )

    if file:
        retriever = embed_file(file)

        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()

        message = st.chat_input("Ask anything about your file.....")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                chain.invoke(message)

    else:
        st.session_state["messages"] = []
        return


try:
    main()

except Exception as e:
    st.error("Check your OpenAI API Key or File")
    st.write(e)
