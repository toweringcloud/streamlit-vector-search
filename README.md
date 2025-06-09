# streamlit-vector-search
rag based vector search using postgresql, opensearch and elasticsearch with python v3.12.8 + streamlit v1.45.1 + langchain v0.3.25

## how to run

### setup

-   install python 3.10 ~ 3.12 LTS and add system path on python & pip

```sh
$ python --version
Python 3.12.8

$ python -m pip install --upgrade pip
$ pip --version
pip 25.1.1 from /usr/lib/python3/dist-packages/pip

$ docker -v
Docker version 28.1.1, build 4eba377

$ docker-compose -v
Docker Compose version v2.35.1-desktop.1

$ docker compose up -d -f docker-compose-opensearch.yml
$ curl -k -u admin:admin https://localhost:9200

$ docker compose up -d -f docker-compose-elasticsearch.yml
$ curl -k https://localhost:19200
```

### config

-   runtime option A: venv

```sh
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ pip list
$ deactivate
```

-   runtime option B: poetry

```sh
$ pip install poetry==1.8.5
$ poetry shell
$ poetry install
$ poetry show
$ exit
```

### launch

-   update runtime secrets in streamlit env

```sh
$ vi .streamlit/secrets.toml
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
...
OPENSEARCH_INDEX_NAME = "streamlit_documents"
OPENSEARCH_CACHE_INDEX_NAME = "embedding_cache"
```

-   run streamlit app in root environment

```sh
$ streamlit run index.py
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```
