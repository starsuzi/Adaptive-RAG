from time import perf_counter
from fastapi import FastAPI, Request

from unified_retriever import UnifiedRetriever

retriever = UnifiedRetriever(host="http://localhost/", port=9200)

app = FastAPI()


@app.get("/")
async def index():
    return {"message": "Hello! This is a retriever server."}


@app.post("/retrieve/")
async def retrieve(arguments: Request):  # see the corresponding method in unified_retriever.py
    arguments = await arguments.json()
    retrieval_method = arguments.pop("retrieval_method")
    assert retrieval_method in ("retrieve_from_elasticsearch")
    start_time = perf_counter()
    retrieval = getattr(retriever, retrieval_method)(**arguments)
    end_time = perf_counter()
    time_in_seconds = round(end_time - start_time, 1)
    return {"retrieval": retrieval, "time_in_seconds": time_in_seconds}
