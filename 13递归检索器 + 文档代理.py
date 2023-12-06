import logging
import sys
from typing import List, Optional

from llama_index.indices.query.query_transform.base import StepDecomposeQueryTransform

from llama_index.callbacks import LlamaDebugHandler, CallbackManager

from llama_index.core import BaseRetriever
from llama_index.embeddings import BaseEmbedding
from llama_index.indices.keyword_table import KeywordTableSimpleRetriever
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.query_engine import RouterQueryEngine, SubQuestionQueryEngine
from llama_index.schema import NodeWithScore, IndexNode

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext, StorageContext, SimpleKeywordTableIndex, QueryBundle, SummaryIndex, LLMPredictor,
    get_response_synthesizer,
)
from llama_index.postprocessor import LLMRerank
from llama_index.llms import HuggingFaceLLM, OpenAI
from llama_index.tools import QueryEngineTool, ToolMetadata

# from llama_index1.llms import HuggingFaceLLM
# from llama_index1.prompts import PromptTemplate
# from llama_index1 import ServiceContext
# from llama_index1.selectors import LLMSingleSelector
#
# system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
# - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
# - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
# - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
# - StableLM will refuse to participate in anything that could harm a human.
# """
# query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")
# llm = HuggingFaceLLM(
#     context_window=4096,
#     max_new_tokens=256,
#     generate_kwargs={"temperature": 0.7, "do_sample": False},
#     system_prompt=system_prompt,
#     query_wrapper_prompt=query_wrapper_prompt,
#     tokenizer_name="/home/qcsun/s2t/S2T_project/stablelm-tuned-alpha-3b",
#     model_name="/home/qcsun/s2t/S2T_project/stablelm-tuned-alpha-3b",
#     device_map="auto",
#     stopping_ids=[50278, 50279, 50277, 1, 0],
#     tokenizer_kwargs={"max_length": 4096},
#     # uncomment this if using CUDA to reduce memory usage
#     # model_kwargs={"torch_dtype": torch.float16}
# )
# service_context = ServiceContext.from_defaults(embed_model="local:/home/qcsun/s2t/S2T_project/bge-large-en", chunk_size=1024, llm=llm)









# from llama_index.prompts import PromptTemplate
# # This will wrap the default prompts that are internal to llama-index
# # taken from https://huggingface.co/Writer/camel-5b-hf
# query_wrapper_prompt = PromptTemplate(
#     "Below is an instruction that describes a task. "
#     "Write a response that appropriately completes the request.\n\n"
#     "### Instruction:\n{query_str}\n\n### Response:"
# )
# llm = HuggingFaceLLM(
#     context_window=2048,
#     max_new_tokens=512,
#     generate_kwargs={"temperature": 0.25, "do_sample": True},
#     query_wrapper_prompt=query_wrapper_prompt,
#     tokenizer_name="/home/qcsun/s2t/S2T_project/Camel-5b",
#     model_name="/home/qcsun/s2t/S2T_project/Camel-5b",
#     device_map="auto",
#     tokenizer_kwargs={"max_length": 2048},
#     # uncomment this if using CUDA to reduce memory usage
#     # model_kwargs={"torch_dtype": torch.float16}
# )
# # initialize service context (set chunk size)
# service_context = ServiceContext.from_defaults(embed_model="local:/home/qcsun/s2t/S2T_project/bge-large-en-v1.5",
#                                                chunk_size=1024, llm=llm)







print(111)
import yaml
import os
with open('resources/application.yaml', 'r') as file:
    data = yaml.safe_load(file)
os.environ["OPENAI_API_KEY"] = data['openai']['api_key']
os.environ["OPENAI_API_BASE"] = data['openai']['base_url']
print(222)







'''
对不同文件创建对应的概括索引和向量索引，每个文件创建对应的代理
创建递归检索器检索所有代理

因为使用了代理，目前仅支持openai
'''





# 设置和下载数据
wiki_titles = ["Toronto", "Seattle", "Chicago", "Boston", "Houston"]
from pathlib import Path
import requests
for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            # 'exintro': True,
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]
    data_path = Path("data")
    if not data_path.exists():
        Path.mkdir(data_path)
    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)
city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")
service_context = ServiceContext.from_defaults(llm=llm)






# 为每个文档构建文档代理
from llama_index.agent import OpenAIAgent
# Build agents dictionary
agents = {}
for wiki_title in wiki_titles:
    # build vector index
    vector_index = VectorStoreIndex.from_documents(
        city_docs[wiki_title], service_context=service_context
    )
    # build summary index
    summary_index = SummaryIndex.from_documents(
        city_docs[wiki_title], service_context=service_context
    )
    # define query engines
    vector_query_engine = vector_index.as_query_engine()
    list_query_engine = summary_index.as_query_engine()
    # define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=(
                    "Useful for summarization questions related to"
                    f" {wiki_title}"
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=list_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=(
                    f"Useful for retrieving specific context from {wiki_title}"
                ),
            ),
        ),
    ]
    # build agent
    function_llm = OpenAI(model="gpt-3.5-turbo-0613")
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
    )
    agents[wiki_title] = agent







# 基于这些代理构建递归检索器
# define top-level nodes
nodes = []
for wiki_title in wiki_titles:
    # define index node that links to these agents
    wiki_summary = (
        f"This content contains Wikipedia articles about {wiki_title}. Use"
        " this index if you need to lookup specific facts about"
        f" {wiki_title}.\nDo not use this index if you want to analyze"
        " multiple cities."
    )
    node = IndexNode(text=wiki_summary, index_id=wiki_title)
    nodes.append(node)
# define top-level retriever
vector_index = VectorStoreIndex(nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=1)
# define recursive retriever
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import get_response_synthesizer
# note: can pass `agents` dict as `query_engine_dict` since every agent can be used as a query engine
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=agents,
    verbose=True,
)




# 定义完整查询引擎
response_synthesizer = get_response_synthesizer(
    # service_context=service_context,
    response_mode="compact",
)
query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever,
    response_synthesizer=response_synthesizer,
    service_context=service_context,
)
response = query_engine.query("Tell me about the sports teams in Boston")
print(response)










