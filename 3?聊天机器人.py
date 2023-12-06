import json
import logging
import sys
import yaml
import os

from llama_index.indices.query.query_transform.base import StepDecomposeQueryTransform
from llama_index.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core import BaseRetriever
from llama_index.embeddings import BaseEmbedding
from llama_index.indices.keyword_table import KeywordTableSimpleRetriever
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.query_engine import RouterQueryEngine, SubQuestionQueryEngine, FLAREInstructQueryEngine
from llama_index.schema import NodeWithScore
from llama_index.prompts import PromptTemplate
from typing import Sequence, List, Optional
from llama_index.tools import BaseTool, FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.llms import OpenAI, ChatMessage
from llama_index.selectors import LLMSingleSelector
from llama_index.postprocessor import LLMRerank
from llama_index.llms import HuggingFaceLLM, OpenAI
from llama_index.agent import ReActAgent, OpenAIAgent
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext, StorageContext, SimpleKeywordTableIndex, QueryBundle, SummaryIndex, LLMPredictor,
    get_response_synthesizer,
)





# setup prompts - specific to StableLM
from llama_index.prompts import PromptTemplate
# This will wrap the default prompts that are internal to llama-index
# taken from https://huggingface.co/Writer/camel-5b-hf
query_wrapper_prompt = PromptTemplate(
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query_str}\n\n### Response:"
)
llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.25, "do_sample": True},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="/home/qcsun/s2t/S2T_project/Camel-5b",
    model_name="/home/qcsun/s2t/S2T_project/Camel-5b",
    device_map="auto",
    tokenizer_kwargs={"max_length": 2048},
    # uncomment this if using CUDA to reduce memory usage
    # model_kwargs={"torch_dtype": torch.float16}
)
# initialize service context (set chunk size)
service_context = ServiceContext.from_defaults(embed_model="local:/home/qcsun/s2t/S2T_project/bge-large-en-v1.5",
                                               chunk_size=1024, llm=llm)








#
# print(111)
# with open('resources/application.yaml', 'r') as file:
#     data = yaml.safe_load(file)
# os.environ["OPENAI_API_KEY"] = data['openai']['api_key']
# os.environ["OPENAI_API_BASE"] = data['openai']['base_url']
# llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")
# service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1024)
# print(222)





years = ["2018", "2019"]
doc_set = {}
all_docs = []
for year in years:
    year_docs = SimpleDirectoryReader(
        input_files=[f"data/{year}.txt"]
    ).load_data()
    # insert year metadata into each year
    for d in year_docs:
        d.metadata = {"year": year}
    doc_set[year] = year_docs
    all_docs.extend(year_docs)








# initialize simple vector indices
from llama_index import VectorStoreIndex, StorageContext
index_set = {}
for year in years:
    storage_context = StorageContext.from_defaults()
    cur_index = VectorStoreIndex.from_documents(
        doc_set[year],
        service_context=service_context,
        storage_context=storage_context,
    )
    index_set[year] = cur_index
    index_set[year].index_struct.index_id = year
    storage_context.persist(persist_dir=f"./storage/{year}")






'''
版本1目前问题：
使用SubQuestionQueryEngine只能适配agent，与10问题一致，在其他llm上无法适配
'''
# # 设置子问题查询引擎以综合 10-K 份申请的答案
# individual_query_engine_tools = [
#     QueryEngineTool(
#         query_engine=index_set[year].as_query_engine(),
#         metadata=ToolMetadata(
#             name=f"vector_index_{year}",
#             description=f"useful for when you want to answer queries about the {year} SEC 10-K for Uber",
#         ),
#     )
#     for year in years
# ]
# query_engine = SubQuestionQueryEngine.from_defaults(
#     query_engine_tools=individual_query_engine_tools,
#     service_context=service_context,
# )
# # query_engine_tool = QueryEngineTool(
# #     query_engine=query_engine,
# #     metadata=ToolMetadata(
# #         name="sub_question_query_engine",
# #         description="useful for when you want to answer queries that require analyzing multiple SEC 10-K documents for Uber",
# #     ),
# # )
# # tools = individual_query_engine_tools + [query_engine_tool]
# # agent = OpenAIAgent.from_tools(tools, verbose=True)
# query_str = "Please compare wql's scores between 2018 and 2019."
# # response_chatgpt = agent.query(query_str)
# response_chatgpt = query_engine.query(query_str)
# print(response_chatgpt)








'''
版本2目前问题：
没有找到对应文档
'''
index_summaries = {}
for year in years:
    # set summary text for city
    index_summaries[year] = (
        f"This content contains knowledge about wql in {year}. "
    )

from llama_index.indices.composability import ComposableGraph

graph = ComposableGraph.from_indices(
    SimpleKeywordTableIndex,
    [index for _, index in index_set.items()],
    [summary for _, summary in index_summaries.items()],
    max_keywords_per_chunk=50,
    service_context=service_context
)
root_index = graph.get_index(
    graph.root_id
)
root_index.set_index_id("years")

# define decompose_transform
from llama_index import LLMPredictor
from llama_index.indices.query.query_transform.base import (
    DecomposeQueryTransform,
)


decompose_transform = DecomposeQueryTransform(
    LLMPredictor(llm=llm), verbose=True
)

# define custom query engines
from llama_index.query_engine.transform_query_engine import (
    TransformQueryEngine,
)

custom_query_engines = {}
for index in index_set.values():
    query_engine = index.as_query_engine(service_context=service_context)
    query_engine = TransformQueryEngine(
        query_engine,
        query_transform=decompose_transform,
        transform_metadata={"index_summary": index.index_struct.summary},
    )
    custom_query_engines[index.index_id] = query_engine
custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
    retriever_mode="simple",
    response_mode="tree_summarize",
    service_context=service_context,
)

# define query engine
query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)

# query the graph
query_str = "what is wql's age in 2019?"
response_chatgpt = query_engine.query(query_str)
print(response_chatgpt)