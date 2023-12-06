import json
import logging
import sys


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
from llama_index.agent import ReActAgent
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext, StorageContext, SimpleKeywordTableIndex, QueryBundle, SummaryIndex, LLMPredictor,
    get_response_synthesizer,
)
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))







# # set service context
# from llama_index1.prompts import PromptTemplate
# from llama_index1 import ServiceContext
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







# This will wrap the default prompts that are internal to llama-index
# taken from https://huggingface.co/Writer/camel-5b-hf
query_wrapper_prompt = PromptTemplate(
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query_str}\n\n### Response:"
)
llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=512,
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






'''
两种方法都可以实现
'''






# print(111)
# import yaml
# import os
# with open('resources/application.yaml', 'r') as file:
#     data = yaml.safe_load(file)
# os.environ["OPENAI_API_KEY"] = data['openai']['api_key']
# os.environ["OPENAI_API_BASE"] = data['openai']['base_url']
# llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")
# service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1024)
# print(222)









# 将所有文件加载到 Document 对象中
wiki_titles = ["Boston", "Houston"]
city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()





# 定义索引集
# Build city document index
vector_indices = {}
for wiki_title in wiki_titles:
    storage_context = StorageContext.from_defaults()
    # build vector index
    vector_indices[wiki_title] = VectorStoreIndex.from_documents(
        city_docs[wiki_title],
        service_context=service_context,
        storage_context=storage_context,
    )
    # set id for vector index
    vector_indices[wiki_title].index_struct.index_id = wiki_title
    # persist to disk
    storage_context.persist(persist_dir=f"./storage/{wiki_title}")










# 为每个向量索引设置“摘要文本”
index_summaries = {}
for wiki_title in wiki_titles:
    # set summary text for city
    index_summaries[wiki_title] = (
        f"This content contains Wikipedia articles about {wiki_title}. Use this index if you need to lookup specific facts about {wiki_title}."
    )









# 在这些向量索引之上使用这些索引和摘要组成一个关键字表，以构建图形
# ComposableGraph是一个建立在所有节点上的一个索引图，用来处理对比所有图的问题
from llama_index.indices.composability import ComposableGraph
graph = ComposableGraph.from_indices(
    SimpleKeywordTableIndex,
    [index for _, index in vector_indices.items()],
    [summary for _, summary in index_summaries.items()],
    # max_keywords_per_chunk=50,
    service_context=service_context
)
# get root index
root_index = graph.get_index(
    graph.root_id
)
# set id of root index
root_index.set_index_id("compare_contrast")
root_summary = (
    "This index contains Wikipedia articles about multiple cities. Use this index if you want to compare multiple cities. "
)










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
for index in vector_indices.values():
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










# 查询该图
# define query engine
query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)
# query the graph
query_str = "Compare and contrast the arts and culture of Houston and Boston. "
response_chatgpt = query_engine.query(query_str)
print(response_chatgpt)





















