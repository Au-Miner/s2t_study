import logging
import sys
from typing import List

from llama_index.core import BaseRetriever
from llama_index.embeddings import BaseEmbedding
from llama_index.indices.keyword_table import KeywordTableSimpleRetriever
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.query_engine import RouterQueryEngine
from llama_index.schema import NodeWithScore

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext, StorageContext, SimpleKeywordTableIndex, QueryBundle, SummaryIndex,
)
from llama_index.postprocessor import LLMRerank
from llama_index.llms import HuggingFaceLLM



'''
定义一个自定义路由器查询引擎，它从多个候选查询引擎中选择一个来执行查询
'''



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






# setup prompts - specific to StableLM
from llama_index.prompts import PromptTemplate
from llama_index.selectors import LLMSingleSelector
from llama_index.selectors.pydantic_selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)

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








# 加载数据
documents = SimpleDirectoryReader("./data/paul_graham").load_data()
# initialize service context (set chunk size)
nodes = service_context.node_parser.get_nodes_from_documents(documents)
# initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)










# 定义相同数据上的汇总索引和向量索引
summary_index = SummaryIndex(nodes, storage_context=storage_context, service_context=service_context)
vector_index = VectorStoreIndex(nodes, storage_context=storage_context, service_context=service_context)







# 为这些索引定义查询引擎和工具
list_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize", use_async=True
)
vector_query_engine = vector_index.as_query_engine(
    response_mode="tree_summarize", use_async=True
)
from llama_index.tools.query_engine import QueryEngineTool
list_tool = QueryEngineTool.from_defaults(
    query_engine=list_query_engine,
    description=(
        "Useful for summarization questions related to Paul Graham eassy on"
        " What I Worked On."
    ),
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from Paul Graham essay on What"
        " I Worked On."
    ),
)








# 定义检索增强路由器查询引擎
from llama_index import VectorStoreIndex
from llama_index.objects import ObjectIndex, SimpleToolNodeMapping
tool_mapping = SimpleToolNodeMapping.from_objects([list_tool, vector_tool])
obj_index = ObjectIndex.from_objects(
    [list_tool, vector_tool],
    tool_mapping,
    VectorStoreIndex,
    service_context=service_context
)
from llama_index.query_engine import ToolRetrieverRouterQueryEngine
query_engine = ToolRetrieverRouterQueryEngine(obj_index.as_retriever(), service_context=service_context)
response = query_engine.query("What is a biography of the author's life?")
print(response)







