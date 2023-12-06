import logging
import sys
from typing import List

from llama_index.callbacks import LlamaDebugHandler, CallbackManager

from llama_index.core import BaseRetriever
from llama_index.embeddings import BaseEmbedding
from llama_index.indices.keyword_table import KeywordTableSimpleRetriever
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.query_engine import RouterQueryEngine, SubQuestionQueryEngine
from llama_index.schema import NodeWithScore

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext, StorageContext, SimpleKeywordTableIndex, QueryBundle, SummaryIndex,
)
from llama_index.postprocessor import LLMRerank
from llama_index.llms import HuggingFaceLLM, OpenAI
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
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])






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
                                               chunk_size=1024, llm=llm, callback_manager=callback_manager)





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







'''
子问题查询引擎：
首先将复杂的查询分解为每个相关数据源的子问题，然后收集所有中间响应并合成最终响应

目前有些问题
'''







# 加载数据
pg_essay = SimpleDirectoryReader(input_dir="../data/paul_graham/").load_data()
# build index and query engine
vector_query_engine = VectorStoreIndex.from_documents(
    pg_essay, use_async=True,
    service_context=service_context
).as_query_engine()






# 设置子问题查询引擎
# setup base query engine as tool
from llama_index.tools import QueryEngineTool, ToolMetadata
query_engine_tools = [
    QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="pg_essay",
            description="Paul Graham essay on What I Worked On",
        ),
    ),
]
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    service_context=service_context,
    use_async=True,
)








# 定义检索增强路由器查询引擎
response = query_engine.query(
    "How was Paul Grahams life different before, during, and after YC?"
)
print("输出如下")
print(response)




