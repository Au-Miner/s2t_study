import logging
import sys
import time
from typing import List

from llama_index.core import BaseRetriever
from llama_index.embeddings import BaseEmbedding
from llama_index.indices.keyword_table import KeywordTableSimpleRetriever
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.schema import NodeWithScore

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext, StorageContext, SimpleKeywordTableIndex, QueryBundle,
)
from llama_index.postprocessor import LLMRerank
from llama_index.llms import HuggingFaceLLM, OpenAI

# # setup prompts - specific to StableLM
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
#     max_new_tokens=256,
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
with open('../resources/application.yaml', 'r') as file:
    data = yaml.safe_load(file)
os.environ["OPENAI_API_KEY"] = data['openai']['api_key']
os.environ["OPENAI_API_BASE"] = data['openai']['base_url']
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1024)
print(222)









# 加载数据
documents = SimpleDirectoryReader("../data/paul_graham").load_data()
node_parser = service_context.node_parser

nodes = node_parser.get_nodes_from_documents(documents)
# initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)






# 在相同数据上定义向量索引和关键字表索引
vector_index = VectorStoreIndex(nodes,
                                storage_context=storage_context,
                                service_context=service_context)
keyword_index = SimpleKeywordTableIndex(nodes,
                                        storage_context=storage_context,
                                        service_context=service_context)




# 定义自定义检索器
class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "AND",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes





# 将检索器插入查询引擎
from llama_index import get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine

# define custom retriever

print("1休息10s")
time.sleep(10)
vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=2)
keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
print("2休息10s")
time.sleep(10)
custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)

# define response synthesizer
print("3休息10s")
time.sleep(10)
response_synthesizer = get_response_synthesizer(service_context=service_context)


# assemble query engine
print("4休息10s")
time.sleep(10)
custom_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer,
)

# vector query engine
vector_query_engine = RetrieverQueryEngine(
    retriever=vector_retriever,
    response_synthesizer=response_synthesizer,
)
# keyword query engine
keyword_query_engine = RetrieverQueryEngine(
    retriever=keyword_retriever,
    response_synthesizer=response_synthesizer,
)


print("5休息10s")
time.sleep(10)
response = custom_query_engine.query(
    "What did the author do during his time at YC?"
)

print("====: ", response)
print(len(response.source_nodes))

response = vector_query_engine.query(
    "What did the author do during his time at YC?"
)
print("====: ", response)
print(len(response.source_nodes))

response = keyword_query_engine.query(
    "What did the author do during his time at YC?"
)
print("====: ", response)
print(len(response.source_nodes))
