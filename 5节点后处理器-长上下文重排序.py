import logging
import sys

from llama_index.embeddings import BaseEmbedding

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.postprocessor import LLMRerank
from llama_index.llms import HuggingFaceLLM










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
# service_context = ServiceContext.from_defaults(embed_model="local:/home/qcsun/s2t/S2T_project/bge-large-en-v1.5",
#                                                chunk_size=1024, llm=llm)








from llama_index import ServiceContext
ctx = ServiceContext.from_defaults(
    llm=llm, embed_model="local:/home/qcsun/s2t/S2T_project/bge-large-en-v1.5"
)









documents = SimpleDirectoryReader(
    input_files=["data/League of Legends.txt"]
).load_data()
index = VectorStoreIndex.from_documents(
    documents, service_context=ctx
)
from llama_index.postprocessor import LongContextReorder
reorder = LongContextReorder()
reorder_engine = index.as_query_engine(
    node_postprocessors=[reorder], similarity_top_k=5
)
base_engine = index.as_query_engine(similarity_top_k=5)







# from llama_index1.response.notebook_utils import display_response
base_response = base_engine.query("what is league of legends?")
print(base_response)
# display_response(base_response)
reorder_response = reorder_engine.query("what is league of legends?")
print(base_response)
# display_response(reorder_response)



























