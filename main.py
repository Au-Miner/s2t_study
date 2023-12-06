# from llama_index1 import (
#     VectorStoreIndex,
#     SimpleDirectoryReader,
#     StorageContext,
#     load_index_from_storage, ServiceContext,
# )
# import yaml
# import os
#
#
# # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# # logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
#
# # with open('resources/application.yaml', 'r') as file:
# #     data = yaml.safe_load(file)
# # os.environ["OPENAI_API_KEY"] = data['openai']['api_key']
# # os.environ["OPENAI_API_BASE"] = data['openai']['base_url']
# #
#
#
#
# from llama_index1.prompts import PromptTemplate
#
# system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
# - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
# - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
# - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
# - StableLM will refuse to participate in anything that could harm a human.
# """
#
# # This will wrap the default prompts that are internal to llama-index
# query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")
#
# import torch
# from llama_index1.llms import HuggingFaceLLM
#
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
# service_context = ServiceContext.from_defaults(
#     chunk_size=1024,
#     llm=llm,
# )
#
#
#
# if_not_use_persist = True
#
#
#
#
# if not os.path.exists("./storage") or if_not_use_persist:
#     print("本地读取")
#     documents = SimpleDirectoryReader("data").load_data()
#     index = VectorStoreIndex.from_documents(documents)
#     index.storage_context.persist()
# else:
#     storage_context = StorageContext.from_defaults(persist_dir="./storage")
#     index = load_index_from_storage(storage_context)
#
# query_engine = index.as_query_engine(service_context=service_context)
# response = query_engine.query("What is my name and my age?")
# print(response)
#
#
#


import logging
import sys


# from llama_index1 import set_global_tokenizer
# from transformers import AutoTokenizer
# set_global_tokenizer(
#     AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").encode
# )

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM


# import yaml
# import os
# with open('resources/application.yaml', 'r') as file:
#     data = yaml.safe_load(file)
# os.environ["OPENAI_API_KEY"] = data['openai']['api_key']
# os.environ["OPENAI_API_BASE"] = data['openai']['base_url']

documents = SimpleDirectoryReader(
    input_files=[f"data/data_info.txt"]
).load_data()



#
# # setup prompts - specific to StableLM
# from llama_index1.prompts import PromptTemplate
# system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
# - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
# - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
# - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
# - StableLM will refuse to participate in anything that could harm a human.
# """
# # This will wrap the default prompts that are internal to llama-index
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
#



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
    generate_kwargs={"temperature": 0.25, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="/home/qcsun/s2t/S2T_project/Camel-5b",
    model_name="/home/qcsun/s2t/S2T_project/Camel-5b",
    device_map="auto",
    tokenizer_kwargs={"max_length": 2048},
    # uncomment this if using CUDA to reduce memory usage
    # model_kwargs={"torch_dtype": torch.float16}
)
service_context = ServiceContext.from_defaults(embed_model="local:/home/qcsun/s2t/S2T_project/bge-large-en", chunk_size=512, llm=llm)




















index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)


# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("What is my name and my age?")


print(response)