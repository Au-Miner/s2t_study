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
from llama_index.schema import NodeWithScore

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
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1024)
print(222)






'''
他们利用最新的llm来评估自己的输出，然后自我纠正以给出更好的回应

TODO:
目前333重试指南查询引擎无法跑通
原因在于FeedbackQueryTransformation的llm_predictor要求BaseLLMPredictor，而不是HuggingFaceLLM，需要实践探索（目前仅支持openai）
'''




documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query = "What did the author do growing up?"
base_query_engine = index.as_query_engine()
response = base_query_engine.query(query)
print("=======000默认=======")
print(response)



# 重试查询引擎
print("=======111重试查询引擎=======")
from llama_index.query_engine import RetryQueryEngine
from llama_index.evaluation import RelevancyEvaluator
query_response_evaluator = RelevancyEvaluator(service_context=service_context)
retry_query_engine = RetryQueryEngine(
    base_query_engine, query_response_evaluator
)
retry_response = retry_query_engine.query(query)
print(retry_response)





# 重试源查询引擎
print("=======222重试源查询引擎=======")
from llama_index.query_engine import RetrySourceQueryEngine
retry_source_query_engine = RetrySourceQueryEngine(
    base_query_engine, query_response_evaluator
)
retry_source_response = retry_source_query_engine.query(query)
print(retry_source_response)






# 重试指南查询引擎
print("=======333重试指南查询引擎=======")
from llama_index.evaluation.guideline import (
    GuidelineEvaluator,
    DEFAULT_GUIDELINES,
)
from llama_index.response.schema import Response
from llama_index.indices.query.query_transform.feedback_transform import (
    FeedbackQueryTransformation,
)
from llama_index.query_engine.retry_query_engine import (
    RetryGuidelineQueryEngine,
)
# Guideline eval
guideline_eval = GuidelineEvaluator(
    service_context=service_context,
    guidelines=DEFAULT_GUIDELINES
    + "\nThe response should not be overly long.\n"
    "The response should try to summarize where possible.\n"
)  # just for example
retry_guideline_query_engine = RetryGuidelineQueryEngine(
    base_query_engine, guideline_eval, resynthesize_query=True,
    # query_transformer=FeedbackQueryTransformation(
    #     llm_predictor=Optional[llm],
    #     resynthesize_query=True
    # )
)
retry_guideline_response = retry_guideline_query_engine.query(query)
print(retry_guideline_response)


