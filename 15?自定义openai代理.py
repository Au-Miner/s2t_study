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
from llama_index.tools import BaseTool, FunctionTool
from llama_index.llms import OpenAI, ChatMessage
from llama_index.selectors import LLMSingleSelector
from llama_index.postprocessor import LLMRerank
from llama_index.llms import HuggingFaceLLM, OpenAI
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext, StorageContext, SimpleKeywordTableIndex, QueryBundle, SummaryIndex, LLMPredictor,
    get_response_synthesizer,
)
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))







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










#
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





'''
自定义openai agent，使用自定义的function
使用提供的openai agent可行
但是使用自定义无法复线在openai和llm中
'''






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










# 为我们的代理定义一些非常简单的计算器工具
def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b
multiply_tool = FunctionTool.from_defaults(fn=multiply)
def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b
add_tool = FunctionTool.from_defaults(fn=add)










# 代理定义
class YourOpenAIAgent:
    def __init__(
        self,
        tools: Sequence[BaseTool] = [],
        llm: OpenAI = OpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
        chat_history: List[ChatMessage] = [],
    ) -> None:
        self._llm = llm
        self._tools = {tool.metadata.name: tool for tool in tools}
        self._chat_history = chat_history

    def reset(self) -> None:
        self._chat_history = []

    def chat(self, message: str) -> str:
        chat_history = self._chat_history
        chat_history.append(ChatMessage(role="user", content=message))
        tools = [
            tool.metadata.to_openai_tool() for _, tool in self._tools.items()
        ]
        print("=================")
        print("chat_history: ", chat_history)
        print("tools: ", tools)
        ai_message2 = self._llm.chat(chat_history, tools=tools)
        ai_message = ai_message2.message
        print("ai_message2: ", ai_message2)
        print("ai_message: ", ai_message)
        print("=================")

        additional_kwargs = ai_message.additional_kwargs
        chat_history.append(ai_message)

        tool_calls = ai_message.additional_kwargs.get("tool_calls", None)
        # parallel function calling is now supported
        if tool_calls is not None:
            for tool_call in tool_calls:
                print("tool_call: ", tool_call)
                function_message = self._call_function(tool_call)
                chat_history.append(function_message)
                ai_message = self._llm.chat(chat_history).message
                chat_history.append(ai_message)

        return ai_message.content

    def _call_function(self, tool_call: dict) -> ChatMessage:
        id_ = tool_call["id"]
        function_call = tool_call["function"]
        tool = self._tools[function_call["name"]]
        output = tool(**json.loads(function_call["arguments"]))
        return ChatMessage(
            name=function_call["name"],
            content=str(output),
            role="tool",
            additional_kwargs={
                "tool_call_id": id_,
                "name": function_call["name"],
            },
        )








# agent = YourOpenAIAgent(llm=llm, tools=[multiply_tool, add_tool])
# response = agent.chat("What is 2123 * 215123")
# print(str(response))








from llama_index.agent import OpenAIAgent
agent = OpenAIAgent.from_tools(
    [multiply_tool, add_tool], llm=llm, verbose=True
)
response = agent.chat("What is (121 * 3) + 42?")
print(str(response))










