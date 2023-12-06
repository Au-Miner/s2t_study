from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
)
from llama_index.llms import HuggingFaceLLM
from llama_index import ServiceContext






engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()
# create city SQL table
table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)
metadata_obj.create_all(engine)
from sqlalchemy import insert
rows = [
    {"city_name": "Toronto", "population": 2731571, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13929286, "country": "Japan"},
    {"city_name": "Berlin", "population": 600000, "country": "Germany"},
]
for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)






# from llama_index.prompts import PromptTemplate
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









# import yaml
# import os
# with open('resources/application.yaml', 'r') as file:
#     data = yaml.safe_load(file)
# os.environ["OPENAI_API_KEY"] = data['openai']['api_key']
# os.environ["OPENAI_API_BASE"] = data['openai']['base_url']



from llama_index import SQLDatabase
sql_database = SQLDatabase(engine, include_tables=["city_stats"])
from llama_index.indices.struct_store import NLSQLTableQueryEngine
query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["city_stats"],
    service_context=service_context
)
query_str = "Please tell me which city has the highest population?"
response = query_engine.query(query_str)
print("================================")
print("the result is: ")
print(response)



# from llama_index1 import SQLDatabase
# sql_database = SQLDatabase(engine, include_tables=["city_stats"])
# from llama_index1.objects import (
#     SQLTableNodeMapping,
#     ObjectIndex,
#     SQLTableSchema,
# )
# table_node_mapping = SQLTableNodeMapping(sql_database)
# table_schema_objs = [
#     (SQLTableSchema(table_name="city_stats")),
# ]  # one SQLTableSchema for each table
# obj_index = ObjectIndex.from_objects(
#     table_schema_objs,
#     table_node_mapping,
# )
# from llama_index1.indices.struct_store import SQLTableRetrieverQueryEngine
# query_engine = SQLTableRetrieverQueryEngine(
#     sql_database, obj_index.as_retriever(similarity_top_k=1)
# )
# response = query_engine.query("Which city has the highest population?")
# print(response)