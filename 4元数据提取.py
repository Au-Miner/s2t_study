from llama_index.extractors.metadata_extractors import EntityExtractor
from llama_index.llms import HuggingFaceLLM
from llama_index.node_parser import SentenceSplitter




# entity_extractor = EntityExtractor(
#     # model_name="/home/qcsun/s2t/S2T_project/span-marker-mbert-base-multinerd",
#     model_name="/home/qcsun/s2t/S2T_project/woc",
#     # model_name="/home/qcsun/woc",
#     prediction_threshold=0.5,
#     label_entities=False,  # include the entity label in the metadata (can be erroneous)
#     device="cpu",  # set to "cuda" if you have a GPU
# )
# node_parser = SentenceSplitter()
# transformations = [node_parser, entity_extractor]



from llama_index import ServiceContext, VectorStoreIndex
# setup prompts - specific to StableLM
from llama_index.prompts import PromptTemplate
system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""
# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="/home/qcsun/s2t/S2T_project/stablelm-tuned-alpha-3b",
    model_name="/home/qcsun/s2t/S2T_project/stablelm-tuned-alpha-3b",
    device_map="auto",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 4096},
    # uncomment this if using CUDA to reduce memory usage
    # model_kwargs={"torch_dtype": torch.float16}
)
service_context = ServiceContext.from_defaults(embed_model="local:/home/qcsun/s2t/S2T_project/bge-large-en", chunk_size=1024, llm=llm)






from llama_index.extractors import QuestionsAnsweredExtractor, TitleExtractor
from llama_index.text_splitter import TokenTextSplitter
text_splitter = TokenTextSplitter(
    separator=" ", chunk_size=512, chunk_overlap=128
)
extractors = [
    TitleExtractor(nodes=5, llm=llm),
    QuestionsAnsweredExtractor(questions=3, llm=llm),
]
transformations = [text_splitter] + extractors









from llama_index import SimpleDirectoryReader
documents = SimpleDirectoryReader(
    input_files=["data/League of Legends.txt"]
).load_data()
from llama_index.node_parser import SentenceSplitter
parser = SentenceSplitter(
    chunk_size=512,
    include_prev_next_rel=False,
)
nodes = parser.get_nodes_from_documents(documents)












index = VectorStoreIndex(nodes, service_context=service_context)
query_engine = index.as_query_engine()
response = query_engine.query("What is wql?")
print(response)

response = query_engine.query("What is league of legends?")
print(response)
