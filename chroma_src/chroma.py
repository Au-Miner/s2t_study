
import chromadb
from chromadb.config import Settings
chroma_client = chromadb.HttpClient(host='localhost', port=8000,
                                    settings=Settings(
                                        chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                                        chroma_client_auth_credentials="test-token"
                                    ))
client = chromadb.PersistentClient(path="/Users/au_miner/opt/system/Chroma/db")
list_collections = chroma_client.list_collections()
if "my_collection" in [list_collections[i].name for i in range(len(list_collections))]:
    chroma_client.delete_collection(name="my_collection")


collection = chroma_client.create_collection(name="my_collection")





collection.add(
    documents=["This is a document2", "This is another document", "wql's age is 21", "wql's sex is male"],
    metadatas=[{"source": "my_source"}, {"source": "my_source"}, {"source": "my_source"}, {"source": "my_source"}],
    ids=["1", "2", "3", "4"]
)



results = collection.query(
    query_texts=["wql's info"],
    n_results=2
)
print(results)


print(collection.peek())

collection.add(
    documents=["This is a document"],
    metadatas=[{"source": "my_source"}],
    ids=["1"]
)

print(collection.peek())

result = collection.query(
    query_texts=["wql's info"],
    n_results=1,
    where={"source": "my_source"},
    where_document={"$contains": "21"},
    include=["documents"],
)
print(result)



from chromadb.utils import embedding_functions
default_ef = embedding_functions.DefaultEmbeddingFunction()


print(default_ef)