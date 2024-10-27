import chromadb
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings



class VectordbManager:
    def read_vectordb_as_df(db_path:str):
        client = chromadb.PersistentClient(path=db_path)
        for collection in client.list_collections():
            data = collection.get(include=['documents', 'metadatas'])
            df = pd.DataFrame({"ids":data["ids"], 
                    "metadatas":data["metadatas"], 
                    "documents":data["documents"]})
            df["first_div"] = df["metadatas"].apply(lambda x: x["First Division"])
            df["second_div"] = df["metadatas"].apply(lambda x: x["Second Division"])
            df["filename"] = df["metadatas"].apply(lambda x: x["File Name"])
            df = df[["ids", "first_div", "second_div","filename","documents", "metadatas"]]
            data = df.to_dict()
        return data
    
    def get_filename(db_path:str):
        client = chromadb.PersistentClient(path=db_path)
        for collection in client.list_collections():
            data = collection.get(include=['documents', 'metadatas'])
            df = pd.DataFrame({"ids":data["ids"], 
                    "metadatas":data["metadatas"], 
                    "documents":data["documents"]})
            df["first_div"] = df["metadatas"].apply(lambda x: x["First Division"])
            df["second_div"] = df["metadatas"].apply(lambda x: x["Second Division"])
            df["filename"] = df["metadatas"].apply(lambda x: x["File Name"])
            df = df[["ids", "first_div", "second_div","filename","documents", "metadatas"]]

        total_results = []

        for i, name in df[["first_div", "second_div", "filename"]].iterrows():
            result = f"{name.iloc[0]}/{name.iloc[1]}/{name.iloc[2]}"
            if result not in total_results:
                total_results.append(result)
            else: pass
            total_results.sort()
        return total_results

    def similarity_search(query:str, db_path:str):
        embed_model = OllamaEmbeddings(base_url="http://ollama:11434", model="bge-m3:latest")
        vector_store = Chroma(collection_name="collection_01", persist_directory=db_path, embedding_function=embed_model)
        results = vector_store.similarity_search_with_relevance_scores(query, k=3)
        return results
    
    def delete_document(filename:str, db_path:str):
        vector_store = Chroma(collection_name="collection_01", persist_directory=db_path, embedding_function=OllamaEmbeddings(model="bge-m3:latest"))
        del_ids = vector_store.get(where={'File Name':filename})["ids"]
        vector_store.delete(del_ids)
        print("Document is deleted")