import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

recipes = json.load(open("recipes.json"))


def short_form(t):
    return t.split("Directions:")[0].strip() if "Directions:" in t else t


docs = [
    Document(page_content=short_form(r["text"]), metadata={"title": r["title"]})
    for r in recipes
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
store = Chroma.from_documents(docs, embeddings, collection_name="chain-demo")

retriever = store.as_retriever(search_kwargs={"k": 5})


def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)


prompt = ChatPromptTemplate.from_template(
    """Here are recipes from a cookbook:

{context}

The user asked for: {question}

Does any recipe above satisfy the request? If not, say so plainly.
If one does, name it and list what they'd need to buy."""
)

# This is LCEL - the pipe operator composes runnables into a chain.
# retriever feeds context, the question passes through untouched,
# both fill the prompt template.
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
)

if __name__ == "__main__":
    q = "a no-bake dessert"
    result = chain.invoke(q)
    print(result.to_string()[:1200])
