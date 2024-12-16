from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
import os
os.environ['HUGGINGFACEHUB_API_TOKEN']="<your hugging face api token>"

# Read the pdfs from the folder

loader=PyPDFDirectoryLoader("/<your path>/pdfs/")
documents=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
final_documents=text_splitter.split_documents(documents)
print(len(final_documents))

## Embedding Using Huggingface

huggingface_embeddings=HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}

)

## VectorStore Creation
vectorstore=FAISS.from_documents(final_documents[:120],huggingface_embeddings)
query = input("Enter your question ")
relevant_docments=vectorstore.similarity_search(query)

#comment this if only llm results
print(relevant_docments[0].page_content)

retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})
model_name="mistralai/Mistral-7B-v0.3"

# path should be same as used in LLMDownload.py
tokenizer= AutoTokenizer.from_pretrained(f"<your path>/mistralmodel/tokenizer/{model_name}")

model=AutoModelForCausalLM.from_pretrained(f"<your path>/mistralmodel/model/{model_name}")

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    #temperature=0.2,
    repetition_penalty=1.1,
    device=0,
    return_full_text=True,
    max_length=500,
    max_new_tokens=300
)

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt_template="""
Use the following piece of context to answer the question asked.
Please try to provide the answer only based on the context
{context}
Question:{question}
Helpful Answers:

"""

prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
retrievalQA=RetrievalQA.from_chain_type(
    llm=mistral_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt":prompt}
)
result = retrievalQA.invoke({"query": query})
print(result['result'])
