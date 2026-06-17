import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from groq import Groq

if len(sys.argv) < 2:
    print("Usage: python main.py <pdf_file>")
    sys.exit(1)

# Load the text file
loader = PyPDFLoader(sys.argv[1])
documents = loader.load()

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

print(f"number of chunks: {len(chunks)}")

# Create embeddings for the chunks
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)

print("vector store created successfully")

#taking question as input
while True:
    question = input("\nenter your question (or 'exit' to quit): ")
    if question.lower() == "exit":
        print("goodbye!")
        break

    #finding the relevant chunks for the question
    relevant_chunks = vectorstore.similarity_search(question, k=2)
    context = " ".join([chunk.page_content for chunk in relevant_chunks])

    print(f"\nRelevant context found:\n{context}\n")

    # Send to Groq AI for answer generation
    client = Groq(api_key="your_groq_api_key_here")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Answer questions based only on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )

    answer = response.choices[0].message.content
    print(f"AI Answer: {answer}")