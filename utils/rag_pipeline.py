from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List, Optional
import streamlit as st

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250
LLM_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an expert video assistant named VideoLens.

RULES — follow these strictly:
1. Answer ONLY using the provided transcript context below.
2. If the answer is NOT available in the transcript, say:
   "I could not find that information in the video."
3. Do NOT hallucinate or use outside knowledge. However, you should make reasonable deductions and inferences from the transcript.
4. Be concise and accurate.
5. When helpful, use bullet points for clarity.
6. If timestamps are visible in the context, mention them.

CONTEXT:
{context}"""

def split_transcript(transcript_text: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.create_documents([transcript_text])

@st.cache_resource(show_spinner="Loading embeddings model...")
def create_embeddings():
    return FastEmbedEmbeddings()

def build_vectorstore(chunks: list[Document], embeddings) -> FAISS:
    return FAISS.from_documents(chunks, embeddings)

class VideoRetriever(BaseRetriever):
    vectorstore: FAISS
    search_type: str = "mmr"
    search_kwargs: dict = {}
    first_chunk: Optional[Document] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        
        # Select the appropriate search method
        if self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(query, **self.search_kwargs)
        else:
            docs = self.vectorstore.similarity_search(query, **self.search_kwargs)

        # Always include the first chunk for necessary context
        if self.first_chunk:
            is_first_chunk_present = any(d.page_content == self.first_chunk.page_content for d in docs)
            if not is_first_chunk_present:
                docs = [self.first_chunk] + docs
                
        return docs

def get_retriever(vectorstore: FAISS, first_chunk: Optional[Document] = None):
    return VideoRetriever(
        vectorstore=vectorstore,
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20},
        first_chunk=first_chunk
    )

def create_llm(api_key: str) -> ChatGroq:
    return ChatGroq(
        groq_api_key=api_key,
        model_name=LLM_MODEL,
        temperature=0.1,
        max_tokens=2048
    )

def _convert_chat_history(chat_history: list[dict]) -> list:
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages

def build_qa_chain(llm: ChatGroq, retriever):
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

def rephrase_query(llm: ChatGroq, question: str, chat_history: list[dict]) -> str:
    # If there is no history, the question is already standalone
    if not chat_history:
        return question

    prompt = (
        "Given the following chat history and a follow-up question, rephrase the follow-up "
        "question to be a standalone question that can be searched for in a video transcript.\n\n"
        "CHAT HISTORY:\n"
    )
    
    # Add the last 5 messages to provide context
    for msg in chat_history[-5:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        prompt += f"{role}: {msg['content']}\n"
        
    prompt += f"\nFollow-up Question: {question}\n"
    prompt += "Standalone Question (output ONLY the standalone question, no explanation):"

    try:
        response = llm.invoke(prompt)
        standalone_question = response.content.strip()
        if standalone_question:
            return standalone_question
    except Exception:
        pass
        
    # Fallback to the original question if anything fails
    return question

def ask_with_history(
    chain, question: str, chat_history: list[dict], llm: Optional[ChatGroq] = None) -> dict:
    
    # Rephrase the question based on chat history if applicable
    search_query = question
    if llm and chat_history:
        search_query = rephrase_query(llm, question, chat_history)
        
    formatted_history = _convert_chat_history(chat_history)
    
    response = chain.invoke({
        "input": search_query,
        "chat_history": formatted_history
    })
    
    return {
        "answer": response["answer"],
        "sources": response.get("context", [])
    }

def ask_question_simple(llm: ChatGroq, retriever, question: str) -> dict:
    qa_chain = build_qa_chain(llm, retriever)
    return ask_with_history(qa_chain, question, chat_history=[])
