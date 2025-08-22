from langchain_core.prompts import PromptTemplate

def get_qa_prompt():
    return PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
You must include 'page' numnber in your answer.
Answer in Korean.

#Context: 
{context}

#Question:
{question}

#Answer:"""
)
