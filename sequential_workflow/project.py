import os 
from typing import TypedDict 

class pipeline_state(TypedDict): 
    raw_input : str 
    edited_text : str 
    script_text : str 
    final_output : str 

from langchain_groq import ChatGroq 
from dotenv import load_dotenv 

load_dotenv() 

# Fixed model to a generative model, otherwise the text generation fails
llm=ChatGroq(model="openai/gpt-oss-20b", api_key=os.environ.get("GROQ_API_KEY"),temperature=0.7,max_tokens=1000) 

def editor_node(state: pipeline_state) -> dict: 
    """STAGE1: CLEANS UP GRAMMAR AND REMOVE ANY TYPOS,AND DOES REFINE ITS TONE """
    # REMOVED outer curly braces to fix the Set compiler error
    prompt=f""" 
    You are a professional editor. Your task is to clean up the grammar, remove any typos, and refine the tone of the following text. Please ensure that the meaning of the text remains unchanged. 
    Text to edit: {state['raw_input']} 
    Please provide the edited text below: 
    """ 
    response = llm.invoke(prompt) 
    return {"edited_text": response.content.strip()} 

def scriptwriter_node(state: pipeline_state) -> dict: 
    """STAGE2: CONVERTS THE EDITED TEXT INTO A SCRIPT FORMAT for a engaing movie or tv show"""
    # REMOVED outer curly braces to fix the Set compiler error
    prompt=f""" 
    You are a professional scriptwriter. Your task is to convert the following text into a script format. Please ensure that the meaning of the text remains unchanged. 
    Text to convert: {state['edited_text']} 
    Please provide the script below: 
    """ 
    response = llm.invoke(prompt) 
    return {"script_text": response.content.strip()} 

def hinenglish_node(state: pipeline_state) -> dict: 
    """STAGE3: CONVERTS THE SCRIPT INTO HINGLISH (A MIX OF HINDI AND ENGLISH)"""
    # REMOVED outer curly braces to fix the Set compiler error
    prompt=f""" 
    You are a professional translator. Your task is to convert the following script into Hinglish (a mix of Hindi and English). Please ensure that the meaning of the text remains unchanged. 
    Script to convert: {state['script_text']} 
    Please provide the Hinglish script below: 
    """ 
    response = llm.invoke(prompt) 
    return {"final_output": response.content.strip()} 

from langgraph.graph import StateGraph, START, END 

graph=StateGraph(pipeline_state) 
graph.add_node("editor", editor_node) 
graph.add_node("scriptwriter", scriptwriter_node) 
graph.add_node("hinenglish", hinenglish_node) 

graph.add_edge(START, "editor") 
graph.add_edge("editor", "scriptwriter") 
graph.add_edge("scriptwriter", "hinenglish") 
graph.add_edge("hinenglish", END) 

app=graph.compile() 

result=app.invoke( 
    {"raw_input": "Once upon a time in a land far, far away, there lived a brave knight named Sir Lancelot. He was known for his courage and honor, and he embarked on many adventures to protect the kingdom from evil forces. One day, he received a quest to rescue a princess who had been captured by a fearsome dragon. With his trusty sword and unwavering determination, Sir Lancelot set out on his journey to save the princess and restore peace to the land."} 
) 

print("your final output is:") 
print(result['final_output'])
