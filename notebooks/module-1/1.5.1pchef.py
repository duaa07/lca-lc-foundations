import os
import base64
import tkinter as tk
from tkinter import filedialog
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# 1. Load Environment Variables
load_dotenv()

# 2. Setup Tools
tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)

# 3. Define the System Prompt
system_prompt = """
You are a personal chef. The user will give you a list of ingredients they have left over in their house, 
or may upload a picture that has the ingredients left. You'll have to identify them, list them, and then 
using the web search tool, search the web for arabic recipes mostly that can be made with the ingredients they have.
Return recipe suggestions and eventually the recipe instructions to the user directly after the picture upload , give them at least 3 recipes with details and instructions, make the recipes suitable for arabs and arabic culture , you may add other needed ingredients if needed.
"""

def main():
  
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite") 
    
    # Using create_agent and system_prompt as your version requires
    agent = create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=system_prompt, 
        checkpointer=InMemorySaver()
    )
    # 5. Open a file dialog to select the image
    root = tk.Tk()
    root.withdraw() # Hides the blank background window

    print("Opening file explorer... Please select your ingredient picture.")
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )

    if not file_path:
        print("No file was selected. Exiting.")
        return # Stop the script if no file is chosen

    print(f"Selected: {file_path}")
    
    # 6. Read the local file and Base64 encode it
    with open(file_path, "rb") as image_file:
        img_bytes = image_file.read()
        
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # 7. Create the LangChain Multimodal Message
    multimodal_question = HumanMessage(content=[
        {"type": "text", "text": "Here are the picture of ingredients I have left. suggest some recipes with details , make the recipes suitable for arabs and arabic cu}lture , you may add other needed ingredients"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        }
    ])

    config = {"configurable": {"thread_id": "1"}}

    # 8. Invoke the Agent
    print("Chef is analyzing your image and searching the web for recipes...")
    response = agent.invoke(
        {"messages": [multimodal_question]},
        config
    )

    # 9. Print the Final Output
    print("\n" + "="*40)
    print("         CHEF's SUGGESTIONS")
    print("="*40)
    print(response['messages'][-1].content)

if __name__ == "__main__":
    main()