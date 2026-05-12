from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

system_prompt = """

You are a personal chef. The user will give you a list of ingredients they have left over in their house.

or may upload a picutre that has the ingredients left , so you'll have to identify them list them and then

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.

"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from ipywidgets import FileUpload
from IPython.display import display


agent = create_agent(
    model="google_genai:gemini-2.5-flash-lite",
    tools=[web_search],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)
uploader = FileUpload(accept='.png', multiple=False)
display(uploader)

import base64

# Get the first (and only) uploaded file dict
uploaded_file = uploader.value[0]

# This is a memoryview
content_mv = uploaded_file["content"]

# Convert memoryview -> bytes
img_bytes = bytes(content_mv)  # or content_mv.tobytes()

# Now base64 encode
img_b64 = base64.b64encode(img_bytes).decode("utf-8")

from langchain.messages import HumanMessage

multimodal_question = HumanMessage(content=[
    {"type": "text", "text": "Here are the picture of ingredients i have left"},
    {"type": "image", "base64": img_b64, "mime_type": "image/png"}
])

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [multimodal_question]},
   # {"messages": [HumanMessage(content="I have some leftover lettecue  and chicken tomato sauce . What can I make?")]},
    config
)

print(response['messages'][-1].content)