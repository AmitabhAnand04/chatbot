from fastapi import FastAPI, Request
from pydantic import BaseModel
from uuid import uuid4
from graph import graph
from langchain_core.messages import HumanMessage

app = FastAPI()

# Request schema
class ChatRequest(BaseModel):
    user_query: str
    thread_id: str | None = None

# Response schema
class ChatResponse(BaseModel):
    response: str
    thread_id: str

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Generate new thread_id if not provided
    thread_id = request.thread_id or str(uuid4())

    # Build config
    config = {"configurable": {"thread_id": thread_id}}

    # Call the graph
    input_message = HumanMessage(content=request.user_query)
    output = graph.invoke({"messages": [input_message]}, config)

    # Extract final message
    response_msg = output['messages'][-1].content if output.get('messages') else "No response."

    return ChatResponse(response=response_msg, thread_id=thread_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
