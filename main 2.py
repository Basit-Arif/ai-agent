from fastapi import HTTPException

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from agents import Agent, Runner,function_tool
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
from datetime import date, time ,datetime

import os
import json
import pytz


# Load environment variables from .env file
load_dotenv()
# Set the OpenAI API key from environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


app = FastAPI()


# Optional: Use CORSMiddleware instead of manual middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
async def read_root():
    agent:Agent=Agent(name="Assistant",instructions="you are helpful assisant to user")
    runner = await Runner.run(agent, input="Hello")
    return {"response": str(runner.final_output)}


class InputData(BaseModel):
    lead_message:str
    lead_id:Optional[int] = None


class OutputData(BaseModel):
    meeting_intent: bool
    meeting_date: Optional[str] = None  # Format: "YYYY-MM-DD"
    meeting_time: Optional[str] = None
    timezone:Optional[str] = None  # Format: "Asia/Karachi"
    confidence: Optional[float] = None  # Format: "HH:MM:SS"

@function_tool
def get_current_time(timezone: str) -> str:
    try:
        print(f"⏰ TOOL CALLED for timezone: {timezone}")
        tz = pytz.timezone(timezone)
        return datetime.now(tz).strftime("%Y-%m-%d")
    except pytz.UnknownTimeZoneError:
        return "Unsupported timezone: " + timezone


@app.post("/process")
async def process_input_with_timezone(input_data: InputData):
    try:
        agent = Agent(
            name="MeetingExtractor",
            instructions=f"""
                You are a helpful assistant. Your task is to:
                1. Read the user's message and determine if they are trying to schedule a meeting.
                2. Extract:
                - meeting_intent (true/false)
                - meeting_date (YYYY-MM-DD or null)
                - meeting_time (HH:MM:SS or null)
                - timezone (e.g., "PST", "EST", "Asia/Kolkata", etc.)
                - confidence (0.0 - 1.0)
                3. If a timezone is found, call the `get_current_time` tool to fetch today's date in that timezone.
                4. Use the date returned from the tool to resolve relative phrases like “tomorrow”, “today”, or “day after tomorrow”.
                Use the resolved date in `meeting_date`.
                
                5 If the message includes an **explicit date** like “26 May”, and **no year is mentioned**, assume the year is the same as the date returned from the tool.
                6 If the message includes **“tomorrow”**, “today”, or “day after tomorrow”, resolve those using the date from the `get_current_time` tool.
                7 Always resolve the final `meeting_date` using this logic.

                Return a valid JSON object exactly in this format:
                {{
                "meeting_intent": true or false,
                "meeting_date": "YYYY-MM-DD" or null,
                "meeting_time": "HH:MM" or null,
                "timezone": "PST" or null,
                "confidence": float
                }}

                🛑 Do NOT include any explanations, YAML, comments, or extra text. Only respond with the JSON object.
                ✅ Begin your response with `{{` and end with `}}`.
"""
            ,
            tools=[get_current_time],
           
        )

        runner = await Runner.run(agent, input=input_data.lead_message)
        print(runner.final_output)
        # Step 1: runner.final_output should be a JSON string from LLM
        parsed = json.loads(runner.final_output)

        # Step 2: Convert to Pydantic model
        structured_output = OutputData(**parsed)


        return {"response": structured_output}

    except Exception as e:
        return {"error": str(e)}

class DirectSummaryRequest(BaseModel):
    summary_date: str
    lead_id: int
    formatted_text: str # format: YYYY-MM-DD


@app.post("/summarize")
async def summarize_direct_input(request: DirectSummaryRequest):
    try:
        SUMMARY_AGENT = Agent(
            name="SummaryAgent",
            instructions = f"""
            You are an assistant helping a CRM admin evaluate a sales conversation from {request.summary_date}.
            Summarize clearly and concisely based on:
                1.	Lead’s interest
                2.	Objections or concerns
                3.	Rep’s handling and response
                4.	Engagement level
                5.	Suggested next step
                6.	One-line comment on today’s progress

            Return 2–4 short bullet points covering all key points. Be direct, useful, and avoid fluff
            """
        )
        runner = await Runner.run(SUMMARY_AGENT, input=request.formatted_text)
        print("🧠 Final summary:\n", runner.final_output)
        return {"summary": runner.final_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)