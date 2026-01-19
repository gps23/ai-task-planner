from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import logging
import asyncio

from fastapi.middleware.cors import CORSMiddleware
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- FASTAPI APP ----------------
app = FastAPI(title="AI Task Planner API")

# ✅ CORS (THIS WAS MISSING)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow Netlify
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- REQUEST / RESPONSE MODELS ----------------
class PlanRequest(BaseModel):
    goal: str = Field(..., min_length=3)
    days: int = Field(..., ge=1, le=365)
    level: str = Field(default="Beginner")

class PlanResponse(BaseModel):
    intent_confirmation: str
    goal_summary: str
    total_days: int
    daily_tasks: List[str]
    tips: List[str]
    why_this_works: str

# ---------------- AI AGENT ----------------
agent = Agent(
    model=GroqModel("llama-3.1-8b-instant"),
    system_prompt=(
        "You are a planning AI agent. "
        "You MUST return valid JSON only."
    )
)

# ---------------- FALLBACK PLAN ----------------
def fallback_plan(goal: str, days: int, level: str) -> PlanResponse:
    logger.warning("Using fallback plan")

    return PlanResponse(
        intent_confirmation=f"I will create a {days}-day {level.lower()} plan for {goal}.",
        goal_summary=f"{days}-day {level.lower()} plan for {goal}",
        total_days=days,
        daily_tasks=[
            "Understand core concepts",
            "Practice key problems",
            "Review mistakes",
            "Revise important topics",
            "Track daily progress"
        ],
        tips=[
            "Stay consistent every day",
            "Focus on weak areas",
            "Revise regularly",
            "Avoid burnout"
        ],
        why_this_works="This plan balances learning, practice, and revision for steady improvement."
    )

# ---------------- HEALTH CHECK ----------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------- MAIN PLAN ENDPOINT ----------------
@app.post("/plan", response_model=PlanResponse)
async def create_plan(req: PlanRequest):
    logger.info(
        f"Plan request | goal={req.goal}, days={req.days}, level={req.level}"
    )

    prompt = f"""
You must return ONLY valid JSON.

JSON schema:
{{
  "intent_confirmation": string,
  "goal_summary": string,
  "daily_tasks": list[string],
  "tips": list[string],
  "why_this_works": string
}}

Task:
Confirm user intent in ONE sentence.
Then generate a {req.days}-day {req.level} level plan for:
"{req.goal}"
"""

    for attempt in range(2):
        try:
            response = await agent.run(prompt)

            if not isinstance(response.data, dict):
                raise ValueError("Agent response is not JSON")

            data = response.data

            return PlanResponse(
                intent_confirmation=data["intent_confirmation"],
                goal_summary=data["goal_summary"],
                total_days=req.days,
                daily_tasks=data["daily_tasks"],
                tips=data["tips"],
                why_this_works=data["why_this_works"]
            )

        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(1)

    return fallback_plan(req.goal, req.days, req.level)
