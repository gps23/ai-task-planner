from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import logging
import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- FASTAPI APP ----------------
app = FastAPI(title="AI Task Planner API")

# ---------------- MODELS ----------------
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

# ---------------- AGENT ----------------
agent = Agent(
    model=GroqModel("llama-3.1-8b-instant"),
    system_prompt=(
        "You are a planning AI agent. "
        "You first confirm user intent, then generate a structured plan."
    )
)

# ---------------- FALLBACK ----------------
def fallback_plan(goal: str, days: int, level: str) -> PlanResponse:
    logger.warning("Fallback plan used")
    return PlanResponse(
        intent_confirmation=f"I will create a {days}-day {level.lower()} plan for {goal}.",
        goal_summary=f"{days}-day plan for {goal}",
        total_days=days,
        daily_tasks=[
            "Review fundamentals",
            "Practice key problems",
            "Revise mistakes",
            "Track progress"
        ],
        tips=[
            "Stay consistent",
            "Focus on weak areas",
            "Revise regularly"
        ],
        why_this_works="This fallback plan ensures steady progress even if AI generation fails."
    )

# ---------------- ROUTES ----------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/plan", response_model=PlanResponse)
async def create_plan(req: PlanRequest):
    logger.info(f"Plan request: {req.goal}, {req.days}, level={req.level}")

    prompt = f"""
Confirm intent in one sentence.
Then generate a {req.days}-day {req.level} level plan for: {req.goal}

Return:
- intent confirmation
- short goal summary
- daily tasks (list)
- smart tips (list)
- why this plan works (1 sentence)
"""

    for attempt in range(2):  # retry once
        try:
            response = await agent.run(prompt)
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
