from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_ai import Agent

app = FastAPI()

# Enable CORS so frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output schema (validated response)
class TaskPlan(BaseModel):
    goal_summary: str
    total_days: int
    daily_tasks: list[str]
    tips: list[str]

# Pydantic AI agent (deterministic demo mode)
agent = Agent(
    model=None,
    system_prompt="You are a task planning agent."
)

# Input schema
class UserInput(BaseModel):
    goal: str
    days: int

# API endpoint
@app.post("/plan")
async def create_plan(user_input: UserInput):
    plan = TaskPlan(
        goal_summary=f"{user_input.days}-day preparation plan for {user_input.goal}",
        total_days=user_input.days,
        daily_tasks=[
            "Revise basic concepts",
            "Practice aptitude questions",
            "Solve reasoning problems",
            "Attempt mock tests",
            "Analyze mistakes"
        ],
        tips=[
            "Be consistent",
            "Focus on weak areas",
            "Revise daily",
            "Track progress weekly"
        ]
    )
    return plan
