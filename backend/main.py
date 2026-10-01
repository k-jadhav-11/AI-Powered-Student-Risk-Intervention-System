# backend/main.py
"""
Provide the FastAPI application and /agent router.
Expose health-check and API endpoints for local development.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .database import init_db
from .routers import agent as agent_router


@asynccontextmanager
async def _lifespan(app: FastAPI):
    init_db()  # idempotent -- creates intervention_log if it doesn't exist yet
    yield


app = FastAPI(
    title="Student Analytics Platform -- Intervention Agent API",
    description="Sprint 5's LangGraph Intervention Agent, wrapped behind REST.",
    version="0.1.0",
    lifespan=_lifespan,
)

@app.get("/")
def root():
    return {
        "message": "Student Analytics Platform API is running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def root_health():
    return {"status": "ok"}


app.include_router(agent_router.router)
