# backend/main.py
"""
Provide the FastAPI application and /agent router.
Expose health-check and API endpoints for local development.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db
from .routers import agent as agent_router
from .routers import sprint2_router, sprint3_router


@asynccontextmanager
async def _lifespan(app: FastAPI):
    init_db()  # idempotent -- creates intervention_log if it doesn't exist yet
    yield


app = FastAPI(
    title="Student Analytics Platform API",
    description="REST endpoints for student analytics, risk prediction, and interventions.",
    version="0.1.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
app.include_router(sprint2_router.router)
app.include_router(sprint3_router.router)
