"""
FastAPI Application for Contraception Counseling System

Uses Compliance-Aware Prompting (Experiment 2 approach) for optimal safety and accuracy.

Based on experimental results:
- Claude Opus 4.5 + Compliance Prompting: 76.25% compliant, 0 critical issues
- RAG approach showed 35% degradation in compliance

Provides REST API endpoints for:
- Query processing
- Conversation management
- Health checks
- Statistics
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import uuid
from loguru import logger
from pathlib import Path

from src.pipeline.compliance_pipeline import CompliancePipeline, CompliancePipelineWithMemory
from src.utils.data_collection import DataCollector
from src.utils.logger import setup_logger

# Initialize logger
setup_logger()

# Create FastAPI app
app = FastAPI(
    title="Contraception Counseling API",
    description="Compliance-aware contraception counseling system (Exp2 approach) with multi-language support. Uses Claude Opus 4.5 for optimal safety and accuracy.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web UI
static_dir = Path(__file__).resolve().parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir.absolute()), html=True), name="static")
    logger.info(f"Static files mounted from: {static_dir}")
else:
    logger.warning(f"Static directory not found: {static_dir}")

# Global pipeline instance
pipeline: Optional[CompliancePipeline] = None
data_collector: Optional[DataCollector] = None


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., min_length=1, max_length=1000, description="User's question")
    language: Optional[str] = Field(None, description="Response language (auto-detected if not provided)")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    user_id: Optional[str] = Field(None, description="User ID for personalization (requires memory system)")
    top_k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    include_sources: bool = Field(True, description="Include source citations")
    collect_data: bool = Field(False, description="Opt-in to data collection")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the side effects of DMPA injection?",
                "language": "english",
                "session_id": "user123",
                "user_id": "user_001",
                "top_k": 5,
                "include_sources": True,
                "collect_data": False
            }
        }


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    response: str = Field(..., description="Generated counseling response")
    sources: List[Dict] = Field(..., description="Source citations")
    language: str = Field(..., description="Response language")
    session_id: Optional[str] = Field(None, description="Session identifier")
    metadata: Dict = Field(..., description="Additional metadata")
    timestamp: str = Field(..., description="Response timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "DMPA injection has several common side effects...",
                "sources": [
                    {"source": "WHO FP Handbook 2022", "page": "45", "relevance_score": 0.89}
                ],
                "language": "english",
                "session_id": "user123",
                "metadata": {"temperature": 0.7, "context_length": 500},
                "timestamp": "2025-11-29T10:30:00Z"
            }
        }


class FeedbackRequest(BaseModel):
    """Request model for feedback submission."""
    session_id: str = Field(..., description="Session identifier")
    rating: int = Field(..., ge=1, le=5, description="Rating (1-5 stars)")
    helpful: bool = Field(..., description="Whether response was helpful")
    comments: Optional[str] = Field(None, max_length=500, description="Optional feedback comments")
    opted_in: bool = Field(False, description="User opted in to data collection")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user123",
                "rating": 5,
                "helpful": True,
                "comments": "Very informative response",
                "opted_in": True
            }
        }


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history."""
    session_id: str
    history: List[Dict]
    total_turns: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    timestamp: str
    components: Dict


class StatsResponse(BaseModel):
    """Response model for statistics."""
    total_sessions: int
    total_conversations: int
    retriever_chunks: int
    uptime: str


# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize compliance-aware pipeline on startup."""
    global pipeline, data_collector

    logger.info("Starting up Contraception Counseling API...")

    try:
        # Initialize Compliance Pipeline (Experiment 2 approach)
        # Uses Claude Opus 4.5 + compliance-aware prompting (no RAG retrieval)
        pipeline = CompliancePipelineWithMemory(
            config_path="configs/config.yaml",
            use_multilingual=True
        )
        logger.info("✓ Compliance Pipeline initialized (Experiment 2 approach)")

        # Initialize data collector (disabled by default)
        data_collector = DataCollector(
            storage_dir="data/collected",
            enabled=False  # Set to True for pilot studies
        )
        logger.info("✓ Data collector initialized (disabled)")

        logger.info("API startup complete")

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Contraception Counseling API...")


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint - serves web UI directly."""
    # Serve the HTML file directly
    html_file = static_dir / "index.html"
    if html_file.exists():
        return FileResponse(str(html_file))
    else:
        # Fallback to API information if UI not available
        return {
            "name": "Contraception Counseling API",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
            "web_ui": "/static/index.html (not available)",
            "endpoints": {
                "query": "/counsel/query",
                "feedback": "/counsel/feedback",
                "conversation": "/counsel/conversation/{session_id}",
                "health": "/health",
                "stats": "/stats"
            }
        }


@app.get("/static/css/style.css")
async def serve_css():
    """Serve CSS file."""
    css_file = static_dir / "css" / "style.css"
    if css_file.exists():
        return FileResponse(str(css_file), media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS file not found")


@app.get("/static/js/chat.js")
async def serve_js():
    """Serve JavaScript file."""
    js_file = static_dir / "js" / "chat.js"
    if js_file.exists():
        return FileResponse(str(js_file), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JS file not found")


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint.

    Returns system status and component health.
    """
    try:
        # Check pipeline
        pipeline_status = "healthy" if pipeline is not None else "not_initialized"

        # Check retriever
        retriever_status = "healthy" if hasattr(pipeline, 'retriever') else "not_initialized"

        # Check generator
        generator_status = "healthy" if hasattr(pipeline, 'generator') else "not_initialized"

        overall_status = "healthy" if pipeline_status == "healthy" else "degraded"

        return {
            "status": overall_status,
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "pipeline": pipeline_status,
                "retriever": retriever_status,
                "generator": generator_status,
                "data_collector": "disabled" if not data_collector.enabled else "enabled"
            }
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )


@app.post("/counsel/query", response_model=QueryResponse, tags=["Counseling"])
async def query_endpoint(request: QueryRequest):
    """
    Main counseling query endpoint.

    Process a user's question and return a counseling response with sources.

    **Multi-language support:** English, French, Kinyarwanda
    **Auto-detection:** Language auto-detected if not specified
    **Conversation tracking:** Provide session_id to maintain conversation context
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized"
        )

    try:
        logger.info(f"Received query: {request.question[:100]}...")

        # Generate session_id if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Process query (with user_id if available for memory system)
        query_kwargs = {
            'question': request.question,
            'language': request.language,
            'session_id': session_id,
            'top_k': request.top_k,
            'include_sources': request.include_sources
        }

        # Add user_id if provided (for memory-enabled pipeline)
        if request.user_id:
            query_kwargs['user_id'] = request.user_id

        result = pipeline.query(**query_kwargs)

        # Collect data if opted in
        if request.collect_data and data_collector:
            data_collector.collect_interaction(
                query=request.question,
                response=result['response'],
                session_id=session_id,
                user_opted_in=True,
                metadata={
                    'language': result['language'],
                    'sources_count': len(result['sources'])
                }
            )

        # Build response
        response = QueryResponse(
            response=result['response'],
            sources=result.get('sources', []),
            language=result['language'],
            session_id=session_id,
            metadata=result.get('metadata', {}),
            timestamp=datetime.now().isoformat()
        )

        logger.info(f"Query processed successfully (session={session_id})")
        return response

    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@app.post("/counsel/feedback", tags=["Counseling"])
async def feedback_endpoint(request: FeedbackRequest):
    """
    Submit feedback for a counseling interaction.

    Helps improve the system through user feedback.
    """
    try:
        logger.info(f"Received feedback for session {request.session_id}")

        # Collect feedback if data collection enabled
        if data_collector:
            success = data_collector.collect_feedback(
                session_id=request.session_id,
                rating=request.rating,
                helpful=request.helpful,
                comments=request.comments,
                user_opted_in=request.opted_in
            )

            if success:
                logger.info(f"Feedback collected for session {request.session_id}")
            else:
                logger.warning(f"Feedback not collected (data collection disabled or user not opted in)")

        return {
            "status": "received",
            "session_id": request.session_id,
            "message": "Thank you for your feedback!"
        }

    except Exception as e:
        logger.error(f"Feedback submission failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback submission failed: {str(e)}"
        )


@app.get("/counsel/conversation/{session_id}", response_model=ConversationHistoryResponse, tags=["Counseling"])
async def get_conversation(session_id: str):
    """
    Get conversation history for a session.

    Retrieve all previous interactions for a given session.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized"
        )

    try:
        history = pipeline.get_conversation_history(session_id)

        return {
            "session_id": session_id,
            "history": history,
            "total_turns": len(history)
        }

    except Exception as e:
        logger.error(f"Failed to retrieve conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation: {str(e)}"
        )


@app.delete("/counsel/conversation/{session_id}", tags=["Counseling"])
async def clear_conversation(session_id: str):
    """
    Clear conversation history for a session.

    Delete all stored conversation turns for privacy.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized"
        )

    try:
        pipeline.clear_conversation(session_id)

        return {
            "status": "cleared",
            "session_id": session_id,
            "message": "Conversation history cleared"
        }

    except Exception as e:
        logger.error(f"Failed to clear conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear conversation: {str(e)}"
        )


# Memory Management Endpoints

# Request model for creating user profile
class CreateProfileRequest(BaseModel):
    """Request model for creating user profile."""
    user_id: str = Field(..., description="User identifier")
    language: str = Field("english", description="Preferred language")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_001",
                "language": "english",
                "metadata": {}
            }
        }


@app.post("/memory/profiles", tags=["Memory"])
async def create_user_profile(request: CreateProfileRequest):
    """
    Create a new user profile.

    Creates or updates user profile for personalization.
    """
    if pipeline is None or not isinstance(pipeline, RAGPipelineWithMemory):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory system not available"
        )

    try:
        profile = pipeline.memory.user_profiles.create_profile(
            user_id=request.user_id,
            language=request.language,
            metadata=request.metadata or {}
        )

        return {
            "status": "created",
            "user_id": request.user_id,
            "profile": profile
        }

    except Exception as e:
        logger.error(f"Failed to create profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create profile: {str(e)}"
        )


@app.get("/memory/profiles/{user_id}", tags=["Memory"])
async def get_user_profile(user_id: str):
    """
    Get user profile.

    Returns user profile and preferences.
    """
    if pipeline is None or not isinstance(pipeline, RAGPipelineWithMemory):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory system not available"
        )

    try:
        profile = pipeline.memory.user_profiles.get_profile(user_id)

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Profile not found for user {user_id}"
            )

        return profile

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get profile: {str(e)}"
        )


# Request model for updating preferences
class UpdatePreferencesRequest(BaseModel):
    """Request model for updating user preferences."""
    preferences: Dict = Field(..., description="User preferences to update")

    class Config:
        json_schema_extra = {
            "example": {
                "preferences": {
                    "language": "french",
                    "detail_level": "high",
                    "include_sources": True
                }
            }
        }


@app.put("/memory/profiles/{user_id}/preferences", tags=["Memory"])
async def update_user_preferences(
    user_id: str,
    request: UpdatePreferencesRequest
):
    """
    Update user preferences.

    Updates language, detail level, and other preferences.
    """
    if pipeline is None or not isinstance(pipeline, RAGPipelineWithMemory):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory system not available"
        )

    try:
        pipeline.memory.update_user_preferences(user_id, request.preferences)

        return {
            "status": "updated",
            "user_id": user_id,
            "preferences": request.preferences
        }

    except Exception as e:
        logger.error(f"Failed to update preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update preferences: {str(e)}"
        )


@app.get("/memory/sessions/{session_id}/summary", tags=["Memory"])
async def get_session_summary(session_id: str):
    """
    Get session summary.

    Returns statistics and summary for a conversation session.
    """
    if pipeline is None or not isinstance(pipeline, RAGPipelineWithMemory):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory system not available"
        )

    try:
        summary = pipeline.memory.get_session_summary(session_id)
        return summary

    except Exception as e:
        logger.error(f"Failed to get session summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session summary: {str(e)}"
        )


@app.get("/memory/users/{user_id}/export", tags=["Memory"])
async def export_user_data(user_id: str):
    """
    Export user data (GDPR data portability).

    Returns all user data in portable format.
    """
    if pipeline is None or not isinstance(pipeline, RAGPipelineWithMemory):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory system not available"
        )

    try:
        data = pipeline.export_user_data(user_id)

        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No data found for user {user_id}"
            )

        return data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export user data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export user data: {str(e)}"
        )


@app.delete("/memory/users/{user_id}", tags=["Memory"])
async def delete_user_data(user_id: str):
    """
    Delete all user data (GDPR right to be forgotten).

    Permanently deletes all user data including profiles and conversations.
    """
    if pipeline is None or not isinstance(pipeline, RAGPipelineWithMemory):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory system not available"
        )

    try:
        pipeline.delete_user_data(user_id)

        return {
            "status": "deleted",
            "user_id": user_id,
            "message": "All user data has been permanently deleted"
        }

    except Exception as e:
        logger.error(f"Failed to delete user data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user data: {str(e)}"
        )


@app.get("/memory/statistics", tags=["Memory"])
async def get_memory_statistics():
    """
    Get memory system statistics.

    Returns statistics about sessions, users, and memory usage.
    """
    if pipeline is None or not isinstance(pipeline, RAGPipelineWithMemory):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory system not available"
        )

    try:
        stats = pipeline.get_memory_statistics()
        return stats

    except Exception as e:
        logger.error(f"Failed to get memory statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get memory statistics: {str(e)}"
        )


@app.get("/stats", response_model=StatsResponse, tags=["General"])
async def get_statistics():
    """
    Get system statistics.

    Returns usage metrics and system information.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized"
        )

    try:
        stats = pipeline.get_statistics()

        return {
            **stats,
            "uptime": "available"  # Can track actual uptime if needed
        }

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


# Run with: uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting development server...")

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
