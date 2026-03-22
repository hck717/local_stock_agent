"""Session and database models."""
import uuid
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass, field, asdict
import json

from app.config import config


@dataclass
class Session:
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_active_at: datetime = field(default_factory=datetime.now)
    
    def update_activity(self):
        self.last_active_at = datetime.now()
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_active_at": self.last_active_at.isoformat()
        }


@dataclass
class QueryLog:
    query_id: str
    session_id: str
    prompt: str
    intent: str
    tickers: List[str]
    runtime_ms: int
    status: str
    warning: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "session_id": self.session_id,
            "prompt": self.prompt,
            "intent": self.intent,
            "tickers": self.tickers,
            "runtime_ms": self.runtime_ms,
            "status": self.status,
            "warning": self.warning,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ChartArtifact:
    chart_id: str
    query_id: str
    chart_type: str
    path: str
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "chart_id": self.chart_id,
            "query_id": self.query_id,
            "chart_type": self.chart_type,
            "path": self.path,
            "created_at": self.created_at.isoformat()
        }


class SessionManager:
    """Manage user sessions and query logs."""
    
    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self.query_logs: list[QueryLog] = []
        self.chart_artifacts: list[ChartArtifact] = []
    
    def create_session(self) -> Session:
        session = Session(session_id=str(uuid.uuid4()))
        self.sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        session = self.sessions.get(session_id)
        if session:
            session.update_activity()
        return session
    
    def log_query(
        self,
        session_id: str,
        prompt: str,
        intent: str,
        tickers: List[str],
        runtime_ms: int,
        status: str,
        warning: Optional[str] = None
    ) -> QueryLog:
        log = QueryLog(
            query_id=str(uuid.uuid4()),
            session_id=session_id,
            prompt=prompt,
            intent=intent,
            tickers=tickers,
            runtime_ms=runtime_ms,
            status=status,
            warning=warning
        )
        self.query_logs.append(log)
        return log
    
    def log_chart(self, query_id: str, chart_type: str, path: str) -> ChartArtifact:
        artifact = ChartArtifact(
            chart_id=str(uuid.uuid4())[:8],
            query_id=query_id,
            chart_type=chart_type,
            path=path
        )
        self.chart_artifacts.append(artifact)
        return artifact
    
    def get_session_history(self, session_id: str) -> List[QueryLog]:
        return [log for log in self.query_logs if log.session_id == session_id]
    
    def export_logs(self) -> str:
        data = {
            "sessions": [s.to_dict() for s in self.sessions.values()],
            "query_logs": [log.to_dict() for log in self.query_logs],
            "chart_artifacts": [a.to_dict() for a in self.chart_artifacts]
        }
        return json.dumps(data, indent=2)


session_manager = SessionManager()
