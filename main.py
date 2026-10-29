# main.py
import json
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from groq import AsyncGroq

from core.config import SettingsLoader
from core.schemas import UnderwritingRequest, AgentStreamEvent
from core.qloo_client import QlooGateway, DiskCacheLayer, RequestHasher
from core.arbitrage_math import DemandScorer, ArbitrageCalculator
from core.citation_validator import GroundingAuditor
from core.agent_orchestrator import ToolExecutor, UnderwritingAgent

app = FastAPI(title="YieldGraph Autonomous Underwriter")


class StreamFormatter:
    """Handles the serialization of agent states into network-ready NDJSON chunks."""

    def format_event(self, event: AgentStreamEvent) -> str:
        """Serializes a strictly typed event into a newline-delimited JSON string."""
        return json.dumps(event.model_dump()) + "\n"

    def format_error(self, message: str) -> str:
        """Standardizes fatal execution errors into the stream schema."""
        return json.dumps({"event_type": "fatal_error", "data": {"reason": message}}) + "\n"


class UnderwritingController:
    """Binds the HTTP request context to the multi-hop Groq agent."""

    def __init__(self, agent_orchestrator: UnderwritingAgent, formatter: StreamFormatter) -> None:
        self._agent = agent_orchestrator
        self._formatter = formatter

    async def stream_execution(self, request: UnderwritingRequest) -> StreamingResponse:
        """Initiates the agentic workflow and yields NDJSON formatted chunks."""
        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                async for event in self._agent.plan_and_execute(request):
                    yield self._formatter.format_event(event)
            except Exception as e:
                yield self._formatter.format_error(str(e))

        return StreamingResponse(
            event_generator(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )


# Dependency Injection Setup
settings_loader = SettingsLoader()
app_config = settings_loader.validate_keys(settings_loader.load_env())

# Initialize the Groq inference engine and domain services
groq_client = AsyncGroq(api_key=app_config.groq_api_key.get_secret_value())
stream_formatter = StreamFormatter()

cache_layer = DiskCacheLayer()
request_hasher = RequestHasher()
qloo_gateway = QlooGateway(
    api_key=app_config.qloo_api_key.get_secret_value(),
    cache_mode=app_config.qloo_cache_mode,
    cache_layer=cache_layer,
    hasher=request_hasher,
    base_url=app_config.qloo_base_url,
)
demand_scorer = DemandScorer()
arbitrage_calculator = ArbitrageCalculator(demand_scorer=demand_scorer)
grounding_auditor = GroundingAuditor()
tool_executor = ToolExecutor(
    qloo_gateway=qloo_gateway,
    calculator=arbitrage_calculator,
    auditor=grounding_auditor,
)
underwriting_agent = UnderwritingAgent(
    groq_client=groq_client,
    tool_executor=tool_executor,
    model_name=app_config.groq_model,
)
controller = UnderwritingController(
    agent_orchestrator=underwriting_agent,
    formatter=stream_formatter,
)


@app.post("/api/underwrite")
async def execute_underwriting(request: UnderwritingRequest) -> StreamingResponse:
    """Public boundary for triggering the autonomous cultural arbitrage workflow."""
    if not controller._agent:
        raise HTTPException(status_code=500, detail="Orchestrator not injected.")
    return await controller.stream_execution(request)


# Serve the reactive zero-dependency UI
app.mount("/", StaticFiles(directory="static", html=True), name="static")
