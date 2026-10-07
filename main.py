"""
Customer Support AI Agent
=========================
Completed implementation for the Udacity AgentCore + Strands project.
"""

from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse
import json
import os
import asyncio
import boto3
from strands.hooks import (
    HookProvider,
    AfterInvocationEvent,
    HookRegistry,
    MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# AgentCore Runtime application.
app = BedrockAgentCoreApp()

# Suppress interactive tool-consent prompts in headless deployments.
os.environ["BYPASS_TOOL_CONSENT"] = "true"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GATEWAY_URL = "https://customersupportgateway-j0irp0wiqi.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID = "PR9BJNQ2KY"
REGION = "us-east-1"
MEMORY_ID = "CustomerSupportMemory-zTnIaF8527"

model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------
def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type to namespace template."""
    if not memory_id or memory_id.startswith("<"):
        return {}

    strategies = mem_client.get_memory_strategies(memory_id=memory_id)
    if isinstance(strategies, dict):
        strategies = strategies.get("strategies", strategies.get("memoryStrategies", []))

    namespaces = {}
    for strategy in strategies:
        strategy_type = (
            strategy.get("strategyType")
            or strategy.get("type")
            or strategy.get("strategy", {}).get("type")
        )

        templates = strategy.get("namespaceTemplates")
        if not templates:
            templates = strategy.get("namespaces")

        if strategy_type and templates:
            namespaces[strategy_type] = templates[0]

    return namespaces


def _plain_message_text(message) -> str:
    """Return text only when a message contains no tool-use/result blocks."""
    if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
        return ""

    content = message.get("content", [])
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ""

    texts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in {"toolUse", "toolResult"}:
            return ""
        if isinstance(block.get("text"), str):
            texts.append(block["text"])

    return "".join(texts)


class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant customer memories and prepend them to the user message."""
        if not self.memory_id or self.memory_id.startswith("<"):
            return

        message = getattr(event, "message", None)
        if not isinstance(message, dict) or message.get("role") != "user":
            return

        query = _plain_message_text(message)
        if not query:
            return

        memories = []

        for strategy_type, namespace_template in self.namespaces.items():
            namespace = namespace_template.format(actorId=self.actor_id)
            try:
                response = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=query,
                    top_k=5,
                )
            except Exception as exc:
                logger.warning("Memory retrieval failed for %s: %s", strategy_type, exc)
                continue

            if isinstance(response, dict):
                results = response.get("memories", response.get("results", []))
            else:
                results = response
            for item in results:
                memory_text = (
                    item.get("content", {}).get("text")
                    if isinstance(item.get("content"), dict)
                    else item.get("text")
                )
                if memory_text:
                    memories.append(f"[{strategy_type}] {memory_text}")

        if memories:
            context = "Customer Context:\n" + "\n".join(memories)
            original = query
            message["content"] = [{"text": f"{context}\n\n{original}"}]

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Persist the last customer query and assistant response."""
        if not self.memory_id or self.memory_id.startswith("<"):
            return

        messages = getattr(event.agent, "messages", [])
        customer_query = ""
        assistant_response = ""

        for message in reversed(messages):
            if not customer_query and isinstance(message, dict) and message.get("role") == "user":
                customer_query = _plain_message_text(message)
            elif not assistant_response and isinstance(message, dict) and message.get("role") == "assistant":
                assistant_response = _plain_message_text(message)

            if customer_query and assistant_response:
                break

        if not customer_query or not assistant_response:
            return

        try:
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[
                    (customer_query, "USER"),
                    (assistant_response, "ASSISTANT"),
                ],
            )
        except Exception as exc:
            logger.warning("Could not save support interaction: %s", exc)

    def register_hooks(self, registry: HookRegistry) -> None:
        """Register memory retrieval and persistence callbacks."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID or KB_ID.startswith("<"):
        return "Knowledge base not configured."

    try:
        response = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
        results = response.get("retrievalResults", [])
        if not results:
            return "No relevant information was found in the knowledge base."

        chunks = []
        for result in results:
            text_value = result.get("content", {}).get("text")
            if text_value:
                chunks.append(text_value)

        if not chunks:
            return "The knowledge base returned results without readable text."

        return "\n---\n".join(chunks)

    except Exception as exc:
        logger.exception("Knowledge base retrieval failed")
        return f"Knowledge base search failed: {exc}"


# ---------------------------------------------------------------------------
# Loyalty discount / Code Interpreter
# ---------------------------------------------------------------------------
@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points: Customer's current points balance
        tier: Customer tier — Silver, Gold, or Platinum
        order_total: Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    code = f"""
import json
import math

loyalty_points = {int(loyalty_points)}
tier = {tier!r}
order_total = {float(order_total)!r}
product_category = {product_category!r}

earn_rates = {{
    "standard": 1,
    "device": 2,
    "fresh": 5,
}}
tier_rates = {{
    "Silver": 0.00,
    "Gold": 0.10,
    "Platinum": 0.15,
}}

tier_discount_pct = tier_rates.get(tier, 0.00)

# 100 points = $1, with redemption in 500-point increments.
# Redemption cannot exceed 50% of the order subtotal.
max_redeem_points = math.floor((order_total * 0.50) * 100 / 500) * 500
points_redeemed = min(
    math.floor(loyalty_points / 500) * 500,
    max_redeem_points,
)

points_discount = points_redeemed / 100
subtotal_after_points = max(order_total - points_discount, 0.0)

tier_discount = subtotal_after_points * tier_discount_pct
final_total = max(subtotal_after_points - tier_discount, 0.0)
total_savings = order_total - final_total

# Points are earned on the discounted final purchase amount.
points_earned = math.floor(final_total * earn_rates.get(product_category, 1))
remaining_points = loyalty_points - points_redeemed + points_earned

result = {{
    "points_redeemed": points_redeemed,
    "tier_discount_pct": tier_discount_pct * 100,
    "final_total": round(final_total, 2),
    "remaining_points": remaining_points,
    "total_savings": round(total_savings, 2),
    "points_earned": points_earned,
}}

print(json.dumps(result))
"""

    try:
        with code_session(REGION) as code_client:
            response = code_client.invoke(
                "executeCode",
                {
                    "code": code,
                    "language": "python",
                    "clearContext": True,
                },
            )

        for event in response.get("stream", []):
            if "result" in event:
                return json.dumps(event["result"])

        return "Code Interpreter returned no result."

    except Exception as exc:
        # Required fallback: calculate only the tier discount locally.
        tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        tier_discount_pct = tier_rates.get(tier, 0.00)
        final_total = round(order_total * (1 - tier_discount_pct), 2)

        fallback = {
            "points_redeemed": 0,
            "tier_discount_pct": tier_discount_pct * 100,
            "final_total": final_total,
            "remaining_points": loyalty_points,
            "total_savings": round(order_total - final_total, 2),
            "points_earned": 0,
            "calculation_mode": "tier-only fallback",
            "reason": str(exc),
        }
        return json.dumps(fallback)


# ---------------------------------------------------------------------------
# Agent entrypoint
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a helpful e-commerce customer support assistant.

Use the available tools deliberately:
- Use Gateway order tools for real order/customer information and refund tools
  for return/refund requests. Do not invent order details.
- For an order ID such as ORD-001, use order-tracker___get_customer with
  {"order_id": "..."} to retrieve the order status, tracking number, carrier,
  items, total, and estimated delivery.
- Use order-tracker___get_customer_orders with {"customer_id": "..."} when the
  customer asks for their orders or order history.
- Use order-tracker___get_order with {"customer_id": "..."} when customer
  profile information such as name, loyalty points, or tier is needed.
- Use the knowledge base for product information, policies, loyalty tiers,
  warranties, and order-status definitions. Treat retrieved knowledge-base
  content as the authoritative catalog/support source.
- Use calculate_loyalty_discount for loyalty calculations instead of doing
  the arithmetic yourself. Treat its returned JSON values as authoritative:
  report points_redeemed, tier_discount_pct, final_total, total_savings,
  remaining_points, and points_earned exactly as returned. Do not recalculate
  or reinterpret the dollar value of redeemed points.
- Always provide product_category when calling calculate_loyalty_discount:
  classify electronics/devices such as Wireless Headphones Pro, Kindle
  Paperwhite, Echo Dot 5th Gen, and Amazon Halo Rise as "device"; classify
  grocery/fresh products as "fresh"; use "standard" for other products.
- Use the browser when the customer explicitly asks you to visit a live web
  page or retrieve current information from a URL. For browser tasks, always
  call the browser with an "init_session" action first using a unique
  session_name of at least 10 characters, then use "navigate" with the
  requested URL and the same session_name. Use "get_text" or other browser
  actions as needed to answer the request.
- Use customer context supplied by memory to personalize responses, but do not
  claim to remember something that is not present in the supplied context.

Be concise, accurate, and transparent about tool failures. Never fabricate
tracking numbers, refund IDs, prices, policy details, or customer information.
"""


@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    try:
        user_input = payload.get("prompt")
        if not isinstance(user_input, str) or not user_input.strip():
            return "Error: 'prompt' must be a non-empty string."

        actor_id = str(payload.get("customer_id") or "anonymous")
        session_id = str(payload.get("session_id") or uuid.uuid4())

        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )

        agent_core_browser = AgentCoreBrowser(region=REGION)

        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            agent_core_browser.browser,
        ]

        mcp_client = MCPClient(
            lambda: streamable_http_client(GATEWAY_URL)
        )

        # The MCP connection must remain open while the agent uses its tools.
        with mcp_client:
            gateway_tools = mcp_client.list_tools_sync()
            tools.extend(gateway_tools)

            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=SYSTEM_PROMPT,
            )
            response = agent(user_input)

        return response.message["content"][0]["text"]

    except Exception as exc:
        logger.exception("Agent invocation failed")
        return f"Sorry, I could not complete that request: {exc}"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()