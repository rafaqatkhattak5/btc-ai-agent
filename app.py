import os
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import requests
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from tensorflow.keras.models import load_model

BASE = Path(__file__).parent
WINDOW = 60
MODEL_NAME = "gemini-2.5-flash-lite"   # use the model that works for you
MAX_QUESTIONS_PER_DAY = 8              # protects your free Gemini quota

gru_model = load_model(BASE / "btc_gru_returns.keras")
r_scaler = joblib.load(BASE / "r_scaler.pkl")


@tool
def get_btc_price() -> dict:
    """Get the current Bitcoin price in USD and its percentage change over the last 24 hours."""
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin", "vs_currencies": "usd", "include_24hr_change": "true"},
            timeout=30,
        )
        r.raise_for_status()
        d = r.json()["bitcoin"]
        return {"price_usd": d["usd"], "change_24h_pct": round(d["usd_24h_change"], 2)}
    except Exception as e:
        return {"error": str(e)}


@tool
def predict_next_day_btc() -> dict:
    """Predict Bitcoin's next-day price with a GRU deep learning model. Returns the last price, the predicted next-day price and the predicted percent change. Educational estimate only: this model does not beat a simple 'same as today' guess."""
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
            params={"vs_currency": "usd", "days": 90, "interval": "daily"},
            timeout=30,
        )
        r.raise_for_status()
        prices = [p[1] for p in r.json()["prices"]][-(WINDOW + 1):]
        if len(prices) < WINDOW + 1:
            return {"error": "Not enough price history"}
        p = np.array(prices)
        rets = np.log(p[1:] / p[:-1]).reshape(-1, 1)
        x = r_scaler.transform(rets).reshape(1, WINDOW, 1)
        pred_ret = float(r_scaler.inverse_transform(gru_model.predict(x, verbose=0))[0, 0])
        return {
            "last_price_usd": round(float(p[-1]), 2),
            "predicted_next_price_usd": round(float(p[-1] * np.exp(pred_ret)), 2),
            "predicted_change_pct": round(pred_ret * 100, 3),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between currencies using the latest daily exchange rate. Use 3-letter codes such as USD, PKR, EUR, GBP, AED, SAR."""
    try:
        base, target = from_currency.upper(), to_currency.upper()
        r = requests.get(f"https://open.er-api.com/v6/latest/{base}", timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("result") != "success":
            return {"error": "Exchange rate service error"}
        rate = data["rates"].get(target)
        if rate is None:
            return {"error": f"Unknown currency code: {target}"}
        return {"amount": amount, "from": base, "to": target, "rate": rate,
                "converted": round(amount * rate, 2)}
    except Exception as e:
        return {"error": str(e)}


SYSTEM_PROMPT = (
    "You are a helpful crypto market assistant. Be short and clear. "
    "Use tools for live data and never invent prices. "
    "For prices in another currency, get the USD price first, then use convert_currency. "
    "If the user asks for a price again, fetch fresh data. "
    "When you show a prediction, say it is an educational estimate, and explain in plain words "
    "that very small changes (under about 0.5%) are within normal noise. "
    "If a request needs data you have no tool for (other coins, news), say so instead of guessing."
)

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, api_version="v1beta")
agent = create_agent(
    model=llm,
    tools=[get_btc_price, predict_next_day_btc, convert_currency],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)

TOOL_LABELS = {
    "get_btc_price": "Checked the live Bitcoin price",
    "predict_next_day_btc": "Ran the GRU prediction model",
    "convert_currency": "Converted the currency",
}

usage = {"day": None, "count": 0}


def allowed():
    today = datetime.now(timezone.utc).date()
    if usage["day"] != today:
        usage["day"], usage["count"] = today, 0
    if usage["count"] >= MAX_QUESTIONS_PER_DAY:
        return False
    usage["count"] += 1
    return True


def respond(message, history, request: gr.Request):
    if not allowed():
        return "Today's free demo limit has been reached. Please come back tomorrow."

    message = (message or "")[:500]          # cap the input length
    thread = getattr(request, "session_hash", None) or "default"

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            {"configurable": {"thread_id": thread}},
        )
    except Exception as e:
        print("Agent error:", e)               # visible in the Space's Logs tab
        return "Sorry, the AI service is busy or has reached its daily limit. Please try again later."

    msgs = result["messages"]
    last_human = max(i for i, m in enumerate(msgs) if m.type == "human")
    steps = []
    for m in msgs[last_human + 1:]:
        if m.type == "ai" and m.tool_calls:
            for c in m.tool_calls:
                steps.append(TOOL_LABELS.get(c["name"], c["name"]))

    text = msgs[-1].content
    if isinstance(text, list):
        text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)
    if steps:
        text += "\n\n_Steps: " + " → ".join(steps) + "_"
    return text


demo = gr.ChatInterface(
    fn=respond,
    title="Bitcoin Market Assistant",
    description=(
        "Ask for the live Bitcoin price, a next-day estimate from a GRU model, "
        "or convert amounts into other currencies. Educational demo, not financial advice."
    ),
    examples=[
        "What is Bitcoin trading at right now?",
        "Convert the Bitcoin price to Pakistani rupees",
        "What does the model predict for tomorrow?",
    ],
    cache_examples=False,
)

demo.launch()
