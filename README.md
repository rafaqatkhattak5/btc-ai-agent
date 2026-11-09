# Bitcoin Market Assistant: a tool-calling AI agent

A chat agent built with LangChain and Gemini. It decides which tool to call, runs it,
reads the result and answers. It remembers each conversation and shows which tools it used.

## Demo
(video link and screenshots go here)

## How it works
1. **Gradio chat page:** the user types a question. Each visitor gets a separate session.
2. **LangChain agent:** Gemini chooses a tool, the agent runs it and sends the result back.
3. **Three tools:**
   - `get_btc_price`: live Bitcoin price and 24h change (CoinGecko)
   - `predict_next_day_btc`: next-day estimate from a GRU model I trained (TensorFlow/Keras)
   - `convert_currency`: daily exchange rates (Exchange Rate API)
4. **Memory:** LangGraph checkpointer keyed by session ID, so follow-ups like "and in rupees?" work.

## Safety and reliability
- Daily question budget to protect the free Gemini quota
- Input length limit
- Friendly error messages (technical errors only appear in the logs)
- Tools return error dictionaries instead of crashing
- API key is read from an environment variable, never stored in the code

## Honest limitations
- The GRU model does **not** beat a simple "same as today" baseline. Bitcoin's next-day
  move is close to a random walk. It is included to show how an agent uses a trained model
  as a tool, not as financial advice.
- Uses the free Gemini tier, so daily limits apply.
- The demo is not hosted permanently.

## Run it yourself
    pip install -r requirements.txt
    export GOOGLE_API_KEY="your-key-from-aistudio"
    python app.py

In Google Colab, store the key in Secrets and copy it into `GOOGLE_API_KEY` first.

## What I learned
- How tool calling works: docstrings, arguments, results fed back to the model
- Writing the agent loop by hand first, then using LangChain's `create_agent`
- Conversation memory with thread IDs
- Building and protecting a public chat interface
- Testing every tool on its own before trusting the agent

## Credits
Prices from CoinGecko. Rates by Exchange Rate API (exchangerate-api.com).
Educational project, not financial advice.
