# Krishi.ai backend

Flask backend matched to the supplied single-file `krishi.html` frontend.

## What it includes

- Serves the complete frontend at `/` and `/chat`.
- Implements the exact `POST /chat` and `POST /translate` JSON contracts used by the HTML.
- Calls OpenRouter using a server-side environment variable. The default model is a named free model (`nvidia/nemotron-3-ultra-550b-a55b:free`) rather than a router that randomly selects a free model. Ordered model fallbacks are configured by `OPENROUTER_FALLBACK_MODELS`, using API-compatible models (the fallback list avoids agent-harness-only models). Free model availability and quality still vary; a paid endpoint is more appropriate for production reliability.
- Uses prior chat messages sent by the frontend, with validation and size limits.
- Retrieves relevant snippets from local knowledge files (RAG). The default lightweight retrieval requires no extra ML package. Optional ChromaDB + Sentence Transformers support is available.
- Includes health checks at `/health` and `/api/health`.
- Fetches a current/3-day weather forecast from Open-Meteo when the farmer explicitly names an Indian location in the question. The answer is labelled as model forecast data, not an official IMD warning.
- Supports optional live government mandi-price lookup through the data.gov.in API when its API key and resource ID are configured.
- Avoids inventing prices or weather when no matching provider data was returned, and avoids unsafe generic pesticide dosage/fertilizer claims.
- Adds limited optional CORS for a separately hosted frontend.

## Run on Windows PowerShell

1. Extract this folder somewhere convenient.
2. Create and activate a virtual environment:

   ```powershell
   cd path\to\krishi_backend
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Create your `.env` from the example and add your key:

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

   Set `OPENROUTER_API_KEY` to the key from your OpenRouter account. If you already have a `.env` file, keep your existing key and update the `OPENROUTER_MODEL` and `OPENROUTER_FALLBACK_MODELS` lines to match `.env.example`. Do not put the key in `krishi.html`, GitHub, or a public frontend. The default `OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free` is a named free model. Recommended API fallback models are configured in `.env.example`. If you use a different model in `.env`, it must be an API model enabled for your OpenRouter key.

4. Start the backend:

   ```powershell
   python app.py
   ```

5. Open `http://127.0.0.1:5000`. Check `http://127.0.0.1:5000/health` for backend status.

## API contract

### `POST /chat`

Request:

```json
{
  "message": "My tomato leaves have yellow spots. What should I check?",
  "language": "en",
  "history": []
}
```

Response:

```json
{
  "success": true,
  "response": "...",
  "suggestions": [],
  "sources": [],
  "model": "nvidia/nemotron-3-ultra-550b-a55b:free"
}
```

The frontend only requires `success` and `response`; additional response fields are optional. Valid language codes include `en`, `hi`, `mr`, `kn`, `ta`, `te`, `ml`, `gu`, `pa`, `bho`, `kok`, `ha`, and `bn`.

### `POST /translate`

Request:

```json
{"text": "What fertilizer should I use for my crops?", "language": "mr"}
```

Response:

```json
{"success": true, "translated": "..."}
```

If translation service is unavailable, the endpoint returns a non-2xx response. The current HTML already catches this and uses its original suggestion.

## Retrieval / RAG

Put trusted `.md`, `.txt`, `.rst`, or `.csv` references into `knowledge/`. On startup the backend loads them and splits long files into overlapping chunks. By default it uses lexical retrieval to keep local installation quick and light.

To enable vector retrieval with a local persistent Chroma database:

```powershell
pip install -r requirements-rag.txt
```

Then set `USE_CHROMA=true` in `.env` and start the backend. The first run may download the Sentence Transformers embedding model. If vector dependencies fail to initialize, the server logs a warning and falls back to lexical retrieval.

Do not assume RAG alone verifies truth. Keep source files authoritative and up to date. Crop-specific fertilizer and pesticide recommendations should be grounded in reliable local sources.

## Live weather and mandi price setup

**Weather:** No API key is needed. If the farmer includes a clear place name, such as `weather in Chikodi, Karnataka today`, the backend looks up the place and retrieves current conditions plus a three-day forecast from Open-Meteo. If no explicit location is provided, the assistant should ask for one instead of guessing. Open-Meteo data is a model forecast; for high-stakes farm decisions, cross-check current IMD/KVK advisories. Attribute the weather source as Open-Meteo.

**Mandi prices:** Get an API key from the Government of India's [Open Government Data Platform](https://data.gov.in/), then open the current daily mandi-price dataset published by the Directorate of Marketing & Inspection and copy its API resource ID. Set `DATA_GOV_IN_API_KEY` and `DATA_GOV_IN_RESOURCE_ID` in `.env`. The endpoint supports a limited set of common commodities and requires an explicit crop plus a town/mandi/district in the user's question. Always check the returned arrival date, market, variety, and units. If either setting is missing or no record matches, the assistant is instructed not to make up a price.

Other limitations:

- Voice input uses browser speech recognition. Assistant speech output is optional and starts only when the farmer taps the speaker icon.
- PDFs must currently be converted to `.txt` or Markdown to be indexed.
- If the AI provider is unavailable, the app returns a short localized service-unavailable message rather than exposing raw retrieval chunks as if they answered the question. AI-powered answers and translations require a valid `OPENROUTER_API_KEY`.

## Deployment

For a hosted deployment, set environment variables in the host dashboard rather than committing `.env`. Use a production WSGI server such as Gunicorn (add it to your deployment requirements if your host needs it), persist `data/chroma` if using Chroma, and ensure `knowledge/` is included in the deployment. If the frontend is hosted separately, set `CORS_ORIGINS` to its exact HTTPS origin and update `API_BASE` inside the HTML to your backend URL.
