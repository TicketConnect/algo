# TicketConnect Algo

A FastAPI application that provides OpenAI-compatible vector store endpoints using PGVector and LiteLLM proxy for embeddings.

---

## TicketConnect Recommendation Pipeline

This service is the embedding layer **and the search index** for TicketConnect's "similar events" feature. The full data flow:

```
                                                       ┌──────────┐
                                                       │ LiteLLM  │
                                                       │  proxy   │
                                                       └────▲─────┘
                                                            │
┌──────────────┐  create event   ┌─────────────┐   /embed   │
│  Organizer   │ ─ name + tags ─►│   Backend   │────────────┘
│   (panel)    │   description   │  (Elysia)   │
└──────────────┘                 └──┬─────┬────┘
                                    │     │  /v1/vector_stores/{id}/upsert
                          Mongo:    │     │   id=eventId, embedding, content
                Event.embedding ◄───┘     ▼
                                       ┌────────────────┐
                                       │ litellm-       │
                                       │ pgvector       │
                                       │ (this service) │
                                       └─────▲────┬─────┘
                                             │    │
                                             │    │  hits = [{id, score}]
              user opens event detail        │    ▼
              GET /similar/:eventId          │  ┌──────────────────┐
                              ▼              │  │  Backend phase 2: │
                  ┌──────────────────────┐   │  │  Mongo find()     │
                  │ source event's       │   │  │  filter by current│
                  │ embedding (Mongo)    │───┘  │  state, return    │
                  └──────────────────────┘      │  full event docs  │
                  POST /search-by-vector        └────────┬──────────┘
                                                         ▼
                                                ┌────────────────┐
                                                │  App           │
                                                │  SimilarEvents │
                                                │  scroller      │
                                                └────────────────┘
```

### How a "similar event" gets ranked (end to end)

1. **Organizer creates the event** in `ticketconnect-panel` with tags (curated list: rock, festival, outdoor, family-friendly, …) plus a free-text description.

2. **Backend persists to Mongo** and fires a non-blocking `RecommendationService.generateEventEmbedding(eventId)`.

3. **`buildEventText(event)`** assembles the embedding input. Tags are deliberately **repeated 4×** so the organizer's curated signal isn't drowned by long descriptions:
   ```
   "{name} {category} {tags*4} {description} {venue} {city}"
   ```

4. **Backend calls `POST /embed`** on this service:
   ```json
   POST http://algo:8000/embed
   { "text": "Sunset Boat Party house electronic ..." }
   → { "embedding": [0.012, -0.034, ...], "dimensions": 1536 }
   ```
   Internally LiteLLM routes to the configured embedding model (default `text-embedding-ada-002`, 1536 dims).

5. **Backend writes the vector to two stores:**
   - `Event.embedding` in Mongo (`select: false` — heavy field, never returned in normal queries). Used as the cache for the source-event vector and for taste-profile math.
   - `litellm-pgvector` via `POST /v1/vector_stores/{events_id}/upsert` with `id=eventId`. This is the search index — pgvector's HNSW ANN index runs the actual nearest-neighbor query. The index is auto-created at startup (HNSW, with an IVFFlat fallback for older pgvector versions); index creation is best-effort and never blocks startup.

   The pgvector mirror is **best-effort**: a failure logs but doesn't roll back the Mongo write. Worst case, the event is briefly absent from similar-events search until the next regeneration or a backfill.

6. **User behavior feeds the taste profile.** `recalculateUserTaste(walletAddress)` gathers every behavioral signal — likes (`User.likedEvents`, weight 1), recommendation clicks (weight 0.3), ticket purchases (weight 3), and scanned attendance (weight 4, the strongest statement of taste) — and weights each by signal strength × exponential time decay (**30-day half-life**). The result is stored twice on the user: `tasteEmbedding` (overall weighted mean, the v1 ranker input) and `tasteEmbeddings` (up to 3 interest centroids from weighted spherical k-means, the v2 ranker input — so a techno + kids'-theater user gets both interests served instead of their meaningless average). Purchases, scans, and clicks queue a debounced recalc automatically (`queueTasteRecalc`).

7. **User opens event detail.** The app calls `GET /api/recommendations/similar/:eventId`. Backend runs a **two-phase query**:

   - **Phase 1 — pgvector ranking.** Backend reads the source event's embedding from Mongo (cheap), then `POST /v1/vector_stores/{events_id}/search-by-vector` returns the top N nearest neighbors by cosine distance. Over-fetches `limit × 4` so we have headroom for filtering. Filters by a **per-model threshold** (default 0.75 for ada-002 similar-events; see Configuration below — swap the embedding model and the thresholds must be recalibrated).
   - **Phase 2 — Mongo state filter + domain re-rank.** Backend looks up those candidate `eventId`s in Mongo with `isPublished: true`, `eventDate ≥ now`, dropping deleted/unpublished/past candidates. On the **v2 ranker** the survivors are then re-ranked: `cosine × date-proximity boost (≤ +10%) × same-city boost (+10%) × recent-sales boost (≤ +8%)`, followed by **MMR diversity** (λ=0.75, max 2 events per organizer) so the shelf isn't five copies of one organizer's weekly series. The v1 ranker (A/B control, `RECS_V2_PERCENT`) preserves pure cosine order.

   **Why two phases?** pgvector knows vectors but not application state (publish status, date moves). Mongo knows state but doesn't have an index for vector similarity. Splitting the work means each store does what it's best at, with no metadata sync between them.

   **Fallback:** if pgvector is unavailable, backend falls back to the legacy Mongo+JS scan so the feature degrades to "slower but works" instead of "broken."

8. **App renders** the horizontal `SimilarEvents` scroller (`app/components/EventDetail/SimilarEvents.tsx`). Clicks are logged to `/api/recommendations/click` for analytics — the response's `algorithm` field (`semantic_v1` / `semantic_v2` / `popularity_v1` / `mongo_scan_v1`) is echoed with each click so CTR is comparable per ranker (`GET /api/recommendations/analytics`).

   **Cold start:** users without a taste profile get a popularity fallback (top ticket sellers of the last 30 days, padded with the soonest upcoming events) instead of an empty feed.

### Service boundaries

| Concern | Owner | Notes |
|---------|-------|-------|
| Embedding model selection | LiteLLM proxy config | Swap models without backend changes |
| Text → vector | `litellm-pgvector` `/embed` | Stateless, pure transform |
| Vector storage (cache) | MongoDB `Event.embedding`, `User.tasteEmbedding` | Source of truth for taste-profile math |
| Vector storage (index) | `litellm-pgvector` `embeddings` table | Search index keyed by `eventId` |
| Similarity ranking | `litellm-pgvector` `/search-by-vector` | Cosine distance via pgvector index |
| State filtering (published, future) | MongoDB | `eventDate`, `isPublished`, `status` live here |

### TicketConnect-specific endpoints

These extend the OpenAI-compatible API for our use case:

| Endpoint | Purpose |
|----------|---------|
| `POST /embed` | Pure text→vector. No store side effects. |
| `POST /v1/vector_stores/{store}/upsert` | Idempotent insert/update keyed by application id (`eventId`). Replaces the `gen_random_uuid()` PK. |
| `POST /v1/vector_stores/{store}/search-by-vector` | Search by precomputed embedding (no LLM call). Supports `exclude_ids` and metadata `filters`. |
| `DELETE /v1/vector_stores/{store}/embeddings/{id}` | Idempotent delete. Called on event deletion. |
| `GET /v1/vector_stores/by-name/{name}` | Look up store by name; lets backend self-bootstrap (find-or-create). |
| `GET /v1/vector_stores/{store}/stats` | `{ "id", "embeddings" }` row count for the store. Called by the backend's nightly drift reconciler. |

### Bootstrap

The events vector store is created automatically on first need. The backend looks up `name=ticketconnect-events` via `GET /v1/vector_stores/by-name/...`; on 404 it `POST /v1/vector_stores` to create it, then caches the id for the rest of the process lifetime. No manual setup required.

If you need to reset the index (e.g. after a model swap that changed embedding dimensions), drop the embeddings table or delete the store — the backend will recreate it on the next embedding call.

### Operational notes

- **First-time deploy:** if your Mongo already has events with embeddings (e.g. you added pgvector mirroring after the fact), run the one-shot mirror to populate pgvector without re-paying for embedding generation:
  ```bash
  curl -X POST 'http://backend/api/recommendations/backfill?mode=pgvector-only'
  ```

- **Full backfill** (regenerate embeddings + mirror to pgvector) — use after first deploy of the embedding pipeline, or after touching `buildEventText`:
  ```bash
  curl -X POST 'http://backend/api/recommendations/backfill'
  ```

- **Embedding service health:** `GET /api/recommendations/health` on the backend pings `POST /embed` end-to-end.

- **Threshold tuning:** thresholds are keyed on the embedding model and overridable per knob; see Configuration. Raise if results feel loose, lower if shelves are too often empty. Validate any change with the offline eval: `bun src/scripts/eval-recall.ts` (leave-one-out recall@10 / MRR for both rankers, read-only).

- **Configuration (backend env vars):**
  - `EMBEDDING_SERVICE_URL` (default `http://localhost:8000`)
  - `EMBEDDING_SERVICE_API_KEY` (default `sk-1234`) — must match `SERVER_API_KEY` on this service.
  - `EMBEDDING_MODEL` (default `text-embedding-ada-002`) — selects the per-model threshold defaults; must match the LiteLLM proxy's embedding model. After a swap: full reindex (`/backfill`) + recalibrate.
  - `RECS_SIMILAR_THRESHOLD` / `RECS_FEED_THRESHOLD` — explicit threshold overrides.
  - `RECS_V2_PERCENT` (default `100`) — share of traffic on the v2 ranker (multi-centroid + re-rank + MMR); the rest gets v1 as the A/B control.
  - `RECS_TASTE_DEBOUNCE_MS` (default `60000`) — debounce window for behavioral taste recalcs.

- **After pulling this change:** run `prisma generate` — the `UserRating` and `UserPreference` schema models were removed. Optionally `DROP TABLE user_rating; DROP TABLE user_preference;` — the service no longer reads them.

### Drift detection

Mongo and pgvector can drift if a pgvector write fails silently. Two mitigations:

1. **Self-healing on read:** the search-by-vector path tolerates pgvector returning ids that don't exist in Mongo (deleted events) — they're filtered out in phase 2.
2. **Periodic reconciler:** the backend ships a nightly reconciler (`Backend/src/services/recommendationReconciler.ts`) that compares `count(events with embedding in Mongo)` against this service's `GET /v1/vector_stores/{store}/stats` and triggers a `?mode=pgvector-only` backfill when they diverge.

---

## Generic API Reference (below)


## Features

- 🔌 OpenAI-compatible API endpoints
- 🗄️ PGVector for efficient vector storage and similarity search
- 🎛️ Configurable database field mappings
- 🔄 LiteLLM proxy integration for any embedding model
- 🐳 Docker support
- ⚡ FastAPI with async support

## API Endpoints

### 1. Create Vector Store
```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Support FAQ"
  }'
```

### 2. List Vector Stores
```bash
# List all vector stores
curl -X GET \
  http://localhost:8000/v1/vector_stores \
  -H "Authorization: Bearer your-api-key"

# List with pagination (limit and after parameters)
curl -X GET \
  "http://localhost:8000/v1/vector_stores?limit=10&after=vs_abc123" \
  -H "Authorization: Bearer your-api-key"
```

### 3. Add Single Embedding to Vector Store
```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores/vs_abc123/embeddings \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Our return policy allows returns within 30 days of purchase.",
    "embedding": [0.1, 0.2, 0.3, ...],
    "metadata": {
      "category": "returns",
      "source": "faq",
      "id": "return_policy_1"
    }
  }'
```

### 4. Add Multiple Embeddings (Batch)
```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores/vs_abc123/embeddings/batch \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "embeddings": [
      {
        "content": "Our return policy allows returns within 30 days of purchase.",
        "embedding": [0.1, 0.2, 0.3, ...],
        "metadata": {"category": "returns"}
      },
      {
        "content": "Shipping is free for orders over $50.",
        "embedding": [0.4, 0.5, 0.6, ...],
        "metadata": {"category": "shipping"}
      }
    ]
  }'
```

### 5. Search Vector Store
```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores/vs_abc123/search \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the return policy?",
    "limit": 20,
    "filters": {"category": "support"}
  }'
```

## Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```bash
# Database Configuration
DATABASE_URL="postgresql://username:password@localhost:5432/vectordb?schema=public"

# API Configuration
SERVER_API_KEY="your-api-key-here"

# Server Configuration
HOST="0.0.0.0"
PORT=8000

# LiteLLM Proxy Configuration
EMBEDDING__MODEL="text-embedding-ada-002"
EMBEDDING__BASE_URL="http://localhost:4000"
EMBEDDING__API_KEY="sk-1234"
EMBEDDING__DIMENSIONS=1536

# Database Field Configuration (optional)
DB_FIELDS__ID_FIELD="id"
DB_FIELDS__CONTENT_FIELD="content"
DB_FIELDS__METADATA_FIELD="metadata"
DB_FIELDS__EMBEDDING_FIELD="embedding"
DB_FIELDS__VECTOR_STORE_ID_FIELD="vector_store_id"
DB_FIELDS__CREATED_AT_FIELD="created_at"
```

### Database Field Mapping

You can customize the database field names by setting environment variables:

- `DB_FIELDS__ID_FIELD` - Primary key field (default: "id")
- `DB_FIELDS__CONTENT_FIELD` - Text content field (default: "content")
- `DB_FIELDS__METADATA_FIELD` - JSON metadata field (default: "metadata")
- `DB_FIELDS__EMBEDDING_FIELD` - Vector embedding field (default: "embedding")
- `DB_FIELDS__VECTOR_STORE_ID_FIELD` - Foreign key field (default: "vector_store_id")
- `DB_FIELDS__CREATED_AT_FIELD` - Timestamp field (default: "created_at")

### LiteLLM Proxy Configuration

The application uses LiteLLM proxy for embeddings. Configure it with:

- `EMBEDDING__MODEL` - Model name (e.g., "text-embedding-ada-002")
- `EMBEDDING__BASE_URL` - LiteLLM proxy URL (e.g., "http://localhost:4000")
- `EMBEDDING__API_KEY` - LiteLLM proxy API key
- `EMBEDDING__DIMENSIONS` - Embedding dimensions (default: 1536)

## Setup and Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Database Setup

```bash
# Generate Prisma client
prisma generate

# Run database migrations
prisma db push
```

### 3. Set up LiteLLM Proxy

Start LiteLLM proxy pointing to your preferred embedding model:

```bash
# Example: Start LiteLLM proxy for OpenAI
litellm --model text-embedding-ada-002 --port 4000
```

### 4. Run the Application

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker Deployment

### Build and run with Docker:

```bash
# Build the image
docker build -t vector-store-api .

# Run the container
docker run -p 8000:8000 --env-file .env vector-store-api
```

## Database Schema

The application uses two main tables:

### vector_stores
- `id` (string, primary key)
- `name` (string)
- `file_counts` (json)
- `status` (string)
- `usage_bytes` (integer)
- `created_at` (timestamp)
- `expires_after` (json, optional)
- `expires_at` (timestamp, optional)
- `last_active_at` (timestamp, optional)
- `metadata` (json, optional)

### embeddings
- `id` (string, primary key)
- `vector_store_id` (string, foreign key)
- `content` (string)
- `embedding` (vector(1536))
- `metadata` (json, optional)
- `created_at` (timestamp)

## Supported Models

Any embedding model supported by LiteLLM proxy can be used. Examples:

- OpenAI: `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`
- Cohere: `embed-english-v3.0`, `embed-multilingual-v3.0`
- Voyage: `voyage-2`, `voyage-large-2`
- And many more...

## API Response Format

### Vector Store Response
```json
{
  "id": "vs_abc123",
  "object": "vector_store",
  "created_at": 1699024800,
  "name": "Support FAQ",
  "usage_bytes": 0,
  "file_counts": {
    "in_progress": 0,
    "completed": 0,
    "failed": 0,
    "cancelled": 0,
    "total": 0
  },
  "status": "completed",
  "metadata": {}
}
```

### Vector Store List Response
```json
{
  "object": "list",
  "data": [
    {
      "id": "vs_abc123",
      "object": "vector_store",
      "created_at": 1699024800,
      "name": "Support FAQ",
      "usage_bytes": 1024,
      "file_counts": {"completed": 5, "total": 5},
      "status": "completed",
      "metadata": {}
    }
  ],
  "first_id": "vs_abc123",
  "last_id": "vs_def456",
  "has_more": false
}
```

### Search Response
```json
{
  "object": "vector_store.search",
  "data": [
    {
      "id": "emb_123",
      "content": "Return policy text...",
      "score": 0.95,
      "metadata": {"category": "support"}
    }
  ],
  "usage": {
    "total_tokens": 1
  }
}
```

## Example Search Request

```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores/vs_support_faq/search \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I return an item?",
    "limit": 5,
    "return_metadata": true
  }'
```

## Health Check

```bash
curl http://localhost:8000/health
```

## Migrating Existing Data

If you have an existing database with embeddings and content, you can easily migrate using the embedding APIs:

### 1. Create Vector Store
First, create a vector store for your data:

```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Migrated Data",
    "metadata": {"source": "legacy_system"}
  }'
```

### 2. Batch Insert Embeddings
Use the batch endpoint to efficiently insert multiple embeddings:

```bash
curl -X POST \
  http://localhost:8000/v1/vector_stores/vs_your_id/embeddings/batch \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "embeddings": [
      {
        "content": "Your text content here",
        "embedding": [0.1, 0.2, 0.3, ...1536 dimensions...],
        "metadata": {"source_id": "doc_123", "category": "support"}
      }
    ]
  }'
```

### 3. Migration Script Example

Here's a Python script example for migrating from an existing database:

```python
import psycopg2
import requests
import json

# Connect to your existing database
conn = psycopg2.connect("your_existing_db_url")
cur = conn.cursor()

# Fetch existing data
cur.execute("SELECT content, embedding, metadata FROM your_table")
rows = cur.fetchall()

# Prepare batch data
embeddings = []
for content, embedding, metadata in rows:
    embeddings.append({
        "content": content,
        "embedding": embedding.tolist(),  # Convert numpy array to list
        "metadata": metadata or {}
    })

# Send batch to API
response = requests.post(
    "http://localhost:8000/v1/vector_stores/your_vector_store_id/embeddings/batch",
    headers={
        "Authorization": "Bearer your-api-key",
        "Content-Type": "application/json"
    },
    json={"embeddings": embeddings}
)

print(f"Migrated {len(embeddings)} embeddings")
```

## License

MIT License
