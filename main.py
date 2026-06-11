import os
import asyncio
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from prisma import Prisma
from pydantic import BaseModel
from dotenv import load_dotenv

from models import (
    VectorStoreCreateRequest,
    VectorStoreResponse,
    VectorStoreSearchRequest,
    VectorStoreSearchResponse,
    SearchResult,
    EmbeddingCreateRequest,
    EmbeddingResponse,
    EmbeddingBatchCreateRequest,
    EmbeddingBatchCreateResponse,
    VectorStoreListResponse,
    ContentChunk,
    RatingCreateRequest,
    RatingResponse,
    UserPreferenceCreateRequest,
    UserPreferenceResponse,
    SimilarEventsResponse,
    EventScoreRequest,
    FeedRecommendationRequest,
    FeedRecommendationItem,
    FeedRecommendationResponse,
)
from config import settings
from embedding_service import embedding_service

load_dotenv()

app = FastAPI(
    title="OpenAI Vector Stores API",
    description="OpenAI-compatible Vector Stores API using PGVector",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Prisma client
db = Prisma()

security = HTTPBearer()

MAX_EMBED_TEXT_CHARS = 20_000


async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API key from Authorization header"""
    expected_key = settings.server_api_key
    if credentials.credentials != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


@app.on_event("startup")
async def startup():
    """Connect to database on startup"""
    await db.connect()


@app.on_event("shutdown")
async def shutdown():
    """Disconnect from database on shutdown"""
    await db.disconnect()


async def generate_query_embedding(query: str) -> List[float]:
    """
    Generate an embedding for the query using LiteLLM
    """
    return await embedding_service.generate_embedding(query)


@app.post("/v1/vector_stores", response_model=VectorStoreResponse)
async def create_vector_store(
    request: VectorStoreCreateRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Create a new vector store.
    """
    try:
        # Use raw SQL to insert the vector store with configurable table/field names
        vector_store_table = settings.table_names["vector_stores"]
        
        result = await db.query_raw(
            f"""
            INSERT INTO {vector_store_table} (id, name, file_counts, status, usage_bytes, expires_after, metadata, created_at)
            VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, NOW())
            RETURNING id, name, file_counts, status, usage_bytes, expires_after, expires_at, last_active_at, metadata, 
                     EXTRACT(EPOCH FROM created_at)::bigint as created_at_timestamp
            """,
            request.name,
            {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
            "completed",
            0,
            request.expires_after,
            request.metadata or {}
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to create vector store")
            
        vector_store = result[0]
        
        # Convert to response format
        created_at = int(vector_store["created_at_timestamp"])
        expires_at = int(vector_store["expires_at"].timestamp()) if vector_store.get("expires_at") else None
        last_active_at = int(vector_store["last_active_at"].timestamp()) if vector_store.get("last_active_at") else None
        
        return VectorStoreResponse(
            id=vector_store["id"],
            created_at=created_at,
            name=vector_store["name"],
            usage_bytes=vector_store["usage_bytes"] or 0,
            file_counts=vector_store["file_counts"] or {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
            status=vector_store["status"],
            expires_after=vector_store["expires_after"],
            expires_at=expires_at,
            last_active_at=last_active_at,
            metadata=vector_store["metadata"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create vector store: {str(e)}")


@app.get("/v1/vector_stores", response_model=VectorStoreListResponse)
async def list_vector_stores(
    limit: Optional[int] = 20,
    after: Optional[str] = None,
    before: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """
    List vector stores with optional pagination.
    """
    try:
        limit = min(limit or 20, 100)  # Cap at 100 results
        
        vector_store_table = settings.table_names["vector_stores"]
        
        # Build base query
        base_query = f"""
        SELECT id, name, file_counts, status, usage_bytes, expires_after, expires_at, last_active_at, metadata,
               EXTRACT(EPOCH FROM created_at)::bigint as created_at_timestamp
        FROM {vector_store_table}
        """
        
        # Add pagination conditions
        conditions = []
        params = []
        param_count = 1
        
        if after:
            conditions.append(f"id > ${param_count}")
            params.append(after)
            param_count += 1
            
        if before:
            conditions.append(f"id < ${param_count}")
            params.append(before)
            param_count += 1
        
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        # Add ordering and limit
        final_query = base_query + f" ORDER BY created_at DESC LIMIT {limit + 1}"
        
        # Execute query
        results = await db.query_raw(final_query, *params)
        
        # Check if there are more results
        has_more = len(results) > limit
        if has_more:
            results = results[:limit]  # Remove extra result
        
        # Convert to response format
        vector_stores = []
        for row in results:
            created_at = int(row["created_at_timestamp"])
            expires_at = int(row["expires_at"].timestamp()) if row.get("expires_at") else None
            last_active_at = int(row["last_active_at"].timestamp()) if row.get("last_active_at") else None
            
            vector_store = VectorStoreResponse(
                id=row["id"],
                created_at=created_at,
                name=row["name"],
                usage_bytes=row["usage_bytes"] or 0,
                file_counts=row["file_counts"] or {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
                status=row["status"],
                expires_after=row["expires_after"],
                expires_at=expires_at,
                last_active_at=last_active_at,
                metadata=row["metadata"]
            )
            vector_stores.append(vector_store)
        
        # Determine first_id and last_id
        first_id = vector_stores[0].id if vector_stores else None
        last_id = vector_stores[-1].id if vector_stores else None
        
        return VectorStoreListResponse(
            data=vector_stores,
            first_id=first_id,
            last_id=last_id,
            has_more=has_more
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to list vector stores: {str(e)}")


@app.post("/v1/vector_stores/{vector_store_id}/search", response_model=VectorStoreSearchResponse)
@app.post("/vector_stores/{vector_store_id}/search", response_model=VectorStoreSearchResponse)
async def search_vector_store(
    vector_store_id: str,
    request: VectorStoreSearchRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Search a vector store for similar content.
    """
    try:
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1",
            vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")
        
        # Generate embedding for query
        query_embedding = await generate_query_embedding(request.query)
        query_vector_str = "[" + ",".join(map(str, query_embedding)) + "]"
        
        # Build the raw SQL query for vector similarity search
        limit = min(request.limit or 20, 100)  # Cap at 100 results
        
        # Base query with vector similarity using cosine distance
        # Use configurable field names
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]
        
        # Build query with proper parameter placeholders for Prisma
        param_count = 1
        query_params = [query_vector_str, vector_store_id]
        
        base_query = f"""
        SELECT 
            {fields.id_field},
            {fields.content_field},
            {fields.metadata_field},
            ({fields.embedding_field} <=> ${param_count}::vector) as distance
        FROM {table_name} 
        WHERE {fields.vector_store_id_field} = ${param_count + 1}
        """
        param_count += 2
        
        # Add metadata filters if provided
        filter_conditions = []
        
        if request.filters:
            for key, value in request.filters.items():
                filter_conditions.append(f"{fields.metadata_field}->>${param_count} = ${param_count + 1}")
                query_params.extend([key, str(value)])
                param_count += 2
        
        if filter_conditions:
            base_query += " AND " + " AND ".join(filter_conditions)
        
        # Add ordering and limit
        final_query = base_query + f" ORDER BY distance ASC LIMIT {limit}"
        
        # Execute the query
        results = await db.query_raw(final_query, *query_params)
        
        # Get user preferences for personalization
        user_prefs = {}
        try:
            prefs_result = await db.query_raw(
                """
                SELECT preference_key, preference_value
                FROM user_preference
                WHERE user_id = $1
                """,
                api_key  # Using API key as user identifier
            )
            for pref in prefs_result:
                user_prefs[pref["preference_key"]] = pref["preference_value"]
        except Exception:
            # If we can't get preferences, continue without personalization
            pass
        
        # Get user ratings for embeddings in this vector store to boost ranked results
        user_ratings = {}
        try:
            # Get ratings for embeddings that are in our results
            embedding_ids = [row[fields.id_field] for row in results]
            if embedding_ids:
                # Create placeholders for the IN clause
                placeholders = ",".join([f"${i}" for i in range(param_count, param_count + len(embedding_ids))])
                ratings_params = [api_key] + embedding_ids
                
                ratings_result = await db.query_raw(
                    f"""
                    SELECT embedding_id, rating
                    FROM user_rating
                    WHERE user_id = $1 AND embedding_id IN ({placeholders})
                    """,
                    *ratings_params
                )
                for rating in ratings_result:
                    user_ratings[rating["embedding_id"]] = rating["rating"]
        except Exception:
            # If we can't get ratings, continue without rating-based boosting
            pass
        
        # Convert results to SearchResult objects
        search_results = []
        for row in results:
            # Convert distance to similarity score (1 - normalized_distance)
            # Cosine distance ranges from 0 (identical) to 2 (opposite)
            similarity_score = max(0, 1 - (row['distance'] / 2))
            
            # Apply personalization based on user preferences and ratings
            embedding_id = row[fields.id_field]
            adjusted_score = similarity_score
            
            # Boost score based on user ratings (1-5 scale, assuming 3 is neutral)
            if embedding_id in user_ratings:
                rating = user_ratings[embedding_id]
                # Convert rating to a boost factor: 1-2 = negative boost, 4-5 = positive boost
                rating_boost = (rating - 3) * 0.1  # Each point above/below 3 gives 10% boost/penalty
                adjusted_score *= (1 + rating_boost)
            
            # Apply preference-based boosting
            # Example: if user has a preference for certain categories in metadata
            metadata = row[fields.metadata_field] or {}
            for pref_key, pref_value in user_prefs.items():
                if pref_key in metadata and metadata[pref_key] == pref_value:
                    # Boost score by 20% for matching preferences
                    adjusted_score *= 1.2
                    break  # Apply only one preference boost for simplicity
            
            # Ensure score stays in reasonable bounds
            adjusted_score = max(0.0, min(1.0, adjusted_score))
            
            # Extract filename from metadata or use a default
            filename = metadata.get('filename', 'document.txt')
            
            content_chunks = [ContentChunk(type="text", text=row[fields.content_field])]
            
            result = SearchResult(
                file_id=embedding_id,
                filename=filename,
                score=adjusted_score,
                attributes=metadata if request.return_metadata else None,
                content=content_chunks
            )
            search_results.append(result)
        
        # Re-sort results by adjusted score (descending)
        search_results.sort(key=lambda x: x.score, reverse=True)
        
        return VectorStoreSearchResponse(
            search_query=request.query,
            data=search_results,
            has_more=False,  # TODO: Implement pagination
            next_page=None
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/v1/vector_stores/{vector_store_id}/embeddings/{embedding_id}/rate", response_model=RatingResponse)
async def rate_embedding(
    vector_store_id: str,
    embedding_id: str,
    request: RatingCreateRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Rate an embedding (e.g., like/dislike or 1-5 rating).
    """
    try:
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1",
            vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")
        
        # Check if embedding exists and belongs to the vector store
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]
        
        embedding_result = await db.query_raw(
            f"""
            SELECT {fields.id_field} 
            FROM {table_name} 
            WHERE {fields.id_field} = $1 AND {fields.vector_store_id_field} = $2
            """,
            embedding_id,
            vector_store_id
        )
        
        if not embedding_result:
            raise HTTPException(status_code=404, detail="Embedding not found in vector store")
        
        # Check if user has already rated this embedding (using API key as user_id)
        # Upsert: update if exists, insert if not
        rating_result = await db.query_raw(
            f"""
            INSERT INTO user_rating (user_id, embedding_id, rating)
            VALUES ($1, $2, $3)
            ON CONFLICT (user_id, embedding_id) 
            DO UPDATE SET rating = EXCLUDED.rating, created_at = NOW()
            RETURNING id, user_id, embedding_id, rating, EXTRACT(EPOCH FROM created_at)::bigint as created_at_timestamp
            """,
            api_key,  # Using API key as user identifier
            embedding_id,
            request.rating
        )
        
        if not rating_result:
            raise HTTPException(status_code=500, detail="Failed to save rating")
            
        rating = rating_result[0]
        
        return RatingResponse(
            id=rating["id"],
            embedding_id=rating["embedding_id"],
            rating=rating["rating"],
            created_at=int(rating["created_at_timestamp"])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to rate embedding: {str(e)}")


@app.get("/v1/vector_stores/{vector_store_id}/embeddings/{embedding_id}/rating", response_model=RatingResponse)
async def get_user_rating(
    vector_store_id: str,
    embedding_id: str,
    api_key: str = Depends(get_api_key)
):
    """
    Get the current user's rating for an embedding.
    """
    try:
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1",
            vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")
        
        # Check if embedding exists and belongs to the vector store
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]
        
        embedding_result = await db.query_raw(
            f"""
            SELECT {fields.id_field} 
            FROM {table_name} 
            WHERE {fields.id_field} = $1 AND {fields.vector_store_id_field} = $2
            """,
            embedding_id,
            vector_store_id
        )
        
        if not embedding_result:
            raise HTTPException(status_code=404, detail="Embedding not found in vector store")
        
        # Get user's rating for this embedding
        rating_result = await db.query_raw(
            f"""
            SELECT id, user_id, embedding_id, rating, EXTRACT(EPOCH FROM created_at)::bigint as created_at_timestamp
            FROM user_rating
            WHERE user_id = $1 AND embedding_id = $2
            """,
            api_key,  # Using API key as user identifier
            embedding_id
        )
        
        if not rating_result:
            raise HTTPException(status_code=404, detail="Rating not found for this user and embedding")
            
        rating = rating_result[0]
        
        return RatingResponse(
            id=rating["id"],
            embedding_id=rating["embedding_id"],
            rating=rating["rating"],
            created_at=int(rating["created_at_timestamp"])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get rating: {str(e)}")


@app.post("/v1/user/preferences", response_model=UserPreferenceResponse)
async def set_user_preference(
    request: UserPreferenceCreateRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Set a user preference for personalized ranking.
    """
    try:
        # Upsert user preference
        preference_result = await db.query_raw(
            f"""
            INSERT INTO user_preference (user_id, preference_key, preference_value)
            VALUES ($1, $2, $3)
            ON CONFLICT (user_id, preference_key) 
            DO UPDATE SET preference_value = EXCLUDED.preference_value, updated_at = NOW()
            RETURNING id, user_id, preference_key, preference_value, EXTRACT(EPOCH FROM updated_at)::bigint as updated_at_timestamp
            """,
            api_key,  # Using API key as user identifier
            request.preference_key,
            request.preference_value
        )
        
        if not preference_result:
            raise HTTPException(status_code=500, detail="Failed to save preference")
            
        preference = preference_result[0]
        
        return UserPreferenceResponse(
            id=preference["id"],
            preference_key=preference["preference_key"],
            preference_value=preference["preference_value"],
            updated_at=int(preference["updated_at_timestamp"])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to set preference: {str(e)}")


@app.get("/v1/user/preferences/{preference_key}", response_model=UserPreferenceResponse)
async def get_user_preference(
    preference_key: str,
    api_key: str = Depends(get_api_key)
):
    """
    Get a user preference.
    """
    try:
        preference_result = await db.query_raw(
            f"""
            SELECT id, user_id, preference_key, preference_value, EXTRACT(EPOCH FROM updated_at)::bigint as updated_at_timestamp
            FROM user_preference
            WHERE user_id = $1 AND preference_key = $2
            """,
            api_key,  # Using API key as user identifier
            preference_key
        )
        
        if not preference_result:
            raise HTTPException(status_code=404, detail="Preference not found")
            
        preference = preference_result[0]
        
        return UserPreferenceResponse(
            id=preference["id"],
            preference_key=preference["preference_key"],
            preference_value=preference["preference_value"],
            updated_at=int(preference["updated_at_timestamp"])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get preference: {str(e)}")


@app.post("/v1/vector_stores/{vector_store_id}/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    vector_store_id: str,
    request: EmbeddingCreateRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Add a single embedding to a vector store.
    """
    try:
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1",
            vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")
        
        # Convert embedding to vector string format
        embedding_vector_str = "[" + ",".join(map(str, request.embedding)) + "]"
        
        # Insert embedding using configurable field names
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]
        
        result = await db.query_raw(
            f"""
            INSERT INTO {table_name} ({fields.id_field}, {fields.vector_store_id_field}, {fields.content_field}, 
                                     {fields.embedding_field}, {fields.metadata_field}, {fields.created_at_field})
            VALUES (gen_random_uuid(), $1, $2, $3::vector, $4, NOW())
            RETURNING {fields.id_field}, {fields.vector_store_id_field}, {fields.content_field}, 
                     {fields.metadata_field}, EXTRACT(EPOCH FROM {fields.created_at_field})::bigint as created_at_timestamp
            """,
            vector_store_id,
            request.content,
            embedding_vector_str,
            request.metadata or {}
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to create embedding")
            
        embedding = result[0]
        
        # Update vector store statistics
        await db.query_raw(
            f"""
            UPDATE {vector_store_table} 
            SET 
                file_counts = jsonb_set(
                    jsonb_set(
                        COALESCE(file_counts, '{{"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0}}'::jsonb),
                        '{{completed}}',
                        (COALESCE(file_counts->>'completed', '0')::int + 1)::text::jsonb
                    ),
                    '{{total}}',
                    (COALESCE(file_counts->>'total', '0')::int + 1)::text::jsonb
                ),
                usage_bytes = COALESCE(usage_bytes, 0) + LENGTH($2),
                last_active_at = NOW()
            WHERE id = $1
            """,
            vector_store_id,
            request.content
        )
        
        return EmbeddingResponse(
            id=embedding[fields.id_field],
            vector_store_id=embedding[fields.vector_store_id_field],
            content=embedding[fields.content_field],
            metadata=embedding[fields.metadata_field],
            created_at=int(embedding["created_at_timestamp"])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create embedding: {str(e)}")


@app.post("/v1/vector_stores/{vector_store_id}/embeddings/batch", response_model=EmbeddingBatchCreateResponse)
async def create_embeddings_batch(
    vector_store_id: str,
    request: EmbeddingBatchCreateRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Add multiple embeddings to a vector store in batch.
    """
    try:
        # Check if vector store exists
        vector_store_table = settings.table_names["vector_stores"]
        vector_store_result = await db.query_raw(
            f"SELECT id FROM {vector_store_table} WHERE id = $1",
            vector_store_id
        )
        if not vector_store_result:
            raise HTTPException(status_code=404, detail="Vector store not found")
        
        if not request.embeddings:
            raise HTTPException(status_code=400, detail="No embeddings provided")
        
        # Prepare batch insert
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]
        
        # Build VALUES clause for batch insert
        values_clauses = []
        params = []
        param_count = 1
        
        for embedding_req in request.embeddings:
            embedding_vector_str = "[" + ",".join(map(str, embedding_req.embedding)) + "]"
            values_clauses.append(f"(gen_random_uuid(), ${param_count}, ${param_count + 1}, ${param_count + 2}::vector, ${param_count + 3}, NOW())")
            params.extend([
                vector_store_id,
                embedding_req.content,
                embedding_vector_str,
                embedding_req.metadata or {}
            ])
            param_count += 4
        
        values_clause = ", ".join(values_clauses)
        
        # Execute batch insert
        result = await db.query_raw(
            f"""
            INSERT INTO {table_name} ({fields.id_field}, {fields.vector_store_id_field}, {fields.content_field}, 
                                     {fields.embedding_field}, {fields.metadata_field}, {fields.created_at_field})
            VALUES {values_clause}
            RETURNING {fields.id_field}, {fields.vector_store_id_field}, {fields.content_field}, 
                     {fields.metadata_field}, EXTRACT(EPOCH FROM {fields.created_at_field})::bigint as created_at_timestamp
            """,
            *params
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to create embeddings")
        
        # Calculate total content length for usage bytes update
        total_content_length = sum(len(emb.content) for emb in request.embeddings)
        
        # Update vector store statistics
        await db.query_raw(
            f"""
            UPDATE {vector_store_table} 
            SET 
                file_counts = jsonb_set(
                    jsonb_set(
                        COALESCE(file_counts, '{{"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0}}'::jsonb),
                        '{{completed}}',
                        (COALESCE(file_counts->>'completed', '0')::int + $2)::text::jsonb
                    ),
                    '{{total}}',
                    (COALESCE(file_counts->>'total', '0')::int + $2)::text::jsonb
                ),
                usage_bytes = COALESCE(usage_bytes, 0) + $3,
                last_active_at = NOW()
            WHERE id = $1
            """,
            vector_store_id,
            len(request.embeddings),
            total_content_length
        )
        
        # Convert results to response format
        embeddings = []
        for row in result:
            embeddings.append(EmbeddingResponse(
                id=row[fields.id_field],
                vector_store_id=row[fields.vector_store_id_field],
                content=row[fields.content_field],
                metadata=row[fields.metadata_field],
                created_at=int(row["created_at_timestamp"])
            ))
        
        return EmbeddingBatchCreateResponse(
            data=embeddings,
            created=int(time.time())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings batch: {str(e)}")


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: List[float]
    dimensions: int


@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest, api_key: str = Depends(get_api_key)):
    """
    Generate an embedding for arbitrary text.

    Used by the TicketConnect backend to embed event descriptions/tags
    and user taste profiles. No vector store side effects — pure transform.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text must be a non-empty string")
    if len(request.text) > MAX_EMBED_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"text exceeds maximum length of {MAX_EMBED_TEXT_CHARS} characters",
        )

    try:
        vector = await embedding_service.generate_embedding(request.text)
        return EmbedResponse(embedding=vector, dimensions=len(vector))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# TicketConnect-specific endpoints
# ─────────────────────────────────────────────────────────────────────────────
# These extend the OpenAI-compatible API with operations the TicketConnect
# backend needs: idempotent upserts keyed by application ID (eventId), search
# by precomputed vector (skips redundant embedding generation), and a get-
# or-create lookup for self-bootstrapping the events vector store.

class UpsertEmbeddingRequest(BaseModel):
    id: str  # Application-supplied ID (e.g. eventId) — used as the row PK
    content: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None


class SearchByVectorRequest(BaseModel):
    embedding: List[float]
    limit: Optional[int] = 20
    filters: Optional[Dict[str, Any]] = None
    exclude_ids: Optional[List[str]] = None
    return_metadata: Optional[bool] = True


class VectorSearchResultItem(BaseModel):
    id: str
    score: float
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchByVectorResponse(BaseModel):
    object: str = "vector_store.search_by_vector"
    data: List[VectorSearchResultItem]


@app.post("/v1/vector_stores/{vector_store_id}/upsert", response_model=EmbeddingResponse)
async def upsert_embedding(
    vector_store_id: str,
    request: UpsertEmbeddingRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Idempotent upsert keyed by `id`. INSERT on first call, UPDATE on subsequent
    calls. Use this when the caller owns a stable identity (e.g. eventId) and
    wants writes to be safely retryable.

    Replaces gen_random_uuid() PK with the caller-supplied id.
    """
    from typing import Dict as _Dict, Any as _Any

    # Verify vector store exists
    vector_store_table = settings.table_names["vector_stores"]
    vector_store_result = await db.query_raw(
        f"SELECT id FROM {vector_store_table} WHERE id = $1",
        vector_store_id
    )
    if not vector_store_result:
        raise HTTPException(status_code=404, detail="Vector store not found")

    fields = settings.db_fields
    table_name = settings.table_names["embeddings"]
    embedding_vector_str = "[" + ",".join(map(str, request.embedding)) + "]"

    try:
        result = await db.query_raw(
            f"""
            INSERT INTO {table_name}
                ({fields.id_field}, {fields.vector_store_id_field}, {fields.content_field},
                 {fields.embedding_field}, {fields.metadata_field}, {fields.created_at_field})
            VALUES ($1, $2, $3, $4::vector, $5, NOW())
            ON CONFLICT ({fields.id_field}) DO UPDATE SET
                {fields.content_field}   = EXCLUDED.{fields.content_field},
                {fields.embedding_field} = EXCLUDED.{fields.embedding_field},
                {fields.metadata_field}  = EXCLUDED.{fields.metadata_field}
            RETURNING {fields.id_field}, {fields.vector_store_id_field}, {fields.content_field},
                     {fields.metadata_field},
                     EXTRACT(EPOCH FROM {fields.created_at_field})::bigint as created_at_timestamp
            """,
            request.id,
            vector_store_id,
            request.content,
            embedding_vector_str,
            request.metadata or {}
        )

        if not result:
            raise HTTPException(status_code=500, detail="Failed to upsert embedding")

        row = result[0]
        return EmbeddingResponse(
            id=row[fields.id_field],
            vector_store_id=row[fields.vector_store_id_field],
            content=row[fields.content_field],
            metadata=row[fields.metadata_field],
            created_at=int(row["created_at_timestamp"])
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to upsert embedding: {str(e)}")


@app.post("/v1/vector_stores/{vector_store_id}/search-by-vector", response_model=SearchByVectorResponse)
async def search_by_vector(
    vector_store_id: str,
    request: SearchByVectorRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Search by a precomputed embedding vector (no LLM call).

    Used by the TicketConnect "similar events" feature where the source event's
    embedding is already cached on the Event document — regenerating it via
    /search would waste an embedding-model call and add latency.

    Returns ids + scores; the caller is expected to fetch the full domain
    objects from its primary store (Mongo) for display.
    """
    # Verify vector store exists
    vector_store_table = settings.table_names["vector_stores"]
    vector_store_result = await db.query_raw(
        f"SELECT id FROM {vector_store_table} WHERE id = $1",
        vector_store_id
    )
    if not vector_store_result:
        raise HTTPException(status_code=404, detail="Vector store not found")

    fields = settings.db_fields
    table_name = settings.table_names["embeddings"]
    limit = min(request.limit or 20, 200)
    query_vector_str = "[" + ",".join(map(str, request.embedding)) + "]"

    param_count = 1
    query_params: List[Any] = [query_vector_str, vector_store_id]
    base_query = f"""
        SELECT
            {fields.id_field},
            {fields.content_field},
            {fields.metadata_field},
            ({fields.embedding_field} <=> ${param_count}::vector) as distance
        FROM {table_name}
        WHERE {fields.vector_store_id_field} = ${param_count + 1}
    """
    param_count += 2

    if request.exclude_ids:
        placeholders = ",".join(
            f"${i}" for i in range(param_count, param_count + len(request.exclude_ids))
        )
        base_query += f" AND {fields.id_field} NOT IN ({placeholders})"
        query_params.extend(request.exclude_ids)
        param_count += len(request.exclude_ids)

    if request.filters:
        for key, value in request.filters.items():
            base_query += (
                f" AND {fields.metadata_field}->>${param_count} = ${param_count + 1}"
            )
            query_params.extend([key, str(value)])
            param_count += 2

    final_query = base_query + f" ORDER BY distance ASC LIMIT {limit}"

    try:
        rows = await db.query_raw(final_query, *query_params)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    items: List[VectorSearchResultItem] = []
    for row in rows:
        # Cosine distance ranges 0 (identical) .. 2 (opposite). Convert to
        # similarity in [0, 1] using the same formula as the existing /search.
        similarity = max(0.0, 1.0 - (row["distance"] / 2.0))
        items.append(VectorSearchResultItem(
            id=row[fields.id_field],
            score=similarity,
            content=row[fields.content_field],
            metadata=row[fields.metadata_field] if request.return_metadata else None,
        ))

    return SearchByVectorResponse(data=items)


@app.delete("/v1/vector_stores/{vector_store_id}/embeddings/{embedding_id}")
async def delete_embedding(
    vector_store_id: str,
    embedding_id: str,
    api_key: str = Depends(get_api_key)
):
    """
    Delete an embedding by id. Idempotent — returns success even if the row
    doesn't exist (caller doesn't need to track which IDs were ever inserted).
    """
    fields = settings.db_fields
    table_name = settings.table_names["embeddings"]

    try:
        await db.execute_raw(
            f"""
            DELETE FROM {table_name}
            WHERE {fields.id_field} = $1 AND {fields.vector_store_id_field} = $2
            """,
            embedding_id,
            vector_store_id
        )
        return {"deleted": True, "id": embedding_id}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to delete embedding: {str(e)}")


@app.get("/v1/vector_stores/by-name/{name}", response_model=VectorStoreResponse)
async def get_vector_store_by_name(
    name: str,
    api_key: str = Depends(get_api_key)
):
    """
    Look up a vector store by name. Returns the most recently created match,
    or 404 if none exist. Lets clients self-bootstrap (find-or-create) without
    needing to remember a hardcoded UUID across restarts.
    """
    vector_store_table = settings.table_names["vector_stores"]
    rows = await db.query_raw(
        f"""
        SELECT id, name, file_counts, status, usage_bytes, expires_after, expires_at,
               last_active_at, metadata,
               EXTRACT(EPOCH FROM created_at)::bigint as created_at_timestamp
        FROM {vector_store_table}
        WHERE name = $1
        ORDER BY created_at DESC
        LIMIT 1
        """,
        name
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Vector store not found")

    row = rows[0]
    expires_at = int(row["expires_at"].timestamp()) if row.get("expires_at") else None
    last_active_at = int(row["last_active_at"].timestamp()) if row.get("last_active_at") else None

    return VectorStoreResponse(
        id=row["id"],
        created_at=int(row["created_at_timestamp"]),
        name=row["name"],
        usage_bytes=row["usage_bytes"] or 0,
        file_counts=row["file_counts"] or {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
        status=row["status"],
        expires_after=row["expires_after"],
        expires_at=expires_at,
        last_active_at=last_active_at,
        metadata=row["metadata"]
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": int(time.time())}


@app.post("/v1/feed/recommendations", response_model=FeedRecommendationResponse)
async def feed_recommendations(
    request: FeedRecommendationRequest,
    api_key: str = Depends(get_api_key),
):
    """
    Personalized feed recommendations using a precomputed user taste embedding.

    The caller (TicketConnect backend) passes the user's tasteEmbedding — a
    weighted average of the embeddings of events they have liked, with a 30-day
    half-life so recent activity dominates.

    Unlike /search-by-vector (which is keyed on a source *event*), this endpoint
    is keyed on a *user* taste vector and optionally boosts scores using the
    user's per-event ratings stored in user_rating.

    Returns a ranked list of { id (eventId), score } items. The caller is
    responsible for the second phase: filtering by publish status / future date
    in its primary store (Mongo) before returning results to the client.
    """
    fields = settings.db_fields
    table_name = settings.table_names["embeddings"]
    vector_store_table = settings.table_names["vector_stores"]
    limit = min(request.limit or 20, 200)

    query_vector_str = "[" + ",".join(map(str, request.taste_embedding)) + "]"

    # ── Phase 1: resolve the events vector store ──────────────────────────────
    # We search across *all* vector stores that hold event embeddings. In
    # practice there is exactly one (ticketconnect-events), but we look it up
    # dynamically so this doesn't hardcode an ID.
    store_rows = await db.query_raw(
        f"SELECT id FROM {vector_store_table} WHERE name = $1 LIMIT 1",
        "ticketconnect-events",
    )
    if not store_rows:
        raise HTTPException(status_code=404, detail="events vector store not found")

    vector_store_id = store_rows[0]["id"]

    # ── Phase 2: ANN search by cosine distance ────────────────────────────────
    param_count = 1
    query_params: List[Any] = [query_vector_str, vector_store_id]
    base_query = f"""
        SELECT
            {fields.id_field},
            ({fields.embedding_field} <=> ${param_count}::vector) AS distance
        FROM {table_name}
        WHERE {fields.vector_store_id_field} = ${param_count + 1}
    """
    param_count += 2

    if request.exclude_ids:
        placeholders = ",".join(
            f"${i}" for i in range(param_count, param_count + len(request.exclude_ids))
        )
        base_query += f" AND {fields.id_field} NOT IN ({placeholders})"
        query_params.extend(request.exclude_ids)
        param_count += len(request.exclude_ids)

    # Over-fetch so ratings boosting can reorder without falling below `limit`
    final_query = base_query + f" ORDER BY distance ASC LIMIT {limit * 4}"

    try:
        rows = await db.query_raw(final_query, *query_params)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Feed recommendation search failed: {str(e)}")

    # ── Phase 3: optional ratings boost ──────────────────────────────────────
    user_ratings: dict = {}
    if request.user_id and rows:
        try:
            embedding_ids = [row[fields.id_field] for row in rows]
            placeholders = ",".join(f"${i}" for i in range(2, 2 + len(embedding_ids)))
            ratings_rows = await db.query_raw(
                f"""
                SELECT embedding_id, rating
                FROM user_rating
                WHERE user_id = $1 AND embedding_id IN ({placeholders})
                """,
                request.user_id,
                *embedding_ids,
            )
            for r in ratings_rows:
                user_ratings[r["embedding_id"]] = r["rating"]
        except Exception:
            pass  # ratings boost is best-effort; never fail the request

    # ── Phase 4: score, boost, sort, truncate ─────────────────────────────────
    items: List[FeedRecommendationItem] = []
    for row in rows:
        # Cosine distance [0, 2] → similarity [0, 1]
        similarity = max(0.0, 1.0 - (row["distance"] / 2.0))

        event_id = row[fields.id_field]
        if event_id in user_ratings:
            # Scale: rating 1-5, neutral at 3. Each point → ±10% adjustment.
            boost = (user_ratings[event_id] - 3) * 0.1
            similarity = max(0.0, min(1.0, similarity * (1 + boost)))

        items.append(FeedRecommendationItem(id=event_id, score=similarity))

    items.sort(key=lambda x: x.score, reverse=True)

    return FeedRecommendationResponse(data=items[:limit])


@app.post("/v1/events/score", response_model=VectorStoreSearchResponse)
async def score_event(
    request: EventScoreRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Score a newly added event and return similar events based on tags.
    Uses organizer-provided tags to find similar events.
    Falls back to showing available events if there's too little data.
    """
    try:
        # Generate embedding for the event using tags and metadata
        query_text = " ".join(request.tags or [])
        if request.metadata:
            # Add metadata fields to the query text
            for key, value in request.metadata.items():
                query_text += f" {value}"
        
        if not query_text.strip():
            query_text = request.event_id
        
        # Generate embedding for the query
        query_embedding = await generate_query_embedding(query_text)
        query_vector_str = "[" + ",".join(map(str, query_embedding)) + "]"
        
        # Build the raw SQL query for vector similarity search
        limit = 20
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]
        
        # Base query with vector similarity using cosine distance
        param_count = 1
        query_params = [query_vector_str]
        
        base_query = f"""
        SELECT 
            {fields.id_field},
            {fields.content_field},
            {fields.metadata_field},
            ({fields.embedding_field} <=> ${param_count}::vector) as distance
        FROM {table_name}
        """
        param_count += 1
        
        # Add tag-based filtering if tags are provided
        filter_conditions = []
        if request.tags:
            # Search for embeddings with matching tags in metadata
            tag_conditions = []
            for tag in request.tags:
                tag_conditions.append(f"{fields.metadata_field}->>'tag' = ${param_count}")
                query_params.append(tag)
                param_count += 1
            
            if tag_conditions:
                filter_conditions.append("(" + " OR ".join(tag_conditions) + ")")
        
        # Exclude the current event from results
        filter_conditions.append(f"{fields.id_field} != ${param_count}")
        query_params.append(request.event_id)
        param_count += 1
        
        if filter_conditions:
            base_query += " WHERE " + " AND ".join(filter_conditions)
        
        # Add ordering and limit
        final_query = base_query + f" ORDER BY distance ASC LIMIT {limit}"
        
        # Execute the query
        results = await db.query_raw(final_query, *query_params)
        
        # If too few results, fall back to showing all available events
        if len(results) < 3:
            fallback_query = f"""
            SELECT 
                {fields.id_field},
                {fields.content_field},
                {fields.metadata_field},
                ({fields.embedding_field} <=> ${1}::vector) as distance
            FROM {table_name}
            WHERE {fields.id_field} != ${2}
            ORDER BY distance ASC LIMIT {limit}
            """
            results = await db.query_raw(fallback_query, query_vector_str, request.event_id)
        
        # Convert results to SearchResult objects
        search_results = []
        for row in results:
            # Convert distance to similarity score (1 - normalized_distance)
            similarity_score = max(0, 1 - (row['distance'] / 2))
            
            # Extract filename from metadata or use a default
            metadata = row[fields.metadata_field] or {}
            filename = metadata.get('filename', 'event.txt')
            
            content_chunks = [ContentChunk(type="text", text=row[fields.content_field])]
            
            result = SearchResult(
                file_id=row[fields.id_field],
                filename=filename,
                score=similarity_score,
                attributes=metadata,
                content=content_chunks
            )
            search_results.append(result)
        
        # Sort results by score (descending)
        search_results.sort(key=lambda x: x.score, reverse=True)
        
        return VectorStoreSearchResponse(
            search_query=query_text,
            data=search_results,
            has_more=False,
            next_page=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Event scoring failed: {str(e)}")


@app.get("/v1/events/{event_id}/similar", response_model=SimilarEventsResponse)
async def get_similar_events(
    event_id: str,
    limit: Optional[int] = 20,
    api_key: str = Depends(get_api_key)
):
    """
    Get similar events for a given event ID.
    Utilizes tags from the organizer panel to find related events.
    Falls back to showing other available events if insufficient data.
    """
    try:
        fields = settings.db_fields
        table_name = settings.table_names["embeddings"]
        limit = min(limit or 20, 100)
        
        # First, get the event's metadata to extract tags
        event_result = await db.query_raw(
            f"SELECT {fields.metadata_field}, {fields.content_field} FROM {table_name} WHERE {fields.id_field} = $1",
            event_id
        )
        
        if not event_result:
            raise HTTPException(status_code=404, detail="Event not found")
        
        event_metadata = event_result[0][fields.metadata_field] or {}
        event_content = event_result[0][fields.content_field]
        tags = event_metadata.get('tags', [])
        
        # Generate embedding from event content and tags
        query_text = event_content
        if tags:
            query_text += " " + " ".join(tags)
        
        query_embedding = await generate_query_embedding(query_text)
        query_vector_str = "[" + ",".join(map(str, query_embedding)) + "]"
        
        # Search for similar events using vector similarity
        param_count = 1
        query_params = [query_vector_str, event_id]
        
        base_query = f"""
        SELECT 
            {fields.id_field},
            {fields.content_field},
            {fields.metadata_field},
            ({fields.embedding_field} <=> ${param_count}::vector) as distance
        FROM {table_name}
        WHERE {fields.id_field} != ${param_count + 1}
        """
        param_count += 2
        
        # If tags exist, try to boost results with matching tags
        if tags:
            tag_conditions = []
            for tag in tags:
                tag_conditions.append(f"{fields.metadata_field}->>'tag' = ${param_count}")
                query_params.append(tag)
                param_count += 1
            
            if tag_conditions:
                # Use a subquery to first find tag-matching events, then union with vector results
                tag_filter = "(" + " OR ".join(tag_conditions) + ")"
                base_query += f" AND {tag_filter}"
        
        final_query = base_query + f" ORDER BY distance ASC LIMIT {limit}"
        
        results = await db.query_raw(final_query, *query_params)
        
        # Fallback: if too few results, show any available events
        if len(results) < 3:
            fallback_query = f"""
            SELECT 
                {fields.id_field},
                {fields.content_field},
                {fields.metadata_field},
                RANDOM() as distance
            FROM {table_name}
            WHERE {fields.id_field} != $1
            LIMIT {limit}
            """
            results = await db.query_raw(fallback_query, event_id)
        
        # Convert to SearchResult objects
        search_results = []
        for row in results:
            similarity_score = max(0, 1 - (row['distance'] / 2)) if isinstance(row['distance'], float) else 0.5
            metadata = row[fields.metadata_field] or {}
            filename = metadata.get('filename', 'event.txt')
            
            content_chunks = [ContentChunk(type="text", text=row[fields.content_field])]
            
            result = SearchResult(
                file_id=row[fields.id_field],
                filename=filename,
                score=similarity_score,
                attributes=metadata,
                content=content_chunks
            )
            search_results.append(result)
        
        search_results.sort(key=lambda x: x.score, reverse=True)
        
        return SimilarEventsResponse(
            query_event_id=event_id,
            data=search_results,
            has_more=len(search_results) == limit
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get similar events: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True) 