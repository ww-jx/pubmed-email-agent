-- enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- article embeddings table
CREATE TABLE IF NOT EXISTS article_embeddings (
    pmid TEXT PRIMARY KEY,
    title TEXT,
    abstract TEXT,
    embedding vector(384),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- matching function for cosine similarity
CREATE OR REPLACE FUNCTION match_articles (
  query_embedding vector(384),
  match_threshold float,
  match_count int
)
RETURNS TABLE (
  pmid TEXT,
  similarity float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    article_embeddings.pmid,
    1 - (article_embeddings.embedding <=> query_embedding) AS similarity
  FROM article_embeddings
  WHERE 1 - (article_embeddings.embedding <=> query_embedding) > match_threshold
  ORDER BY article_embeddings.embedding <=> query_embedding
  LIMIT match_count;
$$;
