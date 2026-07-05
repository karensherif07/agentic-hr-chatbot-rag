-- Adds a column to store the actual PDF bytes in the database.
-- This replaces the need for a persistent disk on the backend host —
-- Render's free tier doesn't offer one, but Postgres (Neon's free tier)
-- persists rows regardless of how many times the backend container
-- restarts or redeploys.
--
-- Run this once, the same way you ran the first migration (pgAdmin's
-- Query Tool, or Neon's SQL Editor if you're already deployed there).

ALTER TABLE policy_documents
ADD COLUMN IF NOT EXISTS file_data BYTEA;