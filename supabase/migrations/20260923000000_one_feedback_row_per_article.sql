DELETE FROM "public"."article_feedback" AS older
USING "public"."article_feedback" AS newer
WHERE older."user_id" = newer."user_id"
  AND older."pmid" = newer."pmid"
  AND older."id" < newer."id";

ALTER TABLE "public"."article_feedback"
    ADD CONSTRAINT "article_feedback_user_id_pmid_key" UNIQUE ("user_id", "pmid");
