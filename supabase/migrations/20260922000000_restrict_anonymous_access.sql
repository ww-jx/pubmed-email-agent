DROP POLICY IF EXISTS "Enable read access for all users" ON "public"."users";

REVOKE SELECT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER
    ON TABLE "public"."users" FROM "anon";
REVOKE SELECT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER
    ON TABLE "public"."users" FROM "authenticated";

ALTER TABLE "public"."article_embeddings" ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON TABLE "public"."article_embeddings" FROM "anon";
REVOKE ALL ON TABLE "public"."article_embeddings" FROM "authenticated";

REVOKE ALL ON FUNCTION "public"."match_articles"("vector", double precision, integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION "public"."match_articles"("vector", double precision, integer) FROM "anon";
REVOKE ALL ON FUNCTION "public"."match_articles"("vector", double precision, integer) FROM "authenticated";

ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public"
    REVOKE ALL ON TABLES FROM "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public"
    REVOKE ALL ON TABLES FROM "authenticated";

ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public"
    REVOKE ALL ON SEQUENCES FROM "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public"
    REVOKE ALL ON SEQUENCES FROM "authenticated";

ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public"
    REVOKE ALL ON FUNCTIONS FROM "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public"
    REVOKE ALL ON FUNCTIONS FROM "authenticated";
