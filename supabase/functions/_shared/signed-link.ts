// Shared link verification for the two public endpoints.
//
// The canonical string is every signed parameter as `key=value`, sorted by
// key and joined with `&`, unescaped and UTF-8 encoded -- byte for byte what
// LLMTools._sign builds in src/pubmed_email_agent/tools/llm/client.py. A
// disagreement between the two does not raise anywhere; it silently rejects
// every link as invalid, so the two must be changed together.

const encoder = new TextEncoder();

export async function sign(
  secret: string,
  params: Record<string, string>,
): Promise<string> {
  const canonical = Object.keys(params)
    .sort()
    .map((key) => `${key}=${params[key]}`)
    .join("&");

  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );

  const signature = await crypto.subtle.sign("HMAC", key, encoder.encode(canonical));

  return Array.from(new Uint8Array(signature))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

// Compares in time independent of how many leading characters match, so a
// caller cannot discover a valid signature one character at a time.
function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;

  let difference = 0;
  for (let i = 0; i < a.length; i++) {
    difference |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }

  return difference === 0;
}

/**
 * Verifies the token and expiry over exactly `signedKeys`.
 *
 * Throws on anything wrong, because every failure here is the same answer to
 * the caller: redirect to the error page. Never say which check failed -- that
 * tells a forger whether the id exists, whether the link merely expired, or
 * how close their guess was.
 */
export async function verifySignedParams(
  url: URL,
  secret: string,
  signedKeys: string[],
): Promise<Record<string, string>> {
  const params: Record<string, string> = {};

  for (const key of signedKeys) {
    const value = url.searchParams.get(key);
    if (value === null) throw new Error("missing parameter");
    params[key] = value;
  }

  const exp = url.searchParams.get("exp");
  const token = url.searchParams.get("token");
  if (!exp || !token) throw new Error("missing signature");

  params.exp = exp;

  if (!Number.isFinite(Number(exp)) || Number(exp) * 1000 < Date.now()) {
    throw new Error("expired");
  }

  if (!timingSafeEqual(await sign(secret, params), token)) {
    throw new Error("bad signature");
  }

  return params;
}

/**
 * The page a GET lands on.
 *
 * These endpoints must not act on GET. Mail clients and corporate security
 * scanners follow links in email to check them, which would silently
 * unsubscribe readers and file ratings nobody chose. A scanner will not
 * submit a form, so the action moves to POST and the link only ever renders
 * this page.
 */
export function confirmPage(title: string, body: string, action: string): Response {
  return new Response(
    `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>${title}</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; padding: 3rem 1.25rem;
         background: #f7f7f8; color: #1a1a1a; }
  main { max-width: 26rem; margin: 0 auto; background: #fff; padding: 2rem;
         border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  h1 { font-size: 1.15rem; margin: 0 0 .75rem; }
  p { margin: 0 0 1.5rem; line-height: 1.5; color: #444; }
  button { font: inherit; font-weight: 600; cursor: pointer; border: 0;
           border-radius: 8px; padding: .7rem 1.4rem; background: #007bff;
           color: #fff; }
</style>
</head>
<body>
<main>
  <h1>${title}</h1>
  <p>${body}</p>
  <form method="POST" action="${action}"><button type="submit">Confirm</button></form>
</main>
</body>
</html>`,
    { status: 200, headers: { "content-type": "text/html; charset=utf-8" } },
  );
}

/** Escapes text before it goes anywhere near the confirmation page. */
export function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!
  );
}
