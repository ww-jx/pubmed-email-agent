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

function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;

  let difference = 0;
  for (let i = 0; i < a.length; i++) {
    difference |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }

  return difference === 0;
}

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
