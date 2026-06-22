/*
Cloudflare Worker for SPX-IC SpotGamma levels.

Required Worker secrets / variables:
- GITHUB_TOKEN: fine-grained token with Contents read/write on Quique805/SPX-IC
- GITHUB_OWNER: Quique805
- GITHUB_REPO: SPX-IC
- GITHUB_BRANCH: main
- SPX_PIN: private PIN typed from the dashboard
- ALLOWED_ORIGIN: dashboard origin, for example https://quique805.github.io
*/

const FILE_PATH = "data/spotgamma-levels.json";

function corsHeaders(env) {
  return {
    "Access-Control-Allow-Origin": env.ALLOWED_ORIGIN || "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Content-Type": "application/json",
  };
}

function jsonResponse(body, status, env) {
  return new Response(JSON.stringify(body), {
    status,
    headers: corsHeaders(env),
  });
}

function asNullableNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function validatePayload(payload) {
  const date = String(payload.date || "");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return "Fecha invalida";
  if (!Number.isFinite(Number(payload.callWall))) return "Call Wall invalida";
  if (!Number.isFinite(Number(payload.putWall))) return "Put Wall invalida";
  return null;
}

function encodeBase64Utf8(text) {
  const bytes = new TextEncoder().encode(text);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function decodeBase64Utf8(text) {
  const binary = atob(text.replace(/\n/g, ""));
  const bytes = Uint8Array.from(binary, char => char.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

async function githubFetch(env, path, init = {}) {
  const owner = env.GITHUB_OWNER || "Quique805";
  const repo = env.GITHUB_REPO || "SPX-IC";
  const url = `https://api.github.com/repos/${owner}/${repo}${path}`;
  const response = await fetch(url, {
    ...init,
    headers: {
      "Accept": "application/vnd.github+json",
      "Authorization": `Bearer ${env.GITHUB_TOKEN}`,
      "Content-Type": "application/json",
      "User-Agent": "SPX-IC-SpotGamma-Worker",
      "X-GitHub-Api-Version": "2022-11-28",
      ...(init.headers || {}),
    },
  });
  const text = await response.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch (_) {}
  if (!response.ok) {
    throw new Error(data.message || text || `GitHub HTTP ${response.status}`);
  }
  return data;
}

async function saveSpotGammaLevel(env, payload) {
  const branch = env.GITHUB_BRANCH || "main";
  const file = await githubFetch(env, `/contents/${FILE_PATH}?ref=${encodeURIComponent(branch)}`);
  const current = JSON.parse(decodeBase64Utf8(file.content || "") || "{}");
  const byDate = current.byDate || {};
  byDate[payload.date] = {
    callWall: Number(payload.callWall),
    putWall: Number(payload.putWall),
    volTrigger: asNullableNumber(payload.volTrigger),
    gammaFlip: asNullableNumber(payload.gammaFlip),
    source: payload.source || "SpotGamma manual",
    updatedAt: new Date().toISOString(),
  };
  const next = {
    ...current,
    lastUpdated: new Date().toISOString(),
    byDate,
  };
  const content = `${JSON.stringify(next, null, 2)}\n`;
  await githubFetch(env, `/contents/${FILE_PATH}`, {
    method: "PUT",
    body: JSON.stringify({
      message: `Add SpotGamma levels for ${payload.date}`,
      content: encodeBase64Utf8(content),
      sha: file.sha,
      branch,
    }),
  });
  return next.byDate[payload.date];
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(env) });
    }
    if (request.method !== "POST") {
      return jsonResponse({ ok: false, error: "Metodo no permitido" }, 405, env);
    }
    try {
      if (!env.GITHUB_TOKEN || !env.SPX_PIN) {
        return jsonResponse({ ok: false, error: "Worker sin configurar" }, 500, env);
      }
      const payload = await request.json();
      if (payload.pin !== env.SPX_PIN) {
        return jsonResponse({ ok: false, error: "PIN incorrecto" }, 401, env);
      }
      const error = validatePayload(payload);
      if (error) return jsonResponse({ ok: false, error }, 400, env);
      const saved = await saveSpotGammaLevel(env, payload);
      return jsonResponse({ ok: true, saved }, 200, env);
    } catch (error) {
      return jsonResponse({ ok: false, error: error.message || String(error) }, 500, env);
    }
  },
};
