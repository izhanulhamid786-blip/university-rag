const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
      ...options,
    });
  } catch (error) {
    throw new Error(
      "Could not reach the FastAPI backend. Start it with: uvicorn api:app --port 8000"
    );
  }

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed with status ${response.status}`);
  }
  return payload;
}

export function fetchStatus() {
  return request("/status");
}

export function askQuestion({ query, history, answerStyle }) {
  return request("/query", {
    method: "POST",
    body: JSON.stringify({
      query,
      history: history || [],
      answer_style: answerStyle,
    }),
  });
}

