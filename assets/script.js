// Minimal client-side metrics for a static site.
// Keeps directory refs the same. Paste your logging endpoint URL below.
// If left empty, logging is disabled but the on-page IP display still works.
document.addEventListener('DOMContentLoaded', () => {
  const y = document.getElementById('year');
  if (y) y.textContent = new Date().getFullYear();

  const LOG_ENDPOINT = ""; // e.g. "https://your-serverless-function.example/collect"

  const ipEl = document.getElementById('ip-address');

  (async () => {
    let ip = "";
    try {
      const r = await fetch("https://api.ipify.org?format=json", { cache: "no-store" });
      const d = await r.json();
      ip = (d && d.ip) || "";
      if (ipEl) ipEl.textContent = ip || "Unavailable";
    } catch (_) {
      if (ipEl) ipEl.textContent = "Unavailable";
    }

    // Build your analytics payload
    const payload = {
      ip,
      path: location.pathname + location.search + location.hash,
      referrer: document.referrer || "",
      userAgent: navigator.userAgent || "",
      tz: (Intl.DateTimeFormat().resolvedOptions().timeZone || ""),
      t: new Date().toISOString()
    };

    // Send via sendBeacon when possible, else fall back to fetch no-cors
    if (LOG_ENDPOINT) {
      try {
        const body = new Blob([JSON.stringify(payload)], { type: "application/json" });
        if (navigator.sendBeacon) {
          navigator.sendBeacon(LOG_ENDPOINT, body);
        } else {
          fetch(LOG_ENDPOINT, {
            method: "POST",
            mode: "no-cors",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          });
        }
      } catch (_) {
        // Non-fatal. We don't block page usage on analytics.
      }
    }
  })();
});
