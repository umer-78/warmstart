/* Warmstart's cache, running live in your browser.
 * Type a support question; it is matched against the cached questions with TF-IDF cosine
 * similarity (one of the three embedders the project measures) computed in the page — no
 * server, no API key. An exact repeat is an exact hit; a close paraphrase above the floor is
 * a semantic hit served for ~$0; anything below misses and would call the model.
 * The measured dashboard above uses bge-small embeddings; this live matcher uses TF-IDF,
 * which catches reworded questions that share words and is honest about missing the rest.
 */
(() => {
  'use strict';
  const COST_PER = 6.72 / 1000;        // $ per question with no cache (from the replay)
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const norm = (s) => s.toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').replace(/\s+/g, ' ').trim();
  const toks = (s) => norm(s).split(' ').filter(Boolean);

  // tiny TF-IDF index over the cached questions
  function buildIndex(docs) {
    const df = new Map(), N = docs.length;
    const tfs = docs.map((d) => {
      const tf = new Map();
      for (const w of toks(d)) tf.set(w, (tf.get(w) || 0) + 1);
      for (const w of tf.keys()) df.set(w, (df.get(w) || 0) + 1);
      return tf;
    });
    const idf = new Map();
    df.forEach((n, w) => idf.set(w, Math.log((1 + N) / (1 + n)) + 1));
    const vecs = tfs.map((tf) => vec(tf, idf));
    return { idf, vecs };
  }
  function vec(tf, idf) {
    const v = new Map(); let ss = 0;
    tf.forEach((c, w) => { if (idf.has(w)) { const x = c * idf.get(w); v.set(w, x); ss += x * x; } });
    const n = Math.sqrt(ss) || 1; v.forEach((x, w) => v.set(w, x / n));
    return v;
  }
  function queryVec(q, idf) {
    const tf = new Map(); for (const w of toks(q)) tf.set(w, (tf.get(w) || 0) + 1);
    return vec(tf, idf);
  }
  function cosine(a, b) { let s = 0; const [sm, lg] = a.size < b.size ? [a, b] : [b, a]; sm.forEach((x, w) => { if (lg.has(w)) s += x * lg.get(w); }); return s; }

  const ready = (fn) => (document.readyState !== 'loading' ? fn() : document.addEventListener('DOMContentLoaded', fn));
  ready(async () => {
    const inEl = document.getElementById('c-in'), outEl = document.getElementById('c-out');
    const slider = document.getElementById('c-th'), thLabel = document.getElementById('c-th-val');
    if (!inEl || !outEl) return;
    let data; try { data = await (await fetch('data.json')).json(); } catch (e) { return; }
    const examples = data.examples || [];
    const cache = [...new Set(examples.map((e) => e.question))];   // the cached canonical questions
    if (!cache.length) return;
    const idx = buildIndex(cache);

    function run(q) {
      if (!q.trim()) { outEl.innerHTML = '<p class="c-empty">Type a banking support question to see whether the cache answers it.</p>'; return; }
      const floor = slider ? Number(slider.value) / 100 : 0.3;
      const nq = norm(q);
      const exact = cache.findIndex((c) => norm(c) === nq);
      const qv = queryVec(q, idx.idf);
      let best = -1, bi = -1;
      idx.vecs.forEach((v, i) => { const s = cosine(qv, v); if (s > best) { best = s; bi = i; } });
      let kind, cls, headline, detail;
      if (exact >= 0) { kind = 'EXACT HIT'; cls = 'exact'; headline = 'Served from cache — exact repeat'; bi = exact; best = 1; }
      else if (best >= floor) { kind = 'SEMANTIC HIT'; cls = 'semantic'; headline = 'Served from cache — close paraphrase'; }
      else { kind = 'MISS'; cls = 'miss'; headline = 'Not in cache — would call the model'; }
      const hit = kind !== 'MISS';
      let html = `<div class="c-verdict ${cls}">${kind}</div><p class="c-head">${headline}</p>`;
      if (bi >= 0 && (hit || best > 0)) html += `<div class="c-row"><span class="c-k">${hit ? 'Matched cached question' : 'Closest cached question'}</span><span class="c-match">${esc(cache[bi])}</span></div>`;
      html += `<div class="c-row"><span class="c-k">TF-IDF similarity</span><span class="c-bar"><span class="c-fill ${cls}" style="width:${Math.round(Math.max(0, Math.min(1, best)) * 100)}%"></span></span><b>${best.toFixed(2)}</b> <span class="c-floor">floor ${floor.toFixed(2)}</span></div>`;
      html += hit
        ? `<div class="c-row"><span class="c-k">Cost &amp; latency</span><b class="c-good">~$0.000 · ~4 ms</b> <span class="c-sub">answered from cache, nothing sent to the model</span></div>`
        : `<div class="c-row"><span class="c-k">Cost &amp; latency</span><b class="c-bad">~$${COST_PER.toFixed(4)} · ~1.6 s</b> <span class="c-sub">a fresh model call, then cached for next time</span></div>`;
      outEl.innerHTML = html;
    }

    // example chips: real paraphrases that hit, plus an out-of-domain miss
    const chips = document.getElementById('c-chips');
    if (chips) {
      const picks = examples.slice(0, 3).map((e) => e.served_for).filter(Boolean);
      picks.push('How do I reset my online banking password?');
      chips.innerHTML = picks.map((p) => `<button type="button" class="c-ex" data-c-example="${esc(p)}">${esc(p.length > 52 ? p.slice(0, 50) + '…' : p)}</button>`).join('');
    }
    if (slider) slider.addEventListener('input', () => { thLabel.textContent = (Number(slider.value) / 100).toFixed(2); run(inEl.value); });
    let t; inEl.addEventListener('input', () => { clearTimeout(t); t = setTimeout(() => run(inEl.value), 110); });
    document.querySelectorAll('[data-c-example]').forEach((b) => b.addEventListener('click', () => { inEl.value = b.getAttribute('data-c-example'); run(inEl.value); inEl.focus(); }));
    if (thLabel && slider) thLabel.textContent = (Number(slider.value) / 100).toFixed(2);
    run('');
  });
})();
