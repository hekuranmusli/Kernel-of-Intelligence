"""
Intelligence Kernel — Interactive Chat Interface
Run: python chat.py
Then open: http://localhost:5000
"""

import sys, os, json, time, threading
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, Response, stream_with_context
from layers import IntelligenceKernel, LLM_BACKEND

app = Flask(__name__)
kernel = IntelligenceKernel()
history = []

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Intelligence Kernel</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #0e0e0f;
  --surface: #161618;
  --surface2: #1e1e21;
  --border: #2a2a2e;
  --text: #e8e6e0;
  --muted: #888680;
  --faint: #444240;
  --blue: #4a90d9;
  --teal: #2db88a;
  --amber: #e8a030;
  --coral: #e06050;
  --purple: #8b7dd8;
  --green: #5a9e52;
}
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
  height: 100vh;
  display: grid;
  grid-template-columns: 320px 1fr;
  grid-template-rows: 48px 1fr 80px;
  grid-template-areas:
    "header header"
    "sidebar main"
    "sidebar input";
}
/* Header */
.header {
  grid-area: header;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 20px;
  gap: 12px;
}
.header-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--teal);
  animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.header-title {
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.08em;
  color: var(--text);
}
.header-sub {
  font-size: 11px;
  color: var(--muted);
  margin-left: auto;
}
.backend-badge {
  font-size: 10px;
  padding: 2px 8px;
  border-radius: 3px;
  background: var(--surface2);
  border: 1px solid var(--border);
  color: var(--teal);
  letter-spacing: 0.06em;
}
/* Sidebar */
.sidebar {
  grid-area: sidebar;
  background: var(--surface);
  border-right: 1px solid var(--border);
  overflow-y: auto;
  padding: 16px;
}
.sidebar-section {
  margin-bottom: 20px;
}
.sidebar-label {
  font-size: 9px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--faint);
  margin-bottom: 10px;
}
.layer-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 10px 12px;
  margin-bottom: 8px;
  transition: border-color 0.2s;
}
.layer-card.active { border-color: var(--teal); }
.layer-card.error  { border-color: var(--coral); }
.layer-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}
.layer-tag {
  font-size: 10px;
  font-weight: 600;
  color: var(--muted);
  letter-spacing: 0.08em;
  width: 24px;
}
.layer-name {
  font-size: 12px;
  color: var(--text);
  flex: 1;
}
.layer-status {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--faint);
}
.layer-status.running  { background: var(--amber); animation: pulse 0.8s ease-in-out infinite; }
.layer-status.done     { background: var(--teal); }
.layer-status.error    { background: var(--coral); }
.layer-value {
  font-size: 10px;
  color: var(--muted);
  line-height: 1.5;
  word-break: break-all;
}
.layer-value span { color: var(--text); }

.metric-row {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  color: var(--muted);
  padding: 4px 0;
  border-bottom: 1px solid var(--border);
}
.metric-row:last-child { border-bottom: none; }
.metric-val { color: var(--text); }
.metric-val.good  { color: var(--teal); }
.metric-val.warn  { color: var(--amber); }
.metric-val.bad   { color: var(--coral); }

.unc-bar-wrap {
  margin-top: 8px;
}
.unc-bar-label {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  color: var(--muted);
  margin-bottom: 3px;
}
.unc-bar {
  height: 3px;
  background: var(--border);
  border-radius: 2px;
  margin-bottom: 6px;
}
.unc-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.4s ease;
}

.compression-chart {
  height: 60px;
  position: relative;
  margin-top: 8px;
}
.compression-chart canvas {
  width: 100%;
  height: 100%;
}

.example-btn {
  width: 100%;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 7px 10px;
  font-family: inherit;
  font-size: 11px;
  color: var(--muted);
  text-align: left;
  cursor: pointer;
  margin-bottom: 6px;
  transition: border-color 0.15s, color 0.15s;
}
.example-btn:hover { border-color: var(--blue); color: var(--text); }

/* Main chat */
.main {
  grid-area: main;
  overflow-y: auto;
  padding: 20px 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.message {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-width: 780px;
}
.message.user { align-self: flex-end; }
.message.kernel { align-self: flex-start; }

.msg-bubble {
  padding: 12px 16px;
  border-radius: 6px;
  font-size: 13px;
  line-height: 1.7;
}
.message.user .msg-bubble {
  background: var(--blue);
  color: #fff;
  border-radius: 6px 6px 2px 6px;
}
.message.kernel .msg-bubble {
  background: var(--surface2);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 6px 6px 6px 2px;
}
.msg-meta {
  font-size: 10px;
  color: var(--faint);
  padding: 0 4px;
}
.message.user .msg-meta { text-align: right; }

.thinking {
  display: flex;
  gap: 4px;
  align-items: center;
  padding: 12px 16px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 6px;
  width: fit-content;
}
.thinking-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--teal);
  animation: thinking-bounce 1.2s ease-in-out infinite;
}
.thinking-dot:nth-child(2) { animation-delay: 0.2s; }
.thinking-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes thinking-bounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-6px)} }

.layer-trace {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
  font-size: 11px;
}
.trace-header {
  padding: 6px 12px;
  background: var(--surface2);
  border-bottom: 1px solid var(--border);
  font-size: 10px;
  letter-spacing: 0.08em;
  color: var(--muted);
  display: flex;
  justify-content: space-between;
}
.trace-row {
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  display: grid;
  grid-template-columns: 32px 1fr auto;
  gap: 8px;
  align-items: start;
}
.trace-row:last-child { border-bottom: none; }
.trace-layer { color: var(--muted); font-size: 10px; font-weight: 600; padding-top: 1px; }
.trace-content { color: var(--text); line-height: 1.5; }
.trace-time { color: var(--faint); font-size: 10px; white-space: nowrap; }

.action-chip {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 3px;
  font-size: 11px;
  font-weight: 600;
  margin-right: 6px;
  margin-top: 6px;
}
.chip-action   { background: rgba(74,144,217,0.15); color: var(--blue); border: 1px solid rgba(74,144,217,0.3); }
.chip-warn     { background: rgba(232,160,48,0.15); color: var(--amber); border: 1px solid rgba(232,160,48,0.3); }
.chip-confirm  { background: rgba(45,184,138,0.15); color: var(--teal); border: 1px solid rgba(45,184,138,0.3); }

/* Input */
.input-area {
  grid-area: input;
  background: var(--surface);
  border-top: 1px solid var(--border);
  padding: 16px 20px;
  display: flex;
  gap: 10px;
  align-items: center;
}
.input-wrap {
  flex: 1;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 6px;
  display: flex;
  align-items: center;
  padding: 0 12px;
  transition: border-color 0.2s;
}
.input-wrap:focus-within { border-color: var(--blue); }
#user-input {
  flex: 1;
  background: none;
  border: none;
  outline: none;
  color: var(--text);
  font-family: inherit;
  font-size: 13px;
  padding: 12px 0;
}
#user-input::placeholder { color: var(--faint); }
.send-btn {
  background: var(--blue);
  border: none;
  border-radius: 4px;
  padding: 10px 18px;
  color: #fff;
  font-family: inherit;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  letter-spacing: 0.06em;
  transition: opacity 0.15s;
}
.send-btn:hover { opacity: 0.85; }
.send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.iter-badge {
  font-size: 10px;
  color: var(--faint);
  letter-spacing: 0.06em;
  white-space: nowrap;
}
</style>
</head>
<body>

<div class="header">
  <div class="header-dot"></div>
  <div class="header-title">INTELLIGENCE KERNEL</div>
  <div class="header-sub">
    <span class="backend-badge" id="backend-badge">LOADING</span>
  </div>
</div>

<div class="sidebar">

  <div class="sidebar-section">
    <div class="sidebar-label">Layer Status</div>

    <div class="layer-card" id="card-l1">
      <div class="layer-header">
        <div class="layer-tag">L1</div>
        <div class="layer-name">Causal Extractor</div>
        <div class="layer-status" id="status-l1"></div>
      </div>
      <div class="layer-value" id="val-l1">Waiting for input</div>
    </div>

    <div class="layer-card" id="card-l0">
      <div class="layer-header">
        <div class="layer-tag">L0</div>
        <div class="layer-name">Compressor</div>
        <div class="layer-status" id="status-l0"></div>
      </div>
      <div class="layer-value" id="val-l0">Waiting for input</div>
    </div>

    <div class="layer-card" id="card-l2">
      <div class="layer-header">
        <div class="layer-tag">L2</div>
        <div class="layer-name">Simulator</div>
        <div class="layer-status" id="status-l2"></div>
      </div>
      <div class="layer-value" id="val-l2">Waiting for input</div>
    </div>

    <div class="layer-card" id="card-l3">
      <div class="layer-header">
        <div class="layer-tag">L3</div>
        <div class="layer-name">Goal Steerer</div>
        <div class="layer-status" id="status-l3"></div>
      </div>
      <div class="layer-value" id="val-l3">Waiting for input</div>
    </div>
  </div>

  <div class="sidebar-section">
    <div class="sidebar-label">Uncertainty Sources</div>
    <div id="unc-bars">
      <div class="unc-bar-wrap">
        <div class="unc-bar-label"><span>Compression</span><span id="unc-l0-val">—</span></div>
        <div class="unc-bar"><div class="unc-fill" id="unc-l0" style="width:0%;background:#e8a030"></div></div>
      </div>
      <div class="unc-bar-wrap">
        <div class="unc-bar-label"><span>Causal</span><span id="unc-l1-val">—</span></div>
        <div class="unc-bar"><div class="unc-fill" id="unc-l1" style="width:0%;background:#8b7dd8"></div></div>
      </div>
      <div class="unc-bar-wrap">
        <div class="unc-bar-label"><span>Simulation</span><span id="unc-l2-val">—</span></div>
        <div class="unc-bar"><div class="unc-fill" id="unc-l2" style="width:0%;background:#4a90d9"></div></div>
      </div>
    </div>
  </div>

  <div class="sidebar-section">
    <div class="sidebar-label">Session Metrics</div>
    <div class="metric-row">
      <span>Iterations</span>
      <span class="metric-val" id="m-iter">0</span>
    </div>
    <div class="metric-row">
      <span>Causal nodes</span>
      <span class="metric-val" id="m-nodes">0</span>
    </div>
    <div class="metric-row">
      <span>Causal edges</span>
      <span class="metric-val" id="m-edges">0</span>
    </div>
    <div class="metric-row">
      <span>Last compression</span>
      <span class="metric-val" id="m-ratio">—</span>
    </div>
    <div class="metric-row">
      <span>Confidence</span>
      <span class="metric-val" id="m-conf">—</span>
    </div>
    <div class="metric-row">
      <span>Uncertainty source</span>
      <span class="metric-val" id="m-usrc">—</span>
    </div>
  </div>

  <div class="sidebar-section">
    <div class="sidebar-label">Try These</div>
    <button class="example-btn" onclick="ask('How does supply and demand determine prices in a market?')">Supply &amp; demand dynamics</button>
    <button class="example-btn" onclick="ask('Explain how a predator-prey ecosystem maintains balance')">Predator-prey ecosystem</button>
    <button class="example-btn" onclick="ask('What causes inflation and how does it spread through an economy?')">Inflation causality</button>
    <button class="example-btn" onclick="ask('How does the immune system respond to a viral infection?')">Immune response</button>
    <button class="example-btn" onclick="ask('What are the causal factors behind climate change feedback loops?')">Climate feedback loops</button>
    <button class="example-btn" onclick="ask('How does stress affect memory and learning in the brain?')">Stress and memory</button>
  </div>

</div>

<div class="main" id="chat"></div>

<div class="input-area">
  <div class="input-wrap">
    <input type="text" id="user-input"
           placeholder="Ask anything — the kernel will compress, extract causality, simulate, and steer..."
           onkeydown="if(event.key==='Enter')send()">
  </div>
  <button class="send-btn" id="send-btn" onclick="send()">RUN</button>
  <div class="iter-badge" id="iter-badge">iter 0</div>
</div>

<script>
let iteration = 0;
let ratioHistory = [];

function ask(text) {
  document.getElementById('user-input').value = text;
  send();
}

function setLayerStatus(layer, status, value) {
  const card   = document.getElementById('card-' + layer);
  const dot    = document.getElementById('status-' + layer);
  const val    = document.getElementById('val-' + layer);
  card.className = 'layer-card' + (status === 'running' ? ' active' : status === 'error' ? ' error' : '');
  dot.className  = 'layer-status ' + status;
  if (value !== undefined) val.innerHTML = value;
}

function updateUncertainty(l0, l1, l2) {
  const bars = {l0, l1, l2};
  for (const [k, v] of Object.entries(bars)) {
    if (v === undefined) continue;
    const pct = Math.round(v * 100);
    document.getElementById('unc-' + k).style.width = pct + '%';
    document.getElementById('unc-' + k + '-val').textContent = pct + '%';
  }
}

function addMessage(role, content, trace) {
  const chat = document.getElementById('chat');
  const msg  = document.createElement('div');
  msg.className = 'message ' + role;

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.textContent = content;
  msg.appendChild(bubble);

  if (trace) {
    const traceEl = document.createElement('div');
    traceEl.className = 'layer-trace';
    traceEl.innerHTML = trace;
    msg.appendChild(traceEl);
  }

  const meta = document.createElement('div');
  meta.className = 'msg-meta';
  meta.textContent = new Date().toLocaleTimeString();
  msg.appendChild(meta);

  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
  return msg;
}

function addThinking() {
  const chat = document.getElementById('chat');
  const el   = document.createElement('div');
  el.className = 'message kernel';
  el.id = 'thinking';
  el.innerHTML = '<div class="thinking"><div class="thinking-dot"></div><div class="thinking-dot"></div><div class="thinking-dot"></div></div>';
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
}

function removeThinking() {
  const el = document.getElementById('thinking');
  if (el) el.remove();
}

async function send() {
  const input = document.getElementById('user-input');
  const btn   = document.getElementById('send-btn');
  const text  = input.value.trim();
  if (!text) return;

  input.value = '';
  btn.disabled = true;

  addMessage('user', text);
  addThinking();

  // Set layers to running
  setLayerStatus('l1', 'running', 'Extracting causal structure...');
  setLayerStatus('l0', 'running', 'Compressing...');
  setLayerStatus('l2', 'running', 'Simulating futures...');
  setLayerStatus('l3', 'running', 'Steering toward goal...');

  try {
    const res = await fetch('/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, goal: text})
    });
    const data = await res.json();
    removeThinking();

    iteration++;
    document.getElementById('iter-badge').textContent = 'iter ' + iteration;
    document.getElementById('m-iter').textContent = iteration;

    if (data.error) {
      addMessage('kernel', 'Error: ' + data.error);
      setLayerStatus('l1', 'error', data.error);
      setLayerStatus('l0', 'error', '');
      setLayerStatus('l2', 'error', '');
      setLayerStatus('l3', 'error', '');
      btn.disabled = false;
      return;
    }

    const c = data.compressed;
    const ca = data.causal;
    const s = data.simulated;
    const r = data.result;
    const log = data.log_entry;

    // Update layer cards
    setLayerStatus('l1', 'done',
      `<span>${ca.n_nodes} nodes, ${ca.n_edges} edges</span><br>` +
      `Uncertainty: <span>${(ca.uncertainty * 100).toFixed(0)}%</span>`);

    setLayerStatus('l0', 'done',
      `Length: <span>${c.length} chars</span><br>` +
      `Ratio: <span>${c.compression_ratio.toFixed(3)}</span><br>` +
      `Program: <span>${c.program.substring(0, 60)}${c.program.length > 60 ? '...' : ''}</span>`);

    const topNodes = Object.entries(s.distribution || {})
      .sort((a,b) => b[1].mean - a[1].mean).slice(0, 3)
      .map(([n,v]) => `${n}: <span>${(v.mean*100).toFixed(0)}%</span>`).join('<br>');
    setLayerStatus('l2', 'done',
      `${Object.keys(s.distribution||{}).length} outcomes simulated<br>${topNodes}`);

    const actionText = r.action
      ? `Action: <span>${r.action}</span><br>Confidence: <span>${(r.confidence*100).toFixed(0)}%</span>`
      : `Asking: <span>${r.diagnostic || 'clarifying question'}</span>`;
    setLayerStatus('l3', 'done', actionText);

    // Uncertainty bars
    updateUncertainty(
      r.uncertainties?.compression,
      r.uncertainties?.causal,
      r.uncertainties?.simulation
    );

    // Metrics
    document.getElementById('m-nodes').textContent = ca.n_nodes;
    document.getElementById('m-edges').textContent = ca.n_edges;
    const ratio = c.compression_ratio;
    const ratioEl = document.getElementById('m-ratio');
    ratioEl.textContent = ratio.toFixed(3);
    ratioEl.className = 'metric-val ' + (ratio < 0.5 ? 'good' : ratio < 1.0 ? 'warn' : 'bad');

    const confEl = document.getElementById('m-conf');
    confEl.textContent = (r.confidence * 100).toFixed(0) + '%';
    confEl.className = 'metric-val ' + (r.confidence > 0.6 ? 'good' : r.confidence > 0.3 ? 'warn' : 'bad');

    document.getElementById('m-usrc').textContent = r.uncertainty_source || '—';

    // Build trace
    const trace = `
<div class="trace-header">
  <span>LAYER TRACE</span>
  <span>${log.elapsed_s}s</span>
</div>
<div class="trace-row">
  <div class="trace-layer">L1</div>
  <div class="trace-content">Extracted ${ca.n_edges} causal edges from input text (uncertainty: ${(ca.uncertainty*100).toFixed(0)}%)</div>
  <div class="trace-time"></div>
</div>
<div class="trace-row">
  <div class="trace-layer">L0</div>
  <div class="trace-content">Compressed to ${c.length} chars (ratio ${c.compression_ratio.toFixed(3)})<br><span style="color:var(--muted)">${c.program.substring(0,100)}${c.program.length>100?'...':''}</span></div>
  <div class="trace-time"></div>
</div>
<div class="trace-row">
  <div class="trace-layer">L2</div>
  <div class="trace-content">Simulated ${Object.keys(s.distribution||{}).length} outcome nodes (sim uncertainty: ${(s.uncertainty*100).toFixed(0)}%)</div>
  <div class="trace-time"></div>
</div>
<div class="trace-row">
  <div class="trace-layer">L3</div>
  <div class="trace-content">${r.action
    ? `Recommends action: <strong>${r.action}</strong> → <em>${r.target_node}</em> (${(r.confidence*100).toFixed(0)}% confidence)<br>Dominant uncertainty: <strong>${r.uncertainty_source}</strong>`
    : `Uncertainty too high (source: <strong>${r.uncertainty_source}</strong>)<br>${r.diagnostic}`}
  </div>
  <div class="trace-time"></div>
</div>`;

    // Build response text
    let responseText = '';
    if (r.action) {
      responseText = `Recommended action: ${r.action}\n\nTarget: ${r.target_node} (${(r.confidence*100).toFixed(0)}% confidence)\n\nCausal graph: ${ca.n_nodes} nodes, ${ca.n_edges} edges extracted\n\nDominant uncertainty: ${r.uncertainty_source}`;
      if (r.note) responseText += `\n\n${r.note}`;
    } else {
      responseText = `Uncertainty too high to recommend an action.\n\nSource: ${r.uncertainty_source}\n\n${r.diagnostic || ''}`;
    }

    addMessage('kernel', responseText, trace);

  } catch(e) {
    removeThinking();
    addMessage('kernel', 'Connection error: ' + e.message + '. Is the server running?');
    setLayerStatus('l1', 'error', 'Connection failed');
    setLayerStatus('l0', 'error', '');
    setLayerStatus('l2', 'error', '');
    setLayerStatus('l3', 'error', '');
  }

  btn.disabled = false;
  input.focus();
}

// Init
fetch('/status').then(r=>r.json()).then(d=>{
  document.getElementById('backend-badge').textContent = d.backend.toUpperCase();
});
</script>
</body>
</html>'''


@app.route('/')
def index():
    return HTML

@app.route('/status')
def status():
    return jsonify({
        'backend': LLM_BACKEND,
        'iteration': kernel.world_model.iteration,
        'nodes': kernel.world_model.causal_graph.number_of_nodes(),
        'edges': kernel.world_model.causal_graph.number_of_edges(),
    })

@app.route('/run', methods=['POST'])
def run():
    data = request.json
    text = data.get('text', '')
    goal = data.get('goal', text)

    if not text:
        return jsonify({'error': 'No input provided'})

    try:
        result = kernel.forward(text, goal, n_samples=300)

        compressed = result['compressed']
        causal     = result['causal']
        simulated  = result['simulated']
        r          = result['result']
        log        = result['log_entry']

        return jsonify({
            'compressed': {
                'program':          compressed.program,
                'length':           compressed.length,
                'compression_ratio':compressed.compression_ratio,
                'uncertainty':      compressed.uncertainty,
            },
            'causal': {
                'n_nodes':    causal.n_nodes,
                'n_edges':    causal.n_edges,
                'uncertainty':causal.uncertainty,
            },
            'simulated': {
                'distribution': simulated.distribution,
                'uncertainty':  simulated.uncertainty,
            },
            'result': {
                'action':             r.action,
                'target_node':        r.target_node,
                'confidence':         r.confidence,
                'uncertainty_source': r.uncertainty_source,
                'uncertainties':      r.uncertainties,
                'diagnostic':         r.diagnostic,
                'note':               r.note,
            },
            'log_entry': log,
        })
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    print("Intelligence Kernel — Interactive Interface")
    print(f"Backend: {LLM_BACKEND.upper()}")
    print("Open: http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
