# app.py  -- upgraded: delta timer, reset avg, /time_counter_stats
from flask import Flask, request, g, jsonify, render_template_string, Response
import sqlite3, math, time, json, traceback
from collections import defaultdict
from statistics import mean, median

# --------- Configuration ----------
DB_PATH = "stats.db"
TOPK_DEFAULT = 8
MAX_SAVED_ROWS = 2000

# initial word list (same as before)
INITIAL_WORDS = [
  "break","case","catch","class","const","continue","debugger","default","delete","do",
  "else","export","extends","finally","for","function","if","import","in","instanceof",
  "let","new","return","super","switch","this","throw","try","typeof","var","void",
  "while","with","yield","await","async","static","get","set","of",
  "console","math","number","string","boolean","array","object","json","date","promise",
  "map","set","weakmap","weakset","symbol","regexp","error","eval",
  "then","catch","resolve","reject","async","await","finally",
  "module","exports","require","__dirname","__filename",
  "parseint","parsefloat","isnan","isfinite","encodeuri","decodeduri","encodeuricomponent",
  "decodeuricomponent","jsonstringify","jsonparse",
  "floor","ceil","round","random","abs","min","max","pow","sqrt","log","exp",
  "log","warn","error","info","debug","table","time","timeend",
  "window","document","location","history","navigator","localstorage","sessionstorage",
  "alert","confirm","prompt","fetch","addeventlistener","removeeventlistener",
  "queryselector","queryselectorall","getelementbyid","getelementsbyclassname","getelementsbytagname",
  "createelement","appendchild","removechild","replacechild","classlist","classname","dataset",
  "innerhtml","textcontent","value","style",
  "settimeout","cleartimeout","setinterval","clearinterval",
  "push","pop","shift","unshift","splice","slice","map","filter","reduce","foreach",
  "find","findindex","includes","indexof","join","split","concat","sort","reverse",
  "replace","tolowercase","touppercase","trim","substr","substring","startswith","endswith",
  "prototype","constructor","hasownproperty","keys","values","entries","assign","create","defineproperty",
  "addclass","removeclass","toggleclass","append","prepend","closest","matches","contains",
  "jquery","react","vue","angular","redux","rxjs",
  "id","name","type","length","size","index","key","value","payload","props","state","setstate",
  "dispatch","subscribe","unsubscribe","handler","callback","errorhandler","response","request"
]

# ---------- Flask app ----------
app = Flask(__name__)

# store last server-side error trace (for /debug_error)
last_server_error = {"ts": None, "trace": None, "message": None}

# in-memory vocab structures (loaded from DB)
WORDS = []
WORD2IDX = {}
IDX2WORD = {}
NUM_WORDS = 0
INDEX = defaultdict(list)      # first-letter index
BIGRAM_COUNTS = {}             # (prev, next) -> count

# ---------- sqlite helpers ----------
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    db = sqlite3.connect(DB_PATH)
    cur = db.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL,
        payload TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS vocab (
        word TEXT PRIMARY KEY
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS bigrams (
        prev TEXT,
        nxt TEXT,
        cnt INTEGER,
        PRIMARY KEY(prev, nxt)
    )
    """)
    db.commit()
    # seed vocab if empty
    cur.execute("SELECT COUNT(*) as c FROM vocab")
    if cur.fetchone()[0] == 0:
        cur.executemany("INSERT OR IGNORE INTO vocab (word) VALUES (?)", [(w.lower(),) for w in INITIAL_WORDS])
        db.commit()
    db.close()

def load_vocab_from_db():
    global WORDS, WORD2IDX, IDX2WORD, NUM_WORDS, INDEX
    db = sqlite3.connect(DB_PATH)
    cur = db.cursor()
    cur.execute("SELECT word FROM vocab ORDER BY word COLLATE NOCASE")
    rows = cur.fetchall()
    words = [r[0] for r in rows]
    # dedupe & normalize
    seen = set(); unique = []
    for w in words:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            unique.append(wl)
    WORDS = unique
    WORD2IDX = {w: i for i, w in enumerate(WORDS)}
    IDX2WORD = {i: w for i, w in enumerate(WORDS)}
    NUM_WORDS = len(WORDS)
    INDEX = defaultdict(list)
    for i, w in enumerate(WORDS):
        if len(w) > 0:
            INDEX[w[0]].append(i)
    db.close()

def load_bigrams_from_db():
    global BIGRAM_COUNTS
    db = sqlite3.connect(DB_PATH)
    cur = db.cursor()
    cur.execute("SELECT prev, nxt, cnt FROM bigrams")
    BIGRAM_COUNTS = {}
    for prev, nxt, cnt in cur.fetchall():
        BIGRAM_COUNTS[(prev, nxt)] = cnt
    db.close()

def inc_bigram(prev, nxt, delta=1):
    if not accepted_valid(prev) or not accepted_valid(nxt):
        return
    db = sqlite3.connect(DB_PATH)
    cur = db.cursor()
    cur.execute("INSERT INTO bigrams(prev, nxt, cnt) VALUES (?, ?, ?) ON CONFLICT(prev, nxt) DO UPDATE SET cnt = cnt + ?",
                (prev, nxt, delta, delta))
    db.commit()
    db.close()
    BIGRAM_COUNTS[(prev, nxt)] = BIGRAM_COUNTS.get((prev, nxt), 0) + delta

def accepted_valid(s):
    return isinstance(s, str)

# ---------- Predictor: heuristic + bigram boost ----------
def score_word_by_prefix(word: str, prefix: str, context_words=None):
    if prefix is None:
        prefix = ""
    prefix = prefix.lower()
    w = word.lower()
    match_len = 0
    for a, b in zip(w, prefix):
        if a == b:
            match_len += 1
        else:
            break
    full_prefix_match = 1.0 if w.startswith(prefix) and len(prefix) > 0 else 0.0
    prop = (match_len / max(1, len(w)))
    length_penalty = math.log(1.0 + len(w)) * 0.08
    context_bonus = 0.0
    if context_words and len(context_words) > 0:
        prev = context_words[-1].lower()
        cnt = BIGRAM_COUNTS.get((prev, w), 0)
        if cnt > 0:
            context_bonus += math.log(1 + cnt) * 0.5
    score = (match_len * 1.2) + (full_prefix_match * 2.0) + (prop * 0.5) - length_penalty + context_bonus
    return score

def predict(prefix: str, context_words=None, topk=TOPK_DEFAULT):
    prefix = (prefix or "").strip().lower()
    candidates_idx = list(range(NUM_WORDS))
    if len(prefix) >= 1 and prefix[0] in INDEX:
        candidates_idx = INDEX[prefix[0]][:]
        if len(candidates_idx) == 0:
            candidates_idx = list(range(NUM_WORDS))
    scored = []
    for i in candidates_idx:
        w = IDX2WORD[i]
        s = score_word_by_prefix(w, prefix, context_words)
        scored.append((i, s))
    if not scored:
        return {"prefix": prefix, "context": context_words or [], "topk": [], "all_probs": {}, "allowed_mass": 0.0}
    max_s = max(s for _, s in scored)
    exps = [(i, math.exp(s - max_s)) for i, s in scored]
    Z = sum(v for _, v in exps) or 1.0
    probs = [(i, v / Z) for i, v in exps]
    probs.sort(key=lambda x: -x[1])
    topk_list = [(IDX2WORD[i], float(p)) for i, p in probs[:topk]]
    allowed_mass = 0.0
    if len(prefix) >= 1 and prefix[0] in INDEX:
        allowed_set = set(INDEX[prefix[0]])
        for i, p in probs:
            if i in allowed_set:
                allowed_mass += p
    else:
        allowed_mass = sum(p for _, p in probs)
    return {"prefix": prefix, "context": context_words or [], "topk": topk_list,
            "all_probs": {IDX2WORD[i]: float(p) for i, p in probs}, "allowed_mass": allowed_mass}

def _safe_float(x):
    try:
        if x is None: return None
        if isinstance(x, (int, float)):
            if math.isfinite(x):
                return float(x)
            else:
                return None
        return None
    except Exception:
        return None

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/favicon.ico")
def favicon():
    return Response(status=204)

@app.route("/debug_error")
def debug_error():
    return jsonify(last_server_error)

@app.route("/predict", methods=["POST"])
def predict_route():
    global last_server_error
    try:
        data = request.get_json(force=True) or {}
        prefix = data.get("prefix", "")
        context = data.get("context", [])
        target = data.get("target", None)
        topk = int(data.get("topk", TOPK_DEFAULT))
        start_ts = time.time()
        preds = predict(prefix, context_words=context, topk=topk)
        end_ts = time.time()
        metrics = {}
        all_probs = preds.get("all_probs", {})
        allowed_mass = preds.get("allowed_mass", 0.0)
        if target:
            target = target.lower()
            p_true = float(all_probs.get(target, 0.0))
            if p_true > 0:
                nll = -math.log(max(1e-12, p_true))
            else:
                nll = float("inf")
            rank = 1 + sum(1 for v in all_probs.values() if v > p_true)
            index_error = 1.0 - (p_true / allowed_mass) if allowed_mass > 0 else 1.0
            metrics = {
                "p_true": _safe_float(p_true),
                "nll": _safe_float(nll),
                "rank": rank,
                "index_error": _safe_float(index_error)
            }
        response = {
            "preds": preds["topk"],
            "allowed_mass": _safe_float(preds.get("allowed_mass", 0.0)),
            "all_probs_sample": preds.get("all_probs", {}),
            "metrics": metrics,
            "server_ts": end_ts,
            "server_latency": _safe_float(end_ts - start_ts)
        }
        return jsonify(response)
    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error("predict_route exception:\n%s", tb)
        last_server_error.update({"ts": time.time(), "trace": tb, "message": str(e)})
        return jsonify({"ok": False, "error": str(e), "trace": tb}), 500

@app.route("/stats", methods=["POST"])
def stats_route():
    try:
        payload = request.get_json(force=True) or {}
        db = get_db()
        cur = db.cursor()
        cur.execute("INSERT INTO stats (ts, payload) VALUES (?, ?)", (time.time(), json.dumps(payload)))
        db.commit()
        return jsonify({"ok": True})
    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error("stats_route exception:\n%s", tb)
        last_server_error.update({"ts": time.time(), "trace": tb, "message": str(e)})
        return jsonify({"ok": False, "error": str(e), "trace": tb}), 500

@app.route("/add_word", methods=["POST"])
def add_word_route():
    try:
        j = request.get_json(force=True) or {}
        w = (j.get("word") or "").strip().lower()
        if not w:
            return jsonify({"ok": False, "error": "empty word"}), 400
        db = get_db()
        cur = db.cursor()
        cur.execute("INSERT OR IGNORE INTO vocab (word) VALUES (?)", (w,))
        db.commit()
        load_vocab_from_db()
        return jsonify({"ok": True, "word": w, "num_words": NUM_WORDS})
    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error("add_word exception:\n%s", tb)
        last_server_error.update({"ts": time.time(), "trace": tb, "message": str(e)})
        return jsonify({"ok": False, "error": str(e), "trace": tb}), 500

@app.route("/accept", methods=["POST"])
def accept_route():
    try:
        j = request.get_json(force=True) or {}
        prev = (j.get("prev") or "").strip().lower()
        accepted = (j.get("accepted") or "").strip().lower()
        if not accepted:
            return jsonify({"ok": False, "error": "no accepted word"}), 400
        inc_bigram(prev, accepted, delta=1)
        return jsonify({"ok": True})
    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error("accept_route exception:\n%s", tb)
        last_server_error.update({"ts": time.time(), "trace": tb, "message": str(e)})
        return jsonify({"ok": False, "error": str(e), "trace": tb}), 500

@app.route("/time_counter_stats")
def time_counter_stats():
    """
    New page showing aggregated click-time logs saved via /stats.
    It parses payloads (assumed to be JSON with 'samples': [...]) and extracts dt_char and req_latency.
    """
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT id, ts, payload FROM stats ORDER BY id DESC LIMIT 500")
        rows = cur.fetchall()
        # collect flattened samples
        flat = []
        for r in rows:
            pid = r["id"]; pts = r["ts"]
            try:
                payload = json.loads(r["payload"])
            except Exception:
                payload = r["payload"]
            # payload may be {"samples":[...]} or direct array etc.
            if isinstance(payload, dict) and "samples" in payload and isinstance(payload["samples"], list):
                for s in payload["samples"]:
                    s2 = dict(s)
                    s2["_row_id"] = pid
                    s2["_row_ts"] = pts
                    flat.append(s2)
            elif isinstance(payload, list):
                for s in payload:
                    if isinstance(s, dict):
                        s2 = dict(s)
                        s2["_row_id"] = pid
                        s2["_row_ts"] = pts
                        flat.append(s2)
            elif isinstance(payload, dict):
                s2 = dict(payload)
                s2["_row_id"] = pid
                s2["_row_ts"] = pts
                flat.append(s2)
            else:
                # store raw
                flat.append({"raw": str(payload), "_row_id": pid, "_row_ts": pts})
        # compute aggregates for dt_char and req_latency
        dt_vals = [s.get("dt_char") for s in flat if isinstance(s.get("dt_char"), (int, float))]
        req_vals = [s.get("req_latency") for s in flat if isinstance(s.get("req_latency"), (int, float))]
        def agg(arr):
            if not arr: return {"count":0,"avg":None,"median":None,"min":None,"max":None}
            return {"count":len(arr),"avg":round(mean(arr),2),"median":round(median(arr),2),"min":min(arr),"max":max(arr)}
        dt_agg = agg(dt_vals)
        req_agg = agg(req_vals)
        # render a simple HTML
        html = ["<html><head><title>Time counter stats</title><style>body{font-family:Arial;background:#071019;color:#dfeff6}table{border-collapse:collapse;width:100%}td,th{padding:6px;border:1px solid #16323b}thead{background:#08202a}</style></head><body>"]
        html.append(f"<h2>Time counter stats (last DB rows: {len(rows)})</h2>")
        html.append("<h3>Aggregates</h3>")
        html.append("<table><tr><th>metric</th><th>dt_char (ms)</th><th>req_latency (ms)</th></tr>")
        html.append(f"<tr><td>count</td><td>{dt_agg['count']}</td><td>{req_agg['count']}</td></tr>")
        html.append(f"<tr><td>avg</td><td>{dt_agg['avg']}</td><td>{req_agg['avg']}</td></tr>")
        html.append(f"<tr><td>median</td><td>{dt_agg['median']}</td><td>{req_agg['median']}</td></tr>")
        html.append(f"<tr><td>min</td><td>{dt_agg['min']}</td><td>{req_agg['min']}</td></tr>")
        html.append(f"<tr><td>max</td><td>{dt_agg['max']}</td><td>{req_agg['max']}</td></tr>")
        html.append("</table>")
        html.append("<h3>Recent flattened samples (most recent first, limited)</h3>")
        html.append("<table><thead><tr><th>#</th><th>row_id</th><th>row_ts</th><th>dt_char</th><th>req_latency</th><th>prefix</th><th>context</th><th>preds</th></tr></thead><tbody>")
        for i, s in enumerate(flat[:500]):
            dt = s.get("dt_char", "")
            rq = s.get("req_latency", "")
            pref = s.get("prefix", "")
            ctx = s.get("context", "")
            preds = s.get("preds", "")
            html.append(f"<tr><td>{i+1}</td><td>{s.get('_row_id')}</td><td>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(s.get('_row_ts',0)))}</td><td>{dt}</td><td>{rq}</td><td>{pref}</td><td>{json.dumps(ctx)}</td><td>{json.dumps(preds)}</td></tr>")
        html.append("</tbody></table>")
        html.append("</body></html>")
        return Response("\n".join(html), mimetype="text/html")
    except Exception as e:
        tb = traceback.format_exc()
        last_server_error.update({"ts": time.time(), "trace": tb, "message": str(e)})
        return jsonify({"ok": False, "error": str(e), "trace": tb}), 500

# ---------- boot DB and load data ----------
init_db()
load_vocab_from_db()
load_bigrams_from_db()

# ---------- front-end (HTML+JS) ----------
INDEX_HTML = r"""
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Editor + Predictor (delta timer)</title>
<style>
:root{--bg:#0b0f14;--panel:#0f1720;--accent:#28a3ff;--muted:#9aa7b2;--text:#e6eef6}
html,body{height:100%;margin:0;background:linear-gradient(180deg,#071019,#06121a);font-family:Arial;color:var(--text)}
.app{display:flex;height:100vh;gap:12px;padding:12px;box-sizing:border-box}
.editor-wrap{flex:1;display:flex;flex-direction:column;background:var(--panel);border-radius:8px;padding:8px}
.toolbar{display:flex;gap:8px;align-items:center;padding:6px 8px}
.toolbar button{background:#0b1220;border:1px solid rgba(255,255,255,0.04);color:var(--muted);padding:6px 10px;border-radius:6px;cursor:pointer}
.editor{flex:1;margin:8px;border-radius:6px;overflow:auto;background:#04101a;padding:10px;position:relative}
.bg-pred{position:absolute;left:12px;top:8px;z-index:1;opacity:0.5;color:rgba(200,230,255,0.85);font-family:monospace;font-size:48px;pointer-events:none;user-select:none;white-space:pre-wrap;line-height:1.2;transform-origin:left top;transition:opacity 150ms linear}
textarea.code{position:relative;z-index:2;width:100%;height:100%;border:0;outline:0;resize:none;font-family:Consolas,monospace;font-size:14px;color:#dff3ff;background:transparent;line-height:1.4;padding:6px;box-sizing:border-box}
.right-panel{width:360px;min-width:280px;max-width:420px;display:flex;flex-direction:column;gap:8px}
.panel{background:linear-gradient(180deg,#07131a,#041018);padding:10px;border-radius:8px}
.small{color:var(--muted);font-size:12px}
.timer{font-weight:700;font-size:18px;color:var(--accent)}
.pred-list{margin-top:6px;display:flex;flex-direction:column;gap:6px;max-height:300px;overflow:auto}
.pred-item{display:flex;justify-content:space-between;align-items:center;padding:6px 8px;border-radius:6px;background:linear-gradient(90deg,rgba(255,255,255,0.02),transparent)}
.pred-word{font-family:monospace;color:#dff3ff}
.pred-prob{color:var(--muted);font-weight:600}
.metric{background:rgba(255,255,255,0.02);padding:6px 8px;border-radius:6px;color:var(--muted);font-size:13px}
.save-btn{background:linear-gradient(90deg,var(--accent),#7ad3ff);color:#04202b;padding:8px 10px;border-radius:8px;border:0;cursor:pointer;font-weight:700}
.stats-row{display:flex;gap:12px;align-items:center;margin-top:6px}
.avg-time{font-weight:700;color:#a3f0ff}
.delta-time{font-weight:700;color:#ffd37a}
</style>
</head>
<body>
<div class="app">
  <div class="editor-wrap">
    <div class="toolbar">
      <div class="small">Type code; top-1 autocompletes when you pause longer than your avg click time.</div>
      <div style="flex:1"></div>
      <button id="context-toggle">Context: last 1 word</button>
      <button id="reset-avg" style="margin-left:8px;background:#1f2b2f;color:#cfe">Reset Avg</button>
    </div>
    <div class="editor">
      <div id="bg-pred" class="bg-pred"></div>
      <textarea id="code" class="code" spellcheck="false" placeholder="// start typing..."></textarea>
    </div>
    <div style="padding:8px;display:flex;gap:8px">
      <button id="clear-stats" class="save-btn" style="background:#ff6b6b">Clear all samples</button>
      <button id="send-stats" class="save-btn">Send stats to server</button>
      <div style="flex:1"></div>
      <div class="small">Local samples: <span id="sample-count">0</span></div>
    </div>
  </div>

  <div class="right-panel">
    <div class="panel">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div>
          <div class="small">Time since last char</div>
          <div class="timer" id="time-since">0 ms</div>
        </div>
        <div style="width:220px">
          <div class="small">Latency (ms) history</div>
          <canvas id="latency-graph" width="220" height="100" style="background:#011216;border-radius:6px"></canvas>
          <div class="stats-row">
            <div class="small">Avg click</div>
            <div class="avg-time" id="avg-click">— ms</div>
            <div style="width:8px"></div>
            <div class="small">Delta to auto-insert</div>
            <div class="delta-time" id="delta-timer">— ms</div>
          </div>
          <div style="height:6px"></div>
          <div style="height:8px;background:#052023;border-radius:6px;overflow:hidden">
            <div id="delta-bar" style="height:8px;width:0%;background:linear-gradient(90deg,#ffd37a,#ffb86b)"></div>
          </div>
        </div>
      </div>

      <div style="margin-top:10px">
        <div class="small">Current token</div>
        <div style="display:flex;gap:6px;align-items:center;margin-top:4px">
          <div style="flex:1;background:#021b22;padding:8px;border-radius:6px;font-family:monospace;color:#bfefff" id="current-token">—</div>
          <div style="width:72px;text-align:right">
            <div class="small">Top prob</div>
            <div id="top-prob" style="font-weight:700;color:var(--accent)">—</div>
          </div>
        </div>
      </div>

      <div style="margin-top:8px">
        <div class="small">Predictions (top)</div>
        <div class="pred-list" id="pred-list"></div>
        <div id="metrics" class="metrics"></div>
      </div>

      <div style="margin-top:8px;display:flex;justify-content:space-between;align-items:center">
        <div class="small">Stats kept in-memory & localStorage (also POSTable).</div>
        <button id="save-local" class="save-btn">Dump to console</button>
      </div>

      <div style="margin-top:6px">
        <a href="/time_counter_stats" style="color:#9fe;text-decoration:none">View server logs /time_counter_stats</a>
      </div>
    </div>

    <div class="panel small">
      <b>How it works (short)</b>
      <ul>
        <li>Client measures time between your keystrokes (dt) and logs samples.</li>
        <li>If you pause longer than the avg dt, top-1 completion is auto-inserted.</li>
        <li>Accepted completions are reported to the server at <code>/accept</code>, updating bigram stats.</li>
        <li>Use <code>/add_word</code> to grow vocabulary (server-side).</li>
      </ul>
    </div>
  </div>
</div>

<script>
(function(){
  const textarea = document.getElementById('code');
  const predList = document.getElementById('pred-list');
  const currentTokenDiv = document.getElementById('current-token');
  const topProb = document.getElementById('top-prob');
  const timeSince = document.getElementById('time-since');
  const canvas = document.getElementById('latency-graph');
  const ctx = canvas.getContext('2d');
  const sampleCount = document.getElementById('sample-count');
  const metricsDiv = document.getElementById('metrics');
  const sendBtn = document.getElementById('send-stats');
  const saveBtn = document.getElementById('save-local');
  const clearBtn = document.getElementById('clear-stats');
  const contextToggle = document.getElementById('context-toggle');
  const bgPred = document.getElementById('bg-pred');
  const avgClickEl = document.getElementById('avg-click');
  const deltaTimerEl = document.getElementById('delta-timer');
  const deltaBar = document.getElementById('delta-bar');
  const resetAvgBtn = document.getElementById('reset-avg');

  let useContextWords = 1;
  contextToggle.addEventListener('click', ()=>{
    useContextWords = (useContextWords % 3) + 1;
    contextToggle.textContent = `Context: last ${useContextWords} word${useContextWords>1?'s':''}`;
  });

  // timing & sample storage
  let prevCharTs = null;     // time of previous user keystroke
  let lastCharTs = null;     // time of last keystroke (for timer)
  let lastResponseTs = null;
  let ignoreProgrammatic = false;
  const maxPoints = 80;
  let latencyHistory = [];
  const MAX_SAVED_ROWS = 2000;   // <--- add this to avoid ReferenceError in browser
  let samples = [];
  try {
    const saved = JSON.parse(localStorage.getItem('predictor_samples') || "null");
    if (Array.isArray(saved) && saved.length) {
      samples = saved;
      latencyHistory = samples.map(s=>s.req_latency).slice(-maxPoints);
      sampleCount.textContent = samples.length;
    }
  } catch(e){ console.warn(e) }

  function saveLocalStorage(){
    localStorage.setItem('predictor_samples', JSON.stringify(samples));
  }

  function computeAverageClick() {
    const clicks = samples.map(s=>s.dt_char).filter(v=>v!=null && !isNaN(v));
    if (!clicks.length) return null;
    const sum = clicks.reduce((a,b)=>a+b,0);
    return Math.round(sum / clicks.length);
  }

  function updateAvgDisplay() {
    const avg = computeAverageClick();
    avgClickEl.textContent = (avg === null) ? "— ms" : (avg + " ms");
  }

  function drawGraph(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle = "#002328"; ctx.fillRect(0,0,canvas.width,canvas.height);
    if (latencyHistory.length === 0) return;
    const w = canvas.width, h = canvas.height;
    const arr = latencyHistory.slice(-maxPoints);
    const maxv = Math.max(10, ...arr);
    ctx.beginPath(); ctx.strokeStyle = "#3ee2ff"; ctx.lineWidth = 2;
    for (let i=0;i<arr.length;i++){
      const x = (i/(arr.length-1||1)) * (w-6) + 3;
      const y = h - ((arr[i]/maxv) * (h-10) + 5);
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
    const last = arr[arr.length-1] || 0;
    const lx = w - 6;
    const ly = h - ((last/maxv) * (h-10) + 5);
    ctx.fillStyle = "#9effff"; ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = "#86b7bf"; ctx.font = "10px monospace"; ctx.fillText(Math.round(maxv) + "ms", 6, 12);
  }

  function getTokenAtCaret(textarea) {
    const value = textarea.value;
    const pos = textarea.selectionStart;
    let start = pos;
    while (start > 0) {
      const ch = value[start-1];
      if (/\s|[\(\)\{\};,\.]/.test(ch)) break;
      start--;
    }
    let end = pos;
    while (end < value.length) {
      const ch = value[end];
      if (/\s|[\(\)\{\};,\.]/.test(ch)) break;
      end++;
    }
    const token = value.slice(start, pos);
    const left = value.slice(0, start).trim();
    const words = left.length ? left.split(/\s+/).filter(Boolean) : [];
    return { token, pos, start, end, prevWords: words.slice(-useContextWords) };
  }

  // requestPrediction accepts precomputed dt_char (delta between this char and previous char)
  function requestPrediction(prefix, contextWords, dt_char=null) {
    const body = { prefix, context: contextWords, topk: 12 };
    const reqTs = performance.now();
    return fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body)
    }).then(resp => resp.json()).then(json => {
      if (!json) throw new Error("Empty JSON from /predict");
      if (json.ok === false && json.error) {
        console.warn("Server returned error:", json.error);
      }
      const respTs = performance.now();
      const req_latency = Math.round(respTs - reqTs);
      const sample = {
        ts: Date.now(),
        dt_char: dt_char,
        req_latency: req_latency,
        server_latency: json.server_latency || null,
        prefix: prefix,
        context: contextWords,
        preds: json.preds,
        metrics: json.metrics
      };
      samples.push(sample);
      if (samples.length > MAX_SAVED_ROWS) samples.shift();
      saveLocalStorage();
      sampleCount.textContent = samples.length;
      latencyHistory.push(req_latency);
      if (latencyHistory.length > maxPoints) latencyHistory.shift();
      drawGraph();
      updateAvgDisplay();
      renderPredictions(json);
      lastResponseTs = performance.now();

      // AFTER UI update, possibly auto-insert if conditions match
      tryAutoInsert(prefix, contextWords, json);
      return sample;
    }).catch(err=>{
      console.error("predict error", err);
      fetch('/debug_error').then(r=>r.json()).then(d=>{ console.warn("server debug:", d); }).catch(()=>{});
    });
  }

  function renderPredictions(json) {
    if (!json) return;
    const preds = json.preds || [];
    const allowed_mass = json.allowed_mass || 0;
    predList.innerHTML = "";
    let top = null;
    for (const [word, prob] of preds) {
      if (!top) top = prob;
      const el = document.createElement('div');
      el.className = 'pred-item';
      el.innerHTML = `<div class="pred-word">${word}</div><div class="pred-prob">${(prob*100).toFixed(2)}%</div>`;
      predList.appendChild(el);
    }
    currentTokenDiv.textContent = currentToken.token || "—";
    topProb.textContent = top ? ((top*100).toFixed(2) + '%') : '—';
    metricsDiv.innerHTML = "";
    if (json.metrics && Object.keys(json.metrics).length>0) {
      const m = json.metrics;
      const nodes = [
        ['p(true)', m.p_true!=null?m.p_true.toFixed(4):'—'],
        ['NLL', m.nll!=null?m.nll.toFixed(3):'—'],
        ['rank', m.rank||'—'],
        ['index_err', m.index_error!=null?m.index_error.toFixed(3):'—']
      ];
      nodes.forEach(([k,v])=>{
        const d = document.createElement('div'); d.className='metric'; d.textContent = `${k}: ${v}`; metricsDiv.appendChild(d);
      });
    } else {
      const d = document.createElement('div'); d.className='metric'; d.textContent = `allowed_mass: ${allowed_mass.toFixed(3)}`; metricsDiv.appendChild(d);
    }
    if (preds && preds.length > 0 && preds[0][0]) {
      bgPred.textContent = preds[0][0];
      bgPred.style.opacity = 0.5;
    } else {
      bgPred.textContent = "";
    }
  }

  // auto-insert logic with DELTA timer
  function tryAutoInsert(prefix, contextWords, json) {
    if (!prefix || prefix.length === 0) {
      updateDeltaDisplay(null); return;
    }
    const preds = (json && json.preds) || [];
    if (!preds || preds.length === 0) { updateDeltaDisplay(null); return; }
    const topWord = preds[0][0];
    if (!topWord) { updateDeltaDisplay(null); return; }
    if (!topWord.startsWith(prefix)) { updateDeltaDisplay(null); return; }

    if (!lastCharTs) { updateDeltaDisplay(null); return; }
    const sinceLastMs = Math.round(performance.now() - lastCharTs);
    const avg = computeAverageClick();
    if (avg === null) { updateDeltaDisplay(null); return; }

    const delta = Math.max(0, avg - sinceLastMs);
    updateDeltaDisplay(delta);

    // if user paused longer than avg, trigger insertion
    if (sinceLastMs > avg) {
      performAutoInsert(prefix, topWord, contextWords);
    }
  }

  function updateDeltaDisplay(delta) {
    if (delta === null) {
      deltaTimerEl.textContent = "— ms";
      deltaBar.style.width = "0%";
      deltaBar.style.background = "linear-gradient(90deg,#ffd37a,#ffb86b)";
      return;
    }
    deltaTimerEl.textContent = delta + " ms";
    // show progress: 0% when delta==avg (i.e., just started), 100% when delta==0 (fire)
    const avg = computeAverageClick() || 1;
    const pct = Math.max(0, Math.min(100, ((avg - delta) / avg) * 100));
    deltaBar.style.width = pct + "%";
    if (delta === 0) {
      deltaBar.style.background = "linear-gradient(90deg,#9effff,#3ee2ff)";
    } else {
      deltaBar.style.background = "linear-gradient(90deg,#ffd37a,#ffb86b)";
    }
  }

  function performAutoInsert(prefix, topWord, contextWords){
    if (!prefix || !topWord) return;
    // avoid loops
    if (ignoreProgrammatic) return;
    ignoreProgrammatic = true;
    const tok = currentToken;
    const start = tok.start, end = tok.end;
    const val = textarea.value;
    const suffix = topWord.slice(prefix.length);
    const newVal = val.slice(0, start) + topWord + " " + val.slice(end);
    const newCaret = start + topWord.length + 1;
    textarea.value = newVal;
    textarea.setSelectionRange(newCaret, newCaret);
    // record accept to server
    const prevWord = (contextWords && contextWords.length>0) ? contextWords[contextWords.length-1] : "";
    fetch('/accept', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ prev: prevWord || "", accepted: topWord })
    }).then(r=>r.json()).then(j=>{
      // ignore details
    }).catch(()=>{});
    setTimeout(()=>{ ignoreProgrammatic = false; }, 30);
    // update token and request next prediction
    const newTok = getTokenAtCaret(textarea);
    currentToken = newTok;
    // request prediction for next token, use the completed word as context
    requestPrediction(newTok.token, [topWord], null);
    // briefly update UI
    renderPredictions({ preds: [[topWord, 1.0]] });
    // clear delta display so user sees insertion happened
    updateDeltaDisplay(null);
  }

  let currentToken = { token: "", prevWords: [], start:0, end:0 };
  function onInputEvent(e){
    if (ignoreProgrammatic) return;
    const now = performance.now();
    const tok = getTokenAtCaret(textarea);
    const dt_char = (prevCharTs ? Math.round(now - prevCharTs) : null);
    prevCharTs = now;
    lastCharTs = now;
    currentToken = tok;
    timeSince.textContent = "0 ms";
    requestPrediction(tok.token, tok.prevWords, dt_char);
  }

  textarea.addEventListener('input', onInputEvent);

  function syncOverlayScroll() {
    bgPred.style.transform = `translateY(${-textarea.scrollTop}px)`;
  }
  textarea.addEventListener('scroll', syncOverlayScroll);

  setInterval(()=>{
    if (!lastCharTs) { timeSince.textContent = "— ms"; updateDeltaDisplay(null); return; }
    const ms = Math.round(performance.now() - lastCharTs);
    timeSince.textContent = ms + " ms";
    // keep delta display updated even when not receiving preds (compute using latest prefix and avg)
    tryAutoInsert(currentToken.token, currentToken.prevWords, {preds: (predList.children.length?[[bgPred.textContent,1.0]]:[])});
  }, 50);

  drawGraph();
  updateAvgDisplay();

  // reset average: remove dt_char entries from samples (or clear all samples)
  resetAvgBtn.addEventListener('click', ()=>{
    if (!confirm("Reset average click time? This will clear recorded dt samples and reset the average.") ) return;
    // remove dt_char values by filtering them out
    samples = samples.filter(s => s.dt_char == null);
    // also reset latency history and persist
    latencyHistory = [];
    sampleCount.textContent = samples.length;
    saveLocalStorage();
    drawGraph();
    updateAvgDisplay();
    alert("Average click time reset. New typing will be collected.");
  });

  saveBtn.addEventListener('click', ()=>{
    console.log("SAMPLES DUMP:", samples);
    alert("Samples dumped to console. See DevTools.");
  });

  clearBtn.addEventListener('click', ()=>{
    if (!confirm("Clear ALL local samples and reset everything?")) return;
    samples = []; latencyHistory = []; sampleCount.textContent = 0;
    localStorage.removeItem('predictor_samples');
    bgPred.textContent = ""; drawGraph(); updateAvgDisplay();
  });

  sendBtn.addEventListener('click', ()=>{
    if (!samples.length) { alert("No samples to send"); return; }
    fetch('/stats', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({samples})
    }).then(r=>r.json()).then(j=>{
      if (j.ok) alert("Stats saved to server (SQLite)."); else alert("Server error: " + (j.error||"unknown"));
    }).catch(err=>{ alert("Failed to send: " + err); });
  });

  // initial blank prediction
  requestPrediction("", []);

  // expose helper for console
  window.addVocab = function(word){
    fetch('/add_word', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({word})})
      .then(r=>r.json()).then(j=>console.log("add_word:", j));
  };

})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Initializing DB (if needed)...")
    init_db()
    load_vocab_from_db()
    load_bigrams_from_db()
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
