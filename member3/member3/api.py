"""Minimal FastAPI inference service + analyst view for Member 3.

Serves exactly the Member-3 decision path (embedding -> memory retrieval -> score ->
threshold -> multi-stage confirmation -> attribution).  No auth, no database, no
queue: the process is stateless apart from the loaded memory bank, the calibrated
threshold and one :class:`~member3.inference.AlertTracker` per stream.

Run:  ``uvicorn member3.api:app --host 0.0.0.0 --port 8040``   (see README §API)
Optional overrides (explicit only): ``MEMBER3_CONFIG``, ``MEMBER3_RECORDS``, ``MEMBER3_GRAPHS``.
The service has no fallback encoder and no bundled data: without the upstream artefacts it starts
but `/health` reports exactly what is missing.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np

from .contract import (
    ContractError,
    assign_streams,
    load_adapter,
    load_config,
    load_records,
    set_seed,
    validate_records,
)
from .explain import build_explanation
from .inference import AlertTracker, Member3Scorer

_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
 :root{--fg:#14171c;--mut:#5b6472;--bd:#d7dbe2;--bad:#b3261e;--ok:#1b6b3a;--warn:#a86400}
 *{box-sizing:border-box}
 body{margin:0;font:14px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--fg);background:#f6f7f9}
 header{background:#101828;color:#fff;padding:14px 18px;display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
 header h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.2px}
 header .sub{color:#9fb0c8;font-size:12px}
 main{max-width:1080px;margin:16px auto;padding:0 14px;display:grid;grid-template-columns:minmax(280px,1fr) minmax(320px,1.35fr);gap:14px}
 .card{background:#fff;border:1px solid var(--bd);border-radius:10px;padding:14px}
 h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:0 0 10px}
 label{display:block;font-size:12px;color:var(--mut);margin:8px 0 3px}
 input,select,textarea{width:100%;padding:7px 9px;border:1px solid var(--bd);border-radius:7px;font:13px ui-monospace,Menlo,Consolas,monospace;background:#fff}
 textarea{min-height:84px;white-space:pre}
 button{margin-top:12px;width:100%;padding:9px;border:0;border-radius:8px;background:#101828;color:#fff;font-weight:600;cursor:pointer}
 button.sec{background:#eef1f5;color:#101828;margin-top:8px}
 .kpi{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px}
 .kpi div{border:1px solid var(--bd);border-radius:8px;padding:8px 9px}
 .kpi b{display:block;font-size:19px;font-variant-numeric:tabular-nums}
 .kpi span{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em}
 .alert{color:var(--bad);font-weight:700}.benign{color:var(--ok);font-weight:700}.cand{color:var(--warn);font-weight:700}
 table{width:100%;border-collapse:collapse;font-size:12.5px}
 th,td{text-align:left;padding:5px 6px;border-bottom:1px solid var(--bd)}
 th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
 code{background:#f1f3f6;padding:1px 4px;border-radius:4px;font-size:12px}
 .note{font-size:11.5px;color:var(--mut);margin-top:10px}
 .bar{height:8px;border-radius:5px;background:#e8ebf0;overflow:hidden;margin-top:5px}
 .bar i{display:block;height:100%}
 #health{font-size:12px;color:#9fb0c8}
</style></head><body>
<header><h1>__TITLE__</h1><div class="sub"><span id="health">connecting&hellip;</span></div></header>
<main>
 <section class="card">
  <h2>Score a window</h2>
  <label>Mode</label>
  <select id="mode"><option value="sample">sample_id from loaded records</option>
   <option value="embedding">raw embedding vector</option><option value="graph">graph sequence JSON</option></select>
  <label>Sample id</label><input id="sid" placeholder="tes-000042">
  <label>Embedding (comma separated) or graph JSON</label><textarea id="payload"></textarea>
  <label>Stream id (for 3-step confirmation state)</label><input id="stream" value="api-default">
  <button id="go">Score + explain</button>
  <button class="sec" id="next">Next test window from record file</button>
  <div class="note">Read-only with respect to the memory bank unless <code>api.allow_online_memory_updates</code>
   is enabled in <code>config.yaml</code>. Alert state is per <code>stream_id</code> and resets on a benign window.</div>
 </section>
 <section class="card">
  <h2>Decision</h2>
  <div class="kpi">
   <div><b id="score">-</b><span>anomaly score</span></div>
   <div><b id="thr">-</b><span>threshold</span></div>
   <div><b id="conf">-</b><span>confidence</span></div>
  </div>
  <div><b id="state">-</b> <span id="statedetail" style="color:var(--mut)"></span>
   <div class="bar"><i id="bar" style="width:0%;background:#2a6f97"></i></div></div>
  <h2 style="margin-top:14px">Nearest stored normal windows</h2>
  <table id="nn"><tr><td>-</td></tr></table>
  <h2 style="margin-top:14px">Attribution (GATv2 structural attention)</h2>
  <table id="nodes"><tr><td>-</td></tr></table>
  <div class="note" id="edges"></div>
  <div class="note" id="err" style="color:var(--bad)"></div>
 </section>
</main>
<script>
const $=id=>document.getElementById(id);
let idx=0, ids=[];
async function j(u,o){const r=await fetch(u,o);if(!r.ok){throw new Error((await r.json()).detail||r.status)}return r.json()}
function num(v,d=4){return (v===null||v===undefined||Number.isNaN(v))?'-':Number(v).toFixed(d)}
async function boot(){try{const h=await j('/health');$('health').textContent=`mem ${h.memory_size} vec / cap ${h.memory_capacity} - ${h.index_type} - tau ${num(h.threshold,4)} - ${h.data_source}`;ids=h.sample_ids||[];$('sid').value=ids[0]||'tes-000000'}catch(e){$('health').textContent='service offline'}}
function body(){const b={stream_id:$('stream').value};const m=$('mode').value;
 if(m==='sample'){b.sample_id=$('sid').value}
 if(m==='embedding'){b.embedding=$('payload').value.split(',').map(Number)}
 if(m==='graph'){b.graph=JSON.parse($('payload').value)}
 return b}
async function go(){let b;try{b=body()}catch(e){$('err').textContent='payload error: '+e.message;return}
 $('err').textContent='';
 try{const r=await j('/score',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(b)});
  const s=r.alert;$('score').textContent=num(r.anomaly_score);$('thr').textContent=num(r.threshold);$('conf').textContent=num(r.confidence,3);
  const st=$('state');st.textContent=r.confirmed_alert?'CONFIRMED ALERT':(s.alert_state==='candidate'?'CANDIDATE (streak '+s.consecutive_anomalies+'/'+s.required_consecutive+')':'no alert');
  st.className=r.confirmed_alert?'alert':(s.alert_state==='candidate'?'cand':'benign');
  $('statedetail').textContent=` | nearest-normal sim ${num(r.nearest_normal_similarity,4)} | streak ${s.consecutive_anomalies}/${s.required_consecutive} | delay ${r.detection_delay_steps??'-'} windows`;
  const frac=Math.min(1,r.anomaly_score/Math.max(r.threshold*2,1e-9));$('bar').style.width=(100*frac)+'%';
  $('bar').style.background=frac>0.5?'#b3261e':'#2a6f97';
  $('nn').innerHTML=(r.neighbours||[]).map(n=>`<tr><td>${n.memory_index}</td><td>${n.sample_id||'-'}</td><td>${n.dataset_id||'-'}</td><td>cos ${num(n.similarity,4)}</td></tr>`).join('')||'<tr><td>-</td></tr>';
  $('nodes').innerHTML=(r.explanation&&r.explanation.top_nodes||[]).map(n=>`<tr><td>${n.host}</td><td>attn ${num(n.attention,5)}</td><td>share ${num(n.attention_share,4)}</td></tr>`).join('')||'<tr><td>no graph sequence supplied</td></tr>';
  $('edges').innerHTML = r.explanation?('top flows: '+(r.explanation.top_edges||[]).map(e=>`<code>${e.flow_record??e.edge_index}</code>`).join(' ')+' &nbsp; top features: '+(r.explanation.top_features||[]).map(f=>`<code>${f.feature}</code>`).join(' ')) : '';
 }catch(e){$('err').textContent=e.message}}
$('go').onclick=go;
$('next').onclick=async()=>{if(!ids.length)return;const id=ids[(idx++)%ids.length];$('sid').value=id;go()};
boot();
</script></body></html>"""


class ServiceState:
    """Loaded Member-3 runtime: adapter, memory, threshold, per-stream trackers."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.cfg = load_config(config_path)
        set_seed(int(self.cfg.get("seed", 0)))
        # nothing heavy at construction time: the adapter, memory bank and records are resolved
        # lazily on the first request, so `uvicorn member3.api:app` always starts and /health
        # reports the real reason if the artefacts are missing
        self.adapter: Optional[Any] = None
        self.adapter_meta: Dict[str, Any] = {}
        self.results_dir = str(self.cfg.get("output", {}).get("results_dir", "results"))
        self.state_dir = str(self.cfg.get("output", {}).get("state_dir", "artifacts"))
        self.records = None
        self.records_path: Optional[str] = None
        self.graphs: Dict[str, Dict[str, Any]] = {}
        self.scorer: Optional[Member3Scorer] = None
        self.trackers: Dict[str, AlertTracker] = {}
        self.api_cfg = dict(self.cfg.get("api", {}))

    # ---- lazy loading ---------------------------------------------------- #
    def load(self) -> Member3Scorer:
        if self.scorer is not None:
            return self.scorer
        if self.adapter is None:
            self.adapter, self.adapter_meta = load_adapter(self.cfg)
        idx = os.path.join(self.state_dir, "memory_index.bin")
        d = dict(self.cfg.get("data", {}))
        # no implicit fallback: the API serves exactly the files the config (or the env override)
        # points at. Env overrides exist so a demo/self-check run can be served without editing config.
        rp = os.environ.get("MEMBER3_RECORDS") or d.get("records_path")
        gp = os.environ.get("MEMBER3_GRAPHS") or d.get("graph_path")
        self.records_path = str(rp) if rp else None
        if rp and os.path.exists(str(rp)):
            rec = load_records(str(rp))
            validate_records(rec, expected_dim=int(self.adapter_meta["embedding_dim"]))
            if "stream_id" not in rec.columns:
                rec = assign_streams(rec, window_seconds=float(d.get("window_seconds", 30)),
                                     seed=int(self.cfg.get("seed", 0)))
            self.records = rec
        if gp and os.path.exists(str(gp)):
            from .contract import load_graphs
            self.graphs = load_graphs(str(gp))
        if self.records is not None and (self.records["split"] == "test").any():
            self.scorer = Member3Scorer.build(self.cfg, self.adapter, self.adapter_meta, self.records)
        elif os.path.exists(idx):
            from .memory import AdaptiveMemoryBank
            from .inference import ThresholdModel

            with open(os.path.join(self.state_dir, "threshold.json")) as fh:
                blob = json.load(fh)
            thr = ThresholdModel(**{k: v for k, v in blob.items() if k in ThresholdModel.__dataclass_fields__})
            self.scorer = Member3Scorer(self.adapter, self.adapter_meta, AdaptiveMemoryBank.load(idx), thr, self.cfg)
        else:
            raise ContractError(
                "API has nothing to serve: Member-2 records (data.records_path) are absent and no calibrated "
                f"memory bank exists at {idx}. Run "
                "`python run_experiment.py --config {0} --stage calibration` first.".format(self.config_path))
        self.scorer.memory.set_immutable(not bool(self.api_cfg.get("allow_online_memory_updates", False)))
        return self.scorer

    # ---- request handling ------------------------------------------------ #
    def tracker(self, stream_id: str) -> AlertTracker:
        debounce = int(self.cfg.get("alerting", {}).get("debounce_steps", 3))
        return self.trackers.setdefault(str(stream_id), AlertTracker(
            debounce=debounce, window_seconds=float(self.cfg.get("data", {}).get("window_seconds", 30))))

    def resolve(self, payload: Dict[str, Any]):
        """Return (embedding row, graph sequence or None, meta dict)."""
        scorer = self.load()
        dim = int(scorer.memory.stats.dim)
        if payload.get("embedding") is not None:
            z = np.asarray(payload["embedding"], dtype=np.float64).ravel()
            if z.size != dim:
                raise ValueError(f"embedding dimension mismatch: got {z.size}, model expects {dim}")
            return z[None, :], payload.get("graph"), {"source": "client_embedding"}
        if payload.get("graph") is not None:
            z = np.asarray(scorer.adapter.encode(payload["graph"]), dtype=np.float64).reshape(1, -1)
            if z.shape[1] != dim:
                raise ValueError(f"adapter returned dim {z.shape[1]} != memory dim {dim}")
            return z, payload["graph"], {"source": "client_graph"}
        sid = payload.get("sample_id")
        if not sid:
            raise ValueError("POST body needs one of: 'embedding', 'graph', 'sample_id'")
        if self.records is None:
            raise ValueError(
                "no records loaded on the server, so 'sample_id' cannot be resolved: set data.records_path "
                "in the config to the Member-1/2 records export, or post a graph/embedding instead")
        hit = self.records[self.records["sample_id"].astype(str) == str(sid)]
        if len(hit) == 0:
            raise ValueError(f"unknown sample_id '{sid}'")
        row = hit.iloc[0]
        meta = {"source": "record_file", "sample_id": str(row["sample_id"]), "dataset_id": str(row["dataset_id"]),
                "split": str(row["split"]), "timestamp": float(row["timestamp"]),
                "stream_pos": int(row["stream_pos"]), "label": str(row["label"]),
                "attack_family": str(row["attack_family"])}
        # the stored graph sequence travels with the record so /explain works without a re-upload
        return np.asarray(row["embedding"], dtype=np.float64).reshape(1, -1), self.graphs.get(str(sid)), meta


    def score(self, payload: Dict[str, Any], explain: bool = False) -> Dict[str, Any]:
        scorer = self.load()
        z, graph, meta = self.resolve(payload)
        res = scorer.score_embeddings(z)
        score = float(res["anomaly_score"][0])
        conf = float(res["confidence"][0])
        nbr = [{"memory_index": int(j), "similarity": float(res["neighbor_sims"][0][i]),
                **{k: v for k, v in scorer.memory.nearest_metadata(int(j)).items() if k in ("sample_id", "dataset_id", "window_id", "insert_stage")}}
               for i, j in enumerate(res["neighbor_idx"][0])]
        stream_id = payload.get("stream_id", "api-default")
        pos = meta.get("stream_pos")
        alert = self.tracker(stream_id).update(score, float(scorer.threshold.value), pos=pos)
        out = {
            **meta, "embedding_dim": int(z.shape[1]), "anomaly_score": score,
            "nearest_normal_similarity": float(res["similarity"][0]), "threshold": float(scorer.threshold.value),
            "threshold_method": scorer.threshold.method, "confidence": conf, "neighbours": nbr, "alert": alert,
            "confirmed_alert": bool(alert["confirmed_alert"]), "is_anomaly": bool(alert["is_anomaly"]),
            "memory_version": int(scorer.memory.stats.version), "index_type": str(scorer.memory.stats.index_type),
            "checkpoint_version": str(self.adapter_meta.get("checkpoint_version")),
            "dataset_label": str(self.cfg.get("data", {}).get("dataset_label")
                                 or ",".join([str(x) for x in (self.cfg.get("evaluation", {})
                                                                .get("test_datasets") or [])])
                                 or "undeclared"),
            "scoring_definition": "1 - max_j cos(z, m_j)",
            "model_weights_updated": False,
        }
        # optional, strictly gated online memory adaptation
        if bool(self.api_cfg.get("allow_online_memory_updates", False)) and not alert["is_anomaly"]:
            rec = scorer.memory.update_if_benign(
                z[0], confidence=conf, score=score, threshold=float(scorer.threshold.value),
                metadata={"sample_id": meta.get("sample_id"), "label": meta.get("label", "unknown"),
                          "dataset_id": meta.get("dataset_id", "api"), "insert_stage": "api_online"},
                allow_online=True, benign_confidence_max=float(self.cfg.get("adaptation", {}).get("benign_confidence_max", 0.05)))
            out["memory_update"] = rec["action"]
            out["memory_version"] = int(scorer.memory.stats.version)
        else:
            out["memory_update"] = "disabled" if not self.api_cfg.get("allow_online_memory_updates", False) else "rejected_above_threshold"
        if explain:
            if graph is None:
                out["explanation"] = None
                out["explanation_error"] = (
                    "explanation needs the temporal graph sequence of this window (handoff contract §C/§E); "
                    "an embedding alone cannot be attributed")
            else:
                out["explanation"] = build_explanation(
                    scorer.adapter, graph, score, float(scorer.threshold.value), conf,
                    alert_id=f"api:{meta.get('sample_id', 'inline')}",
                    k_nodes=int(self.cfg.get("xai", {}).get("top_nodes", 5)),
                    k_edges=int(self.cfg.get("xai", {}).get("top_edges", 5)),
                    k_features=int(self.cfg.get("xai", {}).get("top_features", 8)),
                    extra={"nearest_normal_similarity": out["nearest_normal_similarity"],
                           "memory_source_dataset": scorer.provenance.get("memory_source")})
        return out

    def health(self) -> Dict[str, Any]:
        scorer = self.load()
        ids = []
        if self.records is not None:
            ids = self.records.loc[self.records["split"] == "test", "sample_id"].astype(str).head(500).tolist()
        e = dict(self.cfg.get("evaluation", {}))
        label = str(dict(self.cfg.get("data", {})).get("dataset_label")
                    or ",".join([str(x) for x in (e.get("test_datasets") or [])]) or "undeclared")
        return {"status": "ok", "adapter_origin": self.adapter_meta.get("adapter_origin"),
                "records_path": str(self.records_path or "") or None,
                "dataset_label": label,
                "embedding_dim": int(scorer.memory.stats.dim), "memory_size": int(scorer.memory.stats.n),
                "memory_capacity": int(scorer.memory.stats.capacity), "index_type": str(scorer.memory.stats.index_type),
                "k": int(scorer.k), "threshold": float(scorer.threshold.value),
                "threshold_method": scorer.threshold.method, "threshold_n_calibration": scorer.threshold.n_calibration,
                "immutable_memory": bool(scorer.memory.stats.immutable), "sample_ids": ids,
                }


def create_app(config_path: Optional[str] = None):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse

    cfg_path = config_path or os.environ.get("MEMBER3_CONFIG", "config.yaml")
    if not os.path.exists(cfg_path):
        raise ContractError(f"config file not found: {cfg_path}")
    state = ServiceState(cfg_path)
    app = FastAPI(title="Member 3 - zero-day IDS inference", version="0.1.0",
                  description="Adaptive memory + FAISS retrieval + confidence-aware threshold + attribution")

    def httpify(fn):
        def wrapped(*a, **kw):
            try:
                return fn(*a, **kw)
            except (ValueError, ContractError) as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            except PermissionError as exc:
                raise HTTPException(status_code=403, detail=str(exc))
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc))
        return wrapped

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        title = str(state.api_cfg.get("dashboard_title", "Member 3 - Analyst View"))
        return _HTML.replace("__TITLE__", title)

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse(httpify(state.health)())   # data_source is inferred, not configured

    @app.post("/score")
    def score(payload: Dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
        return JSONResponse(httpify(state.score)(payload, explain=False))

    @app.post("/explain")
    def explain(payload: Dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
        return JSONResponse(httpify(state.score)(payload, explain=True))

    @app.get("/metrics")
    def metrics() -> JSONResponse:
        p = os.path.join(state.results_dir, "metrics.json")
        if not os.path.exists(p):
            raise HTTPException(status_code=404, detail=f"no results yet: {p} missing (run run_experiment.py)")
        with open(p) as fh:
            return JSONResponse(json.load(fh))

    return app


# `uvicorn member3.api:app` - construction only reads config.yaml; artefacts resolve on first request
app = create_app()
