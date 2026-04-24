import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
import pandas as pd
from PIL import Image
import timm
from ultralytics import YOLO
import io
import os
from datetime import datetime

# ══════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
YOLO_MODEL_PATH  = "best-yolov8s.pt"
SKU_MODEL_PATH   = "best-mobilenetv3large.pth"
MAPPING_PATH     = "label_map.json"
CSV_PATH         = "sku_catalog.csv"
IMG_SIZE         = 224

CLASS_NAMES = [
    'boisson_energetique', 'dessert', 'eau', 'fromage',
    'jus', 'lait', 'soda', 'yaourt'
]

val_transforms = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ══════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════
@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

@st.cache_resource
def load_sku_model(model_path, labels_path):
    with open(labels_path, 'r') as f:
        label_map = json.load(f)
    idx_to_class = {int(k): v for k, v in label_map.items()}
    nc = len(idx_to_class)

    model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=nc)
    if hasattr(model, 'classifier'):
        model.classifier = nn.Linear(model.classifier.in_features, nc)

    state_dict = torch.load(model_path, map_location='cpu')
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model, idx_to_class, nc

@st.cache_data
def load_sku_catalog(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df.set_index("sku_id").to_dict("index"), df
    return {}, pd.DataFrame()

# ══════════════════════════════════════════════════════
# IMAGE HELPERS
# ══════════════════════════════════════════════════════
def prepare_crop(crop_np, target=224, upscale=2.5):
    h, w = crop_np.shape[:2]
    if h < 150 or w < 150:
        nh, nw = int(h * upscale), int(w * upscale)
        crop_np = cv2.resize(crop_np, (nw, nh), interpolation=cv2.INTER_CUBIC)
        h, w = nh, nw
    ratio = w / h
    if ratio > 1:
        nw2, nh2 = target, int(target / ratio)
    else:
        nh2, nw2 = target, int(target * ratio)
    resized = cv2.resize(crop_np, (nw2, nh2), interpolation=cv2.INTER_LANCZOS4)
    sq = np.full((target, target, 3), 128, dtype=np.uint8)
    yo, xo = (target - nh2) // 2, (target - nw2) // 2
    sq[yo:yo+nh2, xo:xo+nw2] = resized
    return sq

def predict_sku(model, crop_np, idx_to_class, upscale=2.5):
    prepared = prepare_crop(crop_np, IMG_SIZE, upscale)
    img_t = val_transforms(Image.fromarray(prepared)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(img_t), dim=1)[0]
        topk  = probs.topk(min(5, len(idx_to_class)))
    skus  = [idx_to_class[i.item()] for i in topk.indices]
    confs = [v.item() for v in topk.values]
    return skus, confs

def np_to_bytes(img_np):
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format='JPEG', quality=88)
    return buf.getvalue()

# ══════════════════════════════════════════════════════
# PIPELINE — STAGE 1 + 2 (YOLO + SKU)
# ══════════════════════════════════════════════════════
def run_pipeline(image_bytes, yolo_model, sku_model, idx_to_class, sku_catalog,
                 conf_thr=0.45, upscale=2.5, disp_thr=0.25):
    arr     = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h, w = img_rgb.shape[:2]
    if max(h, w) < 640:
        sc = 640 / max(h, w)
        img_rgb = cv2.resize(img_rgb, (int(w*sc), int(h*sc)), interpolation=cv2.INTER_CUBIC)

    yolo_res = yolo_model(img_rgb, conf=conf_thr, verbose=False)
    annotated = img_rgb.copy()
    detections, crops_data = [], []

    for r in yolo_res:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_det = float(box.conf[0])
            cls_id   = int(box.cls[0])
            famille  = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"

            crop = img_rgb[y1:y2, x1:x2]
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                skus, confs = ["inconnu"], [0.0]
            else:
                skus, confs = predict_sku(sku_model, crop, idx_to_class, upscale)

            info = sku_catalog.get(skus[0], {})
            nom  = info.get('product_name', skus[0])
            crop_bytes = np_to_bytes(prepare_crop(crop, IMG_SIZE, upscale))

            # Colour by confidence
            if   confs[0] > 0.7: col = (16, 185, 129)
            elif confs[0] > 0.4: col = (245, 158, 11)
            elif confs[0] > 0.2: col = (249, 115, 22)
            else:                 col = (239, 68, 68)

            label = f"{nom} ({confs[0]:.1%})" if confs[0] > disp_thr else famille
            font  = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 3)
            cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+8, y1), (10, 10, 20), -1)
            cv2.putText(annotated, label, (x1+4, y1-6), font, 0.55, (255, 255, 255), 2)

            detections.append({
                "bbox": [x1, y1, x2, y2], "famille": famille, "sku": skus[0],
                "nom_produit": nom,
                "confiance_detection": round(conf_det, 3),
                "confiance_sku":       round(confs[0], 3),
                "top5_predictions":    list(zip(skus, confs)),
                "brand":    info.get('brand',    'N/A'),
                "capacity": info.get('capacity', 'N/A'),
                "emballage":info.get('emballage','N/A'),
                "saveur":   info.get('saveur',   'N/A'),
                "category": info.get('category', 'N/A'),
            })
            crops_data.append({
                "crop_bytes":  crop_bytes,
                "stage1":      f"{famille} ({conf_det:.2f})",
                "stage2_sku":  skus[0],
                "stage2_nom":  nom,
                "stage2_conf": confs[0],
                "top5":        list(zip(skus[:5], confs[:5])),
                "brand":    info.get('brand',    'N/A'),
                "capacity": info.get('capacity', 'N/A'),
                "emballage":info.get('emballage','N/A'),
                "saveur":   info.get('saveur',   'N/A'),
            })

    return annotated, detections, crops_data

# ══════════════════════════════════════════════════════
# PIPELINE — STAGE 2 ONLY (Inventaire)
# ══════════════════════════════════════════════════════
def run_inventory_pipeline(image_bytes, sku_model, idx_to_class, sku_catalog, upscale=2.5):
    arr     = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    skus, confs = predict_sku(sku_model, img_rgb, idx_to_class, upscale)
    info = sku_catalog.get(skus[0], {})
    nom  = info.get('product_name', skus[0])
    crop_bytes = np_to_bytes(prepare_crop(img_rgb, IMG_SIZE, upscale))

    return {
        "sku":       skus[0],
        "nom":       nom,
        "conf":      round(confs[0], 3),
        "top5":      list(zip(skus[:5], confs[:5])),
        "brand":     info.get('brand',    'N/A'),
        "capacity":  info.get('capacity', 'N/A'),
        "emballage": info.get('emballage','N/A'),
        "saveur":    info.get('saveur',   'N/A'),
        "crop_bytes": crop_bytes,
    }

# ══════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════
def conf_color(c):
    if c > 0.7:  return "#10b981"
    if c > 0.4:  return "#f59e0b"
    if c > 0.2:  return "#f97316"
    return "#ef4444"

def conf_icon(c):
    if c > 0.7:  return "🟢"
    if c > 0.4:  return "🟡"
    if c > 0.2:  return "🟠"
    return "🔴"

def render_results(crops_data, detections):
    """Shared result block: metrics + detection cards + full table."""
    if not crops_data:
        st.info("Aucune détection trouvée dans cette image.")
        return

    # ── Metrics ────────────────────────────────────────
    hi = sum(1 for c in crops_data if c['stage2_conf'] >  0.7)
    me = sum(1 for c in crops_data if 0.4 < c['stage2_conf'] <= 0.7)
    lo = sum(1 for c in crops_data if 0.2 < c['stage2_conf'] <= 0.4)
    vl = sum(1 for c in crops_data if c['stage2_conf'] <= 0.2)
    st.markdown("---")
    m_cols = st.columns(5)
    for col, val, lbl, cls in zip(m_cols,
            [len(crops_data), hi, me, lo, vl],
            ["Total", ">70%", "40–70%", "20–40%", "<20%"],
            ["total", "high", "med", "low", "vlow"]):
        with col:
            st.markdown(
                f'<div class="metric-pill {cls}">'
                f'<div class="metric-val">{val}</div>'
                f'<div class="metric-lbl">{lbl}</div></div>',
                unsafe_allow_html=True)

    # ── Detection cards ────────────────────────────────
    st.markdown("---")
    st.markdown('<span class="section-title">🔍 Détail des détections</span>', unsafe_allow_html=True)
    sorted_crops = sorted(enumerate(crops_data), key=lambda x: x[1]['stage2_conf'], reverse=True)

    for rank, (_, cd) in enumerate(sorted_crops):
        conf = cd['stage2_conf']
        with st.expander(f"{conf_icon(conf)} #{rank+1} — {cd['stage2_nom']} · {conf:.1%}"):
            ca, cb = st.columns([1, 2])
            with ca:
                st.image(cd["crop_bytes"], use_container_width=True)
            with cb:
                fields = [
                    ("Stage 1",  cd['stage1']),
                    ("SKU",      cd['stage2_sku']),
                    ("Produit",  cd['stage2_nom']),
                    ("Marque",   cd['brand']),
                    ("Capacité", cd['capacity']),
                    ("Emballage",cd['emballage']),
                    ("Saveur",   cd['saveur']),
                ]
                rows_html = "".join(
                    f'<div class="det-row">'
                    f'<span class="det-key">{k}</span>'
                    f'<span class="det-val{"  sku-val" if k == "SKU" else ""}">{v}</span>'
                    f'</div>'
                    for k, v in fields
                )
                st.markdown(f'<div class="det-card-inner">{rows_html}</div>', unsafe_allow_html=True)
                col_hex = conf_color(conf)
                st.markdown(f'<div class="conf-label" style="color:{col_hex}">Confiance : {conf:.1%}</div>', unsafe_allow_html=True)
                st.progress(conf)
                if cd['top5']:
                    items_html = "".join(
                        f'<div class="top5-row">'
                        f'<span class="top5-rank">{j+1}.</span>'
                        f'<span class="top5-sku">{s}</span>'
                        f'<span class="top5-conf">{c:.1%}</span>'
                        f'</div>'
                        for j, (s, c) in enumerate(cd['top5'][:5])
                    )
                    st.markdown(f'<div class="top5-title">🏆 Top-5 prédictions</div>{items_html}', unsafe_allow_html=True)

    # ── Full report table ──────────────────────────────
    st.markdown("---")
    st.markdown('<span class="section-title">📋 Rapport complet</span>', unsafe_allow_html=True)
    if detections:
        df = pd.DataFrame(detections)
        cols_order = ["nom_produit","brand","capacity","emballage","saveur",
                      "famille","confiance_sku","confiance_detection","sku"]
        df_disp = df[[c for c in cols_order if c in df.columns]].rename(columns={
            "nom_produit":"Produit","brand":"Marque","capacity":"Capacité",
            "emballage":"Emballage","saveur":"Saveur","famille":"Famille",
            "confiance_sku":"Conf. SKU","confiance_detection":"Conf. Dét.","sku":"SKU ID"
        })
        if "Conf. SKU" in df_disp.columns:
            df_disp["Conf. SKU"] = df_disp["Conf. SKU"].apply(lambda x: f"{x:.1%}")
        if "Conf. Dét." in df_disp.columns:
            df_disp["Conf. Dét."] = df_disp["Conf. Dét."].apply(lambda x: f"{x:.1%}")
        st.dataframe(df_disp, use_container_width=True, height=350)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        ec1, ec2 = st.columns(2)
        with ec1:
            st.download_button("📥 Télécharger CSV",
                data=df.to_csv(index=False),
                file_name=f"detections_{ts}.csv", mime="text/csv",
                use_container_width=True)
        with ec2:
            st.download_button("📥 Télécharger JSON",
                data=json.dumps(detections, indent=2, ensure_ascii=False, default=str),
                file_name=f"detections_{ts}.json", mime="application/json",
                use_container_width=True)
    else:
        st.info("Aucune détection dans cette image.")

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="SKU Recognition Pipeline",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════
# GLOBAL CSS  (matching JS app design system)
# ══════════════════════════════════════════════════════
# Font import (separate call — avoids HTML parser confusion)
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500'
    '&amp;family=Syne:wght@400;600;700;800&amp;display=swap" rel="stylesheet">',
    unsafe_allow_html=True
)

st.markdown("""
<style>
/* ── Design tokens ── */
:root {
  --accent:#6c63ff; --accent2:#8b83ff; --accent-light:rgba(108,99,255,.08);
  --green:#10b981;  --yellow:#f59e0b;  --orange:#f97316; --red:#ef4444;
  --bg:#f4f5f8;     --surface:#fff;    --surface2:#f0f2f7;
  --border:#e2e6ef; --border2:#d0d5e3;
  --text:#1a1d2e;   --text-muted:#6b7280; --text-dim:#9ca3af;
  --font-d:'Syne',sans-serif; --font-m:'DM Mono',monospace;
  --r:12px; --shadow:0 1px 3px rgba(0,0,0,.07),0 4px 16px rgba(0,0,0,.05);
}

/* ── Global ── */
html, body, .stApp, .main, section { font-family: var(--font-d) !important; }
.stMarkdown p, .stMarkdown span, .stMarkdown div { font-family: var(--font-d) !important; }
code, pre { font-family: var(--font-m) !important; }
#MainMenu,footer,.stDeployButton{visibility:hidden;}
.stApp{background:var(--bg);}

/* ── Sidebar ── */
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3{font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--text-muted);}

/* ── Logo ── */
.logo-block{display:flex;align-items:center;gap:10px;padding:0 0 16px;}
.logo-icon{width:42px;height:42px;background:linear-gradient(135deg,#6c63ff,#8b83ff);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;box-shadow:0 4px 12px rgba(108,99,255,.3);flex-shrink:0;}
.logo-text{font-size:15px;font-weight:800;color:var(--text);line-height:1.2;}
.logo-sub{font-size:10px;color:var(--text-muted);letter-spacing:.07em;text-transform:uppercase;}

/* ── Status ── */
.status-pill{display:flex;align-items:center;gap:8px;border-radius:8px;padding:8px 12px;font-size:12px;font-family:var(--font-m);margin-bottom:4px;}
.status-ready{background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.25);color:var(--green);}
.status-error{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);color:var(--red);}
.status-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.dot-green{background:var(--green);}
.dot-red{background:var(--red);}

/* ── Badges ── */
.badge{display:inline-block;padding:2px 8px;border-radius:5px;font-size:10px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;font-family:var(--font-m);}
.badge-yolo{background:rgba(245,158,11,.12);color:#f59e0b;}
.badge-sku{background:rgba(108,99,255,.12);color:#6c63ff;}
.badge-inv{background:rgba(16,185,129,.12);color:#10b981;}

/* ── Page header ── */
.page-header h1{font-size:26px;font-weight:800;letter-spacing:-.02em;background:linear-gradient(90deg,var(--text) 0%,var(--accent) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:4px;}
.page-header p{font-size:13px;color:var(--text-muted);}

/* ── Section title ── */
.section-title{font-size:15px;font-weight:800;color:var(--text);margin:14px 0 10px;display:block;}

/* ── Metric pills ── */
.metric-pill{border-radius:var(--r);padding:16px 12px;text-align:center;box-shadow:var(--shadow);border:1px solid var(--border);background:var(--surface);margin-bottom:4px;}
.metric-val{font-size:28px;font-weight:800;font-family:var(--font-m)!important;}
.metric-lbl{font-size:10px;color:var(--text-muted);margin-top:3px;letter-spacing:.07em;text-transform:uppercase;}
.metric-pill.total .metric-val{color:var(--text);}
.metric-pill.high  .metric-val{color:var(--green);}
.metric-pill.med   .metric-val{color:var(--yellow);}
.metric-pill.low   .metric-val{color:var(--orange);}
.metric-pill.vlow  .metric-val{color:var(--red);}

/* ── Image card header ── */
.img-card-hdr{background:var(--surface2);border:1px solid var(--border);border-radius:var(--r) var(--r) 0 0;padding:10px 16px;font-size:11px;font-weight:700;color:var(--text-muted);letter-spacing:.06em;text-transform:uppercase;}

/* ── Detection card inner ── */
.det-card-inner{display:flex;flex-direction:column;gap:4px;margin-bottom:10px;}
.det-row{display:flex;gap:10px;align-items:baseline;font-size:12px;border-bottom:1px solid var(--border);padding:4px 0;}
.det-key{color:var(--text-muted);min-width:80px;font-family:var(--font-m)!important;font-size:11px;flex-shrink:0;}
.det-val{color:var(--text);font-weight:600;word-break:break-word;}
.sku-val{color:var(--accent);font-family:var(--font-m)!important;}
.conf-label{font-size:12px;font-weight:700;margin-bottom:5px;font-family:var(--font-m)!important;}

/* ── Top-5 ── */
.top5-title{font-size:11px;font-weight:700;color:var(--text-muted);margin:10px 0 6px;letter-spacing:.06em;text-transform:uppercase;}
.top5-row{display:flex;gap:8px;align-items:center;font-size:11px;padding:3px 0;border-bottom:1px solid var(--border);font-family:var(--font-m)!important;}
.top5-rank{color:var(--text-dim);width:18px;flex-shrink:0;}
.top5-sku{flex:1;color:var(--accent);word-break:break-all;}
.top5-conf{color:var(--text-muted);}

/* ── Upload hint ── */
.upload-hint{background:var(--surface);border:2px dashed var(--border2);border-radius:var(--r);padding:48px 32px;text-align:center;margin-bottom:20px;}
.upload-hint .u-icon{font-size:42px;margin-bottom:12px;}
.upload-hint h3{font-size:17px;font-weight:800;color:var(--text);margin-bottom:6px;}
.upload-hint p{font-size:13px;color:var(--text-muted);}

/* ── Session nav buttons ── */
.stButton>button{border-radius:9px!important;font-weight:600!important;font-size:12px!important;transition:all .18s!important;}

/* ── Inventory pending card ── */
.inv-pending-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:12px;box-shadow:var(--shadow);margin-top:4px;}
.inv-card-name{font-size:12px;font-weight:700;margin-bottom:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.inv-card-sku{font-size:10px;color:var(--accent);font-family:var(--font-m)!important;margin-bottom:6px;word-break:break-all;}
.inv-card-conf{font-size:15px;font-weight:800;font-family:var(--font-m)!important;}

/* ── Empty state ── */
.empty-state{text-align:center;padding:40px;color:var(--text-muted);background:var(--surface);border:2px dashed var(--border2);border-radius:var(--r);}

/* ── Tab overrides ── */
.stTabs [data-baseweb="tab-list"]{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:4px;gap:3px;}
.stTabs [data-baseweb="tab"]{border-radius:10px!important;font-weight:600!important;font-size:13px!important;color:var(--text-muted)!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;color:white!important;}

/* ── Pipeline info box ── */
.pipeline-box{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:12px;font-size:11px;font-family:var(--font-m);}
.pipeline-stage{display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid var(--border);}
.pipeline-stage:last-child{border-bottom:none;}

/* ── Models file info ── */
.models-info{font-family:var(--font-m)!important;font-size:10px;color:var(--text-muted);line-height:2.2;}
.models-info strong{color:var(--accent);}

/* ── Divider ── */
hr{border:none;border-top:1px solid var(--border);margin:20px 0;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════
_defaults = {
    'sessions':           [],   # Upload multi-session
    'active_sess_id':     None,
    'proc_hashes':        set(),
    'inv_pending':        [],   # Inventory pending queue
    'inv_validated':      [],   # Inventory validated table
    'inv_next_id':        1,
    'inv_upload_key':     0,    # Key reset trick for file_uploader
    'inv_cam_key':        0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="logo-block">
        <div class="logo-icon">🏪</div>
        <div>
            <div class="logo-text">SKU Pipeline</div>
            <div class="logo-sub">YOLO · MobileNetV3</div>
        </div>
    </div>""", unsafe_allow_html=True)

    _status_slot = st.empty()
    st.markdown("---")

    st.markdown("**⚙️ Seuils**")
    conf_threshold    = st.slider("Seuil YOLO",      0.10, 0.95, 0.45, 0.05)
    display_threshold = st.slider("Seuil affichage", 0.10, 0.95, 0.25, 0.05)
    upscale_factor    = st.slider("Agrandissement",  1.0,  4.0,  2.5,  0.5)

    st.markdown("---")
    st.markdown("**📊 Pipeline**")
    st.markdown("""
    <div class="pipeline-box">
        <div class="pipeline-stage"><span class="badge badge-yolo">Stage 1</span> YOLO — famille</div>
        <div class="pipeline-stage"><span class="badge badge-sku">Stage 2</span> MobileNetV3 — SKU</div>
        <div class="pipeline-stage"><span class="badge badge-inv">Inventaire</span> Stage 2 uniquement</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📁 Modèles**")
    st.markdown(f"""
    <div class="models-info">
        <strong>YOLO:</strong> {YOLO_MODEL_PATH}<br>
        <strong>SKU:</strong> {SKU_MODEL_PATH}<br>
        <strong>Labels:</strong> {MAPPING_PATH}<br>
        <strong>Catalog:</strong> {CSV_PATH}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════
models_ok  = False
num_classes = 0
try:
    yolo_model                       = load_yolo_model(YOLO_MODEL_PATH)
    sku_model, idx_to_class, num_classes = load_sku_model(SKU_MODEL_PATH, MAPPING_PATH)
    sku_catalog, _                   = load_sku_catalog(CSV_PATH)
    models_ok = True
    _status_slot.markdown(f"""
    <div class="status-pill status-ready">
        <div class="status-dot dot-green"></div> Prêt · {num_classes} SKU
    </div>""", unsafe_allow_html=True)
except Exception as exc:
    _status_slot.markdown("""
    <div class="status-pill status-error">
        <div class="status-dot dot-red"></div> Erreur chargement
    </div>""", unsafe_allow_html=True)
    st.error(f"❌ {exc}")

# ══════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="page-header">
    <h1>SKU Recognition Pipeline</h1>
    <p>YOLO + MobileNetV3 · Reconnaissance de produits en rayon · Inventaire intelligent</p>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════
tab_upload, tab_camera, tab_inventory = st.tabs([
    "📸 Upload d'image", "📷 Prendre une photo", "📦 Inventaire"
])

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 1 — UPLOAD  (multi-session)                    ║
# ╚══════════════════════════════════════════════════════╝
with tab_upload:
    uploaded = st.file_uploader(
        "Déposez une image de rayon",
        type=["jpg","jpeg","png"],
        key="main_uploader",
        label_visibility="collapsed"
    )

    if uploaded and models_ok:
        raw_bytes = uploaded.getvalue()
        file_hash = hash(raw_bytes[:2048] + uploaded.name.encode())

        if file_hash not in st.session_state.proc_hashes:
            with st.spinner("🔍 Analyse YOLO + SKU en cours…"):
                img_out, detections, crops_data = run_pipeline(
                    raw_bytes, yolo_model, sku_model, idx_to_class, sku_catalog,
                    conf_threshold, upscale_factor, display_threshold
                )
            new_sess = {
                'id':         len(st.session_state.sessions) + 1,
                'name':       uploaded.name,
                'raw_bytes':  raw_bytes,
                'img_out':    img_out,
                'detections': detections,
                'crops_data': crops_data,
                'ts':         datetime.now().strftime('%H:%M:%S'),
            }
            st.session_state.sessions.append(new_sess)
            st.session_state.active_sess_id = new_sess['id']
            st.session_state.proc_hashes.add(file_hash)

    # ── Session navigation ─────────────────────────────
    if st.session_state.sessions:
        st.markdown("---")
        n = len(st.session_state.sessions)
        # Build nav: sessions + remove button
        nav_cols = st.columns(min(n, 6) + 1)
        for i, sess in enumerate(st.session_state.sessions[:6]):
            with nav_cols[i]:
                is_active = (sess['id'] == st.session_state.active_sess_id)
                label = f"{'▶ ' if is_active else ''}📷 {sess['name'][:14]}"
                if st.button(label, key=f"sess_{sess['id']}", use_container_width=True):
                    st.session_state.active_sess_id = sess['id']
                    st.rerun()
        with nav_cols[-1]:
            if st.button("🗑 Retirer", use_container_width=True, key="remove_sess"):
                st.session_state.sessions = [
                    s for s in st.session_state.sessions
                    if s['id'] != st.session_state.active_sess_id
                ]
                if st.session_state.sessions:
                    st.session_state.active_sess_id = st.session_state.sessions[-1]['id']
                else:
                    st.session_state.active_sess_id = None
                st.rerun()

        # ── Show active session results ────────────────
        active = next(
            (s for s in st.session_state.sessions if s['id'] == st.session_state.active_sess_id),
            None
        )
        if active:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="img-card-hdr">📸 Image originale</div>', unsafe_allow_html=True)
                st.image(active['raw_bytes'], use_container_width=True)
            with c2:
                st.markdown('<div class="img-card-hdr">🎯 Résultat annoté</div>', unsafe_allow_html=True)
                st.image(active['img_out'], use_container_width=True)

            render_results(active['crops_data'], active['detections'])

    elif not uploaded:
        st.markdown("""
        <div class="upload-hint">
            <div class="u-icon">📦</div>
            <h3>Déposez une image de rayon</h3>
            <p>JPG, JPEG ou PNG · Glissez-déposez ou cliquez sur "Browse files"</p>
        </div>""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 2 — CAMERA                                     ║
# ╚══════════════════════════════════════════════════════╝
with tab_camera:
    st.info("📱 Utilisez l'appareil photo de votre téléphone ou la webcam de votre ordinateur")
    cam_img = st.camera_input("Cadrez le rayon et prenez la photo", label_visibility="collapsed")

    if cam_img and models_ok:
        raw_bytes = cam_img.getvalue()
        with st.spinner("🔍 Analyse en cours…"):
            img_out, detections, crops_data = run_pipeline(
                raw_bytes, yolo_model, sku_model, idx_to_class, sku_catalog,
                conf_threshold, upscale_factor, display_threshold
            )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="img-card-hdr">📸 Photo capturée</div>', unsafe_allow_html=True)
            st.image(cam_img, use_container_width=True)
        with c2:
            st.markdown('<div class="img-card-hdr">🎯 Résultat annoté</div>', unsafe_allow_html=True)
            st.image(img_out, use_container_width=True)
        render_results(crops_data, detections)

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 3 — INVENTAIRE  (Stage 2 uniquement)           ║
# ╚══════════════════════════════════════════════════════╝
with tab_inventory:
    st.markdown(
        '<p style="font-size:13px;color:var(--text-muted);margin-bottom:16px;">'
        'Scannez des produits individuels — classification directe MobileNetV3 (Stage 2 uniquement).'
        '</p>', unsafe_allow_html=True
    )

    inv_sub_up, inv_sub_cam = st.tabs(["📁 Upload", "📷 Caméra"])

    # ── Sub-tab : Upload multiple files ───────────────
    with inv_sub_up:
        inv_files = st.file_uploader(
            "Scannez un ou plusieurs produits",
            type=["jpg","jpeg","png"],
            accept_multiple_files=True,
            key=f"inv_up_{st.session_state.inv_upload_key}",
            label_visibility="collapsed"
        )
        if inv_files and models_ok:
            prog = st.progress(0, "Traitement…")
            for i, f in enumerate(inv_files):
                raw = f.read()
                result = run_inventory_pipeline(
                    raw, sku_model, idx_to_class, sku_catalog, upscale_factor
                )
                result['id']       = st.session_state.inv_next_id
                result['filename'] = f.name
                st.session_state.inv_next_id += 1
                st.session_state.inv_pending.append(result)
                prog.progress((i+1) / len(inv_files), f"Traitement {i+1}/{len(inv_files)}…")
            prog.empty()
            st.session_state.inv_upload_key += 1   # reset the uploader widget
            st.rerun()

    # ── Sub-tab : Camera capture ───────────────────────
    with inv_sub_cam:
        st.info("📷 Capturez des produits individuellement pour les ajouter à l'inventaire.")
        inv_cam = st.camera_input(
            "Capturer un produit",
            key=f"inv_cam_{st.session_state.inv_cam_key}",
            label_visibility="collapsed"
        )
        if inv_cam and models_ok:
            with st.spinner("Classification SKU…"):
                result = run_inventory_pipeline(
                    inv_cam.getvalue(), sku_model, idx_to_class, sku_catalog, upscale_factor
                )
                result['id']       = st.session_state.inv_next_id
                result['filename'] = f"Photo_{datetime.now().strftime('%H%M%S')}"
                st.session_state.inv_next_id += 1
                st.session_state.inv_pending.append(result)
            st.session_state.inv_cam_key += 1
            st.rerun()

    # ══════════════════════════════════════════════════
    # PENDING ITEMS GRID
    # ══════════════════════════════════════════════════
    pending = st.session_state.inv_pending
    st.markdown("---")

    if pending:
        # Bulk actions row
        bh1, bh2, bh3 = st.columns([4, 1, 1])
        with bh1:
            n_p = len(pending)
            st.markdown(
                f'<span class="section-title">🕐 En attente de validation ({n_p} produit{"s" if n_p>1 else ""})</span>',
                unsafe_allow_html=True)
        with bh2:
            if st.button("➕ Ajouter tout", use_container_width=True, key="add_all"):
                st.session_state.inv_validated.extend(st.session_state.inv_pending)
                st.session_state.inv_pending = []
                st.rerun()
        with bh3:
            if st.button("🗑 Tout supprimer", use_container_width=True, key="del_all"):
                st.session_state.inv_pending = []
                st.rerun()

        # Grid: 3 columns
        to_add, to_del = [], []
        N_COLS = 3
        rows = [pending[i:i+N_COLS] for i in range(0, len(pending), N_COLS)]

        for row_items in rows:
            grid = st.columns(N_COLS)
            for col_el, item in zip(grid, row_items):
                with col_el:
                    col_hex = conf_color(item['conf'])
                    st.image(item['crop_bytes'], use_container_width=True)
                    st.markdown(f"""
                    <div class="inv-pending-card">
                        <div class="inv-card-name" title="{item['nom']}">{item['nom']}</div>
                        <div class="inv-card-sku">{item['sku']}</div>
                        <div class="inv-card-conf" style="color:{col_hex};">{item['conf']:.1%}</div>
                    </div>""", unsafe_allow_html=True)
                    ba, bb = st.columns(2)
                    with ba:
                        if st.button("➕ Ajouter", key=f"iadd_{item['id']}", use_container_width=True):
                            to_add.append(item['id'])
                    with bb:
                        if st.button("🗑 Suppr.", key=f"idel_{item['id']}", use_container_width=True):
                            to_del.append(item['id'])

        # Mutate state after all buttons
        if to_add:
            for iid in to_add:
                obj = next((x for x in st.session_state.inv_pending if x['id'] == iid), None)
                if obj:
                    st.session_state.inv_validated.append(obj)
                    st.session_state.inv_pending = [
                        x for x in st.session_state.inv_pending if x['id'] != iid
                    ]
            st.rerun()
        if to_del:
            st.session_state.inv_pending = [
                x for x in st.session_state.inv_pending if x['id'] not in to_del
            ]
            st.rerun()

    else:
        st.markdown("""
        <div class="empty-state">
            📷 Aucune image en attente — uploadez des photos produit ou utilisez la caméra ci-dessus.
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    # VALIDATED INVENTORY TABLE
    # ══════════════════════════════════════════════════
    st.markdown("---")
    validated = st.session_state.inv_validated

    vh1, vh2 = st.columns([4, 1])
    with vh1:
        n_v = len(validated)
        st.markdown(
            f'<span class="section-title">📋 Inventaire validé ({n_v} produit{"s" if n_v!=1 else ""})</span>',
            unsafe_allow_html=True)
    with vh2:
        if validated and st.button("🗑 Tout effacer", use_container_width=True, key="clear_inv"):
            st.session_state.inv_validated = []
            st.rerun()

    if validated:
        # Dataframe view
        inv_rows = [{
            "Produit":   it['nom'],
            "SKU":       it['sku'],
            "Marque":    it['brand'],
            "Capacité":  it['capacity'],
            "Emballage": it['emballage'],
            "Saveur":    it['saveur'],
            "Confiance": f"{it['conf']:.1%}",
        } for it in validated]
        st.dataframe(pd.DataFrame(inv_rows), use_container_width=True, height=360)

        # Per-item delete rows
        st.markdown("**Supprimer un produit spécifique :**")
        del_ids = []
        for it in validated:
            rc1, rc2, rc3 = st.columns([3, 1, 1])
            with rc1:
                st.markdown(
                    f'<span style="font-weight:700;">{it["nom"]}</span> · '
                    f'<span style="color:var(--accent);font-family:var(--font-m);font-size:11px;">{it["sku"]}</span>',
                    unsafe_allow_html=True)
            with rc2:
                st.markdown(
                    f'<span style="color:{conf_color(it["conf"])};font-weight:700;">{it["conf"]:.1%}</span>',
                    unsafe_allow_html=True)
            with rc3:
                if st.button("🗑", key=f"dv_{it['id']}", use_container_width=True):
                    del_ids.append(it['id'])
        if del_ids:
            st.session_state.inv_validated = [
                x for x in st.session_state.inv_validated if x['id'] not in del_ids
            ]
            st.rerun()

        # Export
        st.markdown("---")
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        ec1, ec2 = st.columns(2)
        with ec1:
            st.download_button(
                "📥 Exporter CSV",
                data=pd.DataFrame(inv_rows).to_csv(index=False),
                file_name=f"inventaire_{ts}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with ec2:
            inv_json = [{k: v for k, v in it.items() if k != 'crop_bytes'} for it in validated]
            st.download_button(
                "📥 Exporter JSON",
                data=json.dumps(inv_json, indent=2, ensure_ascii=False, default=str),
                file_name=f"inventaire_{ts}.json",
                mime="application/json",
                use_container_width=True
            )
    else:
        st.markdown("""
        <div class="empty-state">
            Aucun produit dans l'inventaire. Validez des produits depuis la section ci-dessus.
        </div>""", unsafe_allow_html=True)