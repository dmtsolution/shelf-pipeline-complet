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
import io, os, base64
from datetime import datetime

# ══════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
YOLO_MODEL_PATH = "best-yolov8s.pt"
SKU_MODEL_PATH  = "best-mobilenetv3large.pth"
MAPPING_PATH    = "label_map.json"
CSV_PATH        = "sku_catalog.csv"
IMG_SIZE        = 224

CLASS_NAMES = ['boisson_energetique','dessert','eau','fromage',
               'jus','lait','soda','yaourt']

val_transforms = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ══════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════
@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

@st.cache_resource
def load_sku_model(model_path, labels_path):
    with open(labels_path) as f:
        label_map = json.load(f)
    idx_to_class = {int(k): v for k, v in label_map.items()}
    nc = len(idx_to_class)
    model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=nc)
    if hasattr(model, 'classifier'):
        model.classifier = nn.Linear(model.classifier.in_features, nc)
    sd = torch.load(model_path, map_location='cpu')
    if any(k.startswith('module.') for k in sd):
        sd = {k.replace('module.',''):v for k,v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(DEVICE).eval()
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
def prepare_crop(img_np, target=224, upscale=2.5):
    h, w = img_np.shape[:2]
    if h < 150 or w < 150:
        nh, nw = int(h*upscale), int(w*upscale)
        img_np = cv2.resize(img_np, (nw,nh), interpolation=cv2.INTER_CUBIC)
        h, w = nh, nw
    ratio = w/h
    nw2,nh2 = (target,int(target/ratio)) if ratio>1 else (int(target*ratio),target)
    r = cv2.resize(img_np, (nw2,nh2), interpolation=cv2.INTER_LANCZOS4)
    sq = np.full((target,target,3), 128, dtype=np.uint8)
    yo,xo = (target-nh2)//2, (target-nw2)//2
    sq[yo:yo+nh2, xo:xo+nw2] = r
    return sq

def predict_sku(model, img_np, idx_to_class, upscale=2.5):
    prep = prepare_crop(img_np, IMG_SIZE, upscale)
    t = val_transforms(Image.fromarray(prep)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0]
        topk  = probs.topk(min(5, len(idx_to_class)))
    skus  = [idx_to_class[i.item()] for i in topk.indices]
    confs = [v.item()               for v in topk.values]
    return skus, confs

def np_to_bytes(img_np):
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format='JPEG', quality=88)
    return buf.getvalue()

def b64(data):
    return base64.b64encode(data).decode()

# ══════════════════════════════════════════════════════
# PIPELINES
# ══════════════════════════════════════════════════════
def run_pipeline(image_bytes, yolo_model, sku_model, idx_to_class, sku_catalog,
                 conf_thr=0.45, upscale=2.5, disp_thr=0.25):
    arr     = np.frombuffer(image_bytes, np.uint8)
    img_rgb = cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h,w = img_rgb.shape[:2]
    if max(h,w) < 640:
        sc = 640/max(h,w)
        img_rgb = cv2.resize(img_rgb, (int(w*sc),int(h*sc)), interpolation=cv2.INTER_CUBIC)
    annotated = img_rgb.copy()
    detections, crops_data = [], []
    for r in yolo_model(img_rgb, conf=conf_thr, verbose=False):
        if r.boxes is None: continue
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            conf_det = float(box.conf[0])
            cls_id   = int(box.cls[0])
            famille  = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
            crop = img_rgb[y1:y2, x1:x2]
            if crop.shape[0]<10 or crop.shape[1]<10:
                skus,confs = ["inconnu"],[0.0]
            else:
                skus,confs = predict_sku(sku_model, crop, idx_to_class, upscale)
            info = sku_catalog.get(skus[0], {})
            nom  = info.get('product_name', skus[0])
            crop_bytes = np_to_bytes(prepare_crop(crop, IMG_SIZE, upscale))
            if   confs[0]>0.7: col=(16,185,129)
            elif confs[0]>0.4: col=(245,158,11)
            elif confs[0]>0.2: col=(249,115,22)
            else:               col=(239,68,68)
            label = f"{nom} ({confs[0]:.1%})" if confs[0]>disp_thr else famille
            font  = cv2.FONT_HERSHEY_SIMPLEX
            (tw,th),_ = cv2.getTextSize(label, font, 0.55, 2)
            cv2.rectangle(annotated,(x1,y1),(x2,y2),col,3)
            cv2.rectangle(annotated,(x1,y1-th-8),(x1+tw+8,y1),(10,10,20),-1)
            cv2.putText(annotated,label,(x1+4,y1-6),font,0.55,(255,255,255),2)
            detections.append({
                "bbox":[x1,y1,x2,y2],"famille":famille,"sku":skus[0],
                "nom_produit":nom,"confiance_detection":round(conf_det,3),
                "confiance_sku":round(confs[0],3),
                "top5_predictions":list(zip(skus,confs)),
                "brand":info.get('brand','N/A'),"capacity":info.get('capacity','N/A'),
                "emballage":info.get('emballage','N/A'),"saveur":info.get('saveur','N/A'),
                "category":info.get('category','N/A'),
            })
            crops_data.append({
                "crop_bytes":crop_bytes,
                "stage1":f"{famille} ({conf_det:.2f})",
                "stage2_sku":skus[0],"stage2_nom":nom,"stage2_conf":confs[0],
                "top5":list(zip(skus[:5],confs[:5])),
                "brand":info.get('brand','N/A'),"capacity":info.get('capacity','N/A'),
                "emballage":info.get('emballage','N/A'),"saveur":info.get('saveur','N/A'),
            })
    return annotated, detections, crops_data

def run_inventory_pipeline(image_bytes, sku_model, idx_to_class, sku_catalog, upscale=2.5):
    arr    = np.frombuffer(image_bytes, np.uint8)
    im_rgb = cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    skus,confs = predict_sku(sku_model, im_rgb, idx_to_class, upscale)
    info = sku_catalog.get(skus[0], {})
    return {
        "sku":skus[0],"nom":info.get('product_name',skus[0]),
        "conf":round(confs[0],3),"top5":list(zip(skus[:5],confs[:5])),
        "brand":info.get('brand','N/A'),"capacity":info.get('capacity','N/A'),
        "emballage":info.get('emballage','N/A'),"saveur":info.get('saveur','N/A'),
        "crop_bytes":np_to_bytes(prepare_crop(im_rgb, IMG_SIZE, upscale)),
    }

# ══════════════════════════════════════════════════════
# HTML RENDERERS  (copies conformes du JS)
# ══════════════════════════════════════════════════════
def conf_cls(c):
    if c>0.7: return "conf-high"
    if c>0.4: return "conf-med"
    if c>0.2: return "conf-low"
    return "conf-vlow"

def render_result_images(raw_bytes, annotated_np):
    ann_bytes = np_to_bytes(annotated_np)
    return f"""
<div class="result-images">
  <div class="result-image-card">
    <div class="card-header">📸 Image originale</div>
    <img src="data:image/jpeg;base64,{b64(raw_bytes)}" style="width:100%;display:block;">
  </div>
  <div class="result-image-card">
    <div class="card-header">🎯 Résultat annoté</div>
    <img src="data:image/jpeg;base64,{b64(ann_bytes)}" style="width:100%;display:block;">
  </div>
</div>"""

def render_metrics(total, hi, me, lo, vl):
    return f"""
<div class="metrics-grid">
  <div class="metric-card metric-total"><div class="metric-value">{total}</div><div class="metric-label">Total</div></div>
  <div class="metric-card metric-high"><div class="metric-value">{hi}</div><div class="metric-label">&gt;70%</div></div>
  <div class="metric-card metric-med"><div class="metric-value">{me}</div><div class="metric-label">40–70%</div></div>
  <div class="metric-card metric-low"><div class="metric-value">{lo}</div><div class="metric-label">20–40%</div></div>
  <div class="metric-card metric-vlow"><div class="metric-value">{vl}</div><div class="metric-label">&lt;20%</div></div>
</div>"""

def render_detection_card(cd, rank):
    cc  = conf_cls(cd['stage2_conf'])
    conf = cd['stage2_conf']
    img_src = f"data:image/jpeg;base64,{b64(cd['crop_bytes'])}"
    brand_tag = f'<span class="tag tag-brand">{cd["brand"]}</span>' if cd['brand'] != 'N/A' else ''
    top5_html = "".join(
        f'<div class="top5-item">'
        f'<span class="top5-rank">{j+1}.</span>'
        f'<span class="top5-sku">{s}</span>'
        f'<span class="top5-conf">{c:.1%}</span>'
        f'</div>'
        for j,(s,c) in enumerate(cd['top5'][:5])
    )
    return f"""
<div class="detection-card {cc}" onclick="this.classList.toggle('open')">
  <div class="det-summary">
    <img class="det-crop-thumb" src="{img_src}" alt="">
    <div class="det-info">
      <div class="det-name">{cd['stage2_nom']}</div>
      <div class="det-tags">
        <span class="tag tag-famille">{cd['stage1'].split('(')[0].strip()}</span>
        <span class="tag tag-sku">{cd['stage2_sku']}</span>
        {brand_tag}
      </div>
    </div>
    <div class="det-conf-badge">{conf:.0%}</div>
    <span class="det-chevron">▶</span>
  </div>
  <div class="det-details">
    <div class="det-details-inner">
      <img class="det-crop-large" src="{img_src}" alt="">
      <div class="det-fields">
        <div class="det-field"><span class="det-field-key">Stage 1</span><span class="det-field-val">{cd['stage1']}</span></div>
        <div class="det-field"><span class="det-field-key">SKU</span><span class="det-field-val" style="color:var(--accent)">{cd['stage2_sku']}</span></div>
        <div class="det-field"><span class="det-field-key">Produit</span><span class="det-field-val">{cd['stage2_nom']}</span></div>
        <div class="det-field"><span class="det-field-key">Marque</span><span class="det-field-val">{cd['brand']}</span></div>
        <div class="det-field"><span class="det-field-key">Capacité</span><span class="det-field-val">{cd['capacity']}</span></div>
        <div class="det-field"><span class="det-field-key">Emballage</span><span class="det-field-val">{cd['emballage']}</span></div>
        <div class="det-field"><span class="det-field-key">Saveur</span><span class="det-field-val">{cd['saveur']}</span></div>
        <div class="conf-bar-wrap {cc}">
          <div class="conf-bar-label">Confiance : {conf:.1%}</div>
          <div class="conf-bar"><div class="conf-bar-fill" style="width:{conf*100:.1f}%"></div></div>
        </div>
        <div class="top5-list"><p>🏆 Top-5 :</p>{top5_html}</div>
      </div>
    </div>
  </div>
</div>"""

def render_results_section(crops_data, detections, raw_bytes, annotated_np):
    hi = sum(1 for c in crops_data if c['stage2_conf']>0.7)
    me = sum(1 for c in crops_data if 0.4<c['stage2_conf']<=0.7)
    lo = sum(1 for c in crops_data if 0.2<c['stage2_conf']<=0.4)
    vl = sum(1 for c in crops_data if c['stage2_conf']<=0.2)
    sorted_crops = sorted(crops_data, key=lambda x: x['stage2_conf'], reverse=True)
    cards_html = "\n".join(render_detection_card(cd,i) for i,cd in enumerate(sorted_crops))
    # table
    rows_html = ""
    for d in detections:
        col = "var(--green)" if d['confiance_sku']>0.7 else "var(--yellow)" if d['confiance_sku']>0.4 else "var(--red)"
        rows_html += f"""<tr>
          <td class="td-name">{d['nom_produit']}</td>
          <td>{d['brand']}</td><td>{d['capacity']}</td>
          <td>{d['emballage']}</td><td>{d['saveur']}</td>
          <td>{d['famille']}</td>
          <td style="color:{col};font-weight:700">{d['confiance_sku']:.1%}</td>
          <td>{d['confiance_detection']:.1%}</td>
          <td style="color:var(--accent);font-family:var(--font-mono)">{d['sku']}</td>
        </tr>"""
    return f"""
{render_result_images(raw_bytes, annotated_np)}
{render_metrics(len(crops_data),hi,me,lo,vl)}
<div class="detections-header"><h2>🔍 Détail des détections</h2></div>
<div class="detections-grid">{cards_html}</div>
<div class="table-section">
  <h2>📋 Rapport complet</h2>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Produit</th><th>Marque</th><th>Capacité</th><th>Emballage</th>
        <th>Saveur</th><th>Famille</th><th>Conf.SKU</th><th>Conf.Dét.</th><th>SKU ID</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>"""

def render_inventory_pending_card_html(item):
    conf = item['conf']
    if   conf>0.7: col="var(--green)"
    elif conf>0.4: col="var(--yellow)"
    elif conf>0.2: col="var(--orange)"
    else:          col="var(--red)"
    return f"""
<div class="inventory-pending-card">
  <img class="inventory-pending-img" src="data:image/jpeg;base64,{b64(item['crop_bytes'])}" alt="">
  <div class="inventory-pending-name" title="{item['nom']}">{item['nom']}</div>
  <div class="inventory-pending-sku">{item['sku']}</div>
  <div class="inventory-pending-conf" style="color:{col}">{conf:.1%}</div>
</div>"""

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="SKU Recognition Pipeline",
    page_icon="🏪", layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════
# CSS — copie exacte du JS, adaptée Streamlit
# ══════════════════════════════════════════════════════
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500'
    '&amp;family=Syne:wght@400;600;700;800&amp;display=swap" rel="stylesheet">',
    unsafe_allow_html=True
)
st.markdown("""
<style>
:root{
  --bg:#f4f5f8; --surface:#ffffff; --surface2:#f0f2f7;
  --border:#e2e6ef; --border2:#d0d5e3;
  --accent:#6c63ff; --accent2:#8b83ff; --accent-light:rgba(108,99,255,0.08);
  --green:#10b981; --yellow:#f59e0b; --orange:#f97316; --red:#ef4444;
  --text:#1a1d2e; --text-muted:#6b7280; --text-dim:#9ca3af;
  --font-display:'Syne',sans-serif; --font-mono:'DM Mono',monospace;
  --radius:14px; --radius-sm:8px;
  --shadow:0 1px 3px rgba(0,0,0,0.07),0 4px 16px rgba(0,0,0,0.05);
  --shadow-lg:0 8px 32px rgba(0,0,0,0.12);
  --glow:0 0 0 3px rgba(108,99,255,0.18);
}

/* ── Reset & global ── */
*{box-sizing:border-box;}
html,body,.stApp,section.main,section[data-testid="stSidebarContent"],
.stMarkdown,.element-container{
  font-family:var(--font-display)!important;
}
.stApp{background:var(--bg)!important;}

/* dot grid */
.stApp::before{
  content:'';position:fixed;inset:0;
  background-image:radial-gradient(circle,rgba(108,99,255,0.07) 1px,transparent 1px);
  background-size:28px 28px;pointer-events:none;z-index:0;
}

/* no outlines / underlines on click */
button:focus,button:focus-visible,
[data-testid="baseButton-secondary"]:focus,
[data-testid="baseButton-primary"]:focus,
[data-baseweb="tab"]:focus,
[data-baseweb="tab"]:focus-visible,
a:focus,a:focus-visible,
div:focus,div:focus-visible{
  outline:none!important;
  box-shadow:none!important;
  text-decoration:none!important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
  background:var(--surface)!important;
  border-right:1px solid var(--border)!important;
  box-shadow:2px 0 12px rgba(0,0,0,0.04)!important;
}
[data-testid="stSidebar"]::-webkit-scrollbar{width:4px;}
[data-testid="stSidebar"]::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px;}

/* sidebar sliders */
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{
  background:var(--accent)!important;box-shadow:0 2px 6px rgba(108,99,255,0.4)!important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider-track-fill"]{
  background:var(--accent)!important;
}

/* ── Logo ── */
.logo-block{display:flex;align-items:center;gap:10px;padding-bottom:4px;}
.logo-icon{width:38px;height:38px;background:linear-gradient(135deg,var(--accent),var(--accent2));
  border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;
  flex-shrink:0;box-shadow:0 4px 12px rgba(108,99,255,0.3);}
.logo-text{font-size:15px;font-weight:800;color:var(--text);line-height:1.2;}
.logo-sub{font-size:10px;color:var(--text-muted);font-weight:400;letter-spacing:0.06em;text-transform:uppercase;}

/* ── Status indicator ── */
.status-indicator{display:flex;align-items:center;gap:8px;padding:10px 12px;
  border-radius:var(--radius-sm);font-size:11px;font-family:var(--font-mono);}
.status-ready{background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);color:var(--green);}
.status-error{background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);color:var(--red);}
.status-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.status-ready .status-dot{background:var(--green);}
.status-error .status-dot{background:var(--red);}

/* ── Pipeline badges ── */
.stage-badge{padding:2px 6px;border-radius:4px;font-size:9px;font-weight:700;
  letter-spacing:.05em;text-transform:uppercase;}
.badge-yolo{background:rgba(245,158,11,0.12);color:var(--yellow);}
.badge-sku{background:rgba(108,99,255,0.12);color:var(--accent);}
.badge-inventory{background:rgba(16,185,129,0.12);color:var(--green);}
.pipeline-info{background:var(--surface2);border:1px solid var(--border);
  border-radius:var(--radius-sm);padding:12px;}
.stage{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--text-muted);
  padding:5px 0;border-bottom:1px solid var(--border);font-family:var(--font-mono);}
.stage:last-child{border-bottom:none;}

/* ── Model file info ── */
.model-file{font-family:var(--font-mono);font-size:10px;color:var(--text-muted);line-height:2.1;}
.model-file strong{color:var(--accent);}

/* ── Main tabs (Streamlit native) ── */
.stTabs [data-baseweb="tab-list"]{
  display:flex;gap:3px;background:var(--surface)!important;
  border:1px solid var(--border)!important;border-radius:var(--radius)!important;
  padding:4px!important;width:fit-content;box-shadow:var(--shadow)!important;
}
.stTabs [data-baseweb="tab"]{
  padding:8px 20px!important;border-radius:10px!important;border:none!important;
  background:transparent!important;color:var(--text-muted)!important;
  font-family:var(--font-display)!important;font-size:13px!important;
  font-weight:600!important;cursor:pointer!important;transition:all .2s!important;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,var(--accent),var(--accent2))!important;
  color:white!important;box-shadow:0 2px 8px rgba(108,99,255,0.35)!important;
}
.stTabs [aria-selected="false"]:hover{
  color:var(--text)!important;background:var(--surface2)!important;
}
/* hide the bottom bar */
.stTabs [data-baseweb="tab-highlight"]{display:none!important;}
.stTabs [data-baseweb="tab-border"]{display:none!important;}

/* ── Page header ── */
.page-header h1{
  font-size:26px;font-weight:800;letter-spacing:-.02em;
  background:linear-gradient(90deg,var(--text) 0%,var(--accent) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.page-header p{font-size:13px;color:var(--text-muted);margin-top:4px;}

/* ── Upload zone ── */
[data-testid="stFileUploaderDropzone"]{
  border:2px dashed var(--border2)!important;border-radius:var(--radius)!important;
  padding:44px 32px!important;background:var(--surface)!important;
  box-shadow:var(--shadow)!important;transition:all .25s!important;
}
[data-testid="stFileUploaderDropzone"]:hover{
  border-color:var(--accent)!important;box-shadow:var(--glow),var(--shadow)!important;
}
[data-testid="stFileUploaderDropzone"] span{
  font-family:var(--font-display)!important;color:var(--text-muted)!important;
}

/* ── Camera section ── */
.camera-section{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:24px;text-align:center;box-shadow:var(--shadow);}
[data-testid="stCameraInput"] video,[data-testid="stCameraInput"] img{
  border-radius:var(--radius-sm)!important;margin:12px auto!important;
}

/* ── Streamlit buttons styled ── */
.stButton>button{
  padding:9px 20px!important;border-radius:9px!important;border:none!important;
  font-family:var(--font-display)!important;font-size:13px!important;font-weight:600!important;
  cursor:pointer!important;transition:all .2s!important;
  display:inline-flex!important;align-items:center!important;gap:6px!important;
  outline:none!important;text-decoration:none!important;
}
.stButton>button:hover{transform:translateY(-1px)!important;opacity:.9!important;}
/* primary */
[data-testid="baseButton-primary"]{
  background:linear-gradient(135deg,var(--accent),var(--accent2))!important;
  color:white!important;box-shadow:0 2px 8px rgba(108,99,255,0.3)!important;
}
/* secondary */
[data-testid="baseButton-secondary"]{
  background:var(--surface)!important;color:var(--text-muted)!important;
  border:1px solid var(--border)!important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"]>div>div{
  background:linear-gradient(90deg,var(--accent),var(--accent2))!important;
}

/* ── Result images ── */
.result-images{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:22px;}
.result-image-card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow);}
.card-header{padding:11px 16px;border-bottom:1px solid var(--border);font-size:11px;
  font-weight:700;color:var(--text-muted);letter-spacing:.06em;text-transform:uppercase;
  display:flex;align-items:center;justify-content:space-between;
  font-family:var(--font-display);}

/* ── Metrics ── */
.metrics-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:22px;}
.metric-card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:16px 12px;text-align:center;box-shadow:var(--shadow);}
.metric-value{font-size:26px;font-weight:800;font-family:var(--font-mono);}
.metric-label{font-size:10px;color:var(--text-muted);margin-top:3px;
  letter-spacing:.06em;text-transform:uppercase;}
.metric-total .metric-value{color:var(--text);}
.metric-high  .metric-value{color:var(--green);}
.metric-med   .metric-value{color:var(--yellow);}
.metric-low   .metric-value{color:var(--orange);}
.metric-vlow  .metric-value{color:var(--red);}

/* ── Detection cards ── */
.detections-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;}
.detections-header h2{font-size:15px;font-weight:800;font-family:var(--font-display);}
.detections-grid{display:flex;flex-direction:column;gap:8px;}
.detection-card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);overflow:hidden;transition:all .2s;
  box-shadow:var(--shadow);cursor:pointer;}
.detection-card:hover{box-shadow:var(--shadow-lg);transform:translateY(-1px);}
.detection-card.conf-high{border-left:4px solid var(--green);}
.detection-card.conf-med {border-left:4px solid var(--yellow);}
.detection-card.conf-low {border-left:4px solid var(--orange);}
.detection-card.conf-vlow{border-left:4px solid var(--red);}
.det-summary{display:flex;align-items:center;gap:14px;padding:13px 16px;user-select:none;}
.det-crop-thumb{width:52px;height:52px;border-radius:8px;object-fit:cover;
  flex-shrink:0;background:var(--surface2);border:1px solid var(--border);}
.det-info{flex:1;min-width:0;}
.det-name{font-size:13px;font-weight:700;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;margin-bottom:4px;font-family:var(--font-display);}
.det-tags{display:flex;gap:5px;flex-wrap:wrap;}
.tag{padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600;
  font-family:var(--font-mono);letter-spacing:.03em;}
.tag-famille{background:rgba(245,158,11,0.1);color:var(--yellow);}
.tag-sku{background:var(--accent-light);color:var(--accent);}
.tag-brand{background:rgba(16,185,129,0.08);color:var(--green);}
.det-conf-badge{font-family:var(--font-mono);font-size:17px;font-weight:800;flex-shrink:0;}
.conf-high .det-conf-badge{color:var(--green);}
.conf-med  .det-conf-badge{color:var(--yellow);}
.conf-low  .det-conf-badge{color:var(--orange);}
.conf-vlow .det-conf-badge{color:var(--red);}
.det-chevron{color:var(--text-dim);font-size:11px;transition:transform .2s;flex-shrink:0;}
.detection-card.open .det-chevron{transform:rotate(90deg);}
.det-details{display:none;padding:0 16px 16px;border-top:1px solid var(--border);}
.detection-card.open .det-details{display:block;}
.det-details-inner{display:grid;grid-template-columns:110px 1fr;gap:16px;padding-top:14px;}
.det-crop-large{width:110px;height:110px;object-fit:contain;border-radius:8px;
  background:var(--surface2);padding:4px;border:1px solid var(--border);}
.det-fields{display:flex;flex-direction:column;gap:5px;}
.det-field{display:flex;align-items:flex-start;gap:8px;font-size:12px;}
.det-field-key{color:var(--text-muted);width:82px;flex-shrink:0;
  font-family:var(--font-mono);font-size:11px;padding-top:1px;}
.det-field-val{color:var(--text);font-weight:600;font-family:var(--font-display);}
.conf-bar-wrap{margin-top:10px;}
.conf-bar-label{font-size:11px;color:var(--text-muted);font-family:var(--font-mono);margin-bottom:5px;}
.conf-bar{height:6px;background:var(--border2);border-radius:3px;overflow:hidden;}
.conf-bar-fill{height:100%;border-radius:3px;transition:width .6s ease;}
.conf-high .conf-bar-fill{background:var(--green);}
.conf-med  .conf-bar-fill{background:var(--yellow);}
.conf-low  .conf-bar-fill{background:var(--orange);}
.conf-vlow .conf-bar-fill{background:var(--red);}
.top5-list{margin-top:10px;}
.top5-list p{font-size:11px;color:var(--text-muted);font-family:var(--font-mono);margin-bottom:5px;}
.top5-item{display:flex;align-items:center;gap:8px;font-size:11px;
  font-family:var(--font-mono);padding:3px 0;color:var(--text-muted);}
.top5-item .top5-rank{color:var(--text-dim);width:14px;flex-shrink:0;}
.top5-item .top5-sku{flex:1;color:var(--accent);word-break:break-word;}
.top5-item .top5-conf{color:var(--text-muted);}

/* ── Table ── */
.table-section{margin-top:26px;}
.table-section h2{font-size:15px;font-weight:800;margin-bottom:12px;font-family:var(--font-display);}
.table-wrap{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);overflow:auto;max-height:380px;box-shadow:var(--shadow);}
.table-wrap table{width:100%;border-collapse:collapse;font-size:12px;min-width:800px;}
.table-wrap th{padding:10px 14px;text-align:left;font-size:10px;font-weight:700;
  letter-spacing:.08em;text-transform:uppercase;color:var(--text-muted);
  background:var(--surface2);border-bottom:1px solid var(--border);
  position:sticky;top:0;white-space:nowrap;font-family:var(--font-display);}
.table-wrap td{padding:9px 14px;border-bottom:1px solid rgba(226,230,239,0.6);
  font-family:var(--font-mono);color:var(--text-muted);white-space:nowrap;}
.table-wrap tr:last-child td{border-bottom:none;}
.table-wrap tr:hover td{background:var(--accent-light);}
.table-wrap .td-name{color:var(--text);font-family:var(--font-display);
  font-size:12px;font-weight:700;}
.export-row{display:flex;gap:10px;margin-top:14px;}

/* ── Session nav ── */
.sessions-nav-wrap{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:10px 14px;box-shadow:var(--shadow);margin-bottom:4px;}
.session-tab-custom{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;
  border-radius:8px;border:1px solid var(--border);background:var(--surface);
  font-family:var(--font-display);font-size:12px;font-weight:600;
  color:var(--text-muted);cursor:pointer;transition:all .18s;margin:3px;}
.session-tab-custom.active{background:var(--accent-light);border-color:var(--accent);color:var(--accent);}

/* ── Inventory ── */
.inventory-upload-zone{border:2px dashed var(--border2);border-radius:var(--radius);
  padding:32px;text-align:center;cursor:pointer;transition:all .25s;
  background:var(--surface);margin-bottom:24px;}
.inventory-pending-grid{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:24px;}
.inventory-pending-card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:12px;width:220px;position:relative;
  transition:all .2s;box-shadow:var(--shadow);}
.inventory-pending-card:hover{box-shadow:var(--shadow-lg);transform:translateY(-2px);}
.inventory-pending-img{width:100%;height:120px;object-fit:contain;
  background:var(--surface2);border-radius:var(--radius-sm);margin-bottom:10px;}
.inventory-pending-name{font-size:12px;font-weight:600;margin-bottom:4px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:var(--font-display);}
.inventory-pending-sku{font-size:10px;color:var(--text-muted);font-family:var(--font-mono);
  margin-bottom:6px;word-break:break-all;}
.inventory-pending-conf{font-size:11px;font-weight:700;margin-bottom:8px;font-family:var(--font-mono);}
.inventory-table-wrap{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);overflow:auto;max-height:400px;box-shadow:var(--shadow);}
.inventory-table-wrap table{width:100%;border-collapse:collapse;font-size:12px;min-width:800px;}
.inventory-table-wrap th,.inventory-table-wrap td{padding:12px 14px;text-align:left;
  border-bottom:1px solid var(--border);}
.inventory-table-wrap th{background:var(--surface2);font-size:10px;font-weight:700;
  letter-spacing:.08em;text-transform:uppercase;color:var(--text-muted);
  position:sticky;top:0;font-family:var(--font-display);}
.inventory-table-wrap td{font-family:var(--font-mono);color:var(--text-muted);}
.inventory-table-wrap .td-name{color:var(--text);font-family:var(--font-display);
  font-weight:700;white-space:normal;word-break:break-word;}
.inventory-table-wrap .td-sku{font-family:var(--font-mono);color:var(--accent);word-break:break-all;}

/* ── Empty state ── */
.empty-state{text-align:center;padding:40px;color:var(--text-muted);
  background:var(--surface);border:2px dashed var(--border2);border-radius:var(--radius);}

/* ── Section titles ── */
.section-h2{font-size:15px;font-weight:800;color:var(--text);
  margin:0 0 12px;font-family:var(--font-display);}

/* ── Sidebar divider ── */
.sidebar-divider{border:none;border-top:1px solid var(--border);margin:4px 0;}

/* camera placeholder */
.cam-placeholder{background:var(--surface2);border:1px solid var(--border);
  border-radius:var(--radius-sm);width:100%;max-width:640px;height:240px;
  display:flex;align-items:center;justify-content:center;
  color:var(--text-dim);font-size:13px;margin:12px auto;}

@media(max-width:860px){
  .result-images{grid-template-columns:1fr;}
  .metrics-grid{grid-template-columns:repeat(3,1fr);}
  .det-details-inner{grid-template-columns:1fr;}
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════
_defs = {
    'sessions':[], 'active_sess_id':None, 'proc_hashes':set(),
    'cam_active':False,
    'inv_pending':[], 'inv_validated':[], 'inv_next_id':1,
    'inv_upload_key':0,
    'inv_cam_active':False,
}
for k,v in _defs.items():
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
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--text-muted);margin-bottom:12px;">⚙ Seuils</div>', unsafe_allow_html=True)
    conf_threshold    = st.slider("Seuil YOLO",      0.10, 0.95, 0.45, 0.05)
    display_threshold = st.slider("Seuil affichage", 0.10, 0.95, 0.25, 0.05)
    upscale_factor    = st.slider("Agrandissement",  1.0,  4.0,  2.5,  0.5)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--text-muted);margin-bottom:12px;">📊 Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline-info">
      <div class="stage"><span class="stage-badge badge-yolo">Stage 1</span> YOLO — famille</div>
      <div class="stage"><span class="stage-badge badge-sku">Stage 2</span> MobileNetV3 — SKU</div>
      <div class="stage"><span class="stage-badge badge-inventory">Inventaire</span> Stage 2 uniquement</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--text-muted);margin-bottom:12px;">📁 Modèles</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="model-file">
      <strong>YOLO:</strong> {YOLO_MODEL_PATH}<br>
      <strong>SKU:</strong> {SKU_MODEL_PATH}<br>
      <strong>Labels:</strong> {MAPPING_PATH}<br>
      <strong>Catalog:</strong> {CSV_PATH}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════
models_ok = False
try:
    yolo_model = load_yolo_model(YOLO_MODEL_PATH)
    sku_model, idx_to_class, num_classes = load_sku_model(SKU_MODEL_PATH, MAPPING_PATH)
    sku_catalog, _ = load_sku_catalog(CSV_PATH)
    models_ok = True
    _status_slot.markdown(f"""
    <div class="status-indicator status-ready">
      <div class="status-dot"></div> Prêt · {num_classes} SKU
    </div>""", unsafe_allow_html=True)
except Exception as exc:
    _status_slot.markdown(f"""
    <div class="status-indicator status-error">
      <div class="status-dot"></div> Erreur chargement
    </div>""", unsafe_allow_html=True)
    st.error(f"❌ {exc}")

# ══════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="page-header">
  <h1>SKU Recognition Pipeline</h1>
  <p>YOLO + MobileNetV3 · Cliquez sur une carte de détection pour voir les détails produit</p>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════
tab_up, tab_cam, tab_inv = st.tabs([
    "📸 Upload d'image", "📷 Prendre une photo", "📦 Inventaire"
])

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 1 — UPLOAD (multi-session)                     ║
# ╚══════════════════════════════════════════════════════╝
with tab_up:
    uploaded = st.file_uploader(
        "Déposez une image de rayon",
        type=["jpg","jpeg","png"], key="main_up",
        label_visibility="collapsed"
    )
    if uploaded and models_ok:
        raw = uploaded.getvalue()
        fhash = hash(raw[:2048] + uploaded.name.encode())
        if fhash not in st.session_state.proc_hashes:
            with st.spinner("🔍 Analyse YOLO + SKU en cours…"):
                ann, dets, crops = run_pipeline(
                    raw, yolo_model, sku_model, idx_to_class, sku_catalog,
                    conf_threshold, upscale_factor, display_threshold
                )
            sess = {'id':len(st.session_state.sessions)+1,
                    'name':uploaded.name[:20],
                    'raw':raw,'ann':ann,'dets':dets,'crops':crops}
            st.session_state.sessions.append(sess)
            st.session_state.active_sess_id = sess['id']
            st.session_state.proc_hashes.add(fhash)

    # Session nav
    if st.session_state.sessions:
        # Render session tabs + remove button using Streamlit buttons in columns
        n = len(st.session_state.sessions)
        nav_cols = st.columns(min(n,5)+1)
        for i,sess in enumerate(st.session_state.sessions[:5]):
            with nav_cols[i]:
                active = sess['id']==st.session_state.active_sess_id
                label = f"{'▶ ' if active else ''}📷 {sess['name']}"
                if st.button(label, key=f"sn_{sess['id']}", use_container_width=True,
                             type="primary" if active else "secondary"):
                    st.session_state.active_sess_id = sess['id']
                    st.rerun()
        with nav_cols[-1]:
            if st.button("✕ Retirer", key="rm_sess", use_container_width=True, type="secondary"):
                st.session_state.sessions = [s for s in st.session_state.sessions
                                              if s['id']!=st.session_state.active_sess_id]
                if st.session_state.sessions:
                    st.session_state.active_sess_id = st.session_state.sessions[-1]['id']
                else:
                    st.session_state.active_sess_id = None
                st.rerun()

        active = next((s for s in st.session_state.sessions
                       if s['id']==st.session_state.active_sess_id), None)
        if active:
            st.markdown(render_results_section(
                active['crops'], active['dets'], active['raw'], active['ann']
            ), unsafe_allow_html=True)

            # Export buttons (native widgets)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            df = pd.DataFrame(active['dets'])
            ec1,ec2 = st.columns(2)
            with ec1:
                st.download_button("📥 CSV", df.to_csv(index=False),
                    f"detections_{ts}.csv","text/csv",use_container_width=True)
            with ec2:
                st.download_button("📥 JSON",
                    json.dumps(active['dets'],indent=2,ensure_ascii=False,default=str),
                    f"detections_{ts}.json","application/json",use_container_width=True)
    elif not uploaded:
        st.markdown("""
        <div style="border:2px dashed var(--border2);border-radius:var(--radius);
          padding:44px 32px;text-align:center;background:var(--surface);
          box-shadow:var(--shadow);margin-top:8px;">
          <div style="font-size:38px;margin-bottom:10px;">📦</div>
          <div style="font-size:16px;font-weight:700;color:var(--text);margin-bottom:4px;">
            Déposez une image de rayon</div>
          <div style="font-size:12px;color:var(--text-muted);">
            JPG, JPEG ou PNG · Glissez-déposez ou cliquez sur Browse files</div>
        </div>""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 2 — CAMERA                                      ║
# ╚══════════════════════════════════════════════════════╝
with tab_cam:
    st.markdown('<div class="camera-section">', unsafe_allow_html=True)
    st.markdown('<p style="font-size:13px;color:var(--text-muted);margin-bottom:12px;">📱 Appareil photo ou webcam</p>', unsafe_allow_html=True)

    if not st.session_state.cam_active:
        st.markdown('<div class="cam-placeholder">🎥 Caméra inactive</div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns([1,1,1])
        with c1:
            if st.button("▶ Démarrer", key="cam_start", type="primary", use_container_width=True):
                st.session_state.cam_active = True
                st.rerun()
    else:
        cam_img = st.camera_input("", key="cam_input", label_visibility="collapsed")
        c1,c2,c3 = st.columns([1,1,1])
        with c1:
            if st.button("▶ Démarrer", key="cam_start2", disabled=True, use_container_width=True):
                pass
        with c2:
            if cam_img:
                if st.button("📸 Utiliser cette photo", key="cam_use", type="primary", use_container_width=True):
                    raw = cam_img.getvalue()
                    with st.spinner("🔍 Analyse en cours…"):
                        ann,dets,crops = run_pipeline(
                            raw, yolo_model, sku_model, idx_to_class, sku_catalog,
                            conf_threshold, upscale_factor, display_threshold
                        )
                    st.session_state.cam_result = {'raw':raw,'ann':ann,'dets':dets,'crops':crops}
                    st.session_state.cam_active = False
                    st.rerun()
        with c3:
            if st.button("⏹ Arrêter", key="cam_stop", type="secondary", use_container_width=True):
                st.session_state.cam_active = False
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    if 'cam_result' in st.session_state and st.session_state.cam_result:
        r = st.session_state.cam_result
        st.markdown(render_results_section(r['crops'],r['dets'],r['raw'],r['ann']),
                    unsafe_allow_html=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        df = pd.DataFrame(r['dets'])
        ec1,ec2 = st.columns(2)
        with ec1:
            st.download_button("📥 CSV",df.to_csv(index=False),
                f"photo_{ts}.csv","text/csv",use_container_width=True)
        with ec2:
            st.download_button("📥 JSON",
                json.dumps(r['dets'],indent=2,ensure_ascii=False,default=str),
                f"photo_{ts}.json","application/json",use_container_width=True)

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 3 — INVENTAIRE (Stage 2 uniquement)            ║
# ╚══════════════════════════════════════════════════════╝
with tab_inv:
    # Sub-tabs: Upload / Caméra
    inv_s_up, inv_s_cam = st.tabs(["📁 Upload", "📷 Caméra"])

    with inv_s_up:
        inv_files = st.file_uploader(
            "Scannez un ou plusieurs produits",
            type=["jpg","jpeg","png"], accept_multiple_files=True,
            key=f"inv_up_{st.session_state.inv_upload_key}",
            label_visibility="collapsed"
        )
        if inv_files and models_ok:
            prog = st.progress(0)
            for i,f in enumerate(inv_files):
                res = run_inventory_pipeline(
                    f.read(), sku_model, idx_to_class, sku_catalog, upscale_factor
                )
                res['id'] = st.session_state.inv_next_id
                st.session_state.inv_next_id += 1
                st.session_state.inv_pending.append(res)
                prog.progress((i+1)/len(inv_files))
            prog.empty()
            st.session_state.inv_upload_key += 1
            st.rerun()

    with inv_s_cam:
        st.markdown('<div class="camera-section">', unsafe_allow_html=True)
        if not st.session_state.inv_cam_active:
            st.markdown('<div class="cam-placeholder">🎥 Caméra inactive</div>', unsafe_allow_html=True)
            ic1,ic2,ic3 = st.columns([1,1,1])
            with ic1:
                if st.button("▶ Démarrer", key="inv_cam_start", type="primary", use_container_width=True):
                    st.session_state.inv_cam_active = True
                    st.rerun()
        else:
            inv_cam = st.camera_input("", key="inv_cam_input", label_visibility="collapsed")
            ic1,ic2,ic3 = st.columns([1,1,1])
            with ic1:
                if st.button("▶ Démarrer", key="inv_cam_start2", disabled=True, use_container_width=True):
                    pass
            with ic2:
                if inv_cam and st.button("📸 Capturer & Scanner", key="inv_cam_use",
                                         type="primary", use_container_width=True):
                    with st.spinner("Classification SKU…"):
                        res = run_inventory_pipeline(
                            inv_cam.getvalue(), sku_model, idx_to_class, sku_catalog, upscale_factor
                        )
                        res['id'] = st.session_state.inv_next_id
                        st.session_state.inv_next_id += 1
                        st.session_state.inv_pending.append(res)
                    st.session_state.inv_cam_active = False
                    st.rerun()
            with ic3:
                if st.button("⏹ Arrêter", key="inv_cam_stop", type="secondary", use_container_width=True):
                    st.session_state.inv_cam_active = False
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Pending grid ────────────────────────────────────
    pending = st.session_state.inv_pending
    st.markdown("---")

    if pending:
        bh1,bh2,bh3 = st.columns([4,1,1])
        with bh1:
            n_p=len(pending)
            st.markdown(f'<p class="section-h2">🕐 En attente ({n_p} produit{"s" if n_p>1 else ""})</p>',
                        unsafe_allow_html=True)
        with bh2:
            if st.button("➕ Ajouter tout", use_container_width=True, type="primary", key="add_all"):
                st.session_state.inv_validated.extend(st.session_state.inv_pending)
                st.session_state.inv_pending = []
                st.rerun()
        with bh3:
            if st.button("🗑 Tout supprimer", use_container_width=True, key="del_all"):
                st.session_state.inv_pending = []
                st.rerun()

        to_add, to_del = [], []
        for row_items in [pending[i:i+3] for i in range(0,len(pending),3)]:
            cols = st.columns(3)
            for col_el, item in zip(cols, row_items):
                with col_el:
                    st.markdown(render_inventory_pending_card_html(item), unsafe_allow_html=True)
                    ba,bb = st.columns(2)
                    with ba:
                        if st.button("➕ Ajouter", key=f"ia_{item['id']}",
                                     use_container_width=True, type="primary"):
                            to_add.append(item['id'])
                    with bb:
                        if st.button("🗑 Suppr.", key=f"id_{item['id']}",
                                     use_container_width=True, type="secondary"):
                            to_del.append(item['id'])
        if to_add:
            for iid in to_add:
                obj = next((x for x in st.session_state.inv_pending if x['id']==iid), None)
                if obj:
                    st.session_state.inv_validated.append(obj)
                    st.session_state.inv_pending = [x for x in st.session_state.inv_pending if x['id']!=iid]
            st.rerun()
        if to_del:
            st.session_state.inv_pending = [x for x in st.session_state.inv_pending if x['id'] not in to_del]
            st.rerun()
    else:
        st.markdown("""
        <div class="empty-state">
          Aucune image en attente. Uploadez des photos produit ou utilisez la caméra.
        </div>""", unsafe_allow_html=True)

    # ── Validated inventory table ────────────────────────
    st.markdown("---")
    validated = st.session_state.inv_validated
    vh1,vh2 = st.columns([4,1])
    with vh1:
        nv=len(validated)
        st.markdown(f'<p class="section-h2">📋 Inventaire validé ({nv} produit{"s" if nv!=1 else ""})</p>',
                    unsafe_allow_html=True)
    with vh2:
        if validated and st.button("🗑 Tout effacer", use_container_width=True, key="clear_inv"):
            st.session_state.inv_validated = []
            st.rerun()

    if validated:
        # Render as exact copy of JS inventory table
        rows_html = ""
        for it in validated:
            conf=it['conf']
            col = "var(--green)" if conf>0.7 else "var(--yellow)" if conf>0.4 else "var(--orange)" if conf>0.2 else "var(--red)"
            rows_html += f"""<tr>
              <td class="td-name">{it['nom']}</td>
              <td class="td-sku">{it['sku']}</td>
              <td>{it['brand']}</td><td>{it['capacity']}</td>
              <td>{it['emballage']}</td><td>{it['saveur']}</td>
              <td style="color:{col};font-weight:700">{conf:.1%}</td>
            </tr>"""
        st.markdown(f"""
        <div class="inventory-table-wrap">
          <table>
            <thead><tr>
              <th>Produit</th><th>SKU</th><th>Marque</th><th>Capacité</th>
              <th>Emballage</th><th>Saveur</th><th>Confiance</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)

        # Per-item delete
        st.markdown('<p style="font-size:12px;color:var(--text-muted);margin:12px 0 6px;">Supprimer un produit :</p>',
                    unsafe_allow_html=True)
        del_ids = []
        for it in validated:
            rc1,rc2,rc3 = st.columns([3,1,1])
            with rc1:
                conf=it['conf']
                col = "var(--green)" if conf>0.7 else "var(--yellow)" if conf>0.4 else "var(--orange)" if conf>0.2 else "var(--red)"
                st.markdown(
                    f'<span style="font-weight:700;font-family:var(--font-display)">{it["nom"]}</span>'
                    f' &nbsp; <span style="color:var(--accent);font-family:var(--font-mono);font-size:11px">{it["sku"]}</span>',
                    unsafe_allow_html=True)
            with rc2:
                st.markdown(f'<span style="color:{col};font-weight:700;font-family:var(--font-mono)">{conf:.1%}</span>',
                            unsafe_allow_html=True)
            with rc3:
                if st.button("🗑", key=f"dv_{it['id']}", use_container_width=True):
                    del_ids.append(it['id'])
        if del_ids:
            st.session_state.inv_validated = [x for x in st.session_state.inv_validated if x['id'] not in del_ids]
            st.rerun()

        # Export
        st.markdown("---")
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        inv_rows = [{"Produit":it['nom'],"SKU":it['sku'],"Marque":it['brand'],
                     "Capacité":it['capacity'],"Emballage":it['emballage'],
                     "Saveur":it['saveur'],"Confiance":f"{it['conf']:.1%}"}
                    for it in validated]
        ec1,ec2 = st.columns(2)
        with ec1:
            st.download_button("📥 Exporter CSV",
                pd.DataFrame(inv_rows).to_csv(index=False),
                f"inventaire_{ts}.csv","text/csv",use_container_width=True)
        with ec2:
            inv_json = [{k:v for k,v in it.items() if k!='crop_bytes'} for it in validated]
            st.download_button("📥 Exporter JSON",
                json.dumps(inv_json,indent=2,ensure_ascii=False,default=str),
                f"inventaire_{ts}.json","application/json",use_container_width=True)
    else:
        st.markdown("""
        <div class="empty-state">
          Aucun produit dans l'inventaire. Validez des produits depuis la section ci-dessus.
        </div>""", unsafe_allow_html=True)