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
            if   confs[0]>0.7: col=(5,150,105)
            elif confs[0]>0.4: col=(217,119,6)
            elif confs[0]>0.2: col=(234,88,12)
            else:               col=(220,38,38)
            label = f"{nom} ({confs[0]:.1%})" if confs[0]>disp_thr else famille
            font  = cv2.FONT_HERSHEY_SIMPLEX
            (tw,th),_ = cv2.getTextSize(label, font, 0.52, 2)
            cv2.rectangle(annotated,(x1,y1),(x2,y2),col,2)
            cv2.rectangle(annotated,(x1,y1-th-8),(x1+tw+8,y1),(15,23,42),-1)
            cv2.putText(annotated,label,(x1+4,y1-5),font,0.52,(255,255,255),1)
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
# HTML RENDERERS
# ══════════════════════════════════════════════════════
def conf_cls(c):
    if c > 0.7: return "ch"
    if c > 0.4: return "cm"
    if c > 0.2: return "cl"
    return "cv"

def render_annotated_image(raw_bytes, annotated_np, crops_data, detections):
    """Annotated image with clickable bounding boxes → fixed panel (no scroll)."""
    ann_bytes = np_to_bytes(annotated_np)
    ann_h, ann_w = annotated_np.shape[:2]

    bbox_zones = ""
    panel_data = []

    for i, (det, cd) in enumerate(zip(detections, crops_data)):
        x1, y1, x2, y2 = det['bbox']
        lp = x1 / ann_w * 100
        tp = y1 / ann_h * 100
        wp = (x2 - x1) / ann_w * 100
        hp = (y2 - y1) / ann_h * 100
        cc = conf_cls(cd['stage2_conf'])
        bbox_zones += (
            f'<div class="bbox-z {cc}" '
            f'style="left:{lp:.2f}%;top:{tp:.2f}%;width:{wp:.2f}%;height:{hp:.2f}%;" '
            f'onclick="openBboxPanel({i})" title="{cd[\'stage2_nom\']}"></div>'
        )
        panel_data.append({
            "img":  f"data:image/jpeg;base64,{b64(cd['crop_bytes'])}",
            "nom":  cd['stage2_nom'],
            "sku":  cd['stage2_sku'],
            "conf": round(cd['stage2_conf'], 3),
            "stage1": cd['stage1'],
            "brand": cd['brand'],
            "capacity": cd['capacity'],
            "emballage": cd['emballage'],
            "saveur": cd['saveur'],
            "cc": cc,
        })

    data_json = json.dumps(panel_data, ensure_ascii=False)

    return f"""
<div class="result-images">
  <div class="result-card">
    <div class="result-card-header">Image originale</div>
    <img src="data:image/jpeg;base64,{b64(raw_bytes)}" style="width:100%;display:block;" alt="">
  </div>
  <div class="result-card">
    <div class="result-card-header">Résultat annoté <span>· cliquer sur un produit</span></div>
    <div style="position:relative;line-height:0;cursor:crosshair;">
      <img src="data:image/jpeg;base64,{b64(ann_bytes)}" style="width:100%;display:block;" alt="">
      {bbox_zones}
    </div>
  </div>
</div>

<div id="bpanel" class="bpanel">
  <div class="bpanel-hdr">
    <span class="bpanel-title">Détail produit</span>
    <button class="bpanel-close" onclick="closeBboxPanel()">&#x2715;</button>
  </div>
  <div id="bpanel-body" class="bpanel-body"></div>
</div>

<script>
(function(){{
  var D={data_json};
  var CC={{ch:'#059669',cm:'#D97706',cl:'#EA580C',cv:'#DC2626'}};
  window.openBboxPanel=function(i){{
    var d=D[i]; var col=CC[d.cc]||'#2563EB';
    var pct=(d.conf*100).toFixed(1)+'%';
    document.getElementById('bpanel-body').innerHTML=
      '<div class="bp-top">'+
        '<img src="'+d.img+'" class="bp-img">'+
        '<div class="bp-info">'+
          '<div class="bp-name">'+d.nom+'</div>'+
          '<div class="bp-sku">'+d.sku+'</div>'+
          '<div class="bp-conf" style="color:'+col+'">'+pct+'</div>'+
        '</div>'+
      '</div>'+
      '<div class="bp-fields">'+
        '<div class="bp-row"><span class="bp-k">Stage 1</span><span class="bp-v">'+d.stage1+'</span></div>'+
        '<div class="bp-row"><span class="bp-k">Marque</span><span class="bp-v">'+d.brand+'</span></div>'+
        '<div class="bp-row"><span class="bp-k">Capacité</span><span class="bp-v">'+d.capacity+'</span></div>'+
        '<div class="bp-row"><span class="bp-k">Emballage</span><span class="bp-v">'+d.emballage+'</span></div>'+
        '<div class="bp-row"><span class="bp-k">Saveur</span><span class="bp-v">'+d.saveur+'</span></div>'+
      '</div>'+
      '<div class="bp-bar-wrap"><div class="bp-track"><div class="bp-fill" style="width:'+pct+';background:'+col+'"></div></div></div>';
    document.getElementById('bpanel').style.display='block';
  }};
  window.closeBboxPanel=function(){{
    document.getElementById('bpanel').style.display='none';
  }};
  document.addEventListener('keydown',function(e){{if(e.key==='Escape')window.closeBboxPanel();}});
}})();
</script>"""

def render_metrics(total, hi, me, lo, vl):
    return f"""
<div class="metrics-row">
  <div class="metric metric-total"><div class="metric-val">{total}</div><div class="metric-lbl">Détections</div></div>
  <div class="metric metric-high"><div class="metric-val">{hi}</div><div class="metric-lbl">Haute &gt; 70%</div></div>
  <div class="metric metric-med"><div class="metric-val">{me}</div><div class="metric-lbl">Moyenne 40–70%</div></div>
  <div class="metric metric-low"><div class="metric-val">{lo}</div><div class="metric-lbl">Faible 20–40%</div></div>
  <div class="metric metric-vlow"><div class="metric-val">{vl}</div><div class="metric-lbl">Très faible &lt; 20%</div></div>
</div>"""

def render_detection_card(cd, rank):
    cc   = conf_cls(cd['stage2_conf'])
    conf = cd['stage2_conf']
    img  = f"data:image/jpeg;base64,{b64(cd['crop_bytes'])}"
    brand_pill = (f'<span class="pill pill-b">{cd["brand"]}</span>'
                  if cd['brand'] != 'N/A' else '')
    top5_rows = "".join(
        f'<div class="top5-item"><span class="top5-i">{j+1}</span>'
        f'<span class="top5-s">{s}</span>'
        f'<span class="top5-c">{c:.1%}</span></div>'
        for j,(s,c) in enumerate(cd['top5'][:5])
    )
    return f"""
<div class="det-card {cc}" onclick="this.classList.toggle('open')">
  <div class="det-row">
    <img class="det-thumb" src="{img}" alt="">
    <div class="det-body">
      <div class="det-name">{cd['stage2_nom']}</div>
      <div class="det-pills">
        <span class="pill pill-f">{cd['stage1'].split('(')[0].strip()}</span>
        <span class="pill pill-s">{cd['stage2_sku']}</span>
        {brand_pill}
      </div>
    </div>
    <div class="det-pct">{conf:.0%}</div>
    <div class="det-chevron">&#9658;</div>
  </div>
  <div class="det-expand">
    <div class="det-expand-inner">
      <img class="det-large-img" src="{img}" alt="">
      <div class="det-fields">
        <div class="det-field"><span class="det-fk">Stage 1</span><span class="det-fv">{cd['stage1']}</span></div>
        <div class="det-field"><span class="det-fk">SKU</span><span class="det-fv" style="color:var(--blue)">{cd['stage2_sku']}</span></div>
        <div class="det-field"><span class="det-fk">Produit</span><span class="det-fv">{cd['stage2_nom']}</span></div>
        <div class="det-field"><span class="det-fk">Marque</span><span class="det-fv">{cd['brand']}</span></div>
        <div class="det-field"><span class="det-fk">Capacité</span><span class="det-fv">{cd['capacity']}</span></div>
        <div class="det-field"><span class="det-fk">Emballage</span><span class="det-fv">{cd['emballage']}</span></div>
        <div class="det-field"><span class="det-fk">Saveur</span><span class="det-fv">{cd['saveur']}</span></div>
        <div class="conf-bar-wrap {cc}">
          <div class="conf-bar-lbl">Confiance · {conf:.1%}</div>
          <div class="conf-bar-track"><div class="conf-bar-fill" style="width:{conf*100:.1f}%"></div></div>
        </div>
        <div class="top5"><div class="top5-title">Top 5 prédictions</div>{top5_rows}</div>
      </div>
    </div>
  </div>
</div>"""

def render_results_section(crops_data, detections, raw_bytes, annotated_np):
    hi = sum(1 for c in crops_data if c['stage2_conf'] > 0.7)
    me = sum(1 for c in crops_data if 0.4 < c['stage2_conf'] <= 0.7)
    lo = sum(1 for c in crops_data if 0.2 < c['stage2_conf'] <= 0.4)
    vl = sum(1 for c in crops_data if c['stage2_conf'] <= 0.2)
    sorted_crops = sorted(crops_data, key=lambda x: x['stage2_conf'], reverse=True)
    cards_html = "\n".join(render_detection_card(cd, i) for i, cd in enumerate(sorted_crops))
    rows_html = ""
    for d in detections:
        col = ("var(--green)" if d['confiance_sku'] > 0.7
               else "var(--amber)" if d['confiance_sku'] > 0.4
               else "var(--red)")
        rows_html += (f'<tr><td class="td-name">{d["nom_produit"]}</td>'
                      f'<td>{d["brand"]}</td><td>{d["capacity"]}</td>'
                      f'<td>{d["emballage"]}</td><td>{d["saveur"]}</td>'
                      f'<td>{d["famille"]}</td>'
                      f'<td style="color:{col};font-weight:700">{d["confiance_sku"]:.1%}</td>'
                      f'<td>{d["confiance_detection"]:.1%}</td>'
                      f'<td class="td-sku">{d["sku"]}</td></tr>')
    return f"""
{render_annotated_image(raw_bytes, annotated_np, crops_data, detections)}
{render_metrics(len(crops_data), hi, me, lo, vl)}
<div class="detections-section">
  <div class="section-title">Détections <span>· {len(crops_data)} produit{"s" if len(crops_data)!=1 else ""} — cliquer pour développer</span></div>
  <div class="detections-stack">{cards_html}</div>
</div>
<div class="tbl-section">
  <div class="section-title">Rapport complet</div>
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>Produit</th><th>Marque</th><th>Capacité</th><th>Emballage</th>
        <th>Saveur</th><th>Famille</th><th>Conf. SKU</th><th>Conf. Dét.</th><th>SKU ID</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>"""

def render_inv_card(item):
    conf = item['conf']
    col = ("var(--green)" if conf > 0.7 else "var(--amber)" if conf > 0.4
           else "var(--orange)" if conf > 0.2 else "var(--red)")
    return f"""
<div class="inv-card">
  <img class="inv-card-img" src="data:image/jpeg;base64,{b64(item['crop_bytes'])}" alt="">
  <div class="inv-card-name" title="{item['nom']}">{item['nom']}</div>
  <div class="inv-card-sku">{item['sku']}</div>
  <div class="inv-card-conf" style="color:{col}">{conf:.1%}</div>
</div>"""

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="SKU Recognition",
    page_icon="", layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════
# FONTS
# ══════════════════════════════════════════════════════
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700'
    '&amp;family=JetBrains+Mono:wght@400;500&amp;display=swap" rel="stylesheet">',
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
:root{
  --bg:#F8FAFC;--surface:#FFFFFF;--surface-2:#F1F5F9;--surface-3:#E2E8F0;
  --blue:#2563EB;--blue-dark:#1D4ED8;--blue-light:#EFF6FF;--blue-border:#BFDBFE;
  --green:#059669;--amber:#D97706;--orange:#EA580C;--red:#DC2626;
  --text:#0F172A;--text-2:#334155;--text-muted:#64748B;--text-subtle:#94A3B8;
  --border:#E2E8F0;--border-2:#CBD5E1;
  --font:'Inter',-apple-system,sans-serif;
  --font-mono:'JetBrains Mono','Fira Code',monospace;
  --r:10px;--r-sm:6px;--r-lg:14px;
  --sh:0 1px 3px rgba(0,0,0,.06),0 2px 8px rgba(0,0,0,.04);
  --sh-md:0 4px 12px rgba(0,0,0,.08),0 1px 3px rgba(0,0,0,.05);
  --sh-lg:0 8px 32px rgba(0,0,0,.12);
  --sh-blue:0 4px 14px rgba(37,99,235,.22);
}

*,*::before,*::after{box-sizing:border-box;}
*:focus,*:focus-visible{outline:none!important;box-shadow:none!important;}
button:focus,button:focus-visible{text-decoration:none!important;}

html,body,.stApp,section.main,.stMarkdown,.element-container,
[data-testid="stVerticalBlock"]{font-family:var(--font)!important;}
.stApp{background:var(--bg)!important;}

/* ── No top padding on main area ── */
.main .block-container{padding-top:12px!important;}

/* ── Light sidebar ── */
[data-testid="stSidebar"]{
  background:var(--surface)!important;
  border-right:1px solid var(--border)!important;
  box-shadow:2px 0 8px rgba(0,0,0,.04)!important;
}
[data-testid="stSidebar"]::-webkit-scrollbar{width:3px;}
[data-testid="stSidebar"]::-webkit-scrollbar-thumb{background:var(--border-2);border-radius:3px;}
[data-testid="stSidebar"] [data-baseweb="slider-track"]{background:var(--surface-3)!important;}
[data-testid="stSidebar"] [data-baseweb="slider-track-fill"]{background:var(--blue)!important;}
[data-testid="stSidebar"] [role="slider"]{
  background:var(--blue)!important;border:none!important;
  box-shadow:0 0 0 4px rgba(37,99,235,.18)!important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{
  background:var(--surface)!important;border:1px solid var(--border)!important;
  border-radius:var(--r)!important;padding:4px!important;gap:2px!important;
  width:fit-content!important;box-shadow:var(--sh)!important;
}
.stTabs [data-baseweb="tab"]{
  background:transparent!important;border:none!important;border-radius:var(--r-sm)!important;
  padding:7px 18px!important;font-family:var(--font)!important;font-size:13px!important;
  font-weight:500!important;color:var(--text-muted)!important;transition:all .15s!important;
}
.stTabs [aria-selected="true"]{
  background:var(--blue)!important;color:#fff!important;box-shadow:var(--sh-blue)!important;
}
.stTabs [aria-selected="false"]:hover{background:var(--surface-2)!important;color:var(--text-2)!important;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none!important;}

/* ── Buttons ── */
.stButton>button{
  font-family:var(--font)!important;font-size:13px!important;font-weight:500!important;
  border-radius:var(--r-sm)!important;padding:8px 16px!important;border:none!important;
  transition:all .15s!important;outline:none!important;
}
.stButton>button:hover{opacity:.88!important;transform:translateY(-1px)!important;}
[data-testid="baseButton-primary"]{
  background:var(--blue)!important;color:#fff!important;box-shadow:var(--sh-blue)!important;
}
[data-testid="baseButton-secondary"]{
  background:var(--surface)!important;color:var(--text-2)!important;
  border:1px solid var(--border-2)!important;box-shadow:var(--sh)!important;
}

/* ── Download ── */
[data-testid="stDownloadButton"]>button{
  font-family:var(--font)!important;font-size:13px!important;font-weight:500!important;
  background:var(--surface)!important;color:var(--text-2)!important;
  border:1px solid var(--border-2)!important;border-radius:var(--r-sm)!important;
  padding:8px 16px!important;box-shadow:var(--sh)!important;transition:all .15s!important;
}
[data-testid="stDownloadButton"]>button:hover{background:var(--surface-2)!important;transform:translateY(-1px)!important;}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"]{
  border:1.5px dashed var(--border-2)!important;border-radius:var(--r-lg)!important;
  padding:36px 28px!important;background:var(--surface)!important;
  box-shadow:var(--sh)!important;transition:all .2s!important;
}
[data-testid="stFileUploaderDropzone"]:hover{
  border-color:var(--blue)!important;
  box-shadow:0 0 0 3px var(--blue-border),var(--sh)!important;
}
[data-testid="stFileUploaderDropzone"] span{font-family:var(--font)!important;color:var(--text-muted)!important;}
/* Hide duplicate upload label */
[data-testid="stFileUploaderDropzone"] + div > span{display:none!important;}

/* ── Camera ── */
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img{border-radius:var(--r)!important;}
/* Hide camera label */
[data-testid="stCameraInput"] label{display:none!important;}

/* ── Progress ── */
[data-testid="stProgressBar"]>div>div{background:var(--blue)!important;border-radius:4px!important;}

/* ── HR ── */
hr{border:none!important;border-top:1px solid var(--border)!important;margin:24px 0!important;}

/* ═══════ SIDEBAR ═══════ */
.sb-logo{
  display:flex;align-items:center;gap:12px;
  padding:20px 16px 16px;border-bottom:1px solid var(--border);margin-bottom:4px;
}
.sb-logo-mark{
  width:34px;height:34px;flex-shrink:0;background:var(--blue);border-radius:8px;
  display:flex;align-items:center;justify-content:center;
}
.sb-logo-title{font-size:14px;font-weight:700;color:var(--text);line-height:1.2;}
.sb-logo-sub{font-size:10px;color:var(--text-muted);letter-spacing:.08em;text-transform:uppercase;margin-top:2px;}
.sb-section{
  font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  color:var(--text-subtle);padding:6px 0 8px;
}
.sb-pipeline{display:flex;flex-direction:column;gap:4px;}
.sb-stage{
  display:flex;align-items:center;gap:10px;padding:8px 10px;
  border-radius:6px;background:var(--surface-2);border:1px solid var(--border);
}
.sb-stage-num{
  width:20px;height:20px;border-radius:4px;background:var(--blue);
  color:#fff;font-size:10px;font-weight:700;
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
}
.sb-stage-text{font-size:12px;font-weight:600;color:var(--text);}
.sb-stage-sub{font-size:10px;color:var(--text-muted);}
.sb-files{display:flex;flex-direction:column;gap:2px;}
.sb-file-row{display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid var(--border);}
.sb-file-row:last-child{border-bottom:none;}
.sb-file-key{font-size:10px;color:var(--text-muted);font-family:var(--font-mono);width:46px;flex-shrink:0;}
.sb-file-val{font-size:10px;color:var(--text-2);font-family:var(--font-mono);word-break:break-all;}
.sb-divider{border:none;border-top:1px solid var(--border);margin:14px 0;}
.status-pill{
  display:inline-flex;align-items:center;gap:7px;
  padding:6px 12px;border-radius:20px;font-size:11px;font-family:var(--font-mono);font-weight:500;
}
.s-ok{background:rgba(5,150,105,.1);border:1px solid rgba(5,150,105,.25);color:#059669;}
.s-err{background:rgba(220,38,38,.08);border:1px solid rgba(220,38,38,.2);color:#DC2626;}
.status-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0;}
.s-ok .status-dot{background:#059669;}
.s-err .status-dot{background:#DC2626;}

/* ═══════ PAGE HEADER ═══════ */
.page-header{padding:12px 0 20px;border-bottom:1px solid var(--border);margin-bottom:24px;}
.page-header-title{font-size:21px;font-weight:700;color:var(--text);letter-spacing:-.025em;margin:0 0 4px;}
.page-header-sub{font-size:13px;color:var(--text-muted);margin:0;}

/* ═══════ RESULTS ═══════ */
.result-images{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;}
.result-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden;box-shadow:var(--sh);}
.result-card-header{padding:10px 14px;border-bottom:1px solid var(--border);font-size:11px;font-weight:600;color:var(--text-2);letter-spacing:.04em;text-transform:uppercase;}
.result-card-header span{font-weight:400;color:var(--text-subtle);text-transform:none;letter-spacing:0;}

/* Bbox zones */
.bbox-z{
  position:absolute;border:2px solid transparent;
  border-radius:3px;cursor:pointer;transition:background .12s;
  box-sizing:border-box;
}
.bbox-z.ch{border-color:rgba(5,150,105,.85);}
.bbox-z.cm{border-color:rgba(217,119,6,.85);}
.bbox-z.cl{border-color:rgba(234,88,12,.85);}
.bbox-z.cv{border-color:rgba(220,38,38,.85);}
.bbox-z:hover{background:rgba(255,255,255,.18);}

/* Bbox floating panel */
.bpanel{
  display:none;position:fixed;top:50%;right:24px;transform:translateY(-50%);
  width:296px;background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r-lg);box-shadow:var(--sh-lg);z-index:9999;overflow:hidden;
  font-family:var(--font);
}
.bpanel-hdr{
  display:flex;align-items:center;justify-content:space-between;
  padding:11px 14px;border-bottom:1px solid var(--border);background:var(--surface-2);
}
.bpanel-title{font-size:12px;font-weight:600;color:var(--text);}
.bpanel-close{
  background:none;border:none;cursor:pointer;color:var(--text-muted);
  font-size:13px;padding:2px 6px;border-radius:4px;line-height:1;
}
.bpanel-close:hover{background:var(--surface-3);color:var(--text);}
.bpanel-body{padding:14px;}
.bp-top{display:flex;gap:12px;margin-bottom:12px;}
.bp-img{width:68px;height:68px;object-fit:contain;border-radius:6px;background:var(--surface-2);border:1px solid var(--border);padding:3px;flex-shrink:0;}
.bp-info{flex:1;min-width:0;}
.bp-name{font-size:12px;font-weight:600;color:var(--text);margin-bottom:3px;line-height:1.3;}
.bp-sku{font-size:10px;color:var(--blue);font-family:var(--font-mono);word-break:break-all;margin-bottom:4px;}
.bp-conf{font-size:16px;font-weight:700;font-family:var(--font-mono);}
.bp-fields{display:flex;flex-direction:column;gap:4px;}
.bp-row{display:flex;align-items:flex-start;gap:8px;font-size:12px;}
.bp-k{color:var(--text-subtle);width:64px;flex-shrink:0;font-family:var(--font-mono);font-size:11px;padding-top:1px;}
.bp-v{color:var(--text-2);font-weight:500;}
.bp-bar-wrap{margin-top:12px;}
.bp-track{height:4px;background:var(--border);border-radius:2px;overflow:hidden;}
.bp-fill{height:100%;border-radius:2px;}

/* ═══════ METRICS ═══════ */
.metrics-row{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:24px;}
.metric{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:14px 12px;box-shadow:var(--sh);}
.metric-val{font-size:26px;font-weight:700;font-family:var(--font-mono);letter-spacing:-.02em;}
.metric-lbl{font-size:11px;color:var(--text-muted);margin-top:2px;}
.metric-total .metric-val{color:var(--text);}
.metric-high  .metric-val{color:var(--green);}
.metric-med   .metric-val{color:var(--amber);}
.metric-low   .metric-val{color:var(--orange);}
.metric-vlow  .metric-val{color:var(--red);}

/* ═══════ DETECTION CARDS ═══════ */
.detections-section{margin-bottom:24px;}
.section-title{font-size:13px;font-weight:600;color:var(--text);letter-spacing:-.01em;margin:0 0 12px;}
.section-title span{font-weight:400;color:var(--text-muted);}
.detections-stack{display:flex;flex-direction:column;gap:6px;}
.det-card{background:var(--surface);border:1px solid var(--border);border-left:3px solid transparent;border-radius:var(--r);overflow:hidden;cursor:pointer;transition:box-shadow .15s,transform .15s;box-shadow:var(--sh);}
.det-card:hover{box-shadow:var(--sh-md);transform:translateY(-1px);}
.det-card.ch{border-left-color:var(--green);}
.det-card.cm{border-left-color:var(--amber);}
.det-card.cl{border-left-color:var(--orange);}
.det-card.cv{border-left-color:var(--red);}
.det-row{display:flex;align-items:center;gap:12px;padding:11px 14px;user-select:none;}
.det-thumb{width:46px;height:46px;border-radius:6px;object-fit:cover;flex-shrink:0;background:var(--surface-2);border:1px solid var(--border);}
.det-body{flex:1;min-width:0;}
.det-name{font-size:13px;font-weight:600;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:4px;}
.det-pills{display:flex;gap:4px;flex-wrap:wrap;}
.pill{padding:2px 6px;border-radius:4px;font-size:10px;font-weight:600;font-family:var(--font-mono);letter-spacing:.02em;}
.pill-f{background:rgba(217,119,6,.1);color:var(--amber);}
.pill-s{background:rgba(37,99,235,.08);color:var(--blue);}
.pill-b{background:rgba(5,150,105,.08);color:var(--green);}
.det-pct{font-family:var(--font-mono);font-size:16px;font-weight:700;flex-shrink:0;}
.ch .det-pct{color:var(--green);}
.cm .det-pct{color:var(--amber);}
.cl .det-pct{color:var(--orange);}
.cv .det-pct{color:var(--red);}
.det-chevron{width:16px;height:16px;flex-shrink:0;display:flex;align-items:center;justify-content:center;color:var(--text-subtle);font-size:9px;transition:transform .2s;}
.det-card.open .det-chevron{transform:rotate(90deg);}
.det-expand{display:none;padding:0 14px 14px;border-top:1px solid var(--border);}
.det-card.open .det-expand{display:block;}
.det-expand-inner{display:grid;grid-template-columns:96px 1fr;gap:14px;padding-top:12px;}
.det-large-img{width:96px;height:96px;object-fit:contain;border-radius:7px;background:var(--surface-2);border:1px solid var(--border);padding:4px;}
.det-fields{display:flex;flex-direction:column;gap:4px;}
.det-field{display:flex;align-items:flex-start;gap:8px;font-size:12px;}
.det-fk{color:var(--text-muted);width:72px;flex-shrink:0;font-family:var(--font-mono);font-size:11px;padding-top:1px;}
.det-fv{color:var(--text-2);font-weight:500;}
.conf-bar-wrap{margin-top:8px;}
.conf-bar-lbl{font-size:11px;color:var(--text-muted);font-family:var(--font-mono);margin-bottom:4px;}
.conf-bar-track{height:4px;background:var(--border);border-radius:2px;overflow:hidden;}
.conf-bar-fill{height:100%;border-radius:2px;}
.ch .conf-bar-fill{background:var(--green);}
.cm .conf-bar-fill{background:var(--amber);}
.cl .conf-bar-fill{background:var(--orange);}
.cv .conf-bar-fill{background:var(--red);}
.top5{margin-top:8px;}
.top5-title{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--text-subtle);margin-bottom:5px;}
.top5-item{display:flex;align-items:center;gap:7px;font-size:11px;font-family:var(--font-mono);padding:2px 0;}
.top5-i{color:var(--text-subtle);width:11px;flex-shrink:0;}
.top5-s{flex:1;color:var(--blue);}
.top5-c{color:var(--text-muted);}

/* ═══════ TABLE ═══════ */
.tbl-section{margin-top:24px;}
.tbl-wrap{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:auto;max-height:340px;box-shadow:var(--sh);}
.tbl-wrap table{width:100%;border-collapse:collapse;font-size:12px;min-width:720px;}
.tbl-wrap th{padding:9px 12px;text-align:left;font-size:10px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:var(--text-muted);background:var(--surface-2);border-bottom:1px solid var(--border);position:sticky;top:0;white-space:nowrap;}
.tbl-wrap td{padding:8px 12px;border-bottom:1px solid rgba(226,232,240,.5);font-family:var(--font-mono);font-size:11px;color:var(--text-muted);white-space:nowrap;}
.tbl-wrap tr:last-child td{border-bottom:none;}
.tbl-wrap tr:hover td{background:var(--blue-light);}
.td-name{color:var(--text)!important;font-family:var(--font)!important;font-size:12px!important;font-weight:600!important;}
.td-sku{color:var(--blue)!important;}

/* ═══════ INVENTORY ═══════ */
.inv-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:12px;box-shadow:var(--sh);transition:all .15s;}
.inv-card:hover{box-shadow:var(--sh-md);transform:translateY(-2px);}
.inv-card-img{width:100%;height:100px;object-fit:contain;background:var(--surface-2);border-radius:var(--r-sm);margin-bottom:8px;border:1px solid var(--border);display:block;}
.inv-card-name{font-size:12px;font-weight:600;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:2px;}
.inv-card-sku{font-size:10px;color:var(--blue);font-family:var(--font-mono);margin-bottom:4px;word-break:break-all;}
.inv-card-conf{font-size:11px;font-weight:700;font-family:var(--font-mono);margin-bottom:10px;}
/* Spacing between card and its action buttons */
.inv-btn-row{padding-top:2px;}

/* ═══════ CAMERA ═══════ */
.cam-wrap{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-lg);padding:24px;box-shadow:var(--sh);}
.cam-placeholder{
  background:var(--surface-2);border:1px solid var(--border);border-radius:var(--r);
  width:100%;max-width:540px;height:180px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  color:var(--text-subtle);margin:0 auto 16px;gap:6px;
}
.cam-placeholder-title{font-size:13px;font-weight:500;color:var(--text-muted);}
.cam-placeholder-sub{font-size:12px;}
/* Camera facing toggle */
.cam-facing-row{display:flex;gap:8px;margin-bottom:12px;justify-content:center;}

/* ═══════ EMPTY STATE ═══════ */
.empty-state{
  padding:44px 28px;text-align:center;
  background:var(--surface);border:1.5px dashed var(--border-2);border-radius:var(--r-lg);
}
.empty-title{font-size:14px;font-weight:600;color:var(--text-2);margin-bottom:4px;}
.empty-sub{font-size:12px;color:var(--text-muted);}

@media(max-width:860px){
  .result-images{grid-template-columns:1fr;}
  .metrics-row{grid-template-columns:repeat(3,1fr);}
  .det-expand-inner{grid-template-columns:1fr;}
  .bpanel{right:8px;width:calc(100vw - 16px);transform:translateY(-50%);}
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════
_defs = {
    'sessions':[], 'active_sess_id':None, 'proc_hashes':set(),
    'upload_key':0,
    'cam_active':False, 'cam_facing':'back',
    'inv_pending':[], 'inv_validated':[], 'inv_next_id':1,
    'inv_upload_key':0, 'inv_cam_active':False, 'inv_cam_facing':'back',
}
for k,v in _defs.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
      <div class="sb-logo-mark">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
          <rect x="2" y="2" width="6" height="6" rx="1.5" fill="white"/>
          <rect x="10" y="2" width="6" height="6" rx="1.5" fill="rgba(255,255,255,0.55)"/>
          <rect x="2" y="10" width="6" height="6" rx="1.5" fill="rgba(255,255,255,0.55)"/>
          <rect x="10" y="10" width="6" height="6" rx="1.5" fill="white"/>
        </svg>
      </div>
      <div>
        <div class="sb-logo-title">SKU Recognition</div>
        <div class="sb-logo-sub">YOLO · MobileNetV3</div>
      </div>
    </div>""", unsafe_allow_html=True)

    _status_slot = st.empty()
    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Seuils de détection</div>', unsafe_allow_html=True)
    conf_threshold    = st.slider("Seuil YOLO",      0.10, 0.95, 0.45, 0.05)
    display_threshold = st.slider("Seuil affichage", 0.10, 0.95, 0.25, 0.05)
    upscale_factor    = st.slider("Agrandissement",  1.0,  4.0,  2.5,  0.5)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-pipeline">
      <div class="sb-stage">
        <div class="sb-stage-num">1</div>
        <div><div class="sb-stage-text">YOLO</div><div class="sb-stage-sub">Détection famille</div></div>
      </div>
      <div class="sb-stage" style="margin-top:4px;">
        <div class="sb-stage-num">2</div>
        <div><div class="sb-stage-text">MobileNetV3</div><div class="sb-stage-sub">Classification SKU</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Fichiers modèles</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-files">
      <div class="sb-file-row"><span class="sb-file-key">YOLO</span><span class="sb-file-val">{YOLO_MODEL_PATH}</span></div>
      <div class="sb-file-row"><span class="sb-file-key">SKU</span><span class="sb-file-val">{SKU_MODEL_PATH}</span></div>
      <div class="sb-file-row"><span class="sb-file-key">Labels</span><span class="sb-file-val">{MAPPING_PATH}</span></div>
      <div class="sb-file-row"><span class="sb-file-key">Catalog</span><span class="sb-file-val">{CSV_PATH}</span></div>
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
    _status_slot.markdown(
        f'<div class="status-pill s-ok"><div class="status-dot"></div>Prêt &nbsp;·&nbsp; {num_classes} SKU</div>',
        unsafe_allow_html=True
    )
except Exception as exc:
    _status_slot.markdown(
        '<div class="status-pill s-err"><div class="status-dot"></div>Erreur chargement</div>',
        unsafe_allow_html=True
    )
    st.error(f"{exc}")

# ══════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="page-header">
  <div class="page-header-title">SKU Recognition Pipeline</div>
  <div class="page-header-sub">Détection de produits en rayon · YOLO Stage 1 + MobileNetV3 Stage 2</div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# HELPER — camera facing JS injector
# ══════════════════════════════════════════════════════
def inject_cam_facing(selector_key, facing):
    """Injects JS to switch camera facing mode. facing = 'environment' or 'user'."""
    st.markdown(f"""<script>
(function(){{
  var _f='{facing}';
  function _sw(){{
    var v=document.querySelector('[data-testid="stCameraInput"] video');
    if(!v||!v.srcObject){{setTimeout(_sw,350);return;}}
    if(window.__camFacing_{selector_key}===_f)return;
    window.__camFacing_{selector_key}=_f;
    v.srcObject.getTracks().forEach(function(t){{t.stop();}});
    navigator.mediaDevices.getUserMedia({{video:{{facingMode:{{exact:_f}}}},audio:false}})
    .catch(function(){{return navigator.mediaDevices.getUserMedia({{video:{{facingMode:_f}},audio:false}});}})
    .then(function(s){{if(v&&v.isConnected)v.srcObject=s;}});
  }}
  _sw();
}})();
</script>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════
tab_up, tab_cam, tab_inv = st.tabs(["Upload", "Caméra", "Inventaire"])

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 1 — UPLOAD (multi-session)                     ║
# ╚══════════════════════════════════════════════════════╝
with tab_up:
    # File uploader — key resets after each processed upload so dropzone stays clean
    uploaded = st.file_uploader(
        "",
        type=["jpg","jpeg","png"],
        key=f"main_up_{st.session_state.upload_key}",
        label_visibility="collapsed"
    )

    if uploaded and models_ok:
        raw = uploaded.getvalue()
        fhash = hash(raw[:2048] + uploaded.name.encode())
        if fhash not in st.session_state.proc_hashes:
            with st.spinner("Analyse en cours…"):
                ann, dets, crops = run_pipeline(
                    raw, yolo_model, sku_model, idx_to_class, sku_catalog,
                    conf_threshold, upscale_factor, display_threshold
                )
            sess = {
                'id':  len(st.session_state.sessions) + 1,
                'name': uploaded.name[:22],
                'raw': raw, 'ann': ann, 'dets': dets, 'crops': crops
            }
            st.session_state.sessions.append(sess)
            st.session_state.active_sess_id = sess['id']
            st.session_state.proc_hashes.add(fhash)
            st.session_state.upload_key += 1   # reset dropzone
            st.rerun()

    # ── Session navigation : [name][✕] per session ──────
    if st.session_state.sessions:
        n = min(len(st.session_state.sessions), 5)
        # Build column widths: 4 for name, 1 for ✕, repeated
        col_widths = []
        for _ in range(n):
            col_widths += [4, 1]
        nav_cols = st.columns(col_widths)

        to_remove = None
        for i, sess in enumerate(st.session_state.sessions[:5]):
            active = sess['id'] == st.session_state.active_sess_id
            with nav_cols[i * 2]:
                if st.button(
                    sess['name'], key=f"sn_{sess['id']}",
                    use_container_width=True,
                    type="primary" if active else "secondary"
                ):
                    st.session_state.active_sess_id = sess['id']
                    st.rerun()
            with nav_cols[i * 2 + 1]:
                if st.button("✕", key=f"rm_{sess['id']}", use_container_width=True):
                    to_remove = sess['id']

        if to_remove is not None:
            st.session_state.sessions = [
                s for s in st.session_state.sessions if s['id'] != to_remove
            ]
            if st.session_state.sessions:
                st.session_state.active_sess_id = st.session_state.sessions[-1]['id']
            else:
                st.session_state.active_sess_id = None
            st.rerun()

        active_sess = next(
            (s for s in st.session_state.sessions if s['id'] == st.session_state.active_sess_id),
            None
        )
        if active_sess:
            st.markdown(
                render_results_section(
                    active_sess['crops'], active_sess['dets'],
                    active_sess['raw'], active_sess['ann']
                ),
                unsafe_allow_html=True
            )
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            df = pd.DataFrame(active_sess['dets'])
            ec1, ec2 = st.columns(2)
            with ec1:
                st.download_button("Exporter CSV", df.to_csv(index=False),
                    f"detections_{ts}.csv", "text/csv", use_container_width=True)
            with ec2:
                st.download_button("Exporter JSON",
                    json.dumps(active_sess['dets'], indent=2, ensure_ascii=False, default=str),
                    f"detections_{ts}.json", "application/json", use_container_width=True)

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 2 — CAMERA                                      ║
# ╚══════════════════════════════════════════════════════╝
with tab_cam:
    st.markdown('<div class="cam-wrap">', unsafe_allow_html=True)

    if not st.session_state.cam_active:
        st.markdown("""
        <div class="cam-placeholder">
          <div class="cam-placeholder-title">Caméra inactive</div>
          <div class="cam-placeholder-sub">Cliquez sur Démarrer pour activer l'appareil photo</div>
        </div>""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("Démarrer", key="cam_start", type="primary", use_container_width=True):
                st.session_state.cam_active = True
                st.rerun()
    else:
        # ── Camera facing toggle ──
        cf1, cf2, cf3 = st.columns([1, 1, 1])
        with cf1:
            is_back = (st.session_state.cam_facing == 'back')
            if st.button("Caméra arrière", key="cam_back",
                         type="primary" if is_back else "secondary",
                         use_container_width=True):
                st.session_state.cam_facing = 'back'
                st.rerun()
        with cf2:
            if st.button("Selfie", key="cam_front",
                         type="primary" if not is_back else "secondary",
                         use_container_width=True):
                st.session_state.cam_facing = 'front'
                st.rerun()

        cam_img = st.camera_input("", key="cam_input", label_visibility="collapsed")

        # Inject facing-mode switcher
        facing_val = "environment" if st.session_state.cam_facing == 'back' else "user"
        inject_cam_facing("main", facing_val)

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.button("Démarrer", key="cam_start2", disabled=True, use_container_width=True)
        with c2:
            if cam_img:
                if st.button("Utiliser cette photo", key="cam_use",
                             type="primary", use_container_width=True):
                    raw = cam_img.getvalue()
                    with st.spinner("Analyse en cours…"):
                        ann, dets, crops = run_pipeline(
                            raw, yolo_model, sku_model, idx_to_class, sku_catalog,
                            conf_threshold, upscale_factor, display_threshold
                        )
                    st.session_state.cam_result = {'raw':raw,'ann':ann,'dets':dets,'crops':crops}
                    st.session_state.cam_active = False
                    st.rerun()
        with c3:
            if st.button("Arrêter", key="cam_stop", type="secondary", use_container_width=True):
                st.session_state.cam_active = False
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    if 'cam_result' in st.session_state and st.session_state.cam_result:
        r = st.session_state.cam_result
        st.markdown(
            render_results_section(r['crops'], r['dets'], r['raw'], r['ann']),
            unsafe_allow_html=True
        )
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        df = pd.DataFrame(r['dets'])
        ec1, ec2 = st.columns(2)
        with ec1:
            st.download_button("Exporter CSV", df.to_csv(index=False),
                f"photo_{ts}.csv", "text/csv", use_container_width=True)
        with ec2:
            st.download_button("Exporter JSON",
                json.dumps(r['dets'], indent=2, ensure_ascii=False, default=str),
                f"photo_{ts}.json", "application/json", use_container_width=True)

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 3 — INVENTAIRE                                  ║
# ╚══════════════════════════════════════════════════════╝
with tab_inv:
    inv_s_up, inv_s_cam = st.tabs(["Upload produits", "Caméra"])

    with inv_s_up:
        inv_files = st.file_uploader(
            "",
            type=["jpg","jpeg","png"], accept_multiple_files=True,
            key=f"inv_up_{st.session_state.inv_upload_key}",
            label_visibility="collapsed"
        )
        if inv_files and models_ok:
            prog = st.progress(0)
            for i, f in enumerate(inv_files):
                res = run_inventory_pipeline(
                    f.read(), sku_model, idx_to_class, sku_catalog, upscale_factor
                )
                res['id'] = st.session_state.inv_next_id
                st.session_state.inv_next_id += 1
                st.session_state.inv_pending.append(res)
                prog.progress((i + 1) / len(inv_files))
            prog.empty()
            st.session_state.inv_upload_key += 1
            st.rerun()

    with inv_s_cam:
        st.markdown('<div class="cam-wrap">', unsafe_allow_html=True)
        if not st.session_state.inv_cam_active:
            st.markdown("""
            <div class="cam-placeholder">
              <div class="cam-placeholder-title">Caméra inactive</div>
              <div class="cam-placeholder-sub">Scannez un produit à la fois</div>
            </div>""", unsafe_allow_html=True)
            ic1, ic2, ic3 = st.columns([1, 1, 1])
            with ic1:
                if st.button("Démarrer", key="inv_cam_start", type="primary", use_container_width=True):
                    st.session_state.inv_cam_active = True
                    st.rerun()
        else:
            # ── Camera facing toggle ──
            icf1, icf2, icf3 = st.columns([1, 1, 1])
            with icf1:
                inv_is_back = (st.session_state.inv_cam_facing == 'back')
                if st.button("Caméra arrière", key="inv_cam_back",
                             type="primary" if inv_is_back else "secondary",
                             use_container_width=True):
                    st.session_state.inv_cam_facing = 'back'
                    st.rerun()
            with icf2:
                if st.button("Selfie", key="inv_cam_front",
                             type="primary" if not inv_is_back else "secondary",
                             use_container_width=True):
                    st.session_state.inv_cam_facing = 'front'
                    st.rerun()

            inv_cam = st.camera_input("", key="inv_cam_input", label_visibility="collapsed")

            inv_facing_val = "environment" if st.session_state.inv_cam_facing == 'back' else "user"
            inject_cam_facing("inv", inv_facing_val)

            ic1, ic2, ic3 = st.columns([1, 1, 1])
            with ic1:
                st.button("Démarrer", key="inv_cam_start2", disabled=True, use_container_width=True)
            with ic2:
                if inv_cam and st.button("Capturer", key="inv_cam_use",
                                         type="primary", use_container_width=True):
                    with st.spinner("Classification…"):
                        res = run_inventory_pipeline(
                            inv_cam.getvalue(), sku_model, idx_to_class, sku_catalog, upscale_factor
                        )
                        res['id'] = st.session_state.inv_next_id
                        st.session_state.inv_next_id += 1
                        st.session_state.inv_pending.append(res)
                    st.session_state.inv_cam_active = False
                    st.rerun()
            with ic3:
                if st.button("Arrêter", key="inv_cam_stop", type="secondary", use_container_width=True):
                    st.session_state.inv_cam_active = False
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Pending ──────────────────────────────────────────
    pending = st.session_state.inv_pending
    st.markdown('<hr>', unsafe_allow_html=True)

    if pending:
        bh1, bh2, bh3 = st.columns([4, 1, 1])
        with bh1:
            n_p = len(pending)
            st.markdown(
                f'<div class="section-title">En attente <span>· {n_p} produit{"s" if n_p>1 else ""}</span></div>',
                unsafe_allow_html=True
            )
        with bh2:
            if st.button("Ajouter tout", use_container_width=True, type="primary", key="add_all"):
                st.session_state.inv_validated.extend(st.session_state.inv_pending)
                st.session_state.inv_pending = []
                st.rerun()
        with bh3:
            if st.button("Tout supprimer", use_container_width=True, key="del_all"):
                st.session_state.inv_pending = []
                st.rerun()

        to_add, to_del = [], []
        for row_items in [pending[i:i+3] for i in range(0, len(pending), 3)]:
            cols = st.columns(3)
            for col_el, item in zip(cols, row_items):
                with col_el:
                    st.markdown(render_inv_card(item), unsafe_allow_html=True)
                    # Spacing + buttons in same column
                    ba, bb = st.columns(2)
                    with ba:
                        if st.button("Ajouter", key=f"ia_{item['id']}",
                                     use_container_width=True, type="primary"):
                            to_add.append(item['id'])
                    with bb:
                        if st.button("Retirer", key=f"id_{item['id']}",
                                     use_container_width=True, type="secondary"):
                            to_del.append(item['id'])
                    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

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
          <div class="empty-title">Aucun produit en attente</div>
          <div class="empty-sub">Uploadez des photos ou utilisez la caméra pour scanner des produits</div>
        </div>""", unsafe_allow_html=True)

    # ── Validated inventory ───────────────────────────────
    st.markdown('<hr>', unsafe_allow_html=True)
    validated = st.session_state.inv_validated
    vh1, vh2 = st.columns([4, 1])
    with vh1:
        nv = len(validated)
        st.markdown(
            f'<div class="section-title">Inventaire validé <span>· {nv} produit{"s" if nv!=1 else ""}</span></div>',
            unsafe_allow_html=True
        )
    with vh2:
        if validated and st.button("Tout effacer", use_container_width=True, key="clear_inv"):
            st.session_state.inv_validated = []
            st.rerun()

    if validated:
        rows_html = ""
        for it in validated:
            conf = it['conf']
            col = ("var(--green)" if conf > 0.7 else "var(--amber)" if conf > 0.4
                   else "var(--orange)" if conf > 0.2 else "var(--red)")
            rows_html += (f'<tr><td class="td-name">{it["nom"]}</td>'
                          f'<td class="td-sku">{it["sku"]}</td>'
                          f'<td>{it["brand"]}</td><td>{it["capacity"]}</td>'
                          f'<td>{it["emballage"]}</td><td>{it["saveur"]}</td>'
                          f'<td style="color:{col};font-weight:700">{conf:.1%}</td></tr>')
        st.markdown(f"""
        <div class="tbl-wrap">
          <table>
            <thead><tr>
              <th>Produit</th><th>SKU</th><th>Marque</th><th>Capacité</th>
              <th>Emballage</th><th>Saveur</th><th>Confiance</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)

        st.markdown(
            '<div style="font-size:12px;color:var(--text-muted);margin:16px 0 8px;">Supprimer individuellement</div>',
            unsafe_allow_html=True
        )
        del_ids = []
        for it in validated:
            rc1, rc2, rc3 = st.columns([3, 1, 1])
            conf = it['conf']
            col = ("var(--green)" if conf > 0.7 else "var(--amber)" if conf > 0.4
                   else "var(--orange)" if conf > 0.2 else "var(--red)")
            with rc1:
                st.markdown(
                    f'<span style="font-weight:600;font-size:13px">{it["nom"]}</span>'
                    f'&nbsp;&nbsp;<span style="color:var(--blue);font-family:var(--font-mono);font-size:11px">{it["sku"]}</span>',
                    unsafe_allow_html=True
                )
            with rc2:
                st.markdown(
                    f'<span style="color:{col};font-weight:700;font-family:var(--font-mono);font-size:12px">{conf:.1%}</span>',
                    unsafe_allow_html=True
                )
            with rc3:
                if st.button("✕", key=f"dv_{it['id']}", use_container_width=True):
                    del_ids.append(it['id'])
        if del_ids:
            st.session_state.inv_validated = [
                x for x in st.session_state.inv_validated if x['id'] not in del_ids
            ]
            st.rerun()

        st.markdown('<hr>', unsafe_allow_html=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        inv_rows = [{"Produit": it['nom'], "SKU": it['sku'], "Marque": it['brand'],
                     "Capacité": it['capacity'], "Emballage": it['emballage'],
                     "Saveur": it['saveur'], "Confiance": f"{it['conf']:.1%}"}
                    for it in validated]
        ec1, ec2 = st.columns(2)
        with ec1:
            st.download_button("Exporter CSV",
                pd.DataFrame(inv_rows).to_csv(index=False),
                f"inventaire_{ts}.csv", "text/csv", use_container_width=True)
        with ec2:
            inv_json = [{k: v for k, v in it.items() if k != 'crop_bytes'} for it in validated]
            st.download_button("Exporter JSON",
                json.dumps(inv_json, indent=2, ensure_ascii=False, default=str),
                f"inventaire_{ts}.json", "application/json", use_container_width=True)
    else:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-title">Inventaire vide</div>
          <div class="empty-sub">Validez des produits depuis la section ci-dessus</div>
        </div>""", unsafe_allow_html=True)