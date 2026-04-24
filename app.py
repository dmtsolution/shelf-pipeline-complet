# app.py — SKU Recognition Pipeline · Streamlit · YOLO + MobileNetV3
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
import base64

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SKU Recognition Pipeline",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
YOLO_MODEL_PATH = "best-yolov8s.pt"
SKU_MODEL_PATH = "best-mobilenetv3large.pth"
MAPPING_PATH = "label_map.json"
CSV_PATH = "sku_catalog.csv"
IMG_SIZE = 224

CLASS_NAMES = ['boisson_energetique','dessert','eau','fromage','jus','lait','soda','yaourt']

# ═══════════════════════════════════════════════════════════
# DESIGN CSS — Même style que l'app web vanilla JS
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');
    
    :root {
        --bg: #f4f5f8;
        --surface: #ffffff;
        --surface2: #f0f2f7;
        --border: #e2e6ef;
        --accent: #6c63ff;
        --green: #10b981;
        --yellow: #f59e0b;
        --red: #ef4444;
        --text: #1a1d2e;
        --text-muted: #6b7280;
    }
    
    * { font-family: 'Syne', sans-serif; }
    .stApp { background-color: var(--bg); }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--surface);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] h3 {
        font-size: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
        color: var(--text-muted) !important;
        margin-bottom: 12px !important;
    }
    
    /* Cards */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.05);
    }
    
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px 10px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.07);
    }
    .metric-value {
        font-size: 22px;
        font-weight: 800;
        font-family: 'DM Mono', monospace;
    }
    .metric-label {
        font-size: 9px;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    /* Detection cards */
    .det-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        margin-bottom: 12px;
        overflow: hidden;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
    }
    
    /* Upload zone */
    .upload-zone {
        border: 2px dashed var(--border);
        border-radius: 14px;
        padding: 44px 32px;
        text-align: center;
        cursor: pointer;
        background: var(--surface);
    }
    
    /* Status indicator */
    .status-ready { color: var(--green); }
    .status-loading { color: var(--yellow); }
    .status-error { color: var(--red); }
    
    /* Pipeline badges */
    .badge-yolo { background: rgba(245,158,11,0.12); color: #f59e0b; padding: 2px 8px; border-radius: 4px; font-size: 9px; font-weight: 700; }
    .badge-sku { background: rgba(108,99,255,0.12); color: #6c63ff; padding: 2px 8px; border-radius: 4px; font-size: 9px; font-weight: 700; }
    .badge-inv { background: rgba(16,185,129,0.12); color: #10b981; padding: 2px 8px; border-radius: 4px; font-size: 9px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SESSION STATE (comme vanilla JS)
# ═══════════════════════════════════════════════════════════
if 'sessions' not in st.session_state:
    st.session_state.sessions = []
if 'active_session' not in st.session_state:
    st.session_state.active_session = None
if 'inventory_pending' not in st.session_state:
    st.session_state.inventory_pending = []
if 'inventory_validated' not in st.session_state:
    st.session_state.inventory_validated = []
if 'next_id' not in st.session_state:
    st.session_state.next_id = 1

# ═══════════════════════════════════════════════════════════
# TRANSFORMS
# ═══════════════════════════════════════════════════════════
val_transforms = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ═══════════════════════════════════════════════════════════
# MODELS (cachés)
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_yolo_model(path): return YOLO(path)

@st.cache_resource
def load_sku_model(model_path, labels_path):
    with open(labels_path, 'r') as f:
        label_map = json.load(f)
    idx_to_class = {int(k): v for k, v in label_map.items()}
    nc = len(idx_to_class)
    model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=nc)
    state_dict = torch.load(model_path, map_location='cpu')
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE).eval()
    return model, idx_to_class, nc

@st.cache_data
def load_sku_catalog(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df.set_index("sku_id").to_dict("index"), df
    return {}, pd.DataFrame()

# ═══════════════════════════════════════════════════════════
# SKU PREDICT (inventaire — Stage 2 uniquement)
# ═══════════════════════════════════════════════════════════
def predict_single_sku(image_bytes, sku_model, idx_to_class):
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_t = val_transforms(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = sku_model(img_t)
        probs = torch.softmax(out, dim=1)[0]
        topk = probs.topk(min(5, len(idx_to_class)))
    skus = [idx_to_class[i.item()] for i in topk.indices]
    confs = [v.item() for v in topk.values]
    return skus, confs

# ═══════════════════════════════════════════════════════════
# PIPELINE COMPLET (upload)
# ═══════════════════════════════════════════════════════════
def upscale_crop(crop, scale_factor=2.0):
    h, w = crop.shape[:2]
    return cv2.resize(crop, (int(w*scale_factor), int(h*scale_factor)), interpolation=cv2.INTER_CUBIC)

def prepare_crop(crop, target_size=224, upscale_first=True):
    if upscale_first and crop.shape[0] < 100 and crop.shape[1] < 100:
        crop = upscale_crop(crop, 2.5)
    h, w = crop.shape[:2]
    ratio = w / h
    new_w = target_size if ratio > 1 else int(target_size * ratio)
    new_h = int(target_size / ratio) if ratio > 1 else target_size
    crop_r = cv2.resize(crop, (max(1,new_w), max(1,new_h)), interpolation=cv2.INTER_LANCZOS4)
    square = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
    y_off, x_off = (target_size - crop_r.shape[0])//2, (target_size - crop_r.shape[1])//2
    square[y_off:y_off+crop_r.shape[0], x_off:x_off+crop_r.shape[1]] = crop_r
    return square

def predict_sku_crop(sku_model, idx_to_class, crop_img, upscale_factor=2.0):
    h, w = crop_img.shape[:2]
    if h < 150 or w < 150:
        crop_img = cv2.resize(crop_img, (int(w*upscale_factor), int(h*upscale_factor)), interpolation=cv2.INTER_CUBIC)
    prepared = prepare_crop(crop_img, IMG_SIZE)
    img_t = val_transforms(Image.fromarray(prepared)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = sku_model(img_t)
        probs = torch.softmax(out, dim=1)[0]
        topk = probs.topk(min(5, len(idx_to_class)))
    return [idx_to_class[i.item()] for i in topk.indices], [v.item() for v in topk.values]

def run_pipeline(image_bytes, yolo_model, sku_model, idx_to_class, sku_catalog, 
                conf_threshold=0.45, upscale_factor=2.5, display_threshold=0.25):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    if max(h, w) < 640:
        scale = 640 / max(h, w)
        img_rgb = cv2.resize(img_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    
    results = yolo_model(img_rgb, conf=conf_threshold, verbose=False)
    detections, crops_data = [], []
    img_annotated = img.copy()
    
    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_det = float(box.conf[0])
            cls_id = int(box.cls[0])
            famille = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"classe_{cls_id}"
            
            crop_original = img_rgb[y1:y2, x1:x2]
            if crop_original.shape[0] < 10 or crop_original.shape[1] < 10:
                skus, confs = ["inconnu"], [0.0]
                product_name = "Trop petit"
                top5 = []
                crop_display = crop_original
            else:
                skus, confs = predict_sku_crop(sku_model, idx_to_class, crop_original, upscale_factor)
                info = sku_catalog.get(skus[0], {})
                product_name = info.get('product_name', skus[0])
                top5 = list(zip(skus, confs))
                crop_display = prepare_crop(crop_original, IMG_SIZE)
            
            crop_bytes = io.BytesIO()
            Image.fromarray(crop_display).save(crop_bytes, format='JPEG')
            crop_bytes = crop_bytes.getvalue()
            info = sku_catalog.get(skus[0], {})
            
            detections.append({
                "bbox": [x1, y1, x2, y2], "famille": famille, "sku": skus[0],
                "nom_produit": product_name, "confiance_detection": round(conf_det,3),
                "confiance_sku": round(confs[0],3), "top5": top5,
                "brand": info.get('brand','N/A'), "capacity": info.get('capacity','N/A'),
                "emballage": info.get('emballage','N/A'), "saveur": info.get('saveur','N/A'),
                "category": info.get('category','N/A')
            })
            crops_data.append({
                "crop_bytes": crop_bytes, "stage1": f"{famille} ({conf_det:.0%})",
                "sku": skus[0], "nom": product_name, "conf": confs[0],
                "top5": top5, "brand": info.get('brand','N/A'),
                "capacity": info.get('capacity','N/A'), "emballage": info.get('emballage','N/A'),
                "saveur": info.get('saveur','N/A')
            })
            
            color = ((0,255,0) if confs[0]>.7 else (0,255,255) if confs[0]>.4 else (0,165,255) if confs[0]>.2 else (0,0,255))
            lbl = f"{product_name} ({confs[0]:.0%})" if confs[0] > display_threshold else famille
            cv2.rectangle(img_annotated, (x1,y1), (x2,y2), color, 3)
            (tw,th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_annotated, (x1,y1-th-8), (x1+tw+8,y1), color, -1)
            cv2.putText(img_annotated, lbl, (x1+4,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    return img_annotated, detections, crops_data

# ═══════════════════════════════════════════════════════════
# CHARGEMENT
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_all():
    yolo = load_yolo_model(YOLO_MODEL_PATH)
    sku_model, idx_to_class, num_classes = load_sku_model(SKU_MODEL_PATH, MAPPING_PATH)
    sku_catalog, df_catalog = load_sku_catalog(CSV_PATH)
    return yolo, sku_model, idx_to_class, num_classes, sku_catalog, df_catalog

try:
    yolo_model, sku_model, idx_to_class, num_classes, sku_catalog, df_catalog = load_all()
    models_ready = True
except Exception as e:
    st.error(f"❌ Erreur: {e}")
    models_ready = False
    st.stop()

# ═══════════════════════════════════════════════════════════
# SIDEBAR (même style que vanilla JS)
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
        <div style="width:38px;height:38px;background:linear-gradient(135deg,#6c63ff,#8b83ff);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;box-shadow:0 4px 12px rgba(108,99,255,0.3);">🏪</div>
        <div>
            <div style="font-size:15px;font-weight:800;color:#1a1d2e;">SKU Pipeline</div>
            <div style="font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:0.06em;">YOLO · MobileNetV3</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.success(f"✅ Prêt · {num_classes} SKU")
    
    st.markdown("---")
    st.markdown("### ⚙️ Seuils")
    conf_threshold = st.slider("Seuil YOLO", 0.10, 0.95, 0.45, 0.05, help="Filtre les détections")
    display_threshold = st.slider("Seuil affichage", 0.10, 0.95, 0.25, 0.05, help="Affiche le nom du produit")
    upscale_factor = st.slider("Agrandissement", 1.0, 4.0, 2.5, 0.5, help="Agrandit les petits crops")
    
    st.markdown("---")
    st.markdown("### 📊 Pipeline")
    st.markdown(f"""
    <div style="background:#f0f2f7;border:1px solid #e2e6ef;border-radius:8px;padding:12px;font-size:11px;font-family:'DM Mono',monospace;">
        <div style="display:flex;gap:8px;padding:5px 0;border-bottom:1px solid #e2e6ef;">
            <span class="badge-yolo">Stage 1</span> YOLO — famille
        </div>
        <div style="display:flex;gap:8px;padding:5px 0;border-bottom:1px solid #e2e6ef;">
            <span class="badge-sku">Stage 2</span> MobileNetV3 — SKU
        </div>
        <div style="display:flex;gap:8px;padding:5px 0;">
            <span class="badge-inv">Inventaire</span> Stage 2 uniquement
        </div>
        <div style="font-size:10px;padding-top:7px;">
            Classes <strong style="color:#6c63ff;">{num_classes}</strong> · Images <strong style="color:#6c63ff;">{len(st.session_state.sessions)}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📁 Modèles")
    st.code(f"YOLO: {YOLO_MODEL_PATH}\nSKU: {SKU_MODEL_PATH}\nLabels: {MAPPING_PATH}\nCatalog: {CSV_PATH}", language=None)

# ═══════════════════════════════════════════════════════════
# MAIN TABS (comme vanilla JS)
# ═══════════════════════════════════════════════════════════
st.markdown("""
<h1 style="font-size:26px;font-weight:800;background:linear-gradient(90deg,#1a1d2e 0%,#6c63ff 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px;">
    SKU Recognition Pipeline
</h1>
<p style="font-size:13px;color:#6b7280;margin-bottom:20px;">YOLO + MobileNetV3 · Cliquez sur une détection pour voir les détails</p>
""", unsafe_allow_html=True)

tab_upload, tab_camera, tab_inventory = st.tabs(["📸 Upload d'image", "📷 Prendre une photo", "📦 Inventaire"])

# ═══════════════════════════════════════════════════════════
# TAB 1: UPLOAD (comme vanilla JS)
# ═══════════════════════════════════════════════════════════
with tab_upload:
    uploaded_file = st.file_uploader("Déposez une image de rayon", type=["jpg","jpeg","png"],
                                     help="JPG, JPEG ou PNG · Glissez-déposez ou parcourez",
                                     label_visibility="collapsed")
    
    if uploaded_file is not None:
        img_bytes = uploaded_file.read()
        
        with st.spinner("🔍 Analyse en cours..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Chargement image…")
            progress_bar.progress(8)
            
            status_text.text("YOLO…")
            progress_bar.progress(22)
            
            img_out, detections, crops_data = run_pipeline(
                img_bytes, yolo_model, sku_model, idx_to_class, sku_catalog,
                conf_threshold, upscale_factor, display_threshold
            )
            
            progress_bar.progress(92)
            status_text.text("Rendu…")
            
            # Save session
            session = {
                "id": len(st.session_state.sessions),
                "name": uploaded_file.name.rsplit('.',1)[0][:18],
                "detections": detections,
                "crops_data": crops_data,
                "img_bytes": img_bytes,
                "annotated_bytes": cv2.imencode('.jpg', img_out)[1].tobytes()
            }
            st.session_state.sessions.append(session)
            st.session_state.active_session = len(st.session_state.sessions) - 1
            
            progress_bar.progress(100)
            status_text.text("✅ Terminé !")
            progress_bar.empty()
            status_text.empty()
        
        # Résultats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📸 Image originale**")
            st.image(uploaded_file, use_container_width=True)
        with col2:
            st.markdown("**🎯 Résultat annoté**")
            st.image(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        if crops_data:
            st.markdown("---")
            
            # Métriques
            high = len([c for c in crops_data if c['conf'] > 0.7])
            med = len([c for c in crops_data if 0.4 < c['conf'] <= 0.7])
            low = len([c for c in crops_data if 0.2 < c['conf'] <= 0.4])
            vlow = len([c for c in crops_data if c['conf'] <= 0.2])
            
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total", len(crops_data))
            m2.metric(">70%", high)
            m3.metric("40-70%", med)
            m4.metric("20-40%", low)
            m5.metric("<20%", vlow)
            
            st.markdown("---")
            st.markdown("### 🔍 Détail des détections")
            
            sorted_crops = sorted(crops_data, key=lambda x: x['conf'], reverse=True)
            for i, c in enumerate(sorted_crops):
                emoji = "🟢" if c['conf']>.7 else "🟡" if c['conf']>.4 else "🟠" if c['conf']>.2 else "🔴"
                with st.expander(f"{emoji} {c['nom']} ({c['conf']:.1%})"):
                    ca, cb = st.columns([1,2])
                    with ca:
                        st.image(c['crop_bytes'], width=120)
                    with cb:
                        st.markdown(f"**Stage 1:** {c['stage1']}")
                        st.markdown(f"**SKU:** `{c['sku']}`")
                        st.markdown(f"**Produit:** {c['nom']}")
                        st.markdown(f"**Marque:** {c['brand']} | **Capacité:** {c['capacity']}")
                        st.markdown(f"**Emballage:** {c['emballage']} | **Saveur:** {c['saveur']}")
                        st.progress(min(c['conf'], 1.0))
                        if c['top5']:
                            st.markdown("**🏆 Top-5:**")
                            for j, (sku, conf) in enumerate(c['top5'][:5]):
                                st.markdown(f"{j+1}. `{sku}` — {conf:.1%}")
            
            # Tableau
            st.markdown("---")
            st.markdown("### 📋 Rapport complet")
            if detections:
                df = pd.DataFrame(detections)
                cols = ["nom_produit","brand","capacity","emballage","saveur","famille","confiance_sku","confiance_detection","sku"]
                df = df[[c for c in cols if c in df.columns]]
                st.dataframe(df, use_container_width=True, height=300)
                
                c1, c2 = st.columns(2)
                c1.download_button("📥 CSV", df.to_csv(index=False), f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", use_container_width=True)
                c2.download_button("📥 JSON", json.dumps(detections, indent=2), f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", use_container_width=True)
        else:
            st.info("Aucune détection trouvée.")
    
    # Session nav
    if st.session_state.sessions:
        st.markdown("---")
        st.markdown("**📑 Sessions précédentes**")
        cols = st.columns(min(len(st.session_state.sessions), 6))
        for i, sess in enumerate(st.session_state.sessions):
            with cols[i % 6]:
                if st.button(f"📷 {sess['name']}", key=f"sess_{i}", use_container_width=True,
                            type="primary" if i == st.session_state.active_session else "secondary"):
                    st.session_state.active_session = i
                    st.rerun()

# ═══════════════════════════════════════════════════════════
# TAB 2: CAMERA
# ═══════════════════════════════════════════════════════════
with tab_camera:
    st.info("📱 Utilisez l'appareil photo ou la webcam")
    camera_image = st.camera_input("📸 Cadrez et prenez la photo", label_visibility="collapsed")
    
    if camera_image is not None:
        img_bytes = camera_image.getvalue()
        
        with st.spinner("🔍 Analyse en cours..."):
            img_out, detections, crops_data = run_pipeline(
                img_bytes, yolo_model, sku_model, idx_to_class, sku_catalog,
                conf_threshold, upscale_factor, display_threshold
            )
            st.session_state.sessions.append({
                "id": len(st.session_state.sessions),
                "name": f"Photo_{datetime.now().strftime('%H%M%S')}",
                "detections": detections, "crops_data": crops_data,
                "img_bytes": img_bytes,
                "annotated_bytes": cv2.imencode('.jpg', img_out)[1].tobytes()
            })
            st.session_state.active_session = len(st.session_state.sessions) - 1
        
        col1, col2 = st.columns(2)
        with col1: st.image(camera_image, caption="📸 Photo capturée", use_container_width=True)
        with col2: st.image(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB), caption="🎯 Résultat annoté", use_container_width=True)
        
        if crops_data:
            st.metric("Détections", len(crops_data))
            for c in sorted(crops_data, key=lambda x: x['conf'], reverse=True)[:5]:
                st.markdown(f"• **{c['nom']}** ({c['conf']:.1%}) — `{c['sku']}`")

# ═══════════════════════════════════════════════════════════
# TAB 3: INVENTAIRE (comme vanilla JS)
# ═══════════════════════════════════════════════════════════
with tab_inventory:
    inv_mode = st.radio("Mode", ["📁 Upload", "📷 Caméra"], horizontal=True, label_visibility="collapsed")
    
    if inv_mode == "📁 Upload":
        inv_files = st.file_uploader("Scannez un ou plusieurs produits", type=["jpg","jpeg","png"],
                                     accept_multiple_files=True, help="Stage 2 uniquement — MobileNetV3",
                                     label_visibility="collapsed")
        
        if inv_files:
            for f in inv_files:
                if f not in [p.get('file') for p in st.session_state.inventory_pending]:
                    img_bytes = f.read()
                    with st.spinner(f"Classification de {f.name}..."):
                        skus, confs = predict_single_sku(img_bytes, sku_model, idx_to_class)
                        info = sku_catalog.get(skus[0], {})
                        thumb = io.BytesIO()
                        Image.open(io.BytesIO(img_bytes)).resize((120,120)).save(thumb, format='JPEG')
                        st.session_state.inventory_pending.append({
                            "id": st.session_state.next_id,
                            "file": f,
                            "sku": skus[0],
                            "nom": info.get('product_name', skus[0]),
                            "conf": confs[0],
                            "brand": info.get('brand','N/A'),
                            "capacity": info.get('capacity','N/A'),
                            "emballage": info.get('emballage','N/A'),
                            "saveur": info.get('saveur','N/A'),
                            "thumb": thumb.getvalue(),
                            "top5": list(zip(skus, confs))
                        })
                        st.session_state.next_id += 1
            st.rerun()
    else:
        cam_inv = st.camera_input("📸 Scanner un produit", label_visibility="collapsed")
        if cam_inv:
            img_bytes = cam_inv.getvalue()
            with st.spinner("Classification..."):
                skus, confs = predict_single_sku(img_bytes, sku_model, idx_to_class)
                info = sku_catalog.get(skus[0], {})
                thumb = io.BytesIO()
                Image.open(io.BytesIO(img_bytes)).resize((120,120)).save(thumb, format='JPEG')
                st.session_state.inventory_pending.append({
                    "id": st.session_state.next_id,
                    "file": None,
                    "sku": skus[0],
                    "nom": info.get('product_name', skus[0]),
                    "conf": confs[0],
                    "brand": info.get('brand','N/A'),
                    "capacity": info.get('capacity','N/A'),
                    "emballage": info.get('emballage','N/A'),
                    "saveur": info.get('saveur','N/A'),
                    "thumb": thumb.getvalue(),
                    "top5": list(zip(skus, confs))
                })
                st.session_state.next_id += 1
                st.success(f"✅ {info.get('product_name', skus[0])} ajouté")
            st.rerun()
    
    # Pending items
    if st.session_state.inventory_pending:
        st.markdown("---")
        st.markdown("### 📋 En attente")
        c1, c2 = st.columns(2)
        if c1.button("➕ Ajouter tous les produits", use_container_width=True):
            st.session_state.inventory_validated.extend(st.session_state.inventory_pending)
            st.session_state.inventory_pending = []
            st.rerun()
        if c2.button("🗑 Supprimer tous", use_container_width=True):
            st.session_state.inventory_pending = []
            st.rerun()
        
        cols = st.columns(3)
        for i, item in enumerate(st.session_state.inventory_pending):
            with cols[i % 3]:
                st.image(item['thumb'], width=100)
                st.markdown(f"**{item['nom']}**")
                st.caption(f"`{item['sku']}`")
                conf_color = "green" if item['conf']>.7 else "orange" if item['conf']>.4 else "red"
                st.markdown(f":{conf_color}[{item['conf']:.1%}]")
                ca, cb = st.columns(2)
                if ca.button("➕", key=f"add_{item['id']}", use_container_width=True):
                    st.session_state.inventory_validated.append(item)
                    st.session_state.inventory_pending.pop(i)
                    st.rerun()
                if cb.button("🗑", key=f"del_{item['id']}", use_container_width=True):
                    st.session_state.inventory_pending.pop(i)
                    st.rerun()
    
    # Validated inventory
    st.markdown("---")
    st.markdown("### 📋 Inventaire validé")
    
    if st.session_state.inventory_validated:
        if st.button("🗑 Tout effacer", use_container_width=True):
            st.session_state.inventory_validated = []
            st.rerun()
        
        df_inv = pd.DataFrame([{
            "Produit": i['nom'], "SKU": i['sku'], "Marque": i['brand'],
            "Capacité": i['capacity'], "Emballage": i['emballage'],
            "Saveur": i['saveur'], "Confiance": f"{i['conf']:.1%}"
        } for i in st.session_state.inventory_validated])
        st.dataframe(df_inv, use_container_width=True, height=300)
    else:
        st.info("Aucun produit dans l'inventaire")