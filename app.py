import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import timm
import os
import time
import pandas as pd
import json
import io
from ultralytics import YOLO
from datetime import datetime

# ==============================
# CONFIGURATION
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chemins des modèles
YOLO_MODEL_PATH = "best-yolov8s.pt"
SKU_MODEL_PATH = "best-mobilenetv3large.pth"
MAPPING_PATH = "label_map.json"
CSV_PATH = "sku_catalog.csv"
IMG_SIZE = 224

# Classes YOLO
CLASS_NAMES = [
    'boisson_energetique', 'dessert', 'eau', 'fromage',
    'jus', 'lait', 'soda', 'yaourt'
]

# ==============================
# TRANSFORMS (exactement comme stage 2)
# ==============================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transform():
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# ==============================
# CHARGEMENT DES MODÈLES
# ==============================
@st.cache_resource
def load_yolo_model():
    return YOLO(YOLO_MODEL_PATH)

@st.cache_resource
def load_sku_model():
    with open(MAPPING_PATH, 'r') as f:
        label_map = json.load(f)
    idx_to_class = {int(k): v for k, v in label_map.items()}
    nc = len(idx_to_class)
    
    model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=nc)
    if hasattr(model, 'classifier'):
        model.classifier = nn.Linear(model.classifier.in_features, nc)
    
    state_dict = torch.load(SKU_MODEL_PATH, map_location='cpu')
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model, idx_to_class, nc

@st.cache_data
def load_sku_catalog():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        return df.set_index("sku_id").to_dict("index"), df
    else:
        return {}, pd.DataFrame()

# ==============================
# FONCTIONS D'AGRANDISSEMENT
# ==============================
def upscale_crop(crop, scale_factor=2.0):
    h, w = crop.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

def prepare_crop_for_model(crop, target_size=224, upscale_first=True):
    if upscale_first and crop.shape[0] < 100 and crop.shape[1] < 100:
        crop = upscale_crop(crop, scale_factor=2.5)
    
    h, w = crop.shape[:2]
    ratio = w / h
    
    if ratio > 1:
        new_w = target_size
        new_h = int(target_size / ratio)
    else:
        new_h = target_size
        new_w = int(target_size * ratio)
    
    crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    square = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = crop_resized
    
    return square

def predict_sku_with_upscale(model, crop_image, idx_to_class, transform, upscale_factor=2.0):
    if isinstance(crop_image, np.ndarray):
        h, w = crop_image.shape[:2]
        if h < 150 or w < 150:
            new_h, new_w = int(h * upscale_factor), int(w * upscale_factor)
            crop_upscaled = cv2.resize(crop_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            crop_upscaled = crop_image
        
        crop_prepared = prepare_crop_for_model(crop_upscaled, IMG_SIZE)
        crop_pil = Image.fromarray(crop_prepared)
        img_t = transform(crop_pil).unsqueeze(0).to(DEVICE)
    else:
        img_t = transform(crop_image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        out = model(img_t)
        probs = torch.softmax(out, dim=1)[0]
        topk = probs.topk(min(5, len(idx_to_class)))
    
    skus = [idx_to_class[i.item()] for i in topk.indices]
    confs = [v.item() for v in topk.values]
    
    return skus, confs

# ==============================
# DESSIN SUR IMAGE (style stage 2)
# ==============================
def draw_detections(img_np, detections, display_threshold=0.25):
    img = img_np.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confiance_sku']
        product_name = det['nom_produit']
        
        if conf > 0.7:
            color = (0, 255, 0)
        elif conf > 0.4:
            color = (0, 255, 255)
        elif conf > 0.2:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)
        
        # Rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Texte (nom du produit, pas la catégorie)
        if conf > display_threshold:
            text = f"{product_name} ({conf:.1%})"
        else:
            text = det['famille']
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Fond du texte
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), (0, 0, 0), -1)
        cv2.putText(img, text, (x1 + 4, y1 - 6), font, font_scale, color, thickness)
    
    return img

# ==============================
# PIPELINE COMPLET
# ==============================
def run_pipeline(image_bytes, yolo_model, sku_model, idx_to_class, sku_catalog, 
                transform, conf_threshold=0.5, upscale_factor=2.5, display_threshold=0.25):
    # Charger l'image
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Agrandir si nécessaire pour YOLO
    h, w = img_rgb.shape[:2]
    if max(h, w) < 640:
        scale = 640 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # YOLO détection
    results = yolo_model(img_rgb, conf=conf_threshold, verbose=False)
    
    detections = []
    crops_data = []
    
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_det = float(box.conf[0])
            cls_id = int(box.cls[0])
            famille = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"classe_{cls_id}"
            
            crop_original = img_rgb[y1:y2, x1:x2]
            
            if crop_original.shape[0] < 10 or crop_original.shape[1] < 10:
                skus = ["inconnu"]
                confs = [0.0]
                product_name = "Crop trop petit"
                top5 = []
                crop_for_display = crop_original
            else:
                skus, confs = predict_sku_with_upscale(sku_model, crop_original, idx_to_class, transform, upscale_factor)
                product_info = sku_catalog.get(skus[0], {})
                product_name = product_info.get('product_name', skus[0])
                top5 = list(zip(skus, confs))
                crop_for_display = prepare_crop_for_model(crop_original, IMG_SIZE, upscale_first=True)
            
            # Sauvegarde crop
            crop_display = Image.fromarray(crop_for_display)
            crop_bytes = io.BytesIO()
            crop_display.save(crop_bytes, format='JPEG')
            crop_bytes = crop_bytes.getvalue()
            
            product_full_info = sku_catalog.get(skus[0], {})
            
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "famille": famille,
                "sku": skus[0],
                "nom_produit": product_name,
                "confiance_detection": round(conf_det, 3),
                "confiance_sku": round(confs[0], 3),
                "top5_predictions": top5,
                "brand": product_full_info.get('brand', 'N/A'),
                "capacity": product_full_info.get('capacity', 'N/A'),
                "emballage": product_full_info.get('emballage', 'N/A'),
                "saveur": product_full_info.get('saveur', 'N/A'),
                "category": product_full_info.get('category', 'N/A')
            })
            
            crops_data.append({
                "crop_bytes": crop_bytes,
                "crop_original_size": crop_original.shape[:2],
                "stage1": f"{famille} (conf: {conf_det:.2f})",
                "stage2_sku": skus[0],
                "stage2_nom": product_name,
                "stage2_conf": confs[0],
                "top5": top5,
                "brand": product_full_info.get('brand', 'N/A'),
                "capacity": product_full_info.get('capacity', 'N/A')
            })
    
    # Dessiner les résultats
    img_annotated = draw_detections(img, detections, display_threshold)
    
    return img_annotated, detections, crops_data

# ==============================
# UI STREAMLIT
# ==============================
st.set_page_config(
    page_title="SKU Recognition Pipeline PRO",
    page_icon="🏪",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🏪 SKU Recognition Pipeline PRO</h1>
    <p>YOLO + MobileNetV3-Large | Détection en rayon → Identification SKU</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    conf_threshold = st.slider(
        "Seuil détection YOLO", 
        0.10, 0.95, 0.45, 0.05,
        help="Filtre les détections YOLO en dessous de ce seuil"
    )
    
    display_threshold = st.slider(
        "Seuil affichage sur image", 
        0.10, 0.95, 0.25, 0.05,
        help="Affiche le nom du produit seulement si confiance > seuil"
    )
    
    upscale_factor = st.slider(
        "Facteur d'agrandissement", 
        1.0, 4.0, 2.5, 0.5,
        help="Agrandit les petits crops pour meilleure reconnaissance"
    )
    
    st.markdown("---")
    st.info("**Pipeline:**\n- Stage 1: YOLO (détection)\n- Stage 2: MobileNetV3 (identification)\n- Classes: 120 SKU")

# Chargement des modèles
with st.spinner("🚀 Chargement des modèles..."):
    try:
        yolo_model = load_yolo_model()
        sku_model, idx_to_class, num_classes = load_sku_model()
        sku_catalog, df_catalog = load_sku_catalog()
        transform = get_transform()
        
        st.success(f"✅ Modèles chargés avec succès!")
        st.info(f"📊 {num_classes} classes SKU disponibles | 📦 {len(sku_catalog)} produits dans le catalogue")
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        st.stop()

# Mode
mode = st.radio("📱 Mode d'analyse", ["📸 Upload d'image", "🎥 Webcam live"], horizontal=True)

# ==============================
# MODE UPLOAD
# ==============================
if mode == "📸 Upload d'image":
    uploaded_file = st.file_uploader("📸 Déposez une image de rayon", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        with st.spinner("🔍 Analyse en cours..."):
            img_bytes = uploaded_file.read()
            t0 = time.time()
            img_out, detections, crops_data = run_pipeline(
                img_bytes, yolo_model, sku_model, idx_to_class, sku_catalog, 
                transform, conf_threshold, upscale_factor, display_threshold
            )
            latency = (time.time() - t0) * 1000
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="📸 Image originale", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB), caption="🎯 Résultat annoté", use_container_width=True)
        
        # Métriques
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("📦 Détections", len(detections))
        c2.metric("⚡ Latence", f"{latency:.0f} ms")
        if detections:
            avg_conf = np.mean([d['confiance_sku'] for d in detections])
            c3.metric("📊 Confiance moyenne", f"{avg_conf:.1%}")
        
        if detections:
            st.markdown("---")
            st.markdown("## 📋 Tableau complet des détections")
            
            df_results = pd.DataFrame(detections)
            display_cols = ["nom_produit", "brand", "capacity", "emballage", "saveur", 
                           "confiance_sku", "confiance_detection", "sku"]
            available_cols = [c for c in display_cols if c in df_results.columns]
            df_display = df_results[available_cols].copy()
            
            df_display = df_display.rename(columns={
                "nom_produit": "Produit",
                "brand": "Marque",
                "capacity": "Capacité",
                "emballage": "Emballage",
                "saveur": "Saveur",
                "confiance_sku": "Confiance SKU",
                "confiance_detection": "Confiance Détection",
                "sku": "SKU ID"
            })
            
            df_display["Confiance SKU"] = df_display["Confiance SKU"].apply(lambda x: f"{x:.1%}")
            df_display["Confiance Détection"] = df_display["Confiance Détection"].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(df_display, use_container_width=True, height=400)
            
            # Export
            csv_data = df_results.to_csv(index=False)
            st.download_button(
                "📥 Télécharger CSV",
                data=csv_data,
                file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# ==============================
# MODE WEBCAM
# ==============================
else:
    st.info("📹 Webcam en temps réel - Détection live des produits")
    
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    results_placeholder = st.empty()
    stop_button = st.button("⏹️ Arrêter la caméra")
    
    last_detections = []
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir frame en bytes
        _, buffer = cv2.imencode('.jpg', frame)
        img_bytes = buffer.tobytes()
        
        # Analyse
        img_out, detections, _ = run_pipeline(
            img_bytes, yolo_model, sku_model, idx_to_class, sku_catalog,
            transform, conf_threshold, upscale_factor, display_threshold
        )
        
        # Affichage
        frame_placeholder.image(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB), 
                                caption="Webcam live", use_container_width=True)
        
        if detections:
            df_live = pd.DataFrame(detections)[["nom_produit", "confiance_sku", "famille"]]
            df_live = df_live.rename(columns={
                "nom_produit": "Produit",
                "confiance_sku": "Confiance",
                "famille": "Famille"
            })
            df_live["Confiance"] = df_live["Confiance"].apply(lambda x: f"{x:.1%}")
            results_placeholder.dataframe(df_live, use_container_width=True)
        else:
            results_placeholder.info("🔍 Aucun produit détecté")
    
    cap.release()
    st.success("✅ Caméra arrêtée")