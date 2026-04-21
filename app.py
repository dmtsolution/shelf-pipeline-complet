import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import time
import io
import os
from datetime import datetime
from ultralytics import YOLO

# ==============================
# CONFIGURATION
# ==============================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
# TRANSFORMS
# ==============================
val_transforms = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==============================
# CHARGEMENT DES MODÈLES
# ==============================
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
    else:
        return {}, pd.DataFrame()

# ==============================
# FONCTIONS (PIL uniquement - pas de cv2)
# ==============================
def upscale_image_pil(img_pil, scale_factor=2.0):
    w, h = img_pil.size
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    return img_pil.resize((new_w, new_h), Image.LANCZOS)

def prepare_crop_for_model_pil(crop_pil, target_size=224, upscale_first=True):
    if upscale_first and min(crop_pil.size) < 100:
        crop_pil = upscale_image_pil(crop_pil, scale_factor=2.5)
    
    w, h = crop_pil.size
    ratio = w / h
    
    if ratio > 1:
        new_w = target_size
        new_h = int(target_size / ratio)
    else:
        new_h = target_size
        new_w = int(target_size * ratio)
    
    crop_resized = crop_pil.resize((new_w, new_h), Image.LANCZOS)
    
    square = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    square.paste(crop_resized, (x_offset, y_offset))
    
    return square

def predict_sku_with_upscale(model, crop_pil, idx_to_class, upscale_factor=2.0):
    h, w = crop_pil.size
    if h < 150 or w < 150:
        crop_pil = upscale_image_pil(crop_pil, scale_factor=upscale_factor)
    
    crop_prepared = prepare_crop_for_model_pil(crop_pil, IMG_SIZE)
    img_t = val_transforms(crop_prepared).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        out = model(img_t)
        probs = torch.softmax(out, dim=1)[0]
        topk = probs.topk(min(5, len(idx_to_class)))
    
    skus = [idx_to_class[i.item()] for i in topk.indices]
    confs = [v.item() for v in topk.values]
    
    return skus, confs

def draw_bbox_on_image(img_pil, bbox, label, color_rgb):
    """Dessine un rectangle sur l'image PIL (sans cv2)"""
    draw = ImageDraw.Draw(img_pil)
    x1, y1, x2, y2 = bbox
    
    # Rectangle
    draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)
    
    # Texte
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Position du texte
    bbox_text = draw.textbbox((x1, y1), label, font=font)
    text_w = bbox_text[2] - bbox_text[0]
    text_h = bbox_text[3] - bbox_text[1]
    
    # Fond du texte
    draw.rectangle([x1, y1 - text_h - 8, x1 + text_w + 8, y1], fill=(0, 0, 0))
    draw.text((x1 + 4, y1 - 6), label, fill=color_rgb, font=font)
    
    return img_pil

# ==============================
# PIPELINE COMPLET
# ==============================
def run_pipeline(image_bytes, yolo_model, sku_model, idx_to_class, sku_catalog, 
                conf_threshold=0.5, upscale_factor=2.5, display_threshold=0.10):
    # Charger l'image
    img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_original = img_pil.copy()
    
    # YOLO détection (nécessite numpy mais pas cv2)
    img_np = np.array(img_pil)
    results = yolo_model(img_np, conf=conf_threshold, verbose=False)
    
    detections = []
    crops_data = []
    img_annotated = img_original.copy()
    
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_det = float(box.conf[0])
            cls_id = int(box.cls[0])
            famille = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"classe_{cls_id}"
            
            # Extraire le crop
            crop_pil = img_pil.crop((x1, y1, x2, y2))
            
            if crop_pil.size[0] < 10 or crop_pil.size[1] < 10:
                skus = ["inconnu"]
                confs = [0.0]
                product_name = "Crop trop petit"
                top5 = []
                crop_for_display = crop_pil
            else:
                skus, confs = predict_sku_with_upscale(sku_model, crop_pil, idx_to_class, upscale_factor)
                product_info = sku_catalog.get(skus[0], {})
                product_name = product_info.get('product_name', skus[0])
                top5 = list(zip(skus, confs))
                crop_for_display = prepare_crop_for_model_pil(crop_pil, IMG_SIZE, upscale_first=True)
            
            # Sauvegarde du crop
            crop_bytes = io.BytesIO()
            crop_for_display.save(crop_bytes, format='JPEG')
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
                "crop_original_size": crop_pil.size,
                "stage1": f"{famille} (conf: {conf_det:.2f})",
                "stage2_sku": skus[0],
                "stage2_nom": product_name,
                "stage2_conf": confs[0],
                "top5": top5,
                "brand": product_full_info.get('brand', 'N/A'),
                "capacity": product_full_info.get('capacity', 'N/A'),
                "emballage": product_full_info.get('emballage', 'N/A'),
                "saveur": product_full_info.get('saveur', 'N/A')
            })
            
            # Couleur selon confiance
            if confs[0] > 0.7:
                color = (0, 255, 0)
            elif confs[0] > 0.4:
                color = (255, 255, 0)
            elif confs[0] > 0.2:
                color = (255, 165, 0)
            else:
                color = (255, 0, 0)
            
            label = f"{product_name} ({confs[0]:.1%})" if confs[0] > display_threshold else famille
            img_annotated = draw_bbox_on_image(img_annotated, (x1, y1, x2, y2), label, color)
    
    return img_annotated, detections, crops_data

# ==============================
# UI STREAMLIT
# ==============================
st.set_page_config(
    page_title="SKU Recognition Pipeline",
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
    .stMetric {
        background: white;
        padding: 0.5rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏪 SKU Recognition Pipeline</h1>
    <p>Détection YOLO + Classification MobileNetV3 | 120 classes SKU</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    conf_threshold = st.slider(
        "Seuil détection YOLO", 
        0.10, 0.95, 0.45, 0.05,
        help="Filtre les détections en dessous de ce seuil"
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

# Chargement
with st.spinner("🚀 Chargement des modèles..."):
    try:
        yolo_model = load_yolo_model(YOLO_MODEL_PATH)
        sku_model, idx_to_class, num_classes = load_sku_model(SKU_MODEL_PATH, MAPPING_PATH)
        sku_catalog, df_catalog = load_sku_catalog(CSV_PATH)
        st.success(f"✅ Modèles chargés avec succès!")
        st.info(f"📊 {num_classes} classes SKU disponibles | 📦 {len(sku_catalog)} produits")
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        st.stop()

# Upload
uploaded_file = st.file_uploader(
    "📸 Déposez une image de rayon",
    type=["jpg", "jpeg", "png"],
    help="JPG, JPEG ou PNG"
)

if uploaded_file is not None:
    with st.spinner("🔍 Analyse en cours..."):
        img_bytes = uploaded_file.read()
        img_out, detections, crops_data = run_pipeline(
            img_bytes, yolo_model, sku_model, idx_to_class, sku_catalog, 
            conf_threshold, upscale_factor, display_threshold
        )
    
    # Affichage
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="📸 Image originale", use_container_width=True)
    with col2:
        st.image(img_out, caption="🎯 Résultat annoté", use_container_width=True)
    
    # Statistiques
    if crops_data:
        st.markdown("---")
        st.markdown("## 📊 Résultats")
        
        high = len([c for c in crops_data if c['stage2_conf'] > 0.7])
        medium = len([c for c in crops_data if 0.4 < c['stage2_conf'] <= 0.7])
        low = len([c for c in crops_data if 0.2 < c['stage2_conf'] <= 0.4])
        very_low = len([c for c in crops_data if c['stage2_conf'] <= 0.2])
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("📦 Total", len(crops_data))
        m2.metric("🟢 >70%", high)
        m3.metric("🟡 40-70%", medium)
        m4.metric("🟠 20-40%", low)
        m5.metric("🔴 <20%", very_low)
        
        # Tableau complet
        st.markdown("---")
        st.markdown("## 📋 Détail des détections")
        
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
            "📥 Télécharger les résultats (CSV)",
            data=csv_data,
            file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )