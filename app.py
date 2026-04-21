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
# FONCTIONS D'AMÉLIORATION
# ==============================
def upscale_crop(crop, scale_factor=2.0):
    h, w = crop.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    upscaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return upscaled

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

def predict_sku_with_upscale(model, crop_image, idx_to_class, upscale_factor=2.0):
    if isinstance(crop_image, np.ndarray):
        h, w = crop_image.shape[:2]
        if h < 150 or w < 150:
            new_h, new_w = int(h * upscale_factor), int(w * upscale_factor)
            crop_upscaled = cv2.resize(crop_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            crop_upscaled = crop_image
        
        crop_prepared = prepare_crop_for_model(crop_upscaled, IMG_SIZE)
        crop_pil = Image.fromarray(crop_prepared)
        img_t = val_transforms(crop_pil).unsqueeze(0).to(DEVICE)
    else:
        img_t = val_transforms(crop_image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        out = model(img_t)
        probs = torch.softmax(out, dim=1)[0]
        topk = probs.topk(min(5, len(idx_to_class)))
    
    skus = [idx_to_class[i.item()] for i in topk.indices]
    confs = [v.item() for v in topk.values]
    
    return skus, confs

# ==============================
# PIPELINE COMPLET
# ==============================
def run_pipeline(image_bytes, yolo_model, sku_model, idx_to_class, sku_catalog, 
                conf_threshold=0.5, upscale_factor=2.5, display_threshold=0.10):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img_rgb.shape[:2]
    if max(h, w) < 640:
        scale = 640 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    results = yolo_model(img_rgb, conf=conf_threshold, verbose=False)
    
    detections = []
    crops_data = []
    img_annotated = img.copy()
    
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
                skus, confs = predict_sku_with_upscale(sku_model, crop_original, idx_to_class, upscale_factor)
                product_info = sku_catalog.get(skus[0], {})
                product_name = product_info.get('product_name', skus[0])
                top5 = list(zip(skus, confs))
                crop_for_display = prepare_crop_for_model(crop_original, IMG_SIZE, upscale_first=True)
            
            crop_display = Image.fromarray(crop_for_display)
            crop_bytes = io.BytesIO()
            crop_display.save(crop_bytes, format='JPEG')
            crop_bytes = crop_bytes.getvalue()
            
            # Récupérer les infos complètes du produit
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
                "capacity": product_full_info.get('capacity', 'N/A'),
                "emballage": product_full_info.get('emballage', 'N/A'),
                "saveur": product_full_info.get('saveur', 'N/A')
            })
            
            # Couleur selon confiance (affichage sur image)
            if confs[0] > 0.7:
                color = (0, 255, 0)
            elif confs[0] > 0.4:
                color = (0, 255, 255)
            elif confs[0] > 0.2:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)
            
            # Afficher le NOM DU PRODUIT (stage2) sur l'image, pas la catégorie
            label = f"{product_name} ({confs[0]:.1%})" if confs[0] > display_threshold else famille
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img_annotated, (x1, y1 - text_h - 8), (x1 + text_w + 8, y1), (0, 0, 0), -1)
            cv2.putText(img_annotated, label, (x1 + 4, y1 - 6),
                        font, font_scale, (255, 255, 255), thickness)
    
    return img_annotated, detections, crops_data

# ==============================
# UI STREAMLIT
# ==============================
st.set_page_config(
    page_title="SKU Recognition Pipeline",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS professionnel
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .detection-table {
        margin-top: 2rem;
        background: white;
        border-radius: 10px;
        padding: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🏪 SKU Recognition Pipeline</h1>
    <p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem;">Détection YOLO + Classification MobileNetV3</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    st.markdown("### 🎯 Seuils")
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
    
    st.markdown("### 🔬 Amélioration")
    upscale_factor = st.slider(
        "Facteur d'agrandissement", 
        1.0, 4.0, 2.5, 0.5,
        help="Agrandit les petits crops pour meilleure reconnaissance"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Pipeline")
    st.info("""
    **Stage 1:** YOLO (détection famille)  
    **Stage 2:** MobileNetV3 (identification SKU)  
    **Classes:** 120 SKU  
    """)
    
    st.markdown("---")
    st.markdown("### 📁 Fichiers")
    st.code(f"""
    YOLO: {YOLO_MODEL_PATH}
    SKU: {SKU_MODEL_PATH}
    Labels: {MAPPING_PATH}
    """)

# Chargement des modèles
with st.spinner("🚀 Chargement des modèles..."):
    try:
        yolo_model = load_yolo_model(YOLO_MODEL_PATH)
        sku_model, idx_to_class, num_classes = load_sku_model(SKU_MODEL_PATH, MAPPING_PATH)
        sku_catalog, df_catalog = load_sku_catalog(CSV_PATH)
        
        st.success(f"✅ Modèles chargés avec succès!")
        st.info(f"📊 {num_classes} classes SKU disponibles | 📦 {len(sku_catalog)} produits dans le catalogue")
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")
        st.stop()

# Upload
uploaded_file = st.file_uploader(
    "📸 Déposez une image de rayon",
    type=["jpg", "jpeg", "png"],
    help="JPG, JPEG ou PNG - Max 200MB"
)

if uploaded_file is not None:
    with st.spinner("🔍 Analyse en cours..."):
        img_bytes = uploaded_file.read()
        img_out, detections, crops_data = run_pipeline(
            img_bytes, yolo_model, sku_model, idx_to_class, sku_catalog, 
            conf_threshold, upscale_factor, display_threshold
        )
    
    # Affichage des résultats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📸 Image originale")
        st.image(uploaded_file, use_container_width=True)
    with col2:
        st.markdown("### 🎯 Résultat annoté")
        st.image(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # Statistiques
    if crops_data:
        st.markdown("---")
        st.markdown("## 📈 Analyse des détections")
        
        high_conf = len([c for c in crops_data if c['stage2_conf'] > 0.7])
        medium_conf = len([c for c in crops_data if 0.4 < c['stage2_conf'] <= 0.7])
        low_conf = len([c for c in crops_data if 0.2 < c['stage2_conf'] <= 0.4])
        very_low_conf = len([c for c in crops_data if c['stage2_conf'] <= 0.2])
        
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("📦 Total détections", len(crops_data))
        with m2:
            st.metric("✅ >70%", high_conf)
        with m3:
            st.metric("⚠️ 40-70%", medium_conf)
        with m4:
            st.metric("🟠 20-40%", low_conf)
        with m5:
            st.metric("🔴 <20%", very_low_conf)
        
        # Détails des détections (Top 10 par confiance)
        st.markdown("---")
        st.markdown("## 🔍 Détail des détections")
        
        crops_data_sorted = sorted(crops_data, key=lambda x: x['stage2_conf'], reverse=True)
        
        for i, crop_data in enumerate(crops_data_sorted):
            if crop_data['stage2_conf'] > 0.7:
                icon = "🟢"
            elif crop_data['stage2_conf'] > 0.4:
                icon = "🟡"
            elif crop_data['stage2_conf'] > 0.2:
                icon = "🟠"
            else:
                icon = "🔴"
            
            with st.expander(f"{icon} Détection #{i+1} - {crop_data['stage2_nom']} ({crop_data['stage2_conf']:.1%})"):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.image(crop_data["crop_bytes"], caption="Crop préparé", use_container_width=True)
                with col_b:
                    st.markdown(f"**🎯 Stage 1 (YOLO):** {crop_data['stage1']}")
                    st.markdown(f"**🏷️ SKU:** `{crop_data['stage2_sku']}`")
                    st.markdown(f"**📦 Produit:** {crop_data['stage2_nom']}")
                    st.markdown(f"**🏭 Marque:** {crop_data['brand']}")
                    st.markdown(f"**📏 Capacité:** {crop_data['capacity']}")
                    st.markdown(f"**🥤 Emballage:** {crop_data['emballage']}")
                    st.markdown(f"**🍓 Saveur:** {crop_data['saveur']}")
                    st.markdown(f"**📊 Confiance SKU:** {crop_data['stage2_conf']:.1%}")
                    st.progress(crop_data['stage2_conf'])
                    
                    if crop_data['top5']:
                        st.markdown("**🏆 Top-5 prédictions:**")
                        for j, (sku, conf) in enumerate(crop_data['top5'][:5]):
                            st.markdown(f"{j+1}. `{sku}` - {conf:.1%}")
    
    # ==============================
    # TABLEAU COMPLET (TOUTES DÉTECTIONS)
    # ==============================
    st.markdown("---")
    st.markdown("## 📋 Rapport complet des détections")
    
    if detections:
        df_results = pd.DataFrame(detections)
        
        # Colonnes à afficher
        display_cols = ["nom_produit", "brand", "capacity", "emballage", "saveur", 
                       "famille", "confiance_sku", "confiance_detection", "sku"]
        
        available_cols = [col for col in display_cols if col in df_results.columns]
        df_display = df_results[available_cols]
        
        # Renommer les colonnes
        column_names = {
            "nom_produit": "Produit",
            "brand": "Marque",
            "capacity": "Capacité",
            "emballage": "Emballage",
            "saveur": "Saveur",
            "famille": "Famille",
            "confiance_sku": "Confiance SKU",
            "confiance_detection": "Confiance Détection",
            "sku": "SKU ID"
        }
        df_display = df_display.rename(columns=column_names)
        
        # Formatage des pourcentages
        if "Confiance SKU" in df_display.columns:
            df_display["Confiance SKU"] = df_display["Confiance SKU"].apply(lambda x: f"{x:.1%}")
        if "Confiance Détection" in df_display.columns:
            df_display["Confiance Détection"] = df_display["Confiance Détection"].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(df_display, use_container_width=True, height=400)
        
        # Export
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv_data = df_results.to_csv(index=False)
            st.download_button(
                "📥 Télécharger CSV",
                data=csv_data,
                file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_exp2:
            json_data = json.dumps(detections, indent=2, ensure_ascii=False, default=str)
            st.download_button(
                "📥 Télécharger JSON",
                data=json_data,
                file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    else:
        st.info("Aucune détection trouvée dans cette image")

else:
    st.info("👈 Déposez une image pour commencer l'analyse")