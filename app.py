# app.py — SKU Recognition Pipeline · Version Finale avec Caméra HTML
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
import io, os, base64, hashlib
from datetime import datetime
from streamlit.components.v1 import html as st_html

# ═══════════════════════════════════════ CONFIG
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
YOLO_MODEL_PATH = "best-yolov8s.pt"
SKU_MODEL_PATH  = "best-mobilenetv3large.pth"
MAPPING_PATH    = "label_map.json"
CSV_PATH        = "sku_catalog.csv"
IMG_SIZE = 224
CLASS_NAMES = ['boisson_energetique','dessert','eau','fromage','jus','lait','soda','yaourt']

val_transforms = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)), transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ═══════════════════════════════════════ MODELS
@st.cache_resource
def load_yolo_model(path): return YOLO(path)

@st.cache_resource
def load_sku_model(model_path, labels_path):
    with open(labels_path) as f: label_map = json.load(f)
    idx_to_class = {int(k): v for k, v in label_map.items()}
    nc = len(idx_to_class)
    model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=nc)
    sd = torch.load(model_path, map_location='cpu')
    if any(k.startswith('module.') for k in sd): sd = {k.replace('module.',''):v for k,v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(DEVICE).eval()
    return model, idx_to_class, nc

@st.cache_data
def load_sku_catalog(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df.set_index("sku_id").to_dict("index"), df
    return {}, pd.DataFrame()

# ═══════════════════════════════════════ IMAGE HELPERS
def prepare_crop(img_np, target=224, upscale=2.5):
    h, w = img_np.shape[:2]
    if h < 150 or w < 150: img_np = cv2.resize(img_np, (int(w*upscale),int(h*upscale)), interpolation=cv2.INTER_CUBIC)
    h, w = img_np.shape[:2]
    ratio = w/h
    nw2, nh2 = (target,int(target/ratio)) if ratio>1 else (int(target*ratio),target)
    r = cv2.resize(img_np, (nw2,nh2), interpolation=cv2.INTER_LANCZOS4)
    sq = np.full((target,target,3), 128, dtype=np.uint8)
    yo, xo = (target-nh2)//2, (target-nw2)//2
    sq[yo:yo+nh2, xo:xo+nw2] = r
    return sq

def predict_sku(model, img_np, idx_to_class, upscale=2.5):
    prep = prepare_crop(img_np, IMG_SIZE, upscale)
    t = val_transforms(Image.fromarray(prep)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0]
        topk = probs.topk(min(5, len(idx_to_class)))
    return [idx_to_class[i.item()] for i in topk.indices], [v.item() for v in topk.values]

def np_to_bytes(img_np):
    buf = io.BytesIO(); Image.fromarray(img_np).save(buf, format='JPEG', quality=88); return buf.getvalue()

def b64(data): return base64.b64encode(data).decode()

# ═══════════════════════════════════════ PIPELINES
def run_pipeline(image_bytes, yolo_model, sku_model, idx_to_class, sku_catalog, conf_thr=0.45, upscale=2.5, disp_thr=0.25):
    arr = np.frombuffer(image_bytes, np.uint8)
    img_rgb = cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h,w = img_rgb.shape[:2]
    if max(h,w) < 640:
        sc = 640/max(h,w); img_rgb = cv2.resize(img_rgb, (int(w*sc),int(h*sc)), interpolation=cv2.INTER_CUBIC)
    annotated = img_rgb.copy()
    detections, crops_data = [], []
    for r in yolo_model(img_rgb, conf=conf_thr, verbose=False):
        if r.boxes is None: continue
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            conf_det = float(box.conf[0]); cls_id = int(box.cls[0])
            famille = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
            crop = img_rgb[y1:y2, x1:x2]
            if crop.shape[0]<10 or crop.shape[1]<10: skus,confs = ["inconnu"],[0.0]
            else: skus,confs = predict_sku(sku_model, crop, idx_to_class, upscale)
            info = sku_catalog.get(skus[0], {}); nom = info.get('product_name', skus[0])
            crop_bytes = np_to_bytes(prepare_crop(crop, IMG_SIZE, upscale))
            if confs[0]>0.7: col=(5,150,105)
            elif confs[0]>0.4: col=(217,119,6)
            elif confs[0]>0.2: col=(234,88,12)
            else: col=(220,38,38)
            label = f"{nom} ({confs[0]:.1%})" if confs[0]>disp_thr else famille
            (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            cv2.rectangle(annotated,(x1,y1),(x2,y2),col,2)
            cv2.rectangle(annotated,(x1,y1-th-8),(x1+tw+8,y1),(15,23,42),-1)
            cv2.putText(annotated,label,(x1+4,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1)
            detections.append({"bbox":[x1,y1,x2,y2],"famille":famille,"sku":skus[0],"nom_produit":nom,
                "confiance_detection":round(conf_det,3),"confiance_sku":round(confs[0],3),
                "top5_predictions":list(zip(skus,confs)),"brand":info.get('brand','N/A'),
                "capacity":info.get('capacity','N/A'),"emballage":info.get('emballage','N/A'),
                "saveur":info.get('saveur','N/A'),"category":info.get('category','N/A')})
            crops_data.append({"crop_bytes":crop_bytes,"stage1":f"{famille} ({conf_det:.2f})",
                "stage2_sku":skus[0],"stage2_nom":nom,"stage2_conf":confs[0],
                "top5":list(zip(skus[:5],confs[:5])),"brand":info.get('brand','N/A'),
                "capacity":info.get('capacity','N/A'),"emballage":info.get('emballage','N/A'),
                "saveur":info.get('saveur','N/A')})
    return annotated, detections, crops_data

def run_inventory_pipeline(image_bytes, sku_model, idx_to_class, sku_catalog, upscale=2.5):
    arr = np.frombuffer(image_bytes, np.uint8)
    im_rgb = cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    skus,confs = predict_sku(sku_model, im_rgb, idx_to_class, upscale)
    info = sku_catalog.get(skus[0], {})
    return {"sku":skus[0],"nom":info.get('product_name',skus[0]),"conf":round(confs[0],3),
        "top5":list(zip(skus[:5],confs[:5])),"brand":info.get('brand','N/A'),
        "capacity":info.get('capacity','N/A'),"emballage":info.get('emballage','N/A'),
        "saveur":info.get('saveur','N/A'),"crop_bytes":np_to_bytes(prepare_crop(im_rgb, IMG_SIZE, upscale))}

# ═══════════════════════════════════════ HTML COMPONENTS
def conf_cls(c):
    if c > 0.7: return "ch"
    if c > 0.4: return "cm"
    if c > 0.2: return "cl"
    return "cv"

def cam_widget_html():
    """HTML component that mimics vanilla JS camera: back camera, start/capture/stop buttons."""
    return """
    <!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
    *{margin:0;padding:0;box-sizing:border-box}body{font-family:'Inter',sans-serif;background:#F4F5F8}
    .cam-box{background:#fff;border:1px solid #E2E6EF;border-radius:14px;padding:20px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.06)}
    video{width:100%;max-width:640px;border-radius:8px;background:#000;display:none;margin:0 auto 12px}
    canvas{display:none}
    .cam-ctrls{display:flex;gap:10px;justify-content:center;flex-wrap:wrap}
    button{padding:9px 20px;border-radius:9px;border:none;font-size:13px;font-weight:600;cursor:pointer;transition:all .2s;display:inline-flex;align-items:center;gap:6px}
    .btn-start{background:linear-gradient(135deg,#6C63FF,#8B83FF);color:#fff;box-shadow:0 2px 8px rgba(108,99,255,.3)}
    .btn-capture{background:#fff;color:#6B7280;border:1px solid #E2E6EF}
    .btn-stop{background:#fff;color:#6B7280;border:1px solid #E2E6EF}
    button:hover:not(:disabled){transform:translateY(-1px);opacity:.9}
    button:disabled{opacity:.4;cursor:not-allowed}
    </style></head><body>
    <div class="cam-box">
      <p style="font-size:13px;color:#6B7280;margin-bottom:12px">📱 Appareil photo (caméra arrière)</p>
      <video id="cam-video" autoplay playsinline></video>
      <canvas id="cam-canvas"></canvas>
      <div class="cam-ctrls">
        <button class="btn-start" id="btn-start" onclick="startCam()">▶ Démarrer</button>
        <button class="btn-capture" id="btn-capture" onclick="capture()" disabled>📸 Capturer</button>
        <button class="btn-stop" id="btn-stop" onclick="stopCam()" disabled>⏹ Arrêter</button>
      </div>
      <div id="cam-result" style="margin-top:8px"></div>
    </div>
    <script>
    let stream=null;
    async function startCam(){
      try{
        stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'},audio:false});
        const v=document.getElementById('cam-video');v.srcObject=stream;v.style.display='block';
        document.getElementById('btn-capture').disabled=false;
        document.getElementById('btn-stop').disabled=false;
        document.getElementById('btn-start').disabled=true;
      }catch(e){alert('Caméra inaccessible: '+e.message);}
    }
    function capture(){
      const v=document.getElementById('cam-video');
      const c=document.getElementById('cam-canvas');
      c.width=v.videoWidth;c.height=v.videoHeight;
      c.getContext('2d').drawImage(v,0,0);
      const data=c.toDataURL('image/jpeg',0.9);
      // Send to Streamlit via custom event
      window.parent.postMessage({type:'streamlit:setComponentValue',value:data},'*');
      document.getElementById('cam-result').innerHTML='<span style="color:#10B981;font-size:12px">✅ Photo capturée</span>';
    }
    function stopCam(){
      if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
      document.getElementById('cam-video').style.display='none';
      document.getElementById('btn-capture').disabled=true;
      document.getElementById('btn-stop').disabled=true;
      document.getElementById('btn-start').disabled=false;
    }
    </script></body></html>"""

def render_annotated_image_with_modal(raw_bytes, annotated_np, crops_data, detections):
    ann_bytes = np_to_bytes(annotated_np); ann_h, ann_w = annotated_np.shape[:2]
    bbox_zones = ""
    for i, (det, cd) in enumerate(zip(detections, crops_data)):
        x1,y1,x2,y2 = det['bbox']
        lp = x1/ann_w*100; tp = y1/ann_h*100; wp = (x2-x1)/ann_w*100; hp = (y2-y1)/ann_h*100
        cc = conf_cls(cd['stage2_conf'])
        bbox_zones += f'<div class="bbox-z {cc}" style="left:{lp:.2f}%;top:{tp:.2f}%;width:{wp:.2f}%;height:{hp:.2f}%;" onclick="event.stopPropagation();openModal({i})" title="{cd["stage2_nom"]}"></div>'
    panel_data = [{"img":f"data:image/jpeg;base64,{b64(cd['crop_bytes'])}","nom":cd['stage2_nom'],"sku":cd['stage2_sku'],
        "conf":round(cd['stage2_conf'],3),"stage1":cd['stage1'],"brand":cd['brand'],"capacity":cd['capacity'],
        "emballage":cd['emballage'],"saveur":cd['saveur'],"cc":conf_cls(cd['stage2_conf'])} for cd in crops_data]
    panel_data_json = json.dumps(panel_data, ensure_ascii=False)
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
    *{{margin:0;padding:0;box-sizing:border-box}}body{{background:#F4F5F8;font-family:'Inter',sans-serif}}
    .result-images{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
    .result-card{{background:#fff;border:1px solid #E2E6EF;border-radius:14px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
    .result-card-header{{padding:10px 16px;border-bottom:1px solid #E2E6EF;font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;letter-spacing:.06em}}
    .result-card-header span{{font-weight:500;color:#6C63FF;text-transform:none}}
    .img-container{{position:relative;line-height:0;cursor:crosshair}}
    .img-container img{{width:100%;display:block}}
    .bbox-z{{position:absolute;border:2px solid transparent;border-radius:3px;cursor:pointer;transition:all .15s}}
    .bbox-z:hover{{background:rgba(255,255,255,.2);box-shadow:0 0 0 4px rgba(108,99,255,.25)}}
    .bbox-z.ch{{border-color:rgba(16,185,129,.9)}}.bbox-z.cm{{border-color:rgba(245,158,11,.9)}}
    .bbox-z.cl{{border-color:rgba(249,115,22,.9)}}.bbox-z.cv{{border-color:rgba(239,68,68,.9)}}
    .modal-overlay{{display:none;position:fixed;inset:0;z-index:99999;background:rgba(26,29,46,.5);backdrop-filter:blur(7px);align-items:center;justify-content:center}}
    .modal-overlay.open{{display:flex}}
    .modal{{background:#fff;border-radius:20px;width:520px;max-width:94vw;max-height:90vh;overflow-y:auto;box-shadow:0 8px 32px rgba(0,0,0,.12),0 0 0 1px #E2E6EF;transition:transform .25s}}
    .modal-header{{padding:18px 22px 14px;border-bottom:1px solid #E2E6EF;display:flex;align-items:flex-start;justify-content:space-between;gap:14px;position:sticky;top:0;background:#fff;z-index:1;border-radius:20px 20px 0 0}}
    .modal-title{{font-size:18px;font-weight:800;color:#1A1D2E}}
    .modal-subtitle{{font-size:12px;color:#6B7280;font-family:'JetBrains Mono',monospace;margin-top:2px}}
    .modal-close{{width:32px;height:32px;border-radius:8px;background:#F0F2F7;border:1px solid #E2E6EF;cursor:pointer;font-size:15px;color:#6B7280;display:flex;align-items:center;justify-content:center;transition:all .15s;flex-shrink:0}}
    .modal-close:hover{{background:#EF4444;color:#fff;border-color:#EF4444}}
    .modal-body{{padding:20px 22px;display:flex;flex-direction:column;gap:16px}}
    .modal-crop-wrap{{display:flex;justify-content:center;background:#F0F2F7;border-radius:8px;padding:14px;border:1px solid #E2E6EF}}
    .modal-crop-img{{max-height:160px;max-width:100%;border-radius:6px;object-fit:contain}}
    .modal-conf-row{{display:flex;align-items:center;gap:16px}}
    .modal-conf-value{{font-family:'JetBrains Mono',monospace;font-size:34px;font-weight:800;flex-shrink:0}}
    .modal-conf-bar{{flex:1}}
    .modal-conf-label{{font-size:11px;color:#6B7280;font-family:'JetBrains Mono',monospace;margin-bottom:6px}}
    .modal-conf-track{{height:8px;background:#E2E6EF;border-radius:4px;overflow:hidden}}
    .modal-conf-fill{{height:100%;border-radius:4px;transition:width .6s ease}}
    .modal-fields{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
    .modal-field{{background:#F0F2F7;border:1px solid #E2E6EF;border-radius:8px;padding:10px 12px}}
    .modal-field-key{{font-size:10px;color:#6B7280;font-family:'JetBrains Mono',monospace;letter-spacing:.06em;text-transform:uppercase;margin-bottom:3px}}
    .modal-field-val{{font-size:13px;font-weight:700;color:#1A1D2E}}
    .modal-top5{{background:#F0F2F7;border:1px solid #E2E6EF;border-radius:8px;padding:12px 14px}}
    .modal-top5-title{{font-size:11px;font-weight:700;color:#6B7280;letter-spacing:.07em;text-transform:uppercase;margin-bottom:8px}}
    .modal-top5-item{{display:flex;align-items:center;gap:8px;padding:5px 0;font-family:'JetBrains Mono',monospace;font-size:12px;border-bottom:1px solid #E2E6EF}}
    .modal-top5-item:last-child{{border-bottom:none}}
    .modal-top5-rank{{color:#9CA3AF;width:18px}}.modal-top5-sku{{flex:1;color:#6C63FF}}
    .modal-top5-bar{{width:80px;height:4px;background:#E2E6EF;border-radius:2px;overflow:hidden}}
    .modal-top5-fill{{height:100%;border-radius:2px;background:#6C63FF}}.modal-top5-conf{{color:#6B7280;width:44px;text-align:right}}
    @media(max-width:860px){{.result-images{{grid-template-columns:1fr}}.modal-fields{{grid-template-columns:1fr}}}}
    </style></head><body>
    <div class="result-images">
      <div class="result-card"><div class="result-card-header">📸 Image originale</div><img src="data:image/jpeg;base64,{b64(raw_bytes)}" style="width:100%;display:block"></div>
      <div class="result-card"><div class="result-card-header">🎯 Résultat annoté <span>· cliquer sur un produit</span></div><div class="img-container"><img src="data:image/jpeg;base64,{b64(ann_bytes)}">{bbox_zones}</div></div>
    </div>
    <div class="modal-overlay" id="modalOverlay" onclick="closeModal()"><div class="modal" onclick="event.stopPropagation()">
      <div class="modal-header"><div><div class="modal-title" id="modalTitle">—</div><div class="modal-subtitle" id="modalSubtitle">—</div></div><button class="modal-close" onclick="closeModal()">✕</button></div>
      <div class="modal-body"><div class="modal-crop-wrap"><img class="modal-crop-img" id="modalCrop" src=""></div>
      <div class="modal-conf-row"><div class="modal-conf-value" id="modalConf">—</div><div class="modal-conf-bar"><div class="modal-conf-label">Confiance SKU</div><div class="modal-conf-track"><div class="modal-conf-fill" id="modalConfFill" style="width:0%"></div></div></div></div>
      <div class="modal-fields" id="modalFields"></div><div class="modal-top5"><div class="modal-top5-title">🏆 Top-5 prédictions</div><div id="modalTop5"></div></div></div>
    </div></div>
    <script>
    var DATA={panel_data_json};var CC={{ch:'#10B981',cm:'#F59E0B',cl:'#F97316',cv:'#EF4444'}};
    function openModal(i){{var d=DATA[i];var col=CC[d.cc]||'#6C63FF';var pct=(d.conf*100).toFixed(1)+'%';
    document.getElementById('modalTitle').textContent=d.nom;document.getElementById('modalSubtitle').textContent='SKU: '+d.sku+'  ·  '+d.stage1;
    document.getElementById('modalCrop').src=d.img;document.getElementById('modalConf').textContent=pct;document.getElementById('modalConf').style.color=col;
    document.getElementById('modalConfFill').style.width=pct;document.getElementById('modalConfFill').style.background=col;
    var fields=[['Marque',d.brand],['Capacité',d.capacity],['Emballage',d.emballage],['Saveur',d.saveur],['Stage 1',d.stage1]];
    var fHTML='';for(var f=0;f<fields.length;f++){{fHTML+='<div class="modal-field"><div class="modal-field-key">'+fields[f][0]+'</div><div class="modal-field-val">'+fields[f][1]+'</div></div>'}}
    document.getElementById('modalFields').innerHTML=fHTML;
    var t5HTML='';for(var t=0;t<d.top5.length;t++){{var tp=(d.top5[t].conf*100).toFixed(1)+'%';t5HTML+='<div class="modal-top5-item"><span class="modal-top5-rank">'+(t+1)+'.</span><span class="modal-top5-sku">'+d.top5[t].sku+'</span><div class="modal-top5-bar"><div class="modal-top5-fill" style="width:'+tp+'"></div></div><span class="modal-top5-conf">'+tp+'</span></div>'}}
    document.getElementById('modalTop5').innerHTML=t5HTML;document.getElementById('modalOverlay').classList.add('open')}}
    function closeModal(){{document.getElementById('modalOverlay').classList.remove('open')}}
    document.addEventListener('keydown',function(e){{if(e.key==='Escape')closeModal()}})
    </script></body></html>"""

def render_metrics(total,hi,me,lo,vl):
    return f"""<div class="metrics-row">
      <div class="metric metric-total"><div class="metric-val">{total}</div><div class="metric-lbl">Total</div></div>
      <div class="metric metric-high"><div class="metric-val">{hi}</div><div class="metric-lbl">&gt;70%</div></div>
      <div class="metric metric-med"><div class="metric-val">{me}</div><div class="metric-lbl">40–70%</div></div>
      <div class="metric metric-low"><div class="metric-val">{lo}</div><div class="metric-lbl">20–40%</div></div>
      <div class="metric metric-vlow"><div class="metric-val">{vl}</div><div class="metric-lbl">&lt;20%</div></div></div>"""

def render_detection_card(cd):
    cc=conf_cls(cd['stage2_conf']);conf=cd['stage2_conf'];img=f"data:image/jpeg;base64,{b64(cd['crop_bytes'])}"
    brand_pill=f'<span class="pill pill-b">{cd["brand"]}</span>' if cd['brand']!='N/A' else ''
    nom=cd['stage2_nom'].replace("'","\\'").replace('"','&quot;')
    t5="".join(f'<div class="top5-item"><span class="top5-i">{j+1}.</span><span class="top5-s">{s}</span><span class="top5-c">{c:.1%}</span></div>' for j,(s,c) in enumerate(cd['top5'][:5]))
    return f"""<div class="det-card {cc}" onclick="this.classList.toggle('open')">
      <div class="det-row"><img class="det-thumb" src="{img}"><div class="det-body"><div class="det-name">{nom}</div>
      <div class="det-pills"><span class="pill pill-f">{cd['stage1'].split('(')[0].strip()}</span><span class="pill pill-s">{cd['stage2_sku']}</span>{brand_pill}</div></div>
      <div class="det-pct">{conf:.0%}</div><div class="det-chevron">&#9658;</div></div>
      <div class="det-expand"><div class="det-expand-inner"><img class="det-large-img" src="{img}"><div class="det-fields">
        <div class="det-field"><span class="det-fk">Stage 1</span><span class="det-fv">{cd['stage1']}</span></div>
        <div class="det-field"><span class="det-fk">SKU</span><span class="det-fv" style="color:var(--blue)">{cd['stage2_sku']}</span></div>
        <div class="det-field"><span class="det-fk">Produit</span><span class="det-fv">{nom}</span></div>
        <div class="det-field"><span class="det-fk">Marque</span><span class="det-fv">{cd['brand']}</span></div>
        <div class="det-field"><span class="det-fk">Capacité</span><span class="det-fv">{cd['capacity']}</span></div>
        <div class="det-field"><span class="det-fk">Emballage</span><span class="det-fv">{cd['emballage']}</span></div>
        <div class="det-field"><span class="det-fk">Saveur</span><span class="det-fv">{cd['saveur']}</span></div>
        <div class="conf-bar-wrap {cc}"><div class="conf-bar-lbl">Confiance · {conf:.1%}</div><div class="conf-bar-track"><div class="conf-bar-fill" style="width:{conf*100:.1f}%"></div></div></div>
        <div class="top5"><div class="top5-title">🏆 Top-5</div>{t5}</div></div></div></div></div>"""

def render_inv_card(item):
    conf=item['conf'];col="var(--green)" if conf>0.7 else "var(--amber)" if conf>0.4 else "var(--orange)" if conf>0.2 else "var(--red)"
    return f"""<div class="inv-card"><img class="inv-card-img" src="data:image/jpeg;base64,{b64(item['crop_bytes'])}"><div class="inv-card-name" title="{item['nom']}">{item['nom']}</div><div class="inv-card-sku">{item['sku']}</div><div class="inv-card-conf" style="color:{col}">{conf:.1%}</div></div>"""

# ═══════════════════════════════════════ PAGE
st.set_page_config(page_title="SKU Pipeline",page_icon="🏪",layout="wide",initial_sidebar_state="expanded")

# CSS
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
:root{--bg:#F4F5F8;--surface:#FFFFFF;--surface-2:#F0F2F7;--blue:#6C63FF;--blue-dark:#5A52E0;--green:#10B981;--amber:#F59E0B;--orange:#F97316;--red:#EF4444;--text:#1A1D2E;--text-muted:#6B7280;--border:#E2E6EF;--r:10px;--sh:0 1px 3px rgba(0,0,0,.06)}
*,*::before,*::after{box-sizing:border-box}*:focus,*:focus-visible{outline:none!important;box-shadow:none!important}
.stApp{background:var(--bg)!important}.main .block-container{padding-top:2px!important}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important}
[data-testid="stSidebar"] [data-baseweb="slider-track"]{background:#E2E6EF!important}
[data-testid="stSidebar"] [data-baseweb="slider-track-fill"]{background:var(--blue)!important}
[data-testid="stSidebar"] [role="slider"]{background:var(--blue)!important;border:none!important}
.stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:var(--r)!important;padding:4px!important;width:fit-content!important}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,var(--blue),var(--blue-dark))!important;color:#fff!important}
.stButton>button{font-size:12px!important;font-weight:600!important;border-radius:8px!important}
[data-testid="baseButton-primary"]{background:linear-gradient(135deg,var(--blue),var(--blue-dark))!important;color:#fff!important}
[data-testid="stFileUploaderDropzone"]{border:2px dashed #D0D5E3!important;border-radius:14px!important;padding:32px 24px!important}
[data-testid="stFileUploaderDropzone"] small{display:none!important}
[data-testid="stProgressBar"]>div>div{background:var(--blue)!important}
hr{border:none!important;border-top:1px solid var(--border)!important;margin:14px 0!important}
.sb-logo{display:flex;align-items:center;gap:12px;padding:16px 14px 14px;border-bottom:1px solid var(--border)}
.sb-logo-mark{width:36px;height:36px;background:linear-gradient(135deg,var(--blue),var(--blue-dark));border-radius:10px;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 12px rgba(108,99,255,.3);font-size:18px}
.sb-logo-title{font-size:15px;font-weight:800;color:var(--text)!important}
.status-pill{display:inline-flex;align-items:center;gap:7px;padding:8px 12px;border-radius:8px;font-size:11px;font-family:'JetBrains Mono',monospace}.s-ok{background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.25);color:#10B981!important}
.sb-section{font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--text-muted)!important;padding:8px 0}
.sb-stage{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:8px;background:var(--surface-2, #F0F2F7);border:1px solid var(--border)}
.sb-stage-num{width:22px;height:22px;border-radius:4px;background:var(--blue);color:#fff!important;font-size:10px;font-weight:700;display:flex;align-items:center;justify-content:center}
.sb-file-row{display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid var(--border);font-size:10px;font-family:'JetBrains Mono',monospace}
.page-header{padding:4px 0 14px;border-bottom:1px solid var(--border);margin-bottom:16px}
.page-header-title{font-size:22px;font-weight:800;background:linear-gradient(90deg,var(--text),var(--blue));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.metrics-row{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:18px}
.metric{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:14px 10px;text-align:center}
.metric-val{font-size:22px;font-weight:800;font-family:'JetBrains Mono',monospace}.metric-total .metric-val{color:var(--text)}.metric-high .metric-val{color:var(--green)}.metric-med .metric-val{color:var(--amber)}.metric-low .metric-val{color:var(--orange)}.metric-vlow .metric-val{color:var(--red)}
.section-title{font-size:13px;font-weight:800;color:var(--text);margin:0 0 10px}.section-title span{font-weight:400;color:var(--text-muted)}
.detections-stack{display:flex;flex-direction:column;gap:6px}
.det-card{background:var(--surface);border:1px solid var(--border);border-left:4px solid transparent;border-radius:var(--r);overflow:hidden;cursor:pointer;transition:all .15s}
.det-card.ch{border-left-color:var(--green)}.det-card.cm{border-left-color:var(--amber)}.det-card.cl{border-left-color:var(--orange)}.det-card.cv{border-left-color:var(--red)}
.det-row{display:flex;align-items:center;gap:12px;padding:11px 14px}.det-thumb{width:46px;height:46px;border-radius:6px;object-fit:cover;background:var(--surface-2,#F0F2F7);border:1px solid var(--border)}
.det-name{font-size:12px;font-weight:700;color:var(--text)}.det-pct{font-family:'JetBrains Mono',monospace;font-size:15px;font-weight:800}.ch .det-pct{color:var(--green)}.cm .det-pct{color:var(--amber)}.cl .det-pct{color:var(--orange)}.cv .det-pct{color:var(--red)}
.pill{padding:2px 6px;border-radius:4px;font-size:9px;font-weight:600;font-family:'JetBrains Mono',monospace}.pill-f{background:rgba(245,158,11,.1);color:var(--amber)}.pill-s{background:rgba(108,99,255,.08);color:var(--blue)}.pill-b{background:rgba(16,185,129,.08);color:var(--green)}
.det-expand{display:none;padding:0 14px 14px;border-top:1px solid var(--border)}.det-card.open .det-expand{display:block}.det-card.open .det-chevron{transform:rotate(90deg)}
.det-expand-inner{display:grid;grid-template-columns:96px 1fr;gap:12px;padding-top:12px}.det-large-img{width:96px;height:96px;object-fit:contain;border-radius:6px;background:var(--surface-2);border:1px solid var(--border);padding:4px}
.det-field{display:flex;gap:6px;font-size:11px;padding:2px 0}.det-fk{color:var(--text-muted);width:68px;font-family:'JetBrains Mono',monospace;font-size:10px}.det-fv{color:var(--text);font-weight:600}
.conf-bar-track{height:5px;background:#E2E6EF;border-radius:2px;overflow:hidden}.conf-bar-fill{height:100%}.ch .conf-bar-fill{background:var(--green)}.cm .conf-bar-fill{background:var(--amber)}.cl .conf-bar-fill{background:var(--orange)}.cv .conf-bar-fill{background:var(--red)}
.tbl-wrap{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:auto;max-height:320px;box-shadow:var(--sh, 0 1px 3px rgba(0,0,0,.06))}
.tbl-wrap table{width:100%;border-collapse:collapse;font-size:11px}.tbl-wrap th{padding:8px 12px;text-align:left;font-size:9px;font-weight:700;text-transform:uppercase;color:var(--text-muted);background:var(--surface-2);border-bottom:1px solid var(--border);position:sticky;top:0}
.tbl-wrap td{padding:7px 12px;border-bottom:1px solid rgba(226,230,239,.5);font-family:'JetBrains Mono',monospace;color:var(--text-muted)}.td-name{color:var(--text)!important;font-family:'Inter',sans-serif!important;font-size:11px!important;font-weight:700!important}.td-sku{color:var(--blue)!important}
.inv-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:10px;box-shadow:var(--sh, 0 1px 3px rgba(0,0,0,.06));margin-bottom:8px}.inv-card-img{width:100%;height:110px;object-fit:contain;background:var(--surface-2);border-radius:6px;margin-bottom:8px;border:1px solid var(--border);display:block}
.inv-card-name{font-size:11px;font-weight:600;color:var(--text)}.inv-card-sku{font-size:9px;color:var(--text-muted);font-family:'JetBrains Mono',monospace}.inv-card-conf{font-size:10px;font-weight:700}
.empty-state{padding:36px 24px;text-align:center;background:var(--surface);border:1.5px dashed #D0D5E3;border-radius:14px}
@media(max-width:860px){.metrics-row{grid-template-columns:repeat(3,1fr)}.det-expand-inner{grid-template-columns:1fr}}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════ SESSION STATE
for k,v in {'sessions':[],'active_sess_id':None,'proc_hashes':set(),'upload_key':0,
            'cam_photo':None,'inv_pending':[],'inv_validated':[],'inv_next_id':1,
            'inv_upload_key':0,'inv_cam_photo':None}.items():
    if k not in st.session_state: st.session_state[k]=v

# ═══════════════════════════════════════ LOAD MODELS
models_ok=False
try:
    yolo_model=load_yolo_model(YOLO_MODEL_PATH)
    sku_model,idx_to_class,num_classes=load_sku_model(SKU_MODEL_PATH,MAPPING_PATH)
    sku_catalog,_=load_sku_catalog(CSV_PATH)
    models_ok=True
except: pass

# ═══════════════════════════════════════ SIDEBAR
with st.sidebar:
    st.markdown(f"""<div class="sb-logo"><div class="sb-logo-mark">🏪</div><div><div class="sb-logo-title">SKU Pipeline</div><div style="font-size:10px;color:#6B7280">YOLO · MobileNetV3</div></div></div>
    <div class="status-pill s-ok"><span style="width:7px;height:7px;border-radius:50%;background:#10B981"></span>{'Prêt · '+str(num_classes)+' SKU' if models_ok else 'Erreur'}</div>
    <hr class="sb-divider"><div class="sb-section">⚙ Seuils</div>""",unsafe_allow_html=True)
    conf_threshold=st.slider("Seuil YOLO",0.10,0.95,0.45,0.05)
    display_threshold=st.slider("Seuil affichage",0.10,0.95,0.25,0.05)
    upscale_factor=st.slider("Agrandissement",1.0,4.0,2.5,0.5)
    st.markdown(f"""<hr class="sb-divider"><div class="sb-section">📊 Pipeline</div>
    <div class="sb-stage"><div class="sb-stage-num">1</div><div><div style="font-size:11px;font-weight:600">YOLO</div><div style="font-size:10px;color:#6B7280">Détection famille</div></div></div>
    <div class="sb-stage" style="margin-top:4px"><div class="sb-stage-num">2</div><div><div style="font-size:11px;font-weight:600">MobileNetV3</div><div style="font-size:10px;color:#6B7280">Classification SKU</div></div></div>
    <hr class="sb-divider"><div class="sb-section">📁 Modèles</div>
    <div class="sb-file-row"><span style="color:#6C63FF;font-weight:600;width:46px">YOLO</span><span>{YOLO_MODEL_PATH}</span></div>
    <div class="sb-file-row"><span style="color:#6C63FF;font-weight:600;width:46px">SKU</span><span>{SKU_MODEL_PATH}</span></div>
    <div class="sb-file-row"><span style="color:#6C63FF;font-weight:600;width:46px">Labels</span><span>{MAPPING_PATH}</span></div>
    <div class="sb-file-row" style="border-bottom:none"><span style="color:#6C63FF;font-weight:600;width:46px">Catalog</span><span>{CSV_PATH}</span></div>""",unsafe_allow_html=True)

# ═══════════════════════════════════════ HEADER
st.markdown('<div class="page-header"><div class="page-header-title">SKU Recognition Pipeline</div><div style="font-size:12px;color:#6B7280">YOLO + MobileNetV3 · Détection et identification de produits</div></div>',unsafe_allow_html=True)

# ═══════════════════════════════════════ TABS
tab_up,tab_cam,tab_inv=st.tabs(["📸 Upload","📷 Caméra","📦 Inventaire"])

# ╔══════════════════════ UPLOAD
with tab_up:
    uploaded=st.file_uploader("Déposez une image de rayon · JPG, JPEG ou PNG",type=["jpg","jpeg","png"],key=f"up_{st.session_state.upload_key}")
    if uploaded and models_ok:
        raw=uploaded.getvalue();hh=hashlib.md5(raw[:2048]+uploaded.name.encode()).hexdigest()
        if hh not in st.session_state.proc_hashes:
            with st.spinner("Analyse..."): ann,dets,crops=run_pipeline(raw,yolo_model,sku_model,idx_to_class,sku_catalog,conf_threshold,upscale_factor,display_threshold)
            st.session_state.sessions.append({'id':len(st.session_state.sessions)+1,'name':uploaded.name[:22],'raw':raw,'ann':ann,'dets':dets,'crops':crops})
            st.session_state.active_sess_id=st.session_state.sessions[-1]['id'];st.session_state.proc_hashes.add(hh);st.session_state.upload_key+=1;st.rerun()
    if st.session_state.sessions:
        n=min(len(st.session_state.sessions),8);cols=st.columns(n+1 if n<8 else 9);to_remove=None
        for i,sess in enumerate(st.session_state.sessions[:n]):
            a=sess['id']==st.session_state.active_sess_id
            with cols[i]:
                if st.button(f"📷 {sess['name']}",key=f"sn_{sess['id']}",type="primary" if a else "secondary",use_container_width=True):st.session_state.active_sess_id=sess['id'];st.rerun()
        rm_cols=st.columns(n)
        for i,sess in enumerate(st.session_state.sessions[:n]):
            with rm_cols[i]:
                if st.button("✕",key=f"rm_{sess['id']}",use_container_width=True):to_remove=sess['id']
        if to_remove:
            st.session_state.sessions=[s for s in st.session_state.sessions if s['id']!=to_remove]
            st.session_state.active_sess_id=st.session_state.sessions[-1]['id'] if st.session_state.sessions else None;st.rerun()
        active_sess=next((s for s in st.session_state.sessions if s['id']==st.session_state.active_sess_id),None)
        if active_sess:
            html=render_annotated_image_with_modal(active_sess['raw'],active_sess['ann'],active_sess['crops'],active_sess['dets'])
            st_html(html,height=max(480,active_sess['ann'].shape[0]//2+200),scrolling=False)
            cd=active_sess['crops'];hi=sum(1 for c in cd if c['stage2_conf']>0.7);me=sum(1 for c in cd if 0.4<c['stage2_conf']<=0.7);lo=sum(1 for c in cd if 0.2<c['stage2_conf']<=0.4);vl=sum(1 for c in cd if c['stage2_conf']<=0.2)
            st.markdown(render_metrics(len(cd),hi,me,lo,vl),unsafe_allow_html=True)
            sc=sorted(cd,key=lambda x:x['stage2_conf'],reverse=True)
            cards="".join(render_detection_card(c) for c in sc)
            st.markdown(f'<div class="section-title">🔍 Détections <span>· {len(cd)} produit{"s" if len(cd)!=1 else ""}</span></div><div class="detections-stack">{cards}</div>',unsafe_allow_html=True)
            rows=""
            for d in active_sess['dets']:
                col="var(--green)" if d['confiance_sku']>0.7 else "var(--amber)" if d['confiance_sku']>0.4 else "var(--red)"
                rows+=f'<tr><td class="td-name">{d["nom_produit"]}</td><td>{d["brand"]}</td><td>{d["capacity"]}</td><td>{d["emballage"]}</td><td>{d["saveur"]}</td><td>{d["famille"]}</td><td style="color:{col};font-weight:700">{d["confiance_sku"]:.1%}</td><td>{d["confiance_detection"]:.1%}</td><td class="td-sku">{d["sku"]}</td></tr>'
            st.markdown(f'<div style="margin-top:20px"><div class="section-title">📋 Rapport complet</div><div class="tbl-wrap"><table><thead><tr><th>Produit</th><th>Marque</th><th>Capacité</th><th>Emballage</th><th>Saveur</th><th>Famille</th><th>Conf.SKU</th><th>Conf.Dét.</th><th>SKU ID</th></tr></thead><tbody>{rows}</tbody></table></div></div>',unsafe_allow_html=True)
            ts=datetime.now().strftime('%Y%m%d_%H%M%S');df=pd.DataFrame(active_sess['dets'])
            ec1,ec2=st.columns(2)
            ec1.download_button("📥 CSV",df.to_csv(index=False),f"detections_{ts}.csv","text/csv",use_container_width=True)
            ec2.download_button("📥 JSON",json.dumps(active_sess['dets'],indent=2,ensure_ascii=False,default=str),f"detections_{ts}.json","application/json",use_container_width=True)

# ╔══════════════════════ CAMERA (HTML widget - back camera ONLY)
with tab_cam:
    # HTML camera widget (identical to vanilla JS behavior)
    cam_value = st_html(cam_widget_html(), height=420, scrolling=False)
    
    # st_html returns the base64 data URL from the JS postMessage
    if cam_value and cam_value.startswith("data:image/jpeg;base64,"):
        raw = base64.b64decode(cam_value.split(",",1)[1])
        if raw and models_ok:
            with st.spinner("Analyse..."):
                ann,dets,crops = run_pipeline(raw, yolo_model, sku_model, idx_to_class, sku_catalog, conf_threshold, upscale_factor, display_threshold)
            st.session_state.cam_result = {'raw':raw,'ann':ann,'dets':dets,'crops':crops}
    
    if 'cam_result' in st.session_state and st.session_state.cam_result:
        r = st.session_state.cam_result
        html = render_annotated_image_with_modal(r['raw'],r['ann'],r['crops'],r['dets'])
        st_html(html, height=max(480,r['ann'].shape[0]//2+200), scrolling=False)
        cd = r['crops'];hi=sum(1 for c in cd if c['stage2_conf']>0.7);me=sum(1 for c in cd if 0.4<c['stage2_conf']<=0.7);lo=sum(1 for c in cd if 0.2<c['stage2_conf']<=0.4);vl=sum(1 for c in cd if c['stage2_conf']<=0.2)
        st.markdown(render_metrics(len(cd),hi,me,lo,vl),unsafe_allow_html=True)
        sc=sorted(cd,key=lambda x:x['stage2_conf'],reverse=True)
        cards="".join(render_detection_card(c) for c in sc)
        st.markdown(f'<div class="section-title">🔍 Détections <span>· {len(cd)} produit{"s" if len(cd)!=1 else ""}</span></div><div class="detections-stack">{cards}</div>',unsafe_allow_html=True)
        ts=datetime.now().strftime('%Y%m%d_%H%M%S');df=pd.DataFrame(r['dets'])
        ec1,ec2=st.columns(2)
        ec1.download_button("📥 CSV",df.to_csv(index=False),f"photo_{ts}.csv","text/csv",use_container_width=True)
        ec2.download_button("📥 JSON",json.dumps(r['dets'],indent=2,ensure_ascii=False,default=str),f"photo_{ts}.json","application/json",use_container_width=True)

# ╔══════════════════════ INVENTAIRE
with tab_inv:
    inv_up,inv_cam=st.tabs(["📁 Upload","📷 Caméra"])
    with inv_up:
        inv_files=st.file_uploader("Scannez un ou plusieurs produits",type=["jpg","jpeg","png"],accept_multiple_files=True,key=f"inv_up_{st.session_state.inv_upload_key}")
        if inv_files and models_ok:
            prog=st.progress(0)
            for i,f in enumerate(inv_files):
                res=run_inventory_pipeline(f.read(),sku_model,idx_to_class,sku_catalog,upscale_factor)
                res['id']=st.session_state.inv_next_id;st.session_state.inv_next_id+=1;st.session_state.inv_pending.append(res);prog.progress((i+1)/len(inv_files))
            prog.empty();st.session_state.inv_upload_key+=1;st.rerun()
    with inv_cam:
        inv_cam_val = st_html(cam_widget_html().replace('cam-video','inv-cam-video').replace('cam-canvas','inv-cam-canvas').replace('btn-start','inv-btn-start').replace('btn-capture','inv-btn-capture').replace('btn-stop','inv-btn-stop').replace('startCam()','startInvCam()').replace('capture()','captureInv()').replace('stopCam()','stopInvCam()'), height=420, scrolling=False)
        if inv_cam_val and inv_cam_val.startswith("data:image/jpeg;base64,"):
            raw = base64.b64decode(inv_cam_val.split(",",1)[1])
            if raw and models_ok:
                with st.spinner("Classification..."):
                    res=run_inventory_pipeline(raw,sku_model,idx_to_class,sku_catalog,upscale_factor)
                    res['id']=st.session_state.inv_next_id;st.session_state.inv_next_id+=1;st.session_state.inv_pending.append(res);st.rerun()

    st.markdown('<hr>',unsafe_allow_html=True)
    pending=st.session_state.inv_pending
    if pending:
        bh1,bh2=st.columns([6,2])
        with bh1:st.markdown(f'<div class="section-title">📋 En attente <span>· {len(pending)} produit{"s" if len(pending)>1 else ""}</span></div>',unsafe_allow_html=True)
        with bh2:
            c1,c2=st.columns(2)
            if c1.button("➕ Tout",use_container_width=True,type="primary",key="add_all"):st.session_state.inv_validated.extend(pending);st.session_state.inv_pending=[];st.rerun()
            if c2.button("🗑 Tout",use_container_width=True,key="del_all"):st.session_state.inv_pending=[];st.rerun()
        to_add,to_del=[],[]
        for row_items in [pending[i:i+3] for i in range(0,len(pending),3)]:
            cols=st.columns(3)
            for col_el,item in zip(cols,row_items):
                with col_el:
                    st.markdown(render_inv_card(item),unsafe_allow_html=True);st.markdown('<div style="height:4px"></div>',unsafe_allow_html=True)
                    ba,bb=st.columns(2)
                    if ba.button("➕ Ajouter",key=f"ia_{item['id']}",use_container_width=True,type="primary"):to_add.append(item['id'])
                    if bb.button("✕",key=f"id_{item['id']}",use_container_width=True):to_del.append(item['id'])
        if to_add:st.session_state.inv_pending=[x for x in st.session_state.inv_pending if x['id'] not in to_add];st.session_state.inv_validated.extend([x for x in pending if x['id'] in to_add]);st.rerun()
        if to_del:st.session_state.inv_pending=[x for x in st.session_state.inv_pending if x['id'] not in to_del];st.rerun()
    else:st.markdown('<div class="empty-state"><div class="empty-title">Aucun produit en attente</div><div style="font-size:11px;color:#6B7280">Uploadez des photos ou utilisez la caméra</div></div>',unsafe_allow_html=True)

    st.markdown('<hr>',unsafe_allow_html=True)
    validated=st.session_state.inv_validated
    vh1,vh2=st.columns([6,2])
    with vh1:st.markdown(f'<div class="section-title">✅ Inventaire validé <span>· {len(validated)} produit{"s" if len(validated)!=1 else ""}</span></div>',unsafe_allow_html=True)
    with vh2:
        if validated and st.button("🗑 Effacer",use_container_width=True,key="clear_inv"):st.session_state.inv_validated=[];st.rerun()
    if validated:
        rows=""
        for it in validated:
            col="var(--green)" if it['conf']>0.7 else "var(--amber)" if it['conf']>0.4 else "var(--orange)" if it['conf']>0.2 else "var(--red)"
            rows+=f'<tr><td class="td-name">{it["nom"]}</td><td class="td-sku">{it["sku"]}</td><td>{it["brand"]}</td><td>{it["capacity"]}</td><td>{it["emballage"]}</td><td>{it["saveur"]}</td><td style="color:{col};font-weight:700">{it["conf"]:.1%}</td></tr>'
        st.markdown(f'<div class="tbl-wrap"><table><thead><tr><th>Produit</th><th>SKU</th><th>Marque</th><th>Capacité</th><th>Emballage</th><th>Saveur</th><th>Confiance</th></tr></thead><tbody>{rows}</tbody></table></div>',unsafe_allow_html=True)
        ts=datetime.now().strftime('%Y%m%d_%H%M%S')
        inv_rows=[{"Produit":it['nom'],"SKU":it['sku'],"Marque":it['brand'],"Capacité":it['capacity'],"Emballage":it['emballage'],"Saveur":it['saveur'],"Confiance":f"{it['conf']:.1%}"} for it in validated]
        ec1,ec2=st.columns(2)
        ec1.download_button("📥 CSV",pd.DataFrame(inv_rows).to_csv(index=False),f"inventaire_{ts}.csv","text/csv",use_container_width=True)
        ec2.download_button("📥 JSON",json.dumps([{k:v for k,v in it.items() if k!='crop_bytes'} for it in validated],indent=2,ensure_ascii=False,default=str),f"inventaire_{ts}.json","application/json",use_container_width=True)
    else:st.markdown('<div class="empty-state"><div class="empty-title">Inventaire vide</div><div style="font-size:11px;color:#6B7280">Validez des produits depuis la section ci-dessus</div></div>',unsafe_allow_html=True)