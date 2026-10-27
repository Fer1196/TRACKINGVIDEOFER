"""
Paso 6 — Video final con TODOS los filtros combinados en una sola pasada:
  - Tracking: caja + ID + cola que se desvanece (YOLO + ByteTrack)
  - Mapa de calor acumulado y superpuesto EN VIVO (crece con el tiempo)
  - Línea virtual + contador de ingresos/salidas actualizándose EN VIVO

Genera: resultados/video_final_completo.mp4
"""
import os, time
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# --- Parámetros (mismos del notebook) ---
VIDEO_PATH = "store.mp4"
OUTPUT_DIR = "resultados"
MODEL_NAME = "yolo26n.pt"
PERSON_CLASS = 0
CONF = 0.30
TRAIL_LEN = 30
MAX_FRAMES = None          # None = todo el video; pon p.ej. 1500 para una prueba rápida
HEAT_REFRESH = 5           # recalcular el coloreado del heatmap cada N cuadros (rendimiento)

# Línea virtual (igual que el Paso 4)
LINE_Y_FRAC, X_MIN_FRAC, X_MAX_FRAC = 0.22, 0.30, 0.70

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
cap.release()
LINE_Y = int(H * LINE_Y_FRAC); X_MIN = int(W * X_MIN_FRAC); X_MAX = int(W * X_MAX_FRAC)

model = YOLO(MODEL_NAME)
trails = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
prev_y = {}
cruces_down, cruces_up = set(), set()
heat = np.zeros((H, W), dtype=np.float32)
heat_color_cache = np.zeros((H, W, 3), dtype=np.uint8)
heat_mask_cache = np.zeros((H, W, 1), dtype=np.float32)

rng = np.random.default_rng(42)
colores = {}
def color_de(tid):
    if tid not in colores:
        colores[tid] = tuple(int(c) for c in rng.integers(60, 255, size=3))
    return colores[tid]

out_path = os.path.join(OUTPUT_DIR, "video_final_completo.mp4")
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

t0 = time.time(); n = 0
resultados = model.track(VIDEO_PATH, stream=True, classes=[PERSON_CLASS], conf=CONF,
                         tracker="bytetrack.yaml", device=DEVICE, verbose=False)

for i, r in enumerate(resultados):
    if MAX_FRAMES is not None and i >= MAX_FRAMES:
        break
    n += 1
    frame = r.orig_img.copy()

    # 1) Actualizar acumulador de calor y lógica de cruces con las detecciones actuales
    if r.boxes is not None and r.boxes.id is not None:
        xyxy = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.int().cpu().tolist()
        dets = []
        for (x1, y1, x2, y2), tid in zip(xyxy, ids):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if 0 <= cy < H and 0 <= cx < W:
                heat[cy, cx] += 1.0
            trails[tid].append((cx, cy))
            # conteo de cruces (cada ID una sola vez por dirección)
            if tid in prev_y and X_MIN <= cx <= X_MAX:
                py = prev_y[tid]
                if py < LINE_Y <= cy: cruces_down.add(tid)
                elif py > LINE_Y >= cy: cruces_up.add(tid)
            prev_y[tid] = cy
            dets.append((x1, y1, x2, y2, tid))
    else:
        dets = []

    # 2) Capa de mapa de calor (recalcular el color cada HEAT_REFRESH cuadros)
    if i % HEAT_REFRESH == 0 and heat.max() > 0:
        hb = cv2.GaussianBlur(heat, (0, 0), sigmaX=25, sigmaY=25)
        hn = (hb / hb.max() * 255).astype(np.uint8)
        heat_color_cache = cv2.applyColorMap(hn, cv2.COLORMAP_JET)
        heat_mask_cache = (hn > 15).astype(np.float32)[..., None]
    frame = (frame * (1 - 0.55 * heat_mask_cache) +
             heat_color_cache * (0.55 * heat_mask_cache)).astype(np.uint8)

    # 3) Cajas + ID + cola que se desvanece (encima del heatmap)
    for (x1, y1, x2, y2, tid) in dets:
        col = color_de(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, f"ID {tid}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        pts = list(trails[tid])
        for j in range(1, len(pts)):
            ratio = j / len(pts)
            cv2.line(frame, pts[j - 1], pts[j], col, max(1, int(1 + 3 * ratio)))
        for j, (px, py) in enumerate(pts):
            ratio = (j + 1) / len(pts)
            cv2.circle(frame, (px, py), max(1, int(4 * ratio)), col, -1)

    # 4) Línea virtual + contador EN VIVO (panel sólido, grande y explícito)
    cv2.line(frame, (X_MIN, LINE_Y), (X_MAX, LINE_Y), (0, 0, 255), 3)
    cv2.putText(frame, "LINEA DE CONTEO", (X_MIN, LINE_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    FT = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(frame, (0, 0), (W, 96), (0, 0, 0), -1)           # panel sólido
    cv2.putText(frame, f"PERSONAS QUE INGRESARON: {len(cruces_down)}",
                (15, 38), FT, 1.0, (0, 255, 0), 2)                 # verde
    cv2.putText(frame, f"PERSONAS QUE SALIERON:   {len(cruces_up)}",
                (15, 78), FT, 1.0, (0, 0, 255), 2)                 # rojo
    cv2.putText(frame, f"En escena: {len(dets)}", (W - 330, 38), FT, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame: {i}",            (W - 330, 72), FT, 0.7, (255, 255, 255), 2)

    writer.write(frame)

writer.release()
dt = time.time() - t0
print(f"Procesados {n} cuadros en {dt:.1f} s ({n/dt:.1f} fps)")
print(f"INGRESOS={len(cruces_down)}  SALIDAS={len(cruces_up)}")
print(f"Video final con todos los filtros: {out_path}")
