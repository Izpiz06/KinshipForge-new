import os, random, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from preprocess.align_images import align_face

# ── Paths ────────────────────────────────────────────────────
FFHQ_ROOT  = '/kaggle/input/datasets/greatgamedota/ffhq-face-data-set/thumbnails128x128'
CSV_PATH   = '/kaggle/working/StyleGene/data/fairface_gender_angle.csv'
OUT_PATH   = '/kaggle/working/pool_50samples.pkl'
MAX_SAMPLE = 100

# ── Buckets ──────────────────────────────────────────────────
AGES    = ['0-2', '3-9', '10-19', '20-29']
GENDERS = ['male', 'female']
RACES   = ['White', 'Black', 'Latino_Hispanic',
           'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']

# ── Transform ────────────────────────────────────────────────
T = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ── Load CSV ─────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.replace('Male',   'male',   inplace=True)
df.replace('Female', 'female', inplace=True)
print(f"CSV loaded: {len(df)} rows")

# ── Build pool ───────────────────────────────────────────────
# encoder, w2sub34, mean_latent must already be loaded in scope
pool = {}

for age in AGES:
    for gender in GENDERS:
        for race in RACES:
            key    = f'{age}-{gender}-{race}'
            subset = df.query(
                f'age=="{age}" and gender=="{gender}" and race=="{race}"'
            )
            fids = subset['file_id'].values.tolist()
            random.shuffle(fids)

            samples = []
            for fid in fids:
                if len(samples) >= MAX_SAMPLE:
                    break
                path = os.path.join(
                    FFHQ_ROOT, format(int(fid), '05d') + '.png'
                )
                if not os.path.exists(path):
                    continue
                try:
                    # Step 1: Align face (FFHQ-standard alignment)
                    raw     = np.array(Image.open(path).convert('RGB'))
                    aligned = align_face(raw)

                    # Step 2: Transform to tensor
                    if isinstance(aligned, np.ndarray):
                        aligned = Image.fromarray(aligned)
                    t = T(aligned).unsqueeze(0).to(device)

                    # Step 3: e4e encode → W+ latent
                    with torch.no_grad():
                        w18 = encoder(
                            F.interpolate(t, size=(256, 256))
                        ) + mean_latent

                        # Step 4: LGE encode → (mu, var) gene representation
                        # Shape: (1, 18, 34, 512)
                        mu, var, _ = w2sub34(w18)

                    # Step 5: Store (mu, var) under bucket key
                    samples.append((mu.cpu(), var.cpu()))

                except Exception:
                    continue

            pool[key] = samples
            print(f"  {key}: {len(samples)} samples")

# ── Save ─────────────────────────────────────────────────────
with open(OUT_PATH, 'wb') as f:
    pickle.dump(pool, f, protocol=4)

size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
print(f"\n✅ Saved: {OUT_PATH}")
print(f"   Size: {size_mb:.1f} MB")
print(f"   Keys: {len(pool)}")
print(f"   Samples per key (non-empty):")
for k, v in sorted(pool.items()):
    if len(v) > 0:
        print(f"     {k}: {len(v)}")