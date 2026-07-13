import os
import bz2
from huggingface_hub import hf_hub_download

ckpt_dir = 'C:/tmp/ckpt'
os.makedirs(ckpt_dir, exist_ok=True)

files = [
    'e4e_ffhq_encode.pt',
    'stylegan2-ffhq-config-f.pt',
    'stylegene_N18.ckpt',
    'res34_fair_align_multi_7_20190809.pt',
    'shape_predictor_68_face_landmarks.dat.bz2'
]

print("Starting downloads from HuggingFace wmpscc/StyleGene_CKPT...")
for f in files:
    dest = os.path.join(ckpt_dir, f)
    if not os.path.exists(dest):
        print(f"Downloading {f}...")
        hf_hub_download(repo_id='wmpscc/StyleGene_CKPT', filename=f, local_dir=ckpt_dir)
        print(f"Downloaded {f}")
    else:
        print(f"{f} already exists.")

dat_path = os.path.join(ckpt_dir, 'shape_predictor_68_face_landmarks.dat')
bz2_path = os.path.join(ckpt_dir, 'shape_predictor_68_face_landmarks.dat.bz2')
if not os.path.exists(dat_path):
    print("Extracting bz2 landmark file...")
    with bz2.BZ2File(bz2_path) as fr, open(dat_path, 'wb') as fw:
        fw.write(fr.read())
    print("Extracted shape_predictor_68_face_landmarks.dat")
else:
    print("shape_predictor_68_face_landmarks.dat already exists.")

print("All checkpoints ready!")
