import os, sys, torch
print(f"Python: {sys.version}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

UNC = r'\\sshfs.k\root@1.15.92.20\osrcir'
print(f"UNC exists: {os.path.exists(UNC)}")
print(f"outputs: {os.path.exists(os.path.join(UNC, 'outputs'))}")

f = os.path.join(UNC, 'outputs', 'genecis_change_object_full.json')
print(f"baseline json exists: {os.path.exists(f)}")
if os.path.exists(f):
    sz = os.path.getsize(f)
    print(f"  size: {sz} bytes")

clip_path = r'C:\Users\12427\.cache\clip\ViT-L-14.pt'
print(f"CLIP weights: {os.path.exists(clip_path)}")
print("ALL OK")
