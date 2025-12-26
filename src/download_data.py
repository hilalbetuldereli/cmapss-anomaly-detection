import kagglehub
import shutil
import os

# Dataseti indir
path = kagglehub.dataset_download("behrad3d/nasa-cmaps")
print("Dataset indirildi:", path)

# train_FD001.txt'yi her yerde ara
found = False
for root, dirs, files in os.walk(path):
    if "train_FD001.txt" in files:
        source_file = os.path.join(root, "train_FD001.txt")
        found = True
        break

if not found:
    raise FileNotFoundError("train_FD001.txt bulunamadı!")

# Hedef klasör
target_dir = "../data"
os.makedirs(target_dir, exist_ok=True)

shutil.copy(source_file, os.path.join(target_dir, "train_FD001.txt"))
print("train_FD001.txt data/ klasörüne kopyalandı")
