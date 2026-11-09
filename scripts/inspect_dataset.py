from pathlib import Path
from collections import Counter
from PIL import Image, UnidentifiedImageError
from config.settings import DATASET_DIR

EXTS={".jpg",".jpeg",".png",".bmp",".webp"}
def main():
    files=[p for p in DATASET_DIR.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
    print(f"Dataset root: {DATASET_DIR}\nImages found: {len(files)}")
    counts=Counter(); bad=[]; dimensions=Counter()
    for p in files:
        rel=p.relative_to(DATASET_DIR); parts=rel.parts
        label=parts[1] if len(parts)>2 else "unlabelled"
        counts[(parts[0] if len(parts)>2 else "unspecified",label)]+=1
        try:
            with Image.open(p) as im: im.verify()
            with Image.open(p) as im: dimensions[im.size]+=1
        except (OSError,UnidentifiedImageError): bad.append(str(rel))
    print("\nCounts by split/class:")
    for key,count in sorted(counts.items()): print(f"  {key[0]}/{key[1]}: {count}")
    print(f"\nUnreadable images: {len(bad)}")
    for name in bad: print("  "+name)
    if not files: print("No images yet. Arrange files as data/dataset/train/normal/... and data/dataset/train/defective/...")
if __name__=="__main__": main()
