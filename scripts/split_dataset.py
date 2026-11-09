"""Split class-folder images into train/val/test. Input must be data/raw/<class>/*."""
import argparse, random, shutil
from pathlib import Path
from config.settings import ROOT_DIR
EXTS={".jpg",".jpeg",".png",".bmp",".webp"}
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",default="data/raw",help="Folder containing one subfolder per class")
    ap.add_argument("--destination",default="data/dataset",help="Output folder")
    ap.add_argument("--train",type=float,default=.70); ap.add_argument("--val",type=float,default=.15); ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    if args.train<=0 or args.val<0 or args.train+args.val>=1: ap.error("Require train > 0, val >= 0, and train+val < 1")
    src=(ROOT_DIR/args.source).resolve(); dst=(ROOT_DIR/args.destination).resolve()
    if not src.exists(): ap.error(f"Source folder does not exist: {src}")
    random.seed(args.seed); total=0
    for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        images=[p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
        random.shuffle(images); n=len(images); ntrain=int(n*args.train); nval=int(n*args.val)
        groups={"train":images[:ntrain],"val":images[ntrain:ntrain+nval],"test":images[ntrain+nval:]}
        for split,paths in groups.items():
            out=dst/split/class_dir.name; out.mkdir(parents=True,exist_ok=True)
            for i,p in enumerate(paths): shutil.copy2(p,out/f"{p.stem}_{i:05d}{p.suffix.lower()}")
        print(f"{class_dir.name}: {n} images -> train {len(groups['train'])}, val {len(groups['val'])}, test {len(groups['test'])}"); total+=n
    print(f"Total copied: {total}. NOTE: if multiple images come from the same physical part/video/sequence, split by part/group instead of randomly to prevent leakage.")
if __name__=="__main__": main()
