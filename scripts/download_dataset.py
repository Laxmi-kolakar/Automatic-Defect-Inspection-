"""Download a dataset archive URL supplied by the user; does not assume a license or labels."""
import argparse, shutil, tempfile, urllib.request, zipfile
from pathlib import Path
from config.settings import DATASET_DIR

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url",help="Direct HTTPS URL to a ZIP archive you are allowed to download")
    parser.add_argument("--source-name",default="downloaded-source",help="Subfolder name under data/dataset/sources/")
    parser.add_argument("--allow-http",action="store_true",help="Allow HTTP only if you trust the source")
    args=parser.parse_args()
    if not args.url.startswith("https://") and not (args.allow_http and args.url.startswith("http://")):
        raise SystemExit("Use an HTTPS URL. For a trusted HTTP source, explicitly pass --allow-http.")
    dest=DATASET_DIR/"sources"/args.source_name
    dest.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive=Path(tmp)/"dataset.zip"
        request=urllib.request.Request(args.url,headers={"User-Agent":"DefectDoctorAI-Hackathon/1.0"})
        with urllib.request.urlopen(request,timeout=60) as response, archive.open("wb") as out: shutil.copyfileobj(response,out)
        with zipfile.ZipFile(archive) as zf:
            # Protect against zip-slip/path traversal attacks.
            base=dest.resolve()
            for member in zf.infolist():
                target=(dest/member.filename).resolve()
                if target != base and base not in target.parents:
                    raise SystemExit(f"Unsafe archive path rejected: {member.filename}")
            zf.extractall(dest)
    print(f"Downloaded and extracted archive to: {dest}")
    print("Next: inspect source license, labels and class definitions before training.")
if __name__=="__main__": main()
