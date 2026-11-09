"""Train a small PyTorch CNN on data/dataset/{train,val,test}/{class_name}/image.*."""
import argparse, json, random
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from config.settings import ROOT_DIR, DATASET_DIR, MODEL_PATH, IMAGE_SIZE
from core.inference import BrakeDiscCNN, MODEL_VERSION

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--epochs",type=int,default=8); ap.add_argument("--batch-size",type=int,default=16); ap.add_argument("--lr",type=float,default=.001); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--workers",type=int,default=0)
    args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train_dir=DATASET_DIR/"train"; val_dir=DATASET_DIR/"val"; test_dir=DATASET_DIR/"test"
    for p in (train_dir,val_dir,test_dir):
        if not p.is_dir(): raise SystemExit(f"Missing required folder: {p}")
    tfm=transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),transforms.ToTensor()])
    train=datasets.ImageFolder(train_dir,transform=tfm); val=datasets.ImageFolder(val_dir,transform=tfm); test=datasets.ImageFolder(test_dir,transform=tfm)
    if len(train.classes)<2: raise SystemExit("Need at least two class folders (e.g. defective and normal).")
    if train.classes!=val.classes or train.classes!=test.classes: raise SystemExit(f"Class folder mismatch. train={train.classes}, val={val.classes}, test={test.classes}")
    if len(train)==0 or len(val)==0 or len(test)==0: raise SystemExit("Train, val, and test sets must all contain images.")
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=args.workers); val_loader=DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=args.workers); test_loader=DataLoader(test,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    model=BrakeDiscCNN(len(train.classes)); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    criterion=nn.CrossEntropyLoss(); optimizer=torch.optim.Adam(model.parameters(),lr=args.lr); best=float("inf")
    for epoch in range(args.epochs):
        model.train(); train_loss=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); optimizer.zero_grad(); logits=model(x); loss=criterion(logits,y); loss.backward(); optimizer.step(); train_loss+=loss.item()*len(y)
        model.eval(); val_loss=0; y_true=[]; y_pred=[]
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device); logits=model(x); val_loss+=criterion(logits,y).item()*len(y); y_true.extend(y.cpu().tolist()); y_pred.extend(logits.argmax(1).cpu().tolist())
        val_loss/=len(val); print(f"Epoch {epoch+1}/{args.epochs} train_loss={train_loss/len(train):.4f} val_loss={val_loss:.4f} val_accuracy={accuracy_score(y_true,y_pred):.4f}")
        if val_loss<best:
            best=val_loss; MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"state_dict":model.cpu().state_dict(),"class_names":train.classes,"image_size":IMAGE_SIZE,"model_version":MODEL_VERSION,"seed":args.seed},MODEL_PATH); model.to(device)
    checkpoint=torch.load(MODEL_PATH,map_location=device,weights_only=True); model.load_state_dict(checkpoint["state_dict"]); model.eval(); y_true=[]; y_pred=[]
    with torch.no_grad():
        for x,y in test_loader:
            logits=model(x.to(device)); y_true.extend(y.tolist()); y_pred.extend(logits.argmax(1).cpu().tolist())
    precision,recall,f1,_=precision_recall_fscore_support(y_true,y_pred,average="weighted",zero_division=0)
    metrics={"model_version":MODEL_VERSION,"class_names":train.classes,"train_images":len(train),"validation_images":len(val),"test_images":len(test),"test_accuracy":float(accuracy_score(y_true,y_pred)),"test_weighted_precision":float(precision),"test_weighted_recall":float(recall),"test_weighted_f1":float(f1),"confusion_matrix":confusion_matrix(y_true,y_pred).tolist(),"classification_report":classification_report(y_true,y_pred,target_names=train.classes,output_dict=True,zero_division=0),"caveat":"Metrics apply only to this dataset split; they do not establish factory readiness."}
    out=ROOT_DIR/"models"/"evaluation_metrics.json"; out.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in metrics.items() if k!="classification_report"},indent=2)); print(f"Saved checkpoint: {MODEL_PATH}\nSaved metrics: {out}")
if __name__=="__main__": main()
