import uuid
from datetime import datetime, timezone
import pandas as pd
import plotly.express as px
import streamlit as st
from config.settings import APP_NAME, CONFIDENCE_THRESHOLD, DATASET_DIR
from core.database import init_db, get_inspections, save_inspection
from core.image_utils import load_rgb_image
from core.inference import predict, model_status
from core.decisions import make_decision

st.set_page_config(page_title=APP_NAME, page_icon="🔎", layout="wide")
init_db()
st.markdown("""<style>
.stApp {background:#f4f7fb;color:#14263d} section[data-testid="stSidebar"]{background:#10243a}
section[data-testid="stSidebar"] *{color:#f4f8fc!important}
div[data-testid="stMetric"]{background:white;border:1px solid #dce6ef;padding:16px;border-radius:12px}
.dd-title{font-size:2.15rem;font-weight:800;color:#10243a;margin-bottom:0}.dd-sub{color:#526579;margin-top:0}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("# 🔎 Defect Doctor AI")
    st.caption("Brake-disc quality inspection prototype")
    page = st.radio("Navigation", ["Overview", "Single inspection", "Batch inspection", "Dataset explorer", "History", "Analytics & reports", "Model info & evaluation", "Settings & help"])
    st.divider()
    status = model_status()
    st.write("**Model status**")
    st.success("Checkpoint found") if status["available"] else st.warning("No trained model — review only")
    st.caption("This prototype does not control factory machinery or reject parts automatically.")

def run_inspection(file, component_id, batch_id, line, operator, notes=""):
    image = load_rgb_image(file)
    result = predict(image)
    decision = make_decision(result["predicted_class"], result["confidence"], result["model_available"], CONFIDENCE_THRESHOLD)
    record = {"inspection_id": str(uuid.uuid4())[:8].upper(), "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "component_id": component_id, "batch_id": batch_id, "production_line": line, "operator": operator,
              "filename": getattr(file, "name", "uploaded-image"), "predicted_class": result["predicted_class"],
              "confidence": result["confidence"], "decision": decision, "model_version": result["model_version"], "notes": notes}
    save_inspection(record)
    return image, result, decision, record

def show_result(result, decision, record):
    if decision == "PASS": st.success("## PASS — no defect predicted above the configured threshold")
    elif decision == "FAIL": st.error("## FAIL — defect predicted above the configured threshold")
    else: st.warning("## REVIEW REQUIRED — do not use as an automatic accept/reject decision")
    a,b,c = st.columns(3)
    a.metric("Predicted class", result["predicted_class"] or "Unavailable")
    b.metric("Confidence", f"{result['confidence']:.1%}" if result["confidence"] is not None else "N/A")
    c.metric("Model version", result["model_version"])
    st.caption(result["message"])
    if result["probabilities"]: st.json(result["probabilities"])
    st.download_button("Download inspection record (CSV)", pd.DataFrame([record]).to_csv(index=False).encode(), file_name=f"inspection_{record['inspection_id']}.csv", mime="text/csv")

if page == "Overview":
    st.markdown('<p class="dd-title">Defect Doctor AI</p><p class="dd-sub">AI-assisted visual inspection dashboard for brake discs</p>', unsafe_allow_html=True)
    df = get_inspections()
    total = len(df); passed = int((df.decision == "PASS").sum()) if total else 0; failed = int((df.decision == "FAIL").sum()) if total else 0; review = int((df.decision == "REVIEW REQUIRED").sum()) if total else 0
    cols = st.columns(4); cols[0].metric("Inspections", total); cols[1].metric("PASS", passed); cols[2].metric("FAIL", failed); cols[3].metric("Review required", review)
    if total:
        left,right = st.columns(2)
        with left:
            counts=df.decision.value_counts().rename_axis("Decision").reset_index(name="Count")
            st.plotly_chart(px.pie(counts,names="Decision",values="Count",title="Inspection dispositions",hole=.45),use_container_width=True)
        with right:
            by_batch=df.groupby("batch_id",dropna=False).size().reset_index(name="Inspections").tail(12)
            st.plotly_chart(px.bar(by_batch,x="batch_id",y="Inspections",title="Recent batch volumes"),use_container_width=True)
        st.subheader("Recent inspections"); st.dataframe(df.head(10),use_container_width=True,hide_index=True)
    else:
        st.info("No inspections yet. Open **Single inspection** or **Batch inspection** to get started.")
    st.subheader("Inspection workflow")
    st.write("1. Upload a brake-disc image. 2. Enter traceability details. 3. Run the available trained model. 4. Review the result and record. 5. Investigate uncertain predictions manually.")

elif page == "Single inspection":
    st.title("Single-image inspection")
    with st.form("single_form"):
        upload=st.file_uploader("Brake-disc image (JPG, JPEG, PNG)",type=["jpg","jpeg","png"])
        a,b=st.columns(2); component=a.text_input("Component ID",placeholder="DISC-0001"); batch=b.text_input("Batch ID",placeholder="BATCH-2026-01")
        c,d=st.columns(2); line=c.text_input("Production line",placeholder="LINE-A"); operator=d.text_input("Operator",placeholder="Operator name")
        notes=st.text_area("Inspection notes (optional)")
        submit=st.form_submit_button("Inspect image",type="primary")
    if submit:
        if upload is None: st.error("Please upload an image first.")
        elif not component.strip() or not batch.strip(): st.error("Component ID and Batch ID are required for traceability.")
        else:
            try:
                image,result,decision,record=run_inspection(upload,component,batch,line,operator,notes)
                st.image(image,caption=upload.name,use_container_width=True); show_result(result,decision,record)
            except ValueError as e: st.error(str(e))

elif page == "Batch inspection":
    st.title("Batch image inspection")
    st.caption("Upload multiple images at once. This prototype saves one record per image; all untrained/uncertain cases remain REVIEW REQUIRED.")
    files=st.file_uploader("Select images",type=["jpg","jpeg","png"],accept_multiple_files=True)
    a,b,c=st.columns(3); batch=a.text_input("Batch ID",key="batch_upload"); line=b.text_input("Production line",key="line_upload"); operator=c.text_input("Operator",key="operator_upload")
    prefix=st.text_input("Component ID prefix",value="DISC")
    if st.button("Inspect batch",type="primary"):
        if not files or not batch.strip(): st.error("Choose at least one image and provide a batch ID.")
        else:
            records=[]; progress=st.progress(0)
            for i,file in enumerate(files):
                try:
                    _,result,decision,record=run_inspection(file,f"{prefix}-{i+1:04d}",batch,line,operator)
                    records.append(record)
                except ValueError as e: st.warning(f"Skipped {file.name}: {e}")
                progress.progress((i+1)/len(files))
            if records:
                out=pd.DataFrame(records); st.dataframe(out,use_container_width=True,hide_index=True)
                st.download_button("Download batch results CSV",out.to_csv(index=False).encode(),file_name=f"batch_{batch}.csv",mime="text/csv")
                st.info(f"Saved {len(records)} inspection records to the local database.")

elif page == "Dataset explorer":
    st.title("Dataset explorer")
    st.write("Put dataset images under `data/dataset/<split>/<class>/image.jpg`, for example `train/normal`, `train/defective`, `val/normal`, `val/defective`.")
    if not DATASET_DIR.exists(): DATASET_DIR.mkdir(parents=True,exist_ok=True)
    rows=[]; exts={".jpg",".jpeg",".png",".bmp",".webp"}
    for p in DATASET_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            parts=p.relative_to(DATASET_DIR).parts
            rows.append({"relative_path":str(p.relative_to(DATASET_DIR)),"split":parts[0] if len(parts)>2 else "unspecified","class":parts[1] if len(parts)>2 else "unspecified","bytes":p.stat().st_size})
    if rows:
        data=pd.DataFrame(rows); st.metric("Image files found",len(data)); st.dataframe(data,use_container_width=True,hide_index=True)
        st.plotly_chart(px.bar(data.groupby(["split","class"]).size().reset_index(name="images"),x="class",y="images",color="split",barmode="group",title="Image counts by class and split"),use_container_width=True)
        st.download_button("Export dataset manifest",data.to_csv(index=False).encode(),file_name="dataset_manifest.csv",mime="text/csv")
    else: st.warning("No images found yet. Use scripts/download_dataset.py or copy labelled images into the expected class folders.")

elif page == "History":
    st.title("Inspection history")
    df=get_inspections()
    if df.empty: st.info("No saved inspections yet.")
    else:
        a,b=st.columns(2); decisions=a.multiselect("Filter by decision",sorted(df.decision.unique()),default=sorted(df.decision.unique())); batches=b.multiselect("Filter by batch",sorted(df.batch_id.fillna("").unique()))
        filtered=df[df.decision.isin(decisions)]
        if batches: filtered=filtered[filtered.batch_id.fillna("").isin(batches)]
        st.dataframe(filtered,use_container_width=True,hide_index=True)
        st.download_button("Export filtered history CSV",filtered.to_csv(index=False).encode(),file_name="inspection_history.csv",mime="text/csv")

elif page == "Analytics & reports":
    st.title("Analytics & reports")
    df=get_inspections()
    if df.empty: st.info("Run an inspection before generating analytics.")
    else:
        total=len(df); fails=int((df.decision=="FAIL").sum()); known=int(df.decision.isin(["PASS","FAIL"]).sum())
        a,b,c=st.columns(3); a.metric("Total inspections",total); b.metric("Confirmed FAIL dispositions",fails); c.metric("Review rate",f"{(df.decision.eq('REVIEW REQUIRED').mean()*100):.1f}%")
        st.caption("Defect rate among decisive PASS/FAIL predictions is not the same as real factory defect prevalence; results depend on the model and validation quality.")
        if known: st.metric("FAIL share among decisive predictions",f"{fails/known:.1%}")
        byday=df.copy(); byday["date"]=pd.to_datetime(byday.timestamp,errors="coerce",utc=True).dt.date
        st.plotly_chart(px.histogram(byday,x="date",color="decision",barmode="group",title="Inspection outcomes over time"),use_container_width=True)
        st.download_button("Download full report CSV",df.to_csv(index=False).encode(),file_name="defect_doctor_report.csv",mime="text/csv")

elif page == "Model info & evaluation":
    st.title("Model information & evaluation")
    s=model_status(); st.write("**Checkpoint path:**",s["path"]); st.write("**Status:**", "Available" if s["available"] else "Not trained / not found"); st.write("**Configured confidence threshold:**",f"{CONFIDENCE_THRESHOLD:.0%}")
    st.info("The included CNN architecture is a starting baseline. Do not describe it as production-ready or claim accuracy until it is trained and evaluated on representative, held-out brake-disc images.")
    st.code("python -m scripts.inspect_dataset\npython -m scripts.split_dataset --help\npython -m scripts.train_classifier --help\npytest -q",language="powershell")
    metrics_file=st.text_input("Optional evaluation metrics JSON path",value="models/evaluation_metrics.json")
    try:
        import json
        from pathlib import Path
        p=Path(metrics_file)
        if p.exists(): st.json(json.loads(p.read_text(encoding="utf-8")))
        else: st.caption("No evaluation metrics file found yet.")
    except Exception as e: st.warning(f"Could not read metrics file: {e}")

else:
    st.title("Settings & help")
    st.subheader("Decision policy")
    st.write(f"Configured confidence threshold: **{CONFIDENCE_THRESHOLD:.0%}**. Predictions below the threshold, unknown class names, missing models, and model load errors return **REVIEW REQUIRED**.")
    st.subheader("Expected dataset structure")
    st.code("data/dataset/\n  train/\n    normal/\n    defective/\n  val/\n    normal/\n    defective/\n  test/\n    normal/\n    defective/",language="text")
    st.subheader("Important limitations")
    st.markdown("- This is a hackathon prototype, not a certified safety inspection system.\n- Dataset labels and image conditions must match the real brake-disc inspection task.\n- The included CNN classifies whole images; it does not generate bounding boxes or heatmaps.\n- Do not connect it to PLCs or automated rejection hardware without engineering validation and safety controls.\n- Keep source/license records for every downloaded dataset.")
