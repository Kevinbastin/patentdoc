import streamlit as st
import os

# ------------ LOCAL MODULE IMPORTS (Phi-3) ---------------
from generate_title import generate_title_from_abstract
from generate_claims import generate_claims_from_abstract
from generate_summary import summarize_abstract
from generate_field_of_invention import generate_field_of_invention
from generate_background import generate_background_locally
from generate_detailed_description import generate_detailed_description
from generate_brief_description import generate_brief_description
from generate_summary_of_drawings import generate_summary as generate_drawing_summary
from cpc_classifier import classify_cpc
from export_to_pdf import create_patent_pdf

# -------------- UI: HEADER & DISCLAIMER ------------------
st.title("🧠 PatentDoc Co-Pilot")

with st.expander("⚠️ Legal Disclaimer & AI Usage Notice", expanded=False):
    st.markdown("""
    This tool uses local AI language models (Phi-3, Llama.cpp, etc.) to help draft patent documents.

    ⚠️ **Warning:** The AI-generated content may contain factual errors, legal inaccuracies, or formatting that does not comply with USPTO guidelines.
    Always consult a qualified **patent attorney** before relying on the generated material for filing or legal use.

    📜 **Model License Notice:** The underlying LLMs (e.g., Phi-3) are subject to their respective open-source licenses.

    🔍 **Responsibility Disclaimer:** This app is a research prototype. It is **not** a substitute for professional legal services or patent counsel.
    """)

# ---------------- MAIN INPUT FIELDS ---------------------
abstract = st.text_area("📄 Enter Invention Abstract", height=200)
st.session_state["abstract_input"] = abstract
drawing_summary = st.text_area("🎨 Enter Drawing Summary (optional)", height=150)

# ----------------- SESSION STATE INIT -------------------
for key in [
    "title", "claims", "summary", "field_of_invention",
    "background", "detailed_description", "brief_description", "summary_drawings", "cpc_result"
]:
    if key not in st.session_state:
        st.session_state[key] = ""

# ------------------- GENERATION BUTTONS -----------------
if st.button("📌 Generate Title"):
    with st.spinner("Generating title..."):
        try:
            st.session_state.title = generate_title_from_abstract(abstract)
            st.success("Done!")
        except Exception as e:
            st.error(f"❌ Title generation failed: {e}")

if st.session_state.title:
    with st.expander("📘 Title"):
        st.write(st.session_state.title)

if st.button("🔖 Generate Claims"):
    with st.spinner("Generating claims..."):
        try:
            st.session_state.claims = generate_claims_from_abstract(abstract)
            st.success("Done!")
        except Exception as e:
            st.error(f"❌ Claim generation failed: {e}")

if st.session_state.claims:
    with st.expander("🧾 Claims"):
        st.write(st.session_state.claims)

if st.button("🧷 Generate Summary"):
    with st.spinner("Generating summary..."):
        try:
            st.session_state.summary = summarize_abstract(abstract)
            st.success("✅ Summary generated!")
        except Exception as e:
            st.error(f"❌ Summary generation failed: {e}")

if st.session_state.summary:
    with st.expander("📄 Summary"):
        st.write(st.session_state.summary)

if st.button("📚 Field of the Invention"):
    with st.spinner("Generating field of the invention..."):
        try:
            st.session_state.field_of_invention = generate_field_of_invention(abstract)
            st.success("Done!")
        except Exception as e:
            st.error(f"❌ Field generation failed: {e}")

if st.session_state.field_of_invention:
    with st.expander("📘 Field of the Invention"):
        st.write(st.session_state.field_of_invention)

if st.button("🧠 Background"):
    with st.spinner("Generating background..."):
        try:
            st.session_state.background = generate_background_locally(abstract)
            st.success("Done!")
        except Exception as e:
            st.error(f"❌ Background generation failed: {e}")

if st.session_state.background:
    with st.expander("🔍 Background"):
        st.write(st.session_state.background)

if st.button("📝 Detailed Description"):
    with st.spinner("Generating detailed description..."):
        try:
            result = generate_detailed_description(
                abstract,
                st.session_state.get("claims", "Claims not generated yet."),
                drawing_summary or "Drawing summary not provided."
            )
            # Always store something, even if result is None/empty:
            st.session_state.detailed_description = result or "⚠️ No output generated (see logs for details)."
            st.success("Done!")
        except Exception as e:
            st.session_state.detailed_description = f"❌ Exception: {e}"
            st.error(f"❌ Detailed description generation failed: {e}")

if st.session_state.get("detailed_description"):
    with st.expander("📘 Detailed Description"):
        st.write(st.session_state.detailed_description)

if st.button("📊 Brief Description of Drawings"):
    if not abstract or not drawing_summary:
        st.warning("Please enter both the abstract and drawing summary.")
    else:
        with st.spinner("Generating brief description of drawings..."):
            try:
                result = generate_brief_description(abstract, drawing_summary).strip()
                st.session_state.brief_description = result or "⚠️ No output generated (see logs for details)."
                st.success("Done!")
            except Exception as e:
                st.session_state.brief_description = f"❌ Exception: {e}"
                st.error(f"❌ Brief description generation failed: {e}")

if st.session_state.get("brief_description"):
    with st.expander("🖼️ Brief Description of the Drawings"):
        st.write(st.session_state.brief_description)

if st.button("🖼️ Summary of Drawings"):
    if not abstract:
        st.warning("Please enter the invention abstract.")
    else:
        with st.spinner("Generating summary of drawings..."):
            try:
                result = generate_drawing_summary(abstract).strip()
                st.session_state.summary_drawings = result or "⚠️ No output generated (see logs for details)."
                st.success("Done!")
            except Exception as e:
                st.session_state.summary_drawings = f"❌ Exception: {e}"
                st.error(f"❌ Drawing summary failed: {e}")

if st.session_state.get("summary_drawings"):
    with st.expander("📷 Summary of Drawings"):
        st.write(st.session_state.summary_drawings)

# ------------------ CPC CLASSIFIER ----------------------
st.markdown("## 📚 CPC Classifier")
if st.button("🏷️ Classify CPC"):
    with st.spinner("Classifying CPC..."):
        try:
            result = classify_cpc(abstract)
            st.session_state.cpc_result = result or "⚠️ No result."
            st.success("Done!")
        except Exception as e:
            st.session_state.cpc_result = f"❌ Exception: {e}"
            st.error(f"❌ CPC classification failed: {e}")

if st.session_state.get("cpc_result"):
    with st.expander("🔍 CPC Classification"):
        st.code(st.session_state.cpc_result)

# ----------------------- EXPORT -------------------------
st.markdown("## 📄 Export Patent Document")
export_abstract = st.session_state.get("abstract_input", "")
pdf_sections = {
    "Title": st.session_state.get("title", "[Not Generated]"),
    "Abstract": export_abstract or "[Not Provided]",
    "Claims": st.session_state.get("claims", "[Not Generated]"),
    "Field of the Invention": st.session_state.get("field_of_invention", "[Not Generated]"),
    "Background": st.session_state.get("background", "[Not Generated]"),
    "Brief Description of Drawings": st.session_state.get("brief_description", "[Not Generated]"),
    "Summary of Drawings": st.session_state.get("summary_drawings", "[Not Generated]"),
    "Detailed Description": st.session_state.get("detailed_description", "[Not Generated]"),
}
st.session_state.generated_sections = pdf_sections
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧾 Generate PDF")
    if export_abstract.strip():
        if st.button("📄 Generate PDF"):
            with st.spinner("Creating PDF..."):
                pdf_path = create_patent_pdf(pdf_sections)
                st.success("✅ PDF Generated!")
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Patent PDF",
                        data=f,
                        file_name="patent_document.pdf",
                        mime="application/pdf"
                    )
    else:
        st.warning("⚠️ Please enter an abstract before generating the PDF.")

with col2:
    st.markdown("### 📝 Export as DOCX")
    if st.button("📝 Generate DOCX"):
        from docx import Document
        from io import BytesIO
        doc = Document()
        doc.add_heading("Patent Document", 0)
        doc.add_paragraph("Generated by PatentDoc Co-Pilot", style='Intense Quote')
        doc.add_page_break()
        for section, content in st.session_state.generated_sections.items():
            doc.add_heading(section, level=1)
            doc.add_paragraph(content.strip() if content and content.strip() else "[Not Generated]")
            doc.add_paragraph("")  # spacing
        doc.add_page_break()
        doc.add_heading("Disclaimer", level=2)
        doc.add_paragraph(
            "This document was generated using an AI-based drafting system for prototype/research use. "
            "It does not constitute legal advice or replace professional patent counsel."
        )
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        st.download_button(
            label="📥 Download Formatted DOCX",
            data=buffer,
            file_name="patent_document_formatted.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

if st.button("🔄 Reset All"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
