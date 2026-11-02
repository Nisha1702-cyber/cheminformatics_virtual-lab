import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, QED
from rdkit.Chem.Draw import rdMolDraw2D

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, PageBreak

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Cheminformatics Virtual Laboratory",
    page_icon="🧪",
    layout="wide"
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_molecule(smiles):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    return mol

def calculate_properties(mol):
    if mol is None:
        return {}
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    formula = Chem.MolFormulaToSmiles(mol)
    qed_score = QED.qed(mol)
    
    # Lipinski's Rule of 5 evaluation
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if h_donors > 5: violations += 1
    if h_acceptors > 10: violations += 1
    
    lipinski_pass = "Pass" if violations <= 1 else "Fail"
    
    return {
        "Molecular Formula": formula,
        "Molecular Weight": round(mw, 2),
        "LogP": round(logp, 2),
        "H-Bond Donors": h_donors,
        "H-Bond Acceptors": h_acceptors,
        "TPSA (Å²)": round(tpsa, 2),
        "Rotatable Bonds": rot_bonds,
        "QED Score (Drug-likeness)": round(qed_score, 3),
        "Lipinski Rule of 5": lipinski_pass,
        "Violations": violations
    }

def generate_multi_final_report(student_name, registration_number, molecule_records):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=38, rightMargin=38, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("MultiReportTitle", parent=styles["Title"], fontSize=19, leading=23, alignment=TA_CENTER, textColor=HexColor("#17365D"))
    heading_style = ParagraphStyle("MultiReportHeading", parent=styles["Heading1"], fontSize=13, leading=16, textColor=HexColor("#17365D"), spaceBefore=7, spaceAfter=7)
    body_style = ParagraphStyle("MultiReportBody", parent=styles["BodyText"], fontSize=9, leading=12, alignment=TA_JUSTIFY)
    
    story = [
        Spacer(1, 0.15 * inch),
        Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY", title_style),
        Spacer(1, 0.08 * inch),
        Paragraph("MULTI-MOLECULE STUDENT ANALYSIS REPORT", heading_style),
        Paragraph(f"Student: {student_name or 'Not provided'} &nbsp;&nbsp;&nbsp; Registration Number: {registration_number or 'Not provided'}", body_style),
        Paragraph(f"Number of molecules analyzed: {len(molecule_records)}", body_style),
        Spacer(1, 0.15 * inch)
    ]
    
    for index, record in enumerate(molecule_records, start=1):
        input_smiles, mol, properties = record
        if index > 1:
            story.append(PageBreak())
        story.append(Paragraph(f"Molecule {index}: {input_smiles}", heading_style))
        basic_data = [
            ["Canonical SMILES", Chem.MolToSmiles(mol)],
            ["Molecular Formula", properties["Molecular Formula"]],
            ["Molecular Weight", str(properties["Molecular Weight"]) + " g/mol"],
            ["QED Score", str(properties.get("QED Score (Drug-likeness)", "N/A"))]
        ]
        t = Table(basic_data, colWidths=[2.0 * inch, 4.7 * inch])
        story.append(t)
        story.append(Spacer(1, 0.15 * inch))
        
    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

# ============================================================
# SIDEBAR NAVIGATION & BRANDING
# ============================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📄 Final Project Report"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 👩‍💻 Developed by")
st.sidebar.markdown("**Nisha Vaishnavi K**")

# ============================================================
# MAIN APP ROUTING
# ============================================================
if page == "🏠 Home":
    st.title("🧪 Cheminformatics Virtual Laboratory")
    st.markdown("Explore molecular properties, 3D visualization, drug-likeness rules, functional groups, and advanced cheminformatics tools.")
    
    st.markdown("### Example Molecules")
    example_df = pd.DataFrame({
        "Molecule": ["Ethanol", "Benzene", "Phenol", "Aspirin", "Caffeine", "Paracetamol"],
        "SMILES": ["CCO", "c1ccccc1", "Oc1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(=O)Nc1ccc(O)cc1"]
    })
    st.table(example_df)
    
    st.markdown("---")
    st.markdown("### 👩‍💻 Developed by")
    st.markdown("**Nisha Vaishnavi K**")

elif page == "📄 Final Project Report":
    st.title("📄 Final Student Molecular Analysis Report")
    name = st.text_input("Student Name for Report", value=st.session_state.get("student_name", ""), key="rep_name")
    reg = st.text_input("Registration Number for Report", value=st.session_state.get("registration_number", ""), key="rep_reg")
    
    multi_smiles_text = st.text_area("Enter SMILES (one per line)", value=st.session_state.get("latest_smiles", "CCO"), height=100, key="rep_smiles")
    smiles_entries = [line.strip() for line in multi_smiles_text.splitlines() if line.strip()]
    
    molecule_records = []
    for s in smiles_entries:
        m = get_molecule(s)
        if m:
            molecule_records.append((s, m, calculate_properties(m)))
            
    if molecule_records:
        st.success(f"{len(molecule_records)} valid molecule(s) ready.")
        
        if st.button("📄 Generate Combined PDF Report"):
            try:
                with st.spinner("Generating PDF report..."):
                    st.session_state['pdf_data'] = generate_multi_final_report(name, reg, molecule_records)
                st.success("✅ Report generated successfully!")
            except Exception as e:
                st.error(f"❌ Error generating PDF: {e}")
        
        if 'pdf_data' in st.session_state and st.session_state['pdf_data']:
            st.download_button(
                label="⬇️ Download PDF Report",
                data=st.session_state['pdf_data'],
                file_name="Cheminformatics_Report.pdf",
                mime="application/pdf",
                key="download_pdf_final"
            )
    else:
        st.warning("Please enter at least one valid SMILES string.")
        
    st.markdown("---")
    st.markdown("### 👩‍💻 Developed by")
    st.markdown("**Nisha Vaishnavi K**")



