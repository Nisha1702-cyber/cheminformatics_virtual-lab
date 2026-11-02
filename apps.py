import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# RDKit
from rdkit import Chem
from rdkit.Chem import (
    Draw,
    AllChem,
    Descriptors,
    Crippen,
    Lipinski,
    rdMolDescriptors,
    QED
)
from rdkit.DataStructs import TanimotoSimilarity

# 3D Visualization
import py3Dmol
import streamlit.components.v1 as components

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    Preformatted
)
from reportlab.lib.units import inch
import tempfile
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cheminformatics Virtual Laboratory",
    page_icon="🧪",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}
.section-title {
    font-size: 25px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SHARED STATE & HELPER FUNCTIONS
# ============================================================

EXAMPLE_MOLECULES = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Acetic Acid": "CC(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1"
}

if "active_smiles" not in st.session_state:
    st.session_state.active_smiles = "CCO"

if "selected_example" not in st.session_state:
    st.session_state.selected_example = "Ethanol"

def load_shared_example():
    st.session_state.active_smiles = EXAMPLE_MOLECULES[
        st.session_state.selected_example
    ]

def get_molecule(smiles):
    if not smiles or not smiles.strip():
        return None
    try:
        return Chem.MolFromSmiles(smiles.strip())
    except Exception:
        return None


# ============================================================
# NEW FEATURE 1: FUNCTIONAL GROUP IDENTIFIER
# ============================================================

def analyze_functional_groups(mol):
    patterns = {
        "Hydroxyl (-OH)": Chem.MolFromSmarts("[#6][OX2H]"),
        "Carboxylic Acid (-COOH)": Chem.MolFromSmarts("C(=O)[OX2H1]"),
        "Carbonyl / Ketone / Aldehyde (C=O)": Chem.MolFromSmarts("[#6]C(=O)[#6]"),
        "Primary Amine (-NH2)": Chem.MolFromSmarts("[NX3H2]"),
        "Ether (R-O-R)": Chem.MolFromSmarts("[#6][OX2][#6]"),
        "Halogen (-F, -Cl, -Br, -I)": Chem.MolFromSmarts("[F,Cl,Br,I]")
    }
    results = {}
    for name, smarts in patterns.items():
        if smarts:
            matches = mol.GetSubstructMatches(smarts)
            results[name] = len(matches)
    return results


# ============================================================
# NEW FEATURE 3: TANIMOTO SIMILARITY CALCULATOR
# ============================================================

def calculate_similarity_with_drugs(mol):
    reference_drugs = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
        "Ethanol": "CCO"
    }
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    similarities = {}
    for name, ref_smi in reference_drugs.items():
        ref_mol = Chem.MolFromSmiles(ref_smi)
        if ref_mol:
            fp2 = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
            sim = TanimotoSimilarity(fp1, fp2)
            similarities[name] = round(sim * 100, 2)
    return similarities


# ============================================================
# CALCULATE PROPERTIES & BONDS
# ============================================================

def calculate_bond_information(mol):
    heavy_atom_bonds = mol.GetNumBonds()
    kekule_mol = Chem.Mol(mol)
    try:
        Chem.Kekulize(kekule_mol, clearAromaticFlags=True)
    except Exception:
        pass

    pi_bonds = 0
    for bond in kekule_mol.GetBonds():
        btype = bond.GetBondType()
        if btype == Chem.BondType.DOUBLE:
            pi_bonds += 1
        elif btype == Chem.BondType.TRIPLE:
            pi_bonds += 2
        elif bond.GetIsAromatic():
            pi_bonds += 0.5

    mol_with_h = Chem.AddHs(Chem.Mol(mol))
    sigma_bonds = mol_with_h.GetNumBonds()
    return {
        "Heavy Atom Bonds": heavy_atom_bonds,
        "Sigma Bonds": sigma_bonds,
        "Pi Bonds": pi_bonds,
        "Total Bonds": sigma_bonds + pi_bonds,
    }

def calculate_properties(mol):
    bond_info = calculate_bond_information(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = Descriptors.MolWt(mol)
    
    properties = {
        "Molecular Formula": formula,
        "Molecular Weight": round(mw, 3),
        "Exact Molecular Weight": round(Descriptors.ExactMolWt(mol), 4),
        "LogP": round(Crippen.MolLogP(mol), 3),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "Hydrogen Bond Donors": Lipinski.NumHDonors(mol),
        "Hydrogen Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),
        "Ring Count": Lipinski.RingCount(mol),
        "QED Score (Drug-likeness)": round(QED.qed(mol), 3),
        "Number of Atoms": mol.GetNumAtoms(),
        "Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Sigma (σ) Bonds": bond_info["Sigma Bonds"],
        "Pi (π) Bonds": bond_info["Pi Bonds"],
    }
    return properties

def generate_2d_image(mol):
    return Draw.MolToImage(mol, size=(700, 500))

def generate_3d_molecule(mol):
    try:
        mol3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        status = AllChem.EmbedMolecule(mol3d, params)
        if status != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(mol3d)
        except Exception:
            pass
        return mol3d
    except Exception:
        return None

def display_3d_structure(mol3d, style):
    mol_block = Chem.MolToMolBlock(mol3d)
    view = py3Dmol.view(width=900, height=600)
    view.addModel(mol_block, "mol")
    if style == "Ball and Stick":
        view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.30}})
    elif style == "Stick":
        view.setStyle({"stick": {"radius": 0.20}})
    elif style == "Space Filling":
        view.setStyle({"sphere": {"scale": 1.0}})
    view.setBackgroundColor("white")
    view.zoomTo()
    components.html(view._make_html(), height=620)

def save_2d_structure_png(mol):
    img = Draw.MolToImage(mol, size=(700, 500))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    img.save(tmp.name)
    return tmp.name

def save_3d_structure_png(mol):
    import matplotlib.pyplot as plt
    import numpy as np
    mol3d = generate_3d_molecule(mol)
    if mol3d is None:
        return None
    conf = mol3d.GetConformer()
    coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(mol3d.GetNumAtoms())], dtype=float)
    centered = coords - coords.mean(axis=0)
    if mol3d.GetNumAtoms() >= 3:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh[:2].T
    else:
        projected = centered[:, :2]
    
    fig, ax = plt.subplots(figsize=(8.8, 6.4), facecolor="white")
    ax.set_facecolor("white")
    for i, atom in enumerate(mol3d.GetAtoms()):
        x, y = projected[i]
        ax.scatter(x, y, s=250, c="#3F3F3F", edgecolors="#222222")
    ax.axis("off")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    fig.savefig(tmp.name, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return tmp.name

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
            ["QED Score", str(properties["QED Score"])]
        ]
        t = Table(basic_data, colWidths=[2.0 * inch, 4.7 * inch])
        story.append(t)
        story.append(Spacer(1, 0.15 * inch))
        
    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🧪 Virtual Lab")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📖 Theory",
        "🧬 Molecular Visualization",
        "📊 Molecular Properties Report",
        "💊 Drug-Likeness (Ro5)",
        "🔍 Functional Groups",
        "⚖️ Drug Similarity & QED",
        "📈 Structure–Property Analysis",
        "📝 Assessment",
        "📄 Final Project Report"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### 👩‍🎓 Student Information")
student_name = st.sidebar.text_input("Student Name", key="student_name")
registration_number = st.sidebar.text_input("Registration Number", key="registration_number")


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":
    st.markdown('<div class="main-title">🧪 Cheminformatics Virtual Laboratory</div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align:center; font-size:24px; font-weight:700; color:#1f4e79;">
        SRM Institute of Science and Technology, Tiruchirappalli
        </div>
        <div style="text-align:center; font-size:18px; font-weight:600; margin-top:5px;">
        Faculty of Engineering and Technology | School of Sciences | Division of Chemistry
        </div>
        """, unsafe_allow_html=True)
    st.divider()
    st.markdown("## Welcome to the Virtual Laboratory")
    st.write("Explore molecular properties, 3D visualization, drug-likeness rules, functional groups, and advanced cheminformatics tools.")
    
    example_df = pd.DataFrame({
        "Molecule": ["Ethanol", "Benzene", "Phenol", "Aspirin", "Caffeine", "Paracetamol"],
        "SMILES": ["CCO", "c1ccccc1", "Oc1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(=O)Nc1ccc(O)cc1"]
    })
    st.dataframe(example_df, use_container_width=True, hide_index=True)
    st.divider()
    st.markdown("### 👨‍🏫 Developed by")
    st.markdown("**Dr. R. Subramanian**\nAssistant Professor, Division of Chemistry, SRMIST Trichy")


# ============================================================
# THEORY PAGE
# ============================================================

elif page == "📖 Theory":
    st.title("📖 Introduction to Cheminformatics")
    st.markdown("""
    Cheminformatics combines chemistry, computer science, and data analysis to solve complex chemical problems, design new drugs, and analyze molecular libraries.
    """)


# ============================================================
# MOLECULAR VISUALIZATION PAGE
# ============================================================

elif page == "🧬 Molecular Visualization":
    st.title("🧬 Molecular Visualization Laboratory")
    smiles = st.text_input("Enter SMILES", key="active_smiles")
    mol = get_molecule(smiles)

    if mol is None:
        st.error("❌ Invalid SMILES notation.")
    else:
        st.success("✅ Valid Molecular Structure!")
        st.session_state.latest_smiles = smiles
        st.session_state.latest_properties = calculate_properties(mol)
        
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.subheader("2D Structure")
            st.image(generate_2d_image(mol))
        with col2:
            st.subheader("Basic Info")
            st.write("**Formula:**", rdMolDescriptors.CalcMolFormula(mol))
            st.write("**Molecular Weight:**", round(Descriptors.MolWt(mol), 3))

        st.divider()
        st.subheader("🌐 Interactive 3D Model")
        style = st.selectbox("Select Model", ["Ball and Stick", "Stick", "Space Filling"])
        if st.button("🚀 Generate 3D Structure"):
            mol3d = generate_3d_molecule(mol)
            if mol3d:
                display_3d_structure(mol3d, style)
            else:
                st.error("Unable to generate 3D structure.")


# ============================================================
# MOLECULAR PROPERTIES REPORT
# ============================================================

elif page == "📊 Molecular Properties Report":
    st.title("📊 Complete Molecular Properties Report")
    smiles = st.text_input("Enter New SMILES", key="active_smiles")
    if smiles:
        mol = get_molecule(smiles)
        if mol is None:
            st.error("❌ Invalid SMILES notation.")
        else:
            st.success("✅ Valid Molecular Structure Detected!")
            props = calculate_properties(mol)
            st.session_state.latest_smiles = smiles
            st.session_state.latest_properties = props
            df = pd.DataFrame(list(props.items()), columns=["Property", "Value"])
            st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================
# DRUG-LIKENESS (Ro5)
# ============================================================

elif page == "💊 Drug-Likeness (Ro5)":
    st.title("💊 Lipinski's Rule of 5 Checker")
    smiles = st.text_input("Enter SMILES", value=st.session_state.get("active_smiles", "CCO"))
    mol = get_molecule(smiles)

    if mol is None:
        st.error("❌ Invalid SMILES notation.")
    else:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MW", f"{mw:.2f}", "≤ 500" if mw <= 500 else "Failed")
        col2.metric("LogP", f"{logp:.2f}", "≤ 5" if logp <= 5 else "Failed")
        col3.metric("HBD", f"{hbd}", "≤ 5" if hbd <= 5 else "Failed")
        col4.metric("HBA", f"{hba}", "≤ 10" if hba <= 10 else "Failed")

        if violations == 0:
            st.success("🎉 **Passed!** Fully satisfies Lipinski's Rule of Five.")
        else:
            st.warning(f"⚠️ **Notice:** Has **{violations}** violation(s) of Lipinski's Rule of Five.")


# ============================================================
# NEW FEATURE 1 PAGE: FUNCTIONAL GROUPS
# ============================================================

elif page == "🔍 Functional Groups":
    st.title("🔍 Functional Group Identifier")
    st.info("Automatically detects key chemical functional groups in the input molecule using SMARTS patterns.")
    
    smiles = st.text_input("Enter SMILES", value=st.session_state.get("active_smiles", "CC(=O)Oc1ccccc1C(=O)O"))
    mol = get_molecule(smiles)

    if mol is None:
        st.error("❌ Invalid SMILES notation.")
    else:
        groups = analyze_functional_groups(mol)
        fg_df = pd.DataFrame(list(groups.items()), columns=["Functional Group", "Count Found"])
        st.dataframe(fg_df, use_container_width=True, hide_index=True)
        st.image(generate_2d_image(mol))


# ============================================================
# NEW FEATURE 2 & 3 PAGE: QED & TANIMOTO SIMILARITY
# ============================================================

elif page == "⚖️ Drug Similarity & QED":
    st.title("⚖️ Drug-Likeness QED Score & Tanimoto Similarity")
    st.info("Analyze molecular drug-likeness desirability (QED) and compare similarity against benchmark drugs.")

    smiles = st.text_input("Enter SMILES", value=st.session_state.get("active_smiles", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"))
    mol = get_molecule(smiles)

    if mol is None:
        st.error("❌ Invalid SMILES notation.")
    else:
        qed_val = QED.qed(mol)
        st.metric("Quantitative Estimate of Drug-likeness (QED)", f"{qed_val:.3f}", "Scale: 0.0 to 1.0")
        
        st.subheader("🤝 Tanimoto Similarity with Reference Drugs (%)")
        sims = calculate_similarity_with_drugs(mol)
        sim_df = pd.DataFrame(list(sims.items()), columns=["Reference Drug", "Similarity (%)"])
        st.dataframe(sim_df, use_container_width=True, hide_index=True)


# ============================================================
# STRUCTURE PROPERTY ANALYSIS
# ============================================================

elif page == "📈 Structure–Property Analysis":
    st.title("📈 Structure–Property Analysis")
    default_data = "Ethanol,CCO\nBenzene,c1ccccc1\nAspirin,CC(=O)Oc1ccccc1C(=O)O\nCaffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    molecule_input = st.text_area("Enter Molecules (Name, SMILES)", value=default_data, height=150)
    
    rows = []
    for line in molecule_input.splitlines():
        if "," not in line: continue
        name, s = line.split(",", 1)
        m = get_molecule(s)
        if m:
            p = calculate_properties(m)
            rows.append({"Molecule": name.strip(), "MW": p["Molecular Weight"], "LogP": p["LogP"], "QED": p["QED Score"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ============================================================
# ASSESSMENT
# ============================================================

elif page == "📝 Assessment":
    st.title("📝 Cheminformatics Assessment")
    q1 = st.radio("What is the full form of SMILES?", ["Simplified Molecular Input Line Entry System", "Standard Molecular Language System"], index=None)
    if st.button("Submit"):
        if q1 == "Simplified Molecular Input Line Entry System":
            st.success("Correct! 🎉")
        elif q1:
            st.error("Incorrect. Try again!")


# ============================================================
# FINAL PROJECT REPORT
# ============================================================

elif page == "📄 Final Project Report":
    st.title("📄 Final Student Molecular Analysis Report")
    name = st.text_input("Student Name for Report", value=st.session_state.get("student_name", ""))
    reg = st.text_input("Registration Number for Report", value=st.session_state.get("registration_number", ""))
    
    multi_smiles_text = st.text_area("Enter SMILES (one per line)", value=st.session_state.get("latest_smiles", "CCO"), height=100)
    smiles_entries = [line.strip() for line in multi_smiles_text.splitlines() if line.strip()]
    
    molecule_records = []
    for s in smiles_entries:
        m = get_molecule(s)
        if m:
            molecule_records.append((s, m, calculate_properties(m)))
            
    if molecule_records and st.button("📄 Generate Combined PDF Report"):
        pdf = generate_multi_final_report(name, reg, molecule_records)
        st.success("✅ Report generated successfully!")
        st.download_button("⬇️ Download PDF Report", data=pdf, file_name="Cheminformatics_Report.pdf", mime="application/pdf")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("🧪 Cheminformatics Virtual Laboratory | Developed using Python, Streamlit and RDKit")
