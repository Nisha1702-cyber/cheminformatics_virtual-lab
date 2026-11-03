import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from rdkit.Chem.Draw import rdMolDraw2D
import io
import os
from PIL import Image

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Page Configuration
st.set_page_config(page_title="Cheminformatics Virtual Laboratory", layout="wide")

st.title("🧪 Cheminformatics Virtual Laboratory")
st.subheader("Automated Molecular Analysis & PDF Report Generation")

# Student Details
student_name = "Nisha Vaishnavi K"
reg_no = "RA261103005031"

st.sidebar.header("User Details")
st.sidebar.text(f"Name: {student_name}")
st.sidebar.text(f"Reg No: {reg_no}")

# SMILES Input
smiles_input = st.text_area(
    "Enter SMILES Strings (one per line):",
    value="Oc1ccccc1\nCC(=O)Oc1ccccc1C(=O)O",
    height=100
)

def generate_molecule_image(mol):
    """Generate 2D Molecule Image using RDKit"""
    drawer = rdMolDraw2D.MolDraw2DCairo(200, 150)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()
    return io.BytesIO(img_data)

def analyze_molecule(smiles_str):
    mol = Chem.MolFromSmiles(smiles_str)
    if not mol:
        return None
    
    canonical_smiles = Chem.MolToSmiles(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    qed_val = QED.qed(mol)
    
    # Lipinski Evaluation
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    
    lipinski_status = "Pass" if violations <= 1 else "Fail"
    
    return {
        "mol": mol,
        "smiles": smiles_str,
        "canonical_smiles": canonical_smiles,
        "formula": formula,
        "mw": round(mw, 2),
        "logp": round(logp, 2),
        "tpsa": round(tpsa, 2),
        "hbd": hbd,
        "hba": hba,
        "rotb": rotb,
        "qed": round(qed_val, 2),
        "lipinski": lipinski_status,
        "violations": violations
    }

def build_pdf_report(results, student_name, reg_no):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=colors.HexColor('#1A365D'),
        alignment=1,
        spaceAfter=10
    )
    h2_style = ParagraphStyle(
        'H2Style',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=10,
        spaceAfter=5
    )
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=11
    )
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=10
    )
    
    story = []
    
    # PAGE 1: Header & Molecular Data Tables with Images
    story.append(Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY", title_style))
    story.append(Paragraph(f"<b>MULTI-MOLECULE STUDENT ANALYSIS REPORT</b>", body_style))
    story.append(Paragraph(f"<b>Student:</b> {student_name} | <b>Registration Number:</b> {reg_no}", body_style))
    story.append(Paragraph(f"<b>Number of molecules analyzed:</b> {len(results)}", body_style))
    story.append(Spacer(1, 10))
    
    for idx, data in enumerate(results, 1):
        story.append(Paragraph(f"<b>Molecule {idx}: Molecular Information & Structure</b>", h2_style))
        
        # Table Data
        table_data = [
            ["Property", "Value", "Interpretation"],
            ["Input SMILES", data['smiles'], "User provided structural code"],
            ["Canonical SMILES", data['canonical_smiles'], "Standardized structural format"],
            ["Molecular Formula", data['formula'], "Elemental composition"],
            ["Molecular Weight", f"{data['mw']} g/mol", "Average molecular mass"],
            ["LogP", str(data['logp']), "Lipophilicity balance"],
            ["TPSA (Å²)", str(data['tpsa']), "Polar surface area for permeability"],
            ["H-Bond Donors", str(data['hbd']), "Number of hydrogen bond donors"],
            ["H-Bond Acceptors", str(data['hba']), "Number of hydrogen bond acceptors"],
            ["Rotatable Bonds", str(data['rotb']), "Indicates molecular flexibility"],
            ["QED Score", str(data['qed']), "Drug-likeness quantitative estimate"],
            ["Lipinski Rule of 5", data['lipinski'], f"Violations: {data['violations']}"]
        ]
        
        t = Table(table_data, colWidths=[110, 160, 240])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A365D')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')]),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('TOPPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))
    
    # PAGE BREAK TO PAGE 2
    story.append(PageBreak())
    
    # PAGE 2: Key Interpretation & Platform Features
    story.append(Paragraph("Key Interpretation & Virtual Lab Features", title_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("<b>1. Scientific Interpretation Guidelines</b>", h2_style))
    interp_text = """
    • <b>Molecular Identity:</b> Evaluated based on SMILES representations and molecular weight.<br/>
    • <b>Lipophilicity (LogP):</b> Balances non-polar and polar solubility environments.<br/>
    • <b>Molecular Polarity (TPSA):</b> Reflects surface polar atoms impacting cell permeability.<br/>
    • <b>Hydrogen Bonding:</b> Donor/Acceptor counts describe binding affinity and solubility.<br/>
    • <b>Flexibility:</b> Rotatable bonds indicate conformational adaptability.<br/>
    • <b>Drug-Likeness:</b> Evaluated via QED Score and Lipinski's Rule of 5.
    """
    story.append(Paragraph(interp_text, body_style))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("<b>2. Virtual Laboratory Platform Features</b>", h2_style))
    features_data = [
        ["Feature Module", "Description / Capabilities"],
        ["Real-time Descriptor Engine", "Computes physicochemical descriptors instantly using RDKit."],
        ["Automated Rule Checking", "Evaluates drug-likeness criteria (Lipinski, QED) automatically."],
        ["Structure Rendering", "Generates clean 2D graphical visual diagrams for molecules."],
        ["PDF Report Generator", "Compiles comprehensive multi-page analytical documentation."]
    ]
    ft_table = Table(features_data, colWidths=[150, 360])
    ft_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')]),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(ft_table)
    
    # PAGE BREAK TO PAGE 3
    story.append(PageBreak())
    
    # PAGE 3: Tech Stack & Source Code
    story.append(Paragraph("Technical Stack & Core Logic Source Code", title_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("<b>1. Python Libraries Used and Their Importance</b>", h2_style))
    lib_data = [
        ["Library", "Importance in this Virtual Lab Project"],
        ["Streamlit", "Provides interactive web interface and UI components."],
        ["RDKit", "Core Cheminformatics engine for structure parsing & descriptor calculation."],
        ["Pandas", "Handles structured data tables and tabular formatting."],
        ["ReportLab", "Generates high-quality PDF report layouts dynamically."]
    ]
    lib_table = Table(lib_data, colWidths=[120, 390])
    lib_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A365D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')]),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(lib_table)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("<b>2. Illustrative Core Python Code</b>", h2_style))
    code_text = """
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    return {
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "QED": QED.qed(mol)
    }

# Execution Example
data = compute_descriptors('CC(=O)Oc1ccccc1C(=O)O')
print(data)
    """
    story.append(Paragraph(code_text.replace('\n', '<br/>').replace(' ', '&nbsp;'), code_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Main Logic
if smiles_input:
    smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]
    results = []
    
    for sm in smiles_list:
        res = analyze_molecule(sm)
        if res:
            results.append(res)
            
    if results:
        st.write("### Molecule Analysis Results")
        
        # Display Molecules in Web UI
        for idx, r in enumerate(results, 1):
            col1, col2 = st.columns([1, 2])
            with col1:
                img
