```python
import os
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, QED
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# ---------------------------------------------------------
# Helper Functions for Chemistry Calculations
# ---------------------------------------------------------
def calculate_properties(smiles_str):
    mol = Chem.MolFromSmiles(smiles_str)
    if not mol:
        return None
    
    # Calculate key cheminformatics descriptors
    canonical_smiles = Chem.MolToSmiles(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    qed_score = QED.qed(mol)
    
    # Lipinski Rule of 5 evaluation
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    
    lipinski_status = "Pass" if violations <= 1 else "Fail"
    
    return {
        "mol": mol,
        "input_smiles": smiles_str,
        "canonical_smiles": canonical_smiles,
        "formula": formula,
        "mw": round(mw, 2),
        "logp": round(logp, 2),
        "tpsa": round(tpsa, 2),
        "hbd": hbd,
        "hba": hba,
        "rotb": rotb,
        "qed": round(qed_score, 3),
        "lipinski": lipinski_status,
        "violations": violations
    }

def generate_structure_image(mol, filename="temp_mol.png"):
    img = Draw.MolToImage(mol, size=(300, 200))
    img.save(filename)
    return filename

# ---------------------------------------------------------
# ReportLab PDF Generation Function
# ---------------------------------------------------------
def generate_enhanced_pdf(data_list, output_filename="Cheminformatics_Comprehensive_Report.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor('#1A365D'), alignment=1, spaceAfter=10)
    subtitle_style = ParagraphStyle('DocSubTitle', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.HexColor('#4A5568'), alignment=1, spaceAfter=15)
    h1_style = ParagraphStyle('Heading1_Custom', parent=styles['Heading2'], fontSize=13, leading=16, textColor=colors.HexColor('#2B6CB0'), spaceBefore=10, spaceAfter=8)
    body_style = ParagraphStyle('Body_Custom', parent=styles['Normal'], fontSize=9, leading=12, textColor=colors.HexColor('#2D3748'))
    code_style = ParagraphStyle('Code_Style', parent=styles['Code'], fontSize=8, leading=10, textColor=colors.HexColor('#1A202C'), backColor=colors.HexColor('#EDF2F7'), borderPadding=6)

    story = []

    # ---------------------------------------------------------
    # PAGE 1: Header & Molecular Tables with Structure Diagrams
    # ---------------------------------------------------------
    story.append(Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY", title_style))
    story.append(Paragraph("Comprehensive Analysis & Website Technical Report", subtitle_style))
    story.append(Spacer(1, 10))

    for idx, item in enumerate(data_list, 1):
        story.append(Paragraph(f"Molecule {idx}: Structural Diagram & Descriptors", h1_style))
        
        # Save temporary molecule image
        img_path = generate_structure_image(item['mol'], f"temp_mol_{idx}.png")
        
        # Build Property Table Data
        prop_data = [
            [Paragraph("<b>Property</b>", body_style), Paragraph("<b>Value</b>", body_style), Paragraph("<b>Interpretation</b>", body_style)],
            ["Formula", item["formula"], "Elemental composition"],
            ["Molecular Weight", f"{item['mw']} g/mol", "Average molecular mass"],
            ["LogP", str(item["logp"]), "Lipophilicity / Partitioning"],
            ["TPSA", f"{item['tpsa']} Å²", "Polar surface area for permeability"],
            ["H-Bond Donors / Acceptors", f"{item['hbd']} / {item['hba']}", "Intermolecular bonding capacity"],
            ["Rotatable Bonds", str(item["rotb"]), "Conformational flexibility"],
            ["QED Score", str(item["qed"]), "Quantitative Estimate of Drug-likeness"],
            ["Lipinski Rule of 5", item["lipinski"], f"Violations: {item['violations']}"]
        ]

        t_table = Table(prop_data, colWidths=[1.8*inch, 1.0*inch, 2.2*inch])
        t_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')])
        ]))

        # Place Diagram and Table Side-by-Side in a Layout Container
        layout_grid = [
            [RLImage(img_path, width=2.1*inch, height=1.4*inch), t_table]
        ]
        layout_table = Table(layout_grid, colWidths=[2.2*inch, 5.0*inch])
        layout_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        
        story.append(layout_table)
        story.append(Spacer(1, 12))

    story.append(PageBreak())

    # ---------------------------------------------------------
    # PAGE 2: Key Features & Platform Technology Details
    # ---------------------------------------------------------
    story.append(Paragraph("Virtual Laboratory Platform Key Features", h1_style))
    
    features_data = [
        [Paragraph("<b>Key Feature</b>", body_style), Paragraph("<b>Functional Description</b>", body_style)],
        ["Interactive SMILES Input", "Allows students/researchers to parse raw chemical strings into valid 2D spatial structures."],
        ["Automated Descriptor Engine", "Instantly computes topological and physicochemical parameters using RDKit."],
        ["Drug-Likeness Profiling", "Evaluates compounds against Lipinski's Rule of 5 and calculates QED scores."],
        ["Visual Structure Rendering", "Generates high-resolution 2D chemical structure diagrams automatically."],
        ["Exportable PDF Reports", "Compiles laboratory data into standardized academic-grade PDF documents."]
    ]
    
    feat_table = Table(features_data, colWidths=[2.2*inch, 5.0*inch])
    feat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("Technologies & Libraries Employed", h1_style))
    tech_data = [
        [Paragraph("<b>Technology / Library</b>", body_style), Paragraph("<b>Core Function in Platform</b>", body_style)],
        ["Python", "Primary backend engine for handling computational chemistry workflows."],
        ["Streamlit", "Web framework used to build the interactive virtual laboratory UI."],
        ["RDKit", "Cheminformatics toolkit for structure parsing, image rendering, and descriptors."],
        ["Pandas", "Structures tabular property outputs for seamlessly rendering data tables."],
        ["ReportLab", "Programmatic PDF engine generating structured multi-page document outputs."]
    ]
    
    tech_table = Table(tech_data, colWidths=[2.2*inch, 5.0*inch])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(tech_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("Key Interpretation & Analytical Findings", h1_style))
    interp_text = (
        "<b>Molecular Identity & Polarity:</b> The evaluated molecular structures reflect varied hydrogen bonding "
        "and polar surface areas (TPSA). TPSA values below 140 Å² suggest favorable cell permeability.<br/>"
        "<b>Drug-Likeness Evaluation:</b> QED and Lipinski indicators highlight the potential bioavailability "
        "and pharmacological suitability of the submitted chemical compounds."
    )
    story.append(Paragraph(interp_text, body_style))
    
    story.append(PageBreak())

    # ---------------------------------------------------------
    # PAGE 3: Technical Implementation & Python Source Code
    # ---------------------------------------------------------
    story.append(Paragraph("Virtual Laboratory Core Execution Code", h1_style))
    story.append(Paragraph("The Python script below illustrates the workflow used to parse SMILES strings and calculate key descriptors programmatically:", body_style))
    story.append(Spacer(1, 8))

    sample_code = """# Core Cheminformatics Pipeline Implementation
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED

def analyze_molecule(smiles: str):
    # Parse SMILES String into Mol Object
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
        
    # Calculate Core Descriptors
    data = {
        'Formula': rdMolDescriptors.CalcMolFormula(mol),
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'QED': QED.qed(mol),
        'HBD': rdMolDescriptors.CalcNumHBD(mol),
        'HBA': rdMolDescriptors.CalcNumHBA(mol)
    }
    return data

# Example Usage
print(analyze_molecule('CC(=O)Oc1ccccc1C(=O)O'))"""

    story.append(Paragraph(sample_code.replace('\n', '<br/>').replace(' ', '&nbsp;'), code_style))

    # Build Document
    doc.build(story)
    
    # Clean up temporary images
    for idx in range(1, len(data_list) + 1):
        temp_file = f"temp_mol_{idx}.png"
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return output_filename


# ---------------------------------------------------------
# Streamlit App UI
# ---------------------------------------------------------
st.set_page_config(page_title="Cheminformatics Virtual Lab", layout="wide")

st.title("🧪 Cheminformatics Virtual Laboratory")
st.markdown("Analyze molecular structures, compute physicochemical descriptors, and generate comprehensive PDF reports.")

# Default input
smiles_input = st.text_area("Enter SMILES Strings (one per line):", "Oc1ccccc1\nCC(=O)Oc1ccccc1C(=O)O")

if st.button("Run Molecular Analysis & Generate Report"):
    smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]
    results = []
    
    for s in smiles_list:
        res = calculate_properties(s)
        if res:
            results.append(res)
            
    if results:
        st.success(f"Successfully processed {len(results)} molecule(s)!")
        
        # Display structures & data in Web UI
        cols = st.columns(len(results))
        for idx, res in enumerate(results):
            with cols[idx]:
                st.subheader(f"Molecule {idx+1}")
                st.image(Draw.MolToImage(res['mol']), use_column_width=True)
                st.json({
                    "Formula": res["formula"],
                    "MW": res["mw"],
                    "LogP": res["logp"],
                    "QED": res["qed"],
                    "Lipinski": res["lipinski"]
                })
        
        # Generate PDF
        pdf_path = generate_enhanced_pdf(results)
        
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download Comprehensive PDF Report",
                data=pdf_file,
                file_name="Cheminformatics_Virtual_Lab_Report.pdf",
                mime="application/pdf"
            )
    else:
        st.error("Please enter valid SMILES strings.")
```
