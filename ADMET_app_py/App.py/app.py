import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from math import pi

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw, AllChem

# Configuring Streamlit page
st.set_page_config(
    page_title="ADMET-AI Enterprise Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Path to the trained model
MODEL_PATH = r"C:\python projects\Project 8\data\hERG_xgboost.pkl"


@st.cache_resource
def load_ai_models():
    models = {}
    # Importing the trained hERG model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            models["hERG"] = pickle.load(f)
    return models


models = load_ai_models()


# RDKit + AI Integration
class MoleculeAnalyzer:
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.valid = self.mol is not None

    def get_properties(self):
        if not self.valid:
            return None

        # 1. Physicochemical Properties
        mw = Descriptors.MolWt(self.mol)
        logp = Descriptors.MolLogP(self.mol)
        tpsa = Descriptors.TPSA(self.mol)
        h_donors = Lipinski.NumHDonors(self.mol)

        # 2.XGBoost
        # Generating Morgan Fingerprints (Radius=2, Bits=2048)
        from rdkit.Chem import AllChem

        fp = AllChem.GetMorganFingerprintAsBitVect(self.mol, 2, nBits=2048)
        X_ai = np.array(fp).reshape(1, -1)

        herg_prob = 0
        if "hERG" in models:
            herg_prob = models["hERG"].predict_proba(X_ai)[0][1]  # Toxicity probability

        return {
            "MW": mw,  # Molecular Weight
            "LogP": logp,  # Lipophilicity
            "TPSA": tpsa,  # Polarity
            "H-Donors": h_donors,
            "hERG_Risk": herg_prob,  # AI Prediction
        }


# Visualization
def plot_radar_chart(props):
    # Normalization graph (0-1 scale для красоты)
    # Condense all properties into a single radar chart for a quick overview
    data = {
        "Lipophilicity": min(max(props["LogP"] / 5, 0), 1),
        "Size (MW)": min(props["MW"] / 500, 1),
        "Polarity": min(props["TPSA"] / 140, 1),
        "Solubility": 1 - min(max(props["LogP"] / 5, 0), 1),
        "Safety (hERG)": 1 - props["hERG_Risk"],  # Lower risk == higher safety
    }

    categories = list(data.keys())
    N = len(categories)
    values = list(data.values())
    values += values[:1]  # End of the loop

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"projection": "polar"})

    # Draw background
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, color="grey", size=8)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=7)
    plt.ylim(0, 1)

    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle="solid", color="#00ff41")
    ax.fill(angles, values, "#0cf747", alpha=0.2)

    # "Dark Mode"
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.spines["polar"].set_color("#333")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    return fig


# Interface (SIDEBAR NAV)
st.sidebar.title(" ADMET-AI Pro")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation", ["Dashboard", "Batch Screening", "Model Internals"]
)
st.sidebar.markdown("---")
st.sidebar.info(
    f"System Status:\n RDKit Core: Active\n{'🟢' if 'hERG' in models else '🔴'} AI Model: {'Loaded' if 'hERG' in models else 'Offline'}"
)

# Page 1: Dashboard
if page == "Dashboard":
    st.title(" Molecular Profiling Dashboard")
    st.markdown(
        "Real-time analysis of physicochemical properties and AI-driven toxicity prediction."
    )

    col_input, col_viz = st.columns([1, 2])

    with col_input:
        st.subheader("Input")
        smiles_input = st.text_area(
            "SMILES String",
            "CC(=O)Oc1ccccc1C(=O)O",
            help="Enter molecule structure here",
        )

        analyze_btn = st.button("RUN FULL ANALYSIS ⚡", use_container_width=True)

        st.markdown("###  History")
        st.caption("No recent queries.")

    if analyze_btn:
        analyzer = MoleculeAnalyzer(smiles_input)

        if analyzer.valid:
            props = analyzer.get_properties()

            with col_viz:
                # Website-like tabs for different sections of the analysis
                tab1, tab2, tab3 = st.tabs(
                    ["📊 Overview", "⚠️ Toxicity AI", "🔬 Chemistry"]
                )

                with tab1:
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.image(Draw.MolToImage(analyzer.mol), caption="2D Structure")
                    with c2:
                        st.pyplot(plot_radar_chart(props))

                with tab2:
                    st.subheader("hERG Cardiotoxicity Prediction")
                    risk = props["hERG_Risk"]

                    # Indicator of risk level with color coding and progress bar
                    st.metric("hERG Blockage Probability", f"{risk*100:.1f}%")

                    if risk > 0.5:
                        st.error("HIGH RISK DETECTED")
                        st.progress(float(risk))
                        st.warning(
                            " This molecule shows structural patterns similar to known hERG blockers."
                        )
                    else:
                        st.success("LOW RISK")
                        st.progress(float(risk))
                        st.info(
                            " The AI model predicts a safe profile regarding hERG inhibition."
                        )

                with tab3:
                    st.dataframe(pd.DataFrame([props]).T.rename(columns={0: "Value"}))

                    # Drug-Likeness Evaluation based on Lipinski's Rule of 5
                    st.markdown("#### Lipinski Rule of 5 Status")
                    fails = 0
                    if props["MW"] > 500:
                        fails += 1
                    if props["LogP"] > 5:
                        fails += 1
                    if props["H-Donors"] > 5:
                        fails += 1

                    if fails == 0:
                        st.success(" Excellent Drug-Likeness (0 violations)")
                    elif fails == 1:
                        st.warning(" Acceptable (1 violation)")
                    else:
                        st.error(f" Poor Bioavailability ({fails} violations)")

        else:
            st.error("Invalid SMILES string.")

## Page 2 (Batch Screening)
elif page == "Batch Screening":
    st.title("🏭 Batch Screening Module")
    st.markdown(
        """
    **High-Throughput Virtual Screening (HTVS)**
    Upload a CSV file containing a column named **'smiles'** (or 'SMILES'). 
    The system will predict hERG toxicity for thousands of molecules in seconds.
    """
    )

    # 1. Importing file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # 2. Searching for the SMILES column
        smiles_col = None
        for col in df.columns:
            if col.lower() == "smiles":
                smiles_col = col
                break

        if smiles_col:
            st.success(f" Found molecule column: '{smiles_col}' | Rows: {len(df)}")

            if st.button(f"Analyze {len(df)} Molecules "):
                # Progress bar and status text
                progress_bar = st.progress(0)
                status_text = st.empty()

                results = []
                risks = []

                # 3. Cycle for rows (Vectorized approach would be faster, but loop is safer for RDKit)
                # For very large datasets, consider batch processing or multiprocessing to speed up analysis. Here we keep it simple for demonstration.
                total = len(df)

                for i, row in df.iterrows():
                    smi = row[smiles_col]
                    analyzer = MoleculeAnalyzer(smi)

                    if analyzer.valid:
                        # Risk assessment using the AI model
                        props = analyzer.get_properties()
                        risk = props["hERG_Risk"]

                        # Result labeling based on risk threshold
                        label = "TOXIC" if risk > 0.5 else "SAFE"
                        results.append(label)
                        risks.append(risk)
                    else:
                        results.append("INVALID_SMILES")
                        risks.append(-1)

                    # Update progress every 10 molecules to avoid excessive UI updates
                    if i % 10 == 0:
                        progress = int((i / total) * 100)
                        progress_bar.progress(float(progress / 100))
                        status_text.text(f"Processing molecule {i+1}/{total}...")

                # Finished processing
                progress_bar.progress(1.0)
                status_text.text("Analysis Complete! ")

                # 4. Add results to DataFrame
                df["AI_Verdict"] = results
                df["hERG_Probability"] = risks

                # Preview
                st.subheader("Results Preview")

                # Beautyful conditional formatting for the verdict column
                def color_danger(val):
                    color = (
                        "#ff4b4b"
                        if val == "TOXIC"
                        else "#00cc66" if val == "SAFE" else "grey"
                    )
                    return f"color: {color}; font-weight: bold"

                st.dataframe(
                    df.head(50).style.applymap(color_danger, subset=["AI_Verdict"])
                )

                # 5. Download option for the full results
                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv,
                    file_name="hERG_screening_results.csv",
                    mime="text/csv",
                )

        else:
            st.error(" Column 'smiles' not found in CSV. Please rename your column.")
            st.write("Available columns:", df.columns.tolist())

# Page 3 (Model Internals)
elif page == "Model Internals":
    st.title("🧠 Model Architecture")
    st.json(
        {
            "Model Type": "XGBoost Classifier (Gradient Boosting)",
            "Input Features": "Morgan Fingerprints (2048 bits, Radius 2)",
            "Training Data": "ChEMBL v33 (18,000+ compounds)",
            "Target": "hERG IC50 < 10uM",
        }
    )
