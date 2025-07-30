
---

## 🔧 Features

### 1. Enrichment Analysis
- KEGG-based metabolite set enrichment
- Hypergeometric test implementation
- P-value correction (Bonferroni, FDR)

### 2. Joint Pathway Analysis
- Combines statistical and enrichment results
- Pathway impact scoring
- Visualization-ready output (JSON/CSV)

### 3. One-Factor Statistical Analysis
- Supports t-test, ANOVA, PCA
- Volcano plot generation
- Normalization and scaling methods

---

## 🛠️ Tech Stack

- **Python 3.x**
- Custom-built statistical and visualization modules
- No use of proprietary or original MetaBioAnalyst libraries

---

## 📁 Data Input Format

All input files must be in CSV format:

- **Metabolite Table**: Rows = metabolites, Columns = samples
- **Group File**: Two-column format: `SampleID, GroupLabel`

---

## 🚀 How to Run

Each module is self-contained. Example usage for statistical analysis:

```bash
cd "Statistical analysis(one factor)"
python main.py --data input_data.csv --group group_file.csv
