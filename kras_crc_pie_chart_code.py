import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from scipy import stats
import gseapy as gp
import re
from numpy.random import RandomState
import sys
import numpy as np
import textwrap

np.seterr(all='ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('gseapy').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = r"C:\Users\Sahaj\Documents\UTSW Westover"
OUTPUT_DIR = "crc_output"
P_VALUE_THRESHOLD = 0.05
TOP_PATHWAYS = 10
GSEA_PADJ_THRESHOLD = 0.25

def load_data(filename):
    filepath = os.path.join(DATA_DIR, filename)
    try:
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"Successfully loaded {filename}")
        logger.info(f"Shape of {filename}: {df.shape}")
        logger.info(f"Columns in {filename}: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"Empty file: {filename}")
        return None
    except Exception as e:
        logger.error(f"Error loading {filename}: {str(e)}")
        return None

def filter_crc(df):
    if 'OncotreePrimaryDisease' not in df.columns:
        logger.error("OncotreePrimaryDisease column not found in MODEL.CSV")
        return None
    crc_df = df[df['OncotreePrimaryDisease'] == 'Colorectal Adenocarcinoma']
    logger.info(f"Number of Colorectal Adenocarcinoma models: {len(crc_df)}")
    if len(crc_df) == 0:
        logger.warning("No Colorectal Adenocarcinoma models found")
    return crc_df

def filter_kras_mutations(mutations_df, crc_models):
    required_columns = ['ModelID', 'HugoSymbol', 'ProteinChange', 'VepImpact']
    if not all(col in mutations_df.columns for col in required_columns):
        logger.error(f"Required columns not found in OmicsSomaticMutations.CSV. Columns present: {mutations_df.columns.tolist()}")
        return None, None
    
    kras_mutations = mutations_df[
        (mutations_df['HugoSymbol'] == 'KRAS') & 
        (mutations_df['ModelID'].isin(crc_models['ModelID'])) &
        (mutations_df['VepImpact'].isin(['HIGH', 'MODERATE']))
    ]
    logger.info(f"Number of KRAS mutations in Colorectal Adenocarcinoma models: {len(kras_mutations)}")
    
    kras_mutation_counts = kras_mutations['ProteinChange'].value_counts()
    significant_mutations = kras_mutation_counts[kras_mutation_counts >= 3].index.tolist()
    
    kras_specific_mutations = {mutation: kras_mutations[kras_mutations['ProteinChange'] == mutation] for mutation in significant_mutations}
    
    for mutation, df in kras_specific_mutations.items():
        logger.info(f"Number of KRAS {mutation} mutations: {len(df)}")
    
    return kras_mutations, kras_specific_mutations

def identify_cell_lines(crc_df, kras_mutations, kras_specific_mutations):
    if 'ModelID' not in crc_df.columns:
        logger.error("ModelID column not found in Colorectal Adenocarcinoma dataframe")
        return None, None
    
    specific_cell_lines = {mutation: crc_df[crc_df['ModelID'].isin(df['ModelID'])] for mutation, df in kras_specific_mutations.items()}
    wt_cell_lines = crc_df[~crc_df['ModelID'].isin(kras_mutations['ModelID'])]
    
    for mutation, cell_lines in specific_cell_lines.items():
        logger.info(f"Number of KRAS {mutation} cell lines: {len(cell_lines)}")
    logger.info(f"Number of KRAS WT cell lines: {len(wt_cell_lines)}")
    
    return specific_cell_lines, wt_cell_lines

def perform_gsea(mutations_df, specific_cell_lines, wt_cell_lines, gene_set_name='c2.cp.kegg_medicus.v2024.1.Hs.symbols.gmt'):
    results = {}
    rng = RandomState(42)
    
    # Pre-process mutations_df to speed up lookups
    mutations_df = mutations_df[['ModelID', 'HugoSymbol']].drop_duplicates()
    
    for mutation, cell_lines in specific_cell_lines.items():
        logger.info(f"Starting GSEA analysis for {mutation}")
        
        if len(cell_lines) == 0 or len(wt_cell_lines) == 0:
            logger.error(f"Cannot perform GSEA for {mutation}: One or both groups have no samples")
            results[mutation] = pd.DataFrame()
            continue

        mutation_models = set(cell_lines['ModelID'].unique())  # Convert to set for faster lookups
        wt_models = set(wt_cell_lines['ModelID'].unique())
        
        logger.info(f"Processing {len(mutation_models)} mutant and {len(wt_models)} WT models for {mutation}")
        
        # Create efficient lookup dictionary
        model_gene_dict = {}
        for _, row in mutations_df.iterrows():
            if row['ModelID'] not in model_gene_dict:
                model_gene_dict[row['ModelID']] = set()
            model_gene_dict[row['ModelID']].add(row['HugoSymbol'])
        
        # Get all genes efficiently
        all_genes = set()
        for genes in model_gene_dict.values():
            all_genes.update(genes)
        all_genes.discard('KRAS')
        all_genes = list(all_genes)  # Convert to list for indexing
        
        logger.info(f"Processing {len(all_genes)} genes for {mutation}")
        
        # Calculate t-statistics in chunks
        chunk_size = 1000
        gene_chunks = [all_genes[i:i + chunk_size] for i in range(0, len(all_genes), chunk_size)]
        
        all_scores = []
        for chunk in gene_chunks:
            chunk_scores = []
            for gene in chunk:
                mut_vals = np.array([1 if gene in model_gene_dict.get(model, set()) else 0 
                                   for model in mutation_models])
                wt_vals = np.array([1 if gene in model_gene_dict.get(model, set()) else 0 
                                  for model in wt_models])
                
                t_stat, _ = stats.ttest_ind(mut_vals, wt_vals)
                chunk_scores.append(t_stat)
            
            all_scores.extend(chunk_scores)
        
        # Add jitter
        scores = np.array(all_scores) + rng.normal(0, 1e-8, size=len(all_scores))
        
        logger.info(f"Running GSEA prerank for {mutation} with {len(all_genes)} genes")
        
        try:
            pre_res = gp.prerank(
                rnk=pd.Series(index=all_genes, data=scores),
                gene_sets=gene_set_name,
                threads=4,
                permutation_num=1000,
                outdir=None,
                seed=42,
                max_size=500,
                min_size=15
            )
            
            logger.info(f"GSEA prerank completed for {mutation}, processing results")
            
            results_df = pre_res.res2d
            results_df['Abs_NES'] = abs(results_df['NES'])
            significant_results = results_df[results_df['FDR q-val'] < 0.25].sort_values('Abs_NES', ascending=False)
            
            logger.info(f"Found {len(significant_results)} significant pathways (FDR q-val < 0.25) for {mutation}")
            
            top_10_results = significant_results.head(10).copy()
            
            if len(top_10_results) < 10:
                logger.warning(f"Only found {len(top_10_results)} significant pathways for {mutation}")
            
            if top_10_results.empty:
                logger.warning(f"No significant pathways found for {mutation}")
            else:
                logger.info(f"Selected top {len(top_10_results)} significant pathways for {mutation}")
            
            results[mutation] = top_10_results
            
        except Exception as e:
            logger.error(f"Error performing GSEA for {mutation}: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            results[mutation] = pd.DataFrame()
    
    return results

def get_top_pathways(gsea_results, TOP_PATHWAYS=10):
    top_pathways = {}
    for mutation, results_df in gsea_results.items():
        if results_df.empty:
            logger.warning(f"No pathways found for {mutation}")
            top_pathways[mutation] = {}
            continue
        
        # Take top 10 pathways based on absolute NES score
        top_paths = results_df.head(TOP_PATHWAYS)
        mutation_top_pathways = dict(zip(top_paths['Term'], top_paths['NES']))
        
        logger.info(f"Top {len(mutation_top_pathways)} pathways for {mutation}: {', '.join(mutation_top_pathways.keys())}")
        top_pathways[mutation] = mutation_top_pathways
    
    return top_pathways

def clean_pathway_name(pathway):
    """Extract meaningful pathway name from the full pathway string."""
    # Remove common prefixes and split
    path = pathway.replace('PID_', '').replace('KEGG_', '')
    parts = re.split(r'[_\s]', path)
    
    # Try to find meaningful identifiers (e.g., PLK1, MAPK, etc.)
    meaningful_parts = []
    for part in parts:
        # Look for parts that contain both letters and numbers, or are all caps
        if (any(c.isalpha() for c in part) and any(c.isdigit() for c in part)) or \
           (part.isupper() and len(part) >= 2):
            meaningful_parts.append(part)
    
    if meaningful_parts:
        return meaningful_parts[0]
    
    # Fallback: use first part if no meaningful identifier found
    return parts[0].upper()

def create_gsea_pie_chart(data, title, output_path, n_cell_lines, TOP_PATHWAYS=10):
    # Convert all values to absolute values
    valid_data = {k: abs(v) for k, v in data.items() if isinstance(k, str)}
    
    if not valid_data:
        logger.warning(f"No valid pathways found for pie chart: {title}")
        return
        
    # Sort by absolute value and get exactly TOP_PATHWAYS
    sorted_items = sorted(valid_data.items(), key=lambda x: x[1], reverse=True)
    top_data = dict(sorted_items[:TOP_PATHWAYS])
    
    logger.info(f"Creating pie chart with top {TOP_PATHWAYS} pathways for {title}")
    
    plt.figure(figsize=(15, 10))
    
    # Generate colors using Pastel1 colormap
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(top_data)))
    
    # Process pathway names - extract meaningful pathway name
    labels = []
    for pathway in top_data.keys():
        # Remove the KEGG_MEDICUS prefix and get the pathway name before "_TO_"
        clean_label = pathway.replace('KEGG_MEDICUS_', '')
        if '_TO_' in clean_label:
            clean_label = clean_label.split('_TO_')[0]
        # Keep only the pathway identifier part
        clean_label = clean_label.split('https://')[0].strip()
        labels.append(clean_label)
    
    sizes = list(top_data.values())
    
    # Calculate percentages
    total = sum(sizes)
    percentages = [size/total * 100 for size in sizes]
    
    # Create the pie chart
    wedges, texts, autotexts = plt.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.6),
        autopct='%1.1f%%',
        pctdistance=0.75,
        explode=[0.02] * len(sizes)
    )
    
    # Adjust label positioning
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))
        
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        
        # Adjust distance based on percentage
        distance = 1.2 if percentages[i] < 10 else 1.15
        x = x * distance
        y = y * distance
        
        # Format label with pathway name
        plt.text(x, y, labels[i],
                horizontalalignment=horizontalalignment,
                verticalalignment="center",
                fontsize=10,
                weight='bold')
    
    # Title formatting
    plt.title(f"{title}\n(n={n_cell_lines})", 
              fontsize=16, 
              fontweight='bold', 
              pad=20,
              bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
    plt.axis('equal')
    
    # Save with high resolution
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='white')
    plt.close()
    
    # Log the pathways included
    logger.info(f"Saved pie chart to {output_path}")
    logger.info(f"Included pathways in {title}:")
    for label, pct in zip(labels, percentages):
        logger.info(f"{label}: {pct:.1f}%")

def save_results(df, filename):
    if df is None or df.empty:
        logger.warning(f"No data to save for {filename}")
        return
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved results to {filepath}")
    
def main():
    try:
        logger.info("Starting analysis pipeline...")
        
        # Load data
        models_df = load_data("MODEL.CSV")
        mutations_df = load_data("OmicsSomaticMutations.CSV")
        
        if models_df is None or mutations_df is None:
            logger.error("Failed to load one or more required files. Exiting.")
            return
        
        # Filter for crc
        crc_models = filter_crc(models_df)
        if crc_models is None or crc_models.empty:
            logger.error("Failed to filter Colorectal Adenocarcinoma models. Exiting.")
            return
        
        # Filter for KRAS and specific mutations
        kras_mutations, kras_specific_mutations = filter_kras_mutations(mutations_df, crc_models)
        if kras_mutations is None or kras_specific_mutations is None:
            logger.error("Failed to filter KRAS mutations. Exiting.")
            return
        
        # Identify cell lines
        specific_cell_lines, wt_cell_lines = identify_cell_lines(crc_models, kras_mutations, kras_specific_mutations)
        if specific_cell_lines is None or wt_cell_lines is None:
            logger.error("Failed to identify cell lines. Exiting.")
            return
    
        # Use built-in KEGG gene sets
        gene_set_name = 'c2.cp.kegg_medicus.v2024.1.Hs.symbols.gmt'
        
        try:
            gsea_results = perform_gsea(mutations_df, specific_cell_lines, wt_cell_lines, gene_set_name)
        except Exception as e:
            logger.error(f"Error performing GSEA: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return

        # Create output directory
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            logger.info(f"Created/verified output directory: {OUTPUT_DIR}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {str(e)}")
            return

        # Process results for each mutation
        for mutation, results in gsea_results.items():
            logger.info(f"Processing output for mutation {mutation}")
            
            if results.empty:
                logger.warning(f"No enriched pathways found for {mutation} using KEGG")
                continue

            # Save CSV results
            csv_filename = f"crc_{mutation.replace('.', '_')}_KEGG_gsea_results.csv"
            save_results(results, csv_filename)
            logger.info(f"Saved CSV results to {csv_filename}")

            # Create and save pie chart
            pathways = dict(zip(results['Term'], results['NES']))
            
            if pathways:
                output_path = os.path.join(OUTPUT_DIR, f"crc_kras_{mutation.replace('.', '_')}_KEGG_enriched_pathways.png")
                logger.info(f"Creating pie chart for {mutation} at {output_path}")
                
                try:
                    create_gsea_pie_chart(
                        pathways, 
                        f"Top Enriched Pathways in KRAS {mutation} Colorectal Adenocarcinoma\n(KEGG)",
                        output_path,
                        len(specific_cell_lines[mutation])
                    )
                    logger.info(f"Successfully created pie chart for {mutation}")
                except Exception as e:
                    logger.error(f"Error creating pie chart for {mutation}: {str(e)}")
                    logger.error("Traceback:", exc_info=True)
            else:
                logger.warning(f"No pathways found for pie chart creation for {mutation}")
            
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred during the analysis: {str(e)}")
        logger.error("Traceback:", exc_info=True)
    finally:
        logger.info("Analysis pipeline finished")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        logger.error("Traceback:", exc_info=True)
    finally:
        logger.info("Program execution completed")
