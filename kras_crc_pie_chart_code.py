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
P_VALUE_THRESHOLD = 0.25
TOP_PATHWAYS = 10
GSEA_PADJ_THRESHOLD = 0.25

def load_data(filename):
    filepath = os.path.join(DATA_DIR, filename)
    try:
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"Successfully loaded {filename}")
        logger.info(f"Shape of {filename}: {df.shape}")
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
        logger.error(f"Required columns not found in OmicsSomaticMutations.CSV")
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

def identify_cell_lines(models_df, mutations_df):
    try:
        # Read the raw CSV content
        csv_content = pd.read_csv('Comparisons.csv', header=None)
        
        # Find the start of Analysis 1 data
        start_idx = csv_content[csv_content[0] == 'Sample A'].index[0] + 1
        
        # Find the end of Analysis 1 data (start of Analysis 2 or empty rows)
        end_idx = None
        for idx in range(start_idx, len(csv_content)):
            if pd.isna(csv_content.iloc[idx][0]) or 'Analysis 2' in str(csv_content.iloc[idx][0]):
                end_idx = idx
                break
        
        if end_idx is None:
            end_idx = len(csv_content)
    
        data = csv_content.iloc[start_idx:end_idx].copy()
        data.columns = ['oncotree_a', 'genotype_a', 'oncotree_b', 'genotype_b']
        
        # Drop any completely empty rows
        data = data.dropna(how='all')
        # Drop header row if it exists
        data = data[data['oncotree_a'] != 'Oncotree code']
        
        logger.info(f"Loaded {len(data)} comparison pairs")
        
        results = {}
        for idx, row in data.iterrows():
            oncotree = row['oncotree_a']
            wt_genotype = row['genotype_a']
            mut_genotype = row['genotype_b']
            
            if pd.isna(oncotree) or pd.isna(wt_genotype) or pd.isna(mut_genotype):
                continue
                
            # Debug logging
            logger.info(f"Processing comparison: {oncotree} - {mut_genotype}")
            
            if ' ' in mut_genotype:
                gene, mutation_type = mut_genotype.split(' ', 1)
            else:
                # Handle cases like KRASG13C
                gene = ''.join(c for c in mut_genotype if not c.isdigit() and not c.isalpha())
                mutation_type = mut_genotype[len(gene):]
            
            gene = gene.strip()
            mutation_type = mutation_type.strip()
            
            if not mutation_type:
                logger.warning(f"Could not extract mutation type from {mut_genotype}")
                continue
            
            logger.info(f"Extracted gene: {gene}, mutation: {mutation_type}")
            
            # Map Oncotree codes to OncotreePrimaryDisease values
            oncotree_mapping = {
                'LUAD': 'Lung Adenocarcinoma',
                'COAD': 'Colorectal Adenocarcinoma',
                'PAAD': 'Pancreatic Adenocarcinoma',
                'PCM': 'Plasma Cell Myeloma',
                'STAD': 'Stomach Adenocarcinoma',
                'IHCH': 'Intrahepatic Cholangiocarcinoma',
                'UCEC': 'Endometrial Carcinoma',
                'READ': 'Colorectal Adenocarcinoma', #?
                'MEL': 'Melanoma'
            }
            
            primary_disease = oncotree_mapping.get(oncotree)
            if not primary_disease:
                logger.warning(f"No mapping found for Oncotree code: {oncotree}")
                continue
            
            # Get WT cell lines - checking for any mutations in the gene
            wt_models = models_df[models_df['OncotreePrimaryDisease'] == primary_disease].copy()
            wt_model_ids = set(wt_models['ModelID'])
            mutated_model_ids = set(mutations_df[
                (mutations_df['HugoSymbol'] == gene) &
                (mutations_df['ModelID'].isin(wt_model_ids))
            ]['ModelID'])
            wt_models = wt_models[~wt_models['ModelID'].isin(mutated_model_ids)]
            
            # Get mutant cell lines - look for the specific mutation
            mut_models = models_df[models_df['OncotreePrimaryDisease'] == primary_disease].copy()
            mut_models = mut_models[mut_models['ModelID'].isin(
                mutations_df[
                    (mutations_df['HugoSymbol'] == gene) &
                    (mutations_df['ProteinChange'].str.contains(mutation_type, na=False, regex=False))
                ]['ModelID']
            )]
            
            if len(wt_models) > 0 and len(mut_models) > 0:
                comparison_name = f"{oncotree}_{gene}_{mutation_type}"
                results[comparison_name] = {
                    'wt': wt_models,
                    'mut': mut_models,
                    'oncotree': oncotree,
                    'mutation': mutation_type,
                    'gene': gene
                }
                logger.info(f"Found {len(wt_models)} WT and {len(mut_models)} {gene} {mutation_type} models for {oncotree}")
            else:
                logger.warning(f"No valid models found for {oncotree} {gene} {mutation_type}")
        
        if not results:
            logger.error("No valid comparison pairs found")
            return None
            
        return results
        
    except Exception as e:
        logger.error(f"Error in identify_cell_lines: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        return None
    
def perform_gsea(mutations_df, comparison_data, gene_set):
    mutation_type = comparison_data['mutation']
    oncotree = comparison_data['oncotree']
    gene = comparison_data['gene']
    mut_models = comparison_data['mut']
    wt_models = comparison_data['wt']
    
    logger.info(f"Starting GSEA analysis for {oncotree} {gene} {mutation_type}")
    
    try:
        mutation_models = set(mut_models['ModelID'].unique())
        wt_models_set = set(wt_models['ModelID'].unique())
        
        logger.info(f"Processing {len(mutation_models)} mutant and {len(wt_models_set)} WT models")
        
        logger.info("Calculating differential mutation frequencies...")
        
        mut_counts = mutations_df[mutations_df['ModelID'].isin(mutation_models)].groupby('HugoSymbol')['ModelID'].nunique()
        wt_counts = mutations_df[mutations_df['ModelID'].isin(wt_models_set)].groupby('HugoSymbol')['ModelID'].nunique()
        
        all_genes = sorted(set(mut_counts.index) | set(wt_counts.index))
        
        mut_freq = pd.Series(0, index=all_genes)
        wt_freq = pd.Series(0, index=all_genes)
        
        mut_freq[mut_counts.index] = mut_counts / len(mutation_models)
        wt_freq[wt_counts.index] = wt_counts / len(wt_models_set)
        
        gene_list = [g for g in all_genes if mut_freq[g] > wt_freq[g] and g not in {gene, 'KRAS', 'NRAS'}]
        
        logger.info(f"Found {len(gene_list)} differentially mutated genes")
        
        if not gene_list:
            logger.warning("No differentially mutated genes found")
            return pd.DataFrame()
            
        try:
            logger.info("Running enrichr analysis...")
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=gene_set,
                outdir=None,
                no_plot=True,
                cutoff=GSEA_PADJ_THRESHOLD
            )
            
            logger.info("Processing enrichr results...")
            results_df = enr.results
            
            if results_df.empty:
                logger.warning("No enrichment results found")
                return pd.DataFrame()
            
            results_df['NES'] = results_df['Combined Score']
            top_pathways = results_df.nlargest(TOP_PATHWAYS, 'Combined Score')
            
            logger.info(f"Found {len(top_pathways)} significant pathways")
            return top_pathways
            
        except Exception as e:
            logger.error(f"Error in enrichr analysis: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error in GSEA analysis: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        return pd.DataFrame()

def save_results(df, filename):
    if df is None or df.empty:
        logger.warning(f"No data to save for {filename}")
        return
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved results to {filepath}")

def create_gsea_pie_chart(data, title, output_path, n_cell_lines, TOP_PATHWAYS=10):
    if data.empty:
        logger.warning(f"No data available for pie chart: {title}")
        return
    
    pathways = dict(zip(data['Term'], data['NES']))
    valid_data = {k: abs(v) for k, v in pathways.items()}
    
    if not valid_data:
        logger.warning(f"No valid pathways found for pie chart: {title}")
        return
    
    sorted_items = sorted(valid_data.items(), key=lambda x: x[1], reverse=True)
    top_data = dict(sorted_items[:TOP_PATHWAYS])
    
    logger.info(f"Creating pie chart with {len(top_data)} pathways")
    
    plt.figure(figsize=(15, 10))
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(top_data)))
    
    sizes = list(top_data.values())
    total = sum(sizes)
    percentages = [size/total * 100 for size in sizes]
    
    # Create pie chart
    wedges, texts, autotexts = plt.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.6),
        autopct='%1.1f%%',
        pctdistance=0.75,
        explode=[0.02] * len(sizes)
    )
    
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))
        
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        distance = 1.2 if percentages[i] < 10 else 1.15
        x = x * distance
        y = y * distance
        
        wrapped_text = textwrap.fill(list(top_data.keys())[i], width=50)
        
        plt.text(x, y, wrapped_text,
                horizontalalignment=horizontalalignment,
                verticalalignment="center",
                fontsize=10,
                weight='bold')
    
    plt.title(f"{title}\n(n={n_cell_lines})", 
              fontsize=16, 
              fontweight='bold', 
              pad=20,
              bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
    plt.axis('equal')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='white')
    plt.close()
    
    logger.info(f"Saved pie chart to {output_path}")
    logger.info(f"Included pathways in {title}:")
    for label, pct in zip(top_data.keys(), percentages):
        logger.info(f"{label}: {pct:.1f}%")

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
        
        # Get comparison pairs
        comparison_results = identify_cell_lines(models_df, mutations_df)
        if not comparison_results:
            logger.error("Failed to identify comparison pairs. Exiting.")
            return
    
        # Use the KEGG Medicus gene set
        gene_set = "c2.cp.kegg_medicus.v2024.1.Hs.symbols.gmt"
        
        # Create output directory
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            logger.info(f"Created/verified output directory: {OUTPUT_DIR}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {str(e)}")
            return

        # Process each comparison pair
        for comparison_name, comparison_data in comparison_results.items():
            try:
                gsea_results = perform_gsea(mutations_df, comparison_data, gene_set)
                
                if not gsea_results.empty:
                    # Save CSV results
                    csv_filename = f"{comparison_name}_KEGG_gsea_results.csv"
                    save_results(gsea_results, csv_filename)

                    # Create and save pie chart
                    output_path = os.path.join(OUTPUT_DIR, f"{comparison_name}_KEGG_enriched_pathways.png")
                    title = f"Enriched Pathways: {comparison_data['oncotree']} {comparison_data['gene']} WT vs {comparison_data['mutation']}"
                    
                    create_gsea_pie_chart(
                        gsea_results, 
                        title,
                        output_path,
                        len(comparison_data['mut'])
                    )
                    
                else:
                    logger.warning(f"No enriched pathways found for {comparison_name}")
                    
            except Exception as e:
                logger.error(f"Error processing comparison {comparison_name}: {str(e)}")
                logger.error("Traceback:", exc_info=True)
                continue
            
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
