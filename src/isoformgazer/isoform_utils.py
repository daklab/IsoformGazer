import os
import re
import hashlib
import sqlite3
import warnings
import traceback
import pandas as pd
import numpy as np
import dash_bio
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Tuple
from data_utils import apply_distance_preprocessing, extract_gtf_attr_val, get_matplotlib_colormap, get_table_prefix
from performance_utils import cached, memory_tracker, plot_optimizer
# suppresses "Mean of empty slice" warnings coming from numpy when computing nanmean on all-NaN arrays for isoform clustergram log(TPM) data display...
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')

def get_lrs_metadata_replicates() -> List[List[str]]:
    """
    Parse LRS metadata CSV to group samples into replicate groups.
    
    Returns:
        List of lists where each inner list contains samples that should be averaged together.
        Single samples (no replicates) are in their own list.
        
        Example:
        [
            ['ENCFF387HPO.mucosa_of_descending_colon'],
            ['ENCFF680XXE.cardiac_septum'], 
            ['ENCFF242WRZ.right_cardiac_atrium'],
            ['ENCFF548JGS.Calu3', 'ENCFF569KOA.Calu3']
        ]
    """
    metadata_path = os.path.join(os.path.dirname(__file__), 'data', 'ENCODE4_LRS_Metadata.csv')
    
    try:
        df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        print(f"Metadata file not found: {metadata_path}")
        return []
    
    replicate_groups = []
    grouped = df.groupby(['Cell_Type', 'Organ'])

    for (cell_type, organ), group in grouped:
        has_replicates = group['Replicate'].notna().any() and (group['Replicate'] != 'None').any()
        
        # Case 1: sample has replicates, so put all samples in same list
        if has_replicates:
            samples = group['Sample_id'].tolist()
            replicate_groups.append(samples)
            
        # Case 2: sample has no replicates, so put each sample in its own list
        else:
            for sample_id in group['Sample_id']:
                replicate_groups.append([sample_id])
    
    return replicate_groups


def get_sample_replicates_mapping(species: str = 'Human') -> List[List[str]]:
    """
    Create a mapping of sample IDs to their replicate groups, represented as nested lists.
    This mapping was obtained using get_lrs_metadata_replicates().

    Args:
        species: 'Human' or 'Mouse' - determines which replicate mapping to use

    Returns:
        List of lists where each sublist contains sample IDs that are replicates
    """
    if species == "Mouse":
        replicates_mapping = [['ENCFF565RLW.left_cerebral_cortex', 'ENCFF325BXV.left_cerebral_cortex'], ['ENCFF605NQH.left_cerebral_cortex','ENCFF937WKF.left_cerebral_cortex'],
                              ['ENCFF518QIV.layer_of_hippocampus', 'ENCFF809BZD.layer_of_hippocampus'], ['ENCFF402OZL.layer_of_hippocampus', 'ENCFF182ZKL.layer_of_hippocampus'],
                              ['ENCFF327DHL.gastrocnemius', 'ENCFF441MLQ.gastrocnemius'], ['ENCFF223JJA.layer_of_hippocampus', 'ENCFF326JTG.layer_of_hippocampus'],
                              ['ENCFF977AQV.adrenal_gland', 'ENCFF626SLN.adrenal_gland'], ['ENCFF167GYW.adrenal_gland', 'ENCFF589EMD.adrenal_gland'],
                              ['ENCFF122PGZ.adrenal_gland', 'ENCFF916NVH.adrenal_gland'],['ENCFF618PJT.gastrocnemius', 'ENCFF124DBU.gastrocnemius'],
                              ['ENCFF873KDL.adrenal_gland', 'ENCFF866CSV.adrenal_gland'], ['ENCFF863AGD.adrenal_gland', 'ENCFF321AJK.adrenal_gland'],
                              ['ENCFF435KPO.adrenal_gland', 'ENCFF671OCJ.adrenal_gland'], ['ENCFF811RRD.adrenal_gland'],
                              ['ENCFF532MMA.layer_of_hippocampus', 'ENCFF175BNQ.layer_of_hippocampus'], ['ENCFF100OAR.adrenal_gland', 'ENCFF604SVZ.adrenal_gland'],
                              ['ENCFF874VSI.F121.9', 'ENCFF667VXS.F121.9', 'ENCFF313VYZ.F121.9'], ['ENCFF169PFY.adrenal_gland', 'ENCFF142VQP.adrenal_gland'],
                              ['ENCFF100UBG.left_cerebral_cortex', 'ENCFF812RTU.left_cerebral_cortex'],['ENCFF337DWM.gastrocnemius', 'ENCFF696VVK.gastrocnemius'],
                              ['ENCFF584WWA.heart', 'ENCFF860CBL.heart'], ['ENCFF429RCV.adrenal_gland', 'ENCFF387GTW.adrenal_gland'],['ENCFF330HVI.layer_of_hippocampus'],
                              ['ENCFF285NUG.left_cerebral_cortex', 'ENCFF977AFJ.left_cerebral_cortex'], ['ENCFF303OLU.adrenal_gland', 'ENCFF856RHM.adrenal_gland'],
                              ['ENCFF836THV.adrenal_gland', 'ENCFF970IHS.adrenal_gland'],['ENCFF524YTF.gastrocnemius', 'ENCFF294TJX.gastrocnemius'],
                              ['ENCFF047OYU.gastrocnemius', 'ENCFF389RMT.gastrocnemius'], ['ENCFF948TOU.gastrocnemius', 'ENCFF369DHX.gastrocnemius'],
                              ['ENCFF520HCL.left_cerebral_cortex', 'ENCFF132MEJ.left_cerebral_cortex'], ['ENCFF836SSV.heart', 'ENCFF031EVU.heart'],
                              ['ENCFF773BXP.left_cerebral_cortex', 'ENCFF468KWN.left_cerebral_cortex'],['ENCFF107CZA.gastrocnemius', 'ENCFF957OYJ.gastrocnemius'],
                              ['ENCFF281GTL.gastrocnemius', 'ENCFF028QQV.gastrocnemius'], ['ENCFF896RNF.layer_of_hippocampus', 'ENCFF346JYL.layer_of_hippocampus'],
                              ['ENCFF338YTG.layer_of_hippocampus', 'ENCFF509RDZ.layer_of_hippocampus'], ['ENCFF271ZDH.gastrocnemius', 'ENCFF095FKV.gastrocnemius'],
                              ['ENCFF865ZOH.gastrocnemius', 'ENCFF421HRA.gastrocnemius'],['ENCFF798HCT.gastrocnemius', 'ENCFF069POP.gastrocnemius'],
                              ['ENCFF177KNR.heart', 'ENCFF468BPP.heart'], ['ENCFF837UMP.gastrocnemius', 'ENCFF556BPF.gastrocnemius'],['ENCFF583MVR.heart', 'ENCFF358AEP.heart'],
                              ['ENCFF221ABY.gastrocnemius', 'ENCFF435GLV.gastrocnemius'], ['ENCFF988MZV.adrenal_gland', 'ENCFF549DGV.adrenal_gland'],
                              ['ENCFF032HOK.heart', 'ENCFF356MNK.heart'], ['ENCFF477GNS.gastrocnemius', 'ENCFF462TIY.gastrocnemius'], ['ENCFF885OVT.heart', 'ENCFF484VDN.heart'],
                              ['ENCFF669LWV.myotube', 'ENCFF003OWX.myotube'], ['ENCFF141ZHE.adrenal_gland', 'ENCFF177RGM.adrenal_gland'], ['ENCFF376LTY.layer_of_hippocampus', 'ENCFF314QJI.layer_of_hippocampus'],
                              ['ENCFF448KBZ.layer_of_hippocampus', 'ENCFF890RGD.layer_of_hippocampus'], ['ENCFF311BUV.adrenal_gland', 'ENCFF698CVI.adrenal_gland'],
                              ['ENCFF095CHU.left_cerebral_cortex', 'ENCFF201HBC.left_cerebral_cortex'], ['ENCFF309DMQ.layer_of_hippocampus', 'ENCFF103DSA.layer_of_hippocampus'],
                              ['ENCFF110OOL.layer_of_hippocampus', 'ENCFF997NDX.layer_of_hippocampus'], ['ENCFF348TRU.forelimb'], ['ENCFF944JXN.left_cerebral_cortex', 'ENCFF148ESR.left_cerebral_cortex'],
                              ['ENCFF479GNP.left_cerebral_cortex', 'ENCFF402ZUC.left_cerebral_cortex'], ['ENCFF019HRC.C2C12', 'ENCFF676BYQ.C2C12'], ['ENCFF978EIJ.layer_of_hippocampus', 'ENCFF228VBQ.layer_of_hippocampus'],
                              ['ENCFF046XAX.left_cerebral_cortex', 'ENCFF319GBG.left_cerebral_cortex'], ['ENCFF560ONE.adrenal_gland', 'ENCFF445VZL.adrenal_gland'], ['ENCFF417EKT.forelimb'],
                              ['ENCFF683BZL.left_cerebral_cortex', 'ENCFF476MLA.left_cerebral_cortex'], ['ENCFF008ATT.C2C12', 'ENCFF238DPX.C2C12']]
    else:
        # Human replicates mapping
        replicates_mapping = [['ENCFF168MIB.A673', 'ENCFF861BKY.A673'], ['ENCFF649CYY.Caco.2', 'ENCFF827OXR.Caco.2'], 
                        ['ENCFF548JGS.Calu3', 'ENCFF569KOA.Calu3'], ['ENCFF417VHJ.GM12878', 'ENCFF281TNJ.GM12878', 
                        'ENCFF475ORL.GM12878', 'ENCFF329AYV.GM12878', 'ENCFF902UIT.GM12878', 'ENCFF450VAU.GM12878', 
                        'ENCFF694DIE.GM12878'], ['ENCFF954UFG.GM23338', 'ENCFF251CBB.GM23338'], ['ENCFF853OFP.H1', 
                        'ENCFF400BQQ.H1', 'ENCFF436GKZ.H1'], ['ENCFF688QGB.H9', 'ENCFF272VSN.H9'], ['ENCFF337VWR.HCT116'], 
                        ['ENCFF728ITF.HFFc6', 'ENCFF288CJF.HFFc6', 'ENCFF385QZZ.HFFc6'], ['ENCFF609QIM.HL.60', 
                        'ENCFF274DYS.HL.60'], ['ENCFF483HTA.HepG2', 'ENCFF427JDY.HepG2', 'ENCFF589SMB.HepG2'], 
                        ['ENCFF197DCI.IMR.90'], ['ENCFF429VVB.K562', 'ENCFF634YSN.K562', 'ENCFF696GDL.K562', 'ENCFF694INI.K562', 
                        'ENCFF763VZC.K562'], ['ENCFF041EGI.MCF_10A', 'ENCFF702KLU.MCF_10A'], ['ENCFF887DGG.MCF.7'], 
                        ['ENCFF511KJB.OCI.LY7', 'ENCFF417UQV.OCI.LY7'], ['ENCFF834KTE.PC.3'], ['ENCFF107YRM.PC.9', 
                        'ENCFF860AWQ.PC.9'], ['ENCFF990CUL.Panc1'], ['ENCFF738UZJ.Right_ventricle_myocardium_inferior'], 
                        ['ENCFF665LBS.Right_ventricle_myocardium_superior'], ['ENCFF563QZR.WTC11', 'ENCFF370NFS.WTC11', 
                        'ENCFF245IPA.WTC11'], ['ENCFF417ALN.adrenal_gland'], ['ENCFF211SQY.adrenal_gland'], 
                        ['ENCFF912HPY.adrenal_gland'], ['ENCFF144KHH.aorta'], ['ENCFF902BIU.aorta'], ['ENCFF316EZQ.astrocyte', 
                        'ENCFF474GEK.astrocyte'], ['ENCFF680XXE.cardiac_septum'], ['ENCFF352CGL.chondrocyte', 
                        'ENCFF342HOS.chondrocyte', 'ENCFF011BFA.chondrocyte'], ['ENCFF206TQZ.dorsolateral_prefrontal_cortex'], 
                        ['ENCFF156TTD.dorsolateral_prefrontal_cortex'], ['ENCFF311CZO.dorsolateral_prefrontal_cortex'], 
                        ['ENCFF785KVJ.dorsolateral_prefrontal_cortex'], ['ENCFF260AWP.dorsolateral_prefrontal_cortex'], 
                        ['ENCFF838DFB.dorsolateral_prefrontal_cortex'], ['ENCFF827DUW.dorsolateral_prefrontal_cortex'], 
                        ['ENCFF708BOP.dorsolateral_prefrontal_cortex'], ['ENCFF446EFU.dorsolateral_prefrontal_cortex'], 
                        ['ENCFF712CBL.endodermal_cell', 'ENCFF731HST.endodermal_cell', 'ENCFF235QXW.endodermal_cell', 
                        'ENCFF142LPL.endodermal_cell', 'ENCFF561HIY.endodermal_cell'], ['ENCFF595PPR.endothelial_cell', 
                        'ENCFF770DXN.endothelial_cell'], ['ENCFF033LRZ.endothelial_cell_of_umbilical_vein', 
                        'ENCFF096UHO.endothelial_cell_of_umbilical_vein'], ['ENCFF919JFJ.glutamatergic_neuron', 
                        'ENCFF982WKN.glutamatergic_neuron'], ['ENCFF429JUP.heart_left_ventricle', 'ENCFF602MAI.heart_left_ventricle', 
                        'ENCFF537NCV.heart_left_ventricle', 'ENCFF185VYD.heart_left_ventricle'], ['ENCFF793PGJ.heart_right_ventricle', 
                        'ENCFF425VDL.heart_right_ventricle', 'ENCFF615FIC.heart_right_ventricle'], ['ENCFF492BYP.kidney'], 
                        ['ENCFF920VXE.left_cardiac_atrium'], ['ENCFF245MBY.left_colon'], ['ENCFF733RRO.left_lung'], 
                        ['ENCFF196WMM.left_lung'], ['ENCFF793CMQ.left_ventricle_myocardium_inferior'], 
                        ['ENCFF624IQY.left_ventricle_myocardium_superior'], ['ENCFF341BSQ.lower_lobe_of_left_lung'], 
                        ['ENCFF552NVU.lower_lobe_of_left_lung'], ['ENCFF250IWT.lower_lobe_of_right_lung'], ['ENCFF237FMP.mammary_epithelial_cell', 
                        'ENCFF617YVE.mammary_epithelial_cell'], ['ENCFF907SZK.mesenteric_fat_pad'], ['ENCFF387HPO.mucosa_of_descending_colon'], 
                        ['ENCFF511AVQ.mucosa_of_descending_colon'], ['ENCFF026VEI.neural_crest_cell', 'ENCFF249GFH.neural_crest_cell'], 
                        ['ENCFF556DYU.osteocyte', 'ENCFF560XTG.osteocyte'], ['ENCFF422XLS.ovary'], ['ENCFF187BTK.ovary'], ['ENCFF756AHG.ovary'], 
                        ['ENCFF960KBO.posterior_vena_cava'], ['ENCFF658OZB.posterior_vena_cava'], ['ENCFF471YEK.progenitor_cell_of_endocrine_pancreas', 
                        'ENCFF988RQM.progenitor_cell_of_endocrine_pancreas'], ['ENCFF750LYC.psoas_muscle'], ['ENCFF630XEC.psoas_muscle'], 
                        ['ENCFF242WRZ.right_cardiac_atrium'], ['ENCFF905RVF.right_cardiac_atrium'], ['ENCFF722JJS.right_cardiac_atrium'], 
                        ['ENCFF899MTI.right_cardiac_atrium'], ['ENCFF318SKH.right_lobe_of_liver'], ['ENCFF306ZPP.right_lobe_of_liver'], 
                        ['ENCFF580BQX.type_B_pancreatic_cell', 'ENCFF489XQJ.type_B_pancreatic_cell'], ['ENCFF934MBW.upper_lobe_of_right_lung']]

    return replicates_mapping


def abbreviate_transcript_name(transcript_name: str) -> str:
    """
    Abbreviate transcript names for plotting display.
    
    Examples:
    - "s-6373022524934343100:e2226920842648636318" -> "s-6373022524934343100:e2226"
    - "s-6373022524934343100:e-7666537037463561649" -> "s-6373022524934343100:e-7666"
    """
    if ':' not in transcript_name:
        return transcript_name
    
    prefix, suffix = transcript_name.split(':', 1)
    
    if suffix.startswith('e-'):
        abbreviated_suffix = 'e-' + suffix[2:6] if len(suffix) > 6 else suffix
    elif suffix.startswith('e'):
        abbreviated_suffix = 'e' + suffix[1:5] if len(suffix) > 5 else suffix
    else:
        abbreviated_suffix = suffix[:4]
    
    return f"{prefix}:{abbreviated_suffix}"


def abbreviate_transcript_names(transcript_names: List[str]) -> List[str]:
    """Apply abbreviation to a list of transcript names"""
    return [abbreviate_transcript_name(name) for name in transcript_names]


def calculate_dynamic_structure_plot_height(num_transcripts: int, base_height: int = 600) -> int:
    """Calculate dynamic height for structure plots based on number of transcripts"""
    calculated_height = base_height + (num_transcripts * 10)
    # Round to nearest hundred (to align with slider values)
    return round(calculated_height / 100) * 100


def calculate_unified_plot_height(transcript_data: pd.DataFrame, gene_data: dict = None, base_height: int = 600) -> int:
    """Calculate unified height for both transcript and junction plots based on available data"""
    num_transcripts = 0
    if not transcript_data.empty:
        if 'id' in transcript_data.columns:
            num_transcripts = transcript_data['id'].nunique()
        elif 'trans_id' in transcript_data.columns:
            num_transcripts = transcript_data['trans_id'].nunique()

    num_junctions = 0
    if gene_data and 'junctions' in gene_data:
        num_junctions = len(gene_data['junctions'])

    max_elements = max(num_transcripts, num_junctions)
    calculated_height = base_height + (max_elements * 20)

    calculated_height = round(calculated_height / 100) * 100

    return min(calculated_height, 1600)


def calculate_clustergram_min_height(num_rows: int, base_height: int = 600) -> int:
    """
    Calculate minimum height needed for clustergram to prevent label overlap.

    For clustergrams, each row needs approximately 15-20 pixels to display labels without overlap.
    This ensures labels are readable even with many transcripts/junctions.

    Parameters:
        num_rows: Number of rows (transcripts or junctions) in the clustergram
        base_height: Minimum base height for the clustergram

    Returns:
        Minimum height needed to display all labels without overlap
    """
    pixels_per_row = 18
    min_heatmap_height = num_rows * pixels_per_row
    min_total_height = int(min_heatmap_height / 0.88)

    calculated_height = max(base_height, min_total_height)
    calculated_height += 100
    # round to nearest hundred for slider compatibility
    calculated_height = round(calculated_height / 100) * 100

    return calculated_height


###################################################################
# VISUALIZATION METHODS
###################################################################
def load_psl_data(psl_file_path: str) -> pd.DataFrame:
    """Load and process PSL file for transcript structure"""
    try:
        # PSL file format: https://useast.ensembl.org/info/website/upload/psl.html
        psl_columns = [
            'matches', 'misMatches', 'repMatches', 'nCount', 'qNumInsert', 'qBaseInsert',
            'tNumInsert', 'tBaseInsert', 'strand', 'qName', 'qSize', 'qStart', 'qEnd',
            'tName', 'tSize', 'tStart', 'tEnd', 'blockCount', 'blockSizes', 'qStarts', 'tStarts'
        ]
        
        psl_df = pd.read_csv(psl_file_path, sep='\t', header=None, names=psl_columns)
        
        # Extract gene_id and trans_id from qName (assuming format: transcript_geneID)
        psl_df['gene_id'] = psl_df['qName'].str.split('_').str[1]
        psl_df['trans_id'] = psl_df['qName'].str.split('_').str[0]
        
        # Calculate transcript length
        psl_df['transcript_length'] = psl_df['tEnd'] - psl_df['tStart']

        return psl_df
    
    except Exception as e:
        print(f"Error loading PSL file: {e}")
        return pd.DataFrame()
    

def get_gene_id_for_gene_name(db_path: str, gene_name: str, species="Human") -> str:
    """Get gene_id for a given gene_name from the isoforms database"""
    conn = sqlite3.connect(db_path)
    table_prefix = get_table_prefix(species)
    query = f"SELECT DISTINCT gene_id FROM {table_prefix}isoforms WHERE gene_name = ? LIMIT 1"
    result = pd.read_sql_query(query, conn, params=[gene_name])
    conn.close()

    if len(result) > 0:
        gene_id = result.iloc[0]['gene_id']
        # Remove version number from gene_id (e.g., ENSG00000100320.16 -> ENSG00000100320)
        gene_id_clean = gene_id.split('.')[0]
        return gene_id_clean

    else:
        return None


def prepare_gene_psl_data(psl_df: pd.DataFrame):
    """Preprocesses result of process_transcript_structure() PSL data query to have expected structure 
    for use with data visualization methods downstream."""
    try: 
        psl_columns = [
                'matches', 'misMatches', 'repMatches', 'nCount', 'qNumInsert', 'qBaseInsert',
                'tNumInsert', 'tBaseInsert', 'strand', 'qName', 'qSize', 'qStart', 'qEnd',
                'tName', 'tSize', 'tStart', 'tEnd', 'blockCount', 'blockSizes', 'qStarts', 'tStarts'
        ]

        psl_df['gene_id'] = psl_df['qName'].str.split('_').str[1]
        psl_df['trans_id'] = psl_df['qName'].str.split('_').str[0]

        psl_df['transcript_length'] = psl_df['tEnd'] - psl_df['tStart']

        return psl_df
    
    except Exception as e:
        print(f"Error preprocessing PSL file data: {e}")
        return pd.DataFrame()
    

def process_transcript_structure(db_path: str, gene_name: str, filtered_ids: list, species="Human") -> pd.DataFrame:
    """Transcript structure processing with caching and memory optimization for faster rendering!"""
    #with ProfilerContext(f"process_transcript_structure_{gene_name}"):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size = 10000")
    conn.execute("PRAGMA temp_store = MEMORY")
    table_prefix = get_table_prefix(species)
    gene_query = f"SELECT DISTINCT gene_id FROM {table_prefix}isoforms WHERE gene_name = ? LIMIT 1"
    gene_result = pd.read_sql_query(gene_query, conn, params=[gene_name])

    if gene_result.empty:
        conn.close()
        return pd.DataFrame()

    gene_id = gene_result.iloc[0]['gene_id'].split('.')[0]
    #memory_tracker.measure(f"after_gene_lookup_{gene_name}")

    if filtered_ids and len(filtered_ids) > 0:
        filtered_ids_int = [int(id) for id in filtered_ids if str(id).isdigit()]
        if filtered_ids_int:
            placeholders = ','.join(['?'] * len(filtered_ids_int))
            isoform_query = f"""
            SELECT id FROM {table_prefix}isoforms
            WHERE gene_id LIKE ? AND id IN ({placeholders})
            ORDER BY isoform_average_tpm DESC NULLS LAST
            """
            params = [f"{gene_id}%"] + filtered_ids_int
        else:
            isoform_query = f"SELECT id FROM {table_prefix}isoforms WHERE gene_id LIKE ? ORDER BY isoform_average_tpm DESC NULLS LAST"
            params = [f"{gene_id}%"]
    else:
        isoform_query = f"SELECT id FROM {table_prefix}isoforms WHERE gene_id LIKE ? ORDER BY isoform_average_tpm DESC NULLS LAST"
        params = [f"{gene_id}%"]

    isoform_ids = pd.read_sql_query(isoform_query, conn, params=params)['id'].tolist()

    if not isoform_ids:
        conn.close()
        return pd.DataFrame()

    #memory_tracker.measure(f"after_isoform_lookup_{gene_name}")

    placeholders = ','.join(['?'] * len(isoform_ids))
    psl_query = f"""
    SELECT
        id, trans_id, gene_id, tName, strand,
        tStart, tEnd, blockSizes, tStarts
    FROM {table_prefix}psl_data
    WHERE id IN ({placeholders})
    ORDER BY tStart, id
    """
    gene_psl = pd.read_sql_query(psl_query, conn, params=isoform_ids)
    conn.close()
    
    #memory_tracker.measure(f"after_psl_query_{gene_name}")
    
    # Use much faster vectorized processing instead of pandas iterrows (super slow)
    transcript_data = []
    
    if not gene_psl.empty:
        gene_psl = gene_psl.copy()
        gene_psl['blockSizes'] = gene_psl['blockSizes'].str.rstrip(',')
        gene_psl['tStarts'] = gene_psl['tStarts'].str.rstrip(',')
        
        # filter out rows with empty block data upfront for some speedup
        valid_mask = (gene_psl['blockSizes'].notna() & 
                        gene_psl['tStarts'].notna() & 
                        (gene_psl['blockSizes'] != '') & 
                        (gene_psl['tStarts'] != ''))
        gene_psl_valid = gene_psl[valid_mask].copy()
        
        for idx, row in gene_psl_valid.iterrows():
            try:
                block_sizes = [int(x) for x in row['blockSizes'].split(',') if x]
                block_starts = [int(x) for x in row['tStarts'].split(',') if x]
                
                if not block_sizes or not block_starts or len(block_sizes) != len(block_starts):
                    continue
                
                base_data = {
                    'id': row['id'],
                    'trans_id': row['trans_id'], 
                    'gene_id': row['gene_id'],
                    'chr': row['tName'],
                    'strand': row['strand'],
                    'transcript_start': row['tStart'],
                    'transcript_end': row['tEnd']
                }
                
                for i, (size, start) in enumerate(zip(block_sizes, block_starts)):
                    transcript_data.append({
                        **base_data,
                        'exon_number': i + 1,
                        'exon_start': start,
                        'exon_end': start + size,
                        'exon_size': size
                    })

            except (ValueError, AttributeError, IndexError, TypeError):
                continue
    
    #memory_tracker.measure(f"after_processing_{gene_name}")
    result_df = pd.DataFrame(transcript_data)
    
    if not result_df.empty:
        int_cols = ['id', 'transcript_start', 'transcript_end', 'exon_number', 
                    'exon_start', 'exon_end', 'exon_size']
        for col in int_cols:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], downcast='integer')
    
    return result_df


def load_expression_data(db_path: str, gene_name: str, data_type: str = 'tpm', species="Human") -> pd.DataFrame:
    """Load expression data from SQLite database with caching"""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size = 10000")

    gene_id = get_gene_id_for_gene_name(db_path, gene_name, species)
    if not gene_id:
        return pd.DataFrame()

    table_prefix = get_table_prefix(species)
    if data_type.lower() == 'tpm':
        table_name = f'{table_prefix}tpm_data'
    elif data_type.lower() == 'log_tpm':
        table_name = f'{table_prefix}log_tpm_data'
    else:
        table_name = f'{table_prefix}ratio_data'

    query = f"""
    SELECT
        exp.*,
        psl.trans_id,
        iso.gene_name,
        iso.isoform_average_tpm
    FROM {table_name} exp
    JOIN {table_prefix}psl_data psl ON exp.id = psl.id
    JOIN {table_prefix}isoforms iso ON exp.id = iso.id
    WHERE iso.gene_id LIKE ?
    ORDER BY iso.isoform_average_tpm ASC NULLS LAST, psl.trans_id
    """

    try:
        df = pd.read_sql_query(query, conn, params=[f"{gene_id}%"])
        df = df.drop(['index'], axis=1)

        numeric_cols = [col for col in df.columns if col not in ['transcript', 'trans_id', 'gene', 'gene_name']]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # For log_tpm data, preserve NaN values. For other data types, fill NaN with 0
        if data_type.lower() != 'log_tpm':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        else:
            # Log TPM: preserve NaN values
            nan_count = df[numeric_cols].isna().sum().sum()

        return df

    except Exception as e:
        print(f"Error loading {data_type} data: {e}")
        return pd.DataFrame()

    finally:
        conn.close()


def create_transcript_structure_plot(db_path: str,
                                     transcript_data: pd.DataFrame,
                                     gene_name: str,
                                     height: int = 600,
                                     show_y_labels: bool = False,
                                     exon_color: str = '#2E86C1',
                                     color_by_abundance: bool = False,
                                     colorscale: str = 'Viridis',
                                     abundance_type: str = 'average',
                                     tissue_name: str = None,
                                     organ_name: str = None,
                                     individual_transcript_colors: dict = None,
                                     species: str = "Human") -> go.Figure:
    """Create transcript structure plot showing all transcripts with 50%+ speed improvement"""

    if transcript_data.empty:
        return create_empty_isoform_message(f"No transcript data for gene: {gene_name}")
    
    #with PlotPerformanceContext(gene_name, "transcript_structure"):
        #memory_tracker.measure("plot_start")

    table_prefix = get_table_prefix(species)
    conn = sqlite3.connect(db_path)
    metadata_query = f"""SELECT gene_id, ORF_perplexity FROM {table_prefix}isoforms
                        WHERE gene_name = ? LIMIT 1"""
    metadata_result = pd.read_sql_query(metadata_query, conn, params=[gene_name])

    if not metadata_result.empty:
        gene_ensembl_id = metadata_result.iloc[0]['gene_id']
        orf_value = metadata_result.iloc[0]['ORF_perplexity']
        orf_perplexity = "None" if pd.isna(orf_value) else f"{orf_value:.3f}"
    else:
        gene_ensembl_id = "Unknown"
        orf_perplexity = "No data available"

    conn.close()

    strand = transcript_data['strand'].iloc[0] if not transcript_data.empty else ""

    transcript_data_opt = plot_optimizer.preprocess_dataframe_for_plotting(transcript_data)

    # Query database for transcript IDs in correct TPM order (like junction_utils.py does)
    conn = sqlite3.connect(db_path)
    # Get gene_id for gene_name
    gene_id = get_gene_id_for_gene_name(db_path, gene_name, species)
    if gene_id:
        transcript_ids_query = f"""
        SELECT DISTINCT id FROM {table_prefix}isoforms
        WHERE gene_id LIKE ?
        ORDER BY isoform_average_tpm DESC NULLS LAST
        """
        ordered_ids_df = pd.read_sql_query(transcript_ids_query, conn, params=[f"{gene_id}%"])
    else:
        ordered_ids_df = pd.DataFrame()
    conn.close()
    ordered_transcript_ids = ordered_ids_df['id'].tolist() if not ordered_ids_df.empty else []
    # Reverse because transcripts are displayed from top to bottom
    ordered_transcript_ids = ordered_transcript_ids[::-1]

    transcript_summary = transcript_data_opt.groupby('id', as_index=False).agg({
        'trans_id': 'first',
        'transcript_start': 'min',
        'transcript_end': 'max'
    })
    transcript_summary['transcript_length'] = (
        transcript_summary['transcript_end'] - transcript_summary['transcript_start']
    )

    # Reindex transcript_summary to match the original sorted order from database
    transcript_summary = transcript_summary.set_index('id').loc[ordered_transcript_ids].reset_index()
    transcript_summary['trans_order'] = range(1, len(transcript_summary) + 1)

    if color_by_abundance:
        if abundance_type == 'tissue' and tissue_name:
            tissue_tpm_dict = get_tissue_tpm_for_isoforms(db_path, gene_name, tissue_name, species)
            transcript_summary['abundance_tpm'] = transcript_summary['id'].map(tissue_tpm_dict).fillna(0)

        elif abundance_type == 'organ' and organ_name:
            organ_tpm_dict = get_organ_tpm_for_isoforms(db_path, gene_name, organ_name, species)
            transcript_summary['abundance_tpm'] = transcript_summary['id'].map(organ_tpm_dict).fillna(0)

        else:
            conn = sqlite3.connect(db_path)
            try:
                # Get gene_id for gene_name
                gene_id_for_tpm = get_gene_id_for_gene_name(db_path, gene_name, species)
                if gene_id_for_tpm:
                    tpm_query = f"""
                    SELECT DISTINCT id, isoform_average_tpm
                    FROM {table_prefix}isoforms
                    WHERE gene_id LIKE ?
                    """
                    tpm_df = pd.read_sql_query(tpm_query, conn, params=[f"{gene_id_for_tpm}%"])
                    transcript_summary = transcript_summary.merge(tpm_df, on='id', how='left')
                    transcript_summary['abundance_tpm'] = transcript_summary['isoform_average_tpm']

            finally:
                conn.close()

    # Calculate dynamic height if not provided or if using default slider value
    if height is None or height == 600:
        num_transcripts = len(transcript_summary)
        height = calculate_dynamic_structure_plot_height(num_transcripts)
    
    plot_data = transcript_data_opt.merge(transcript_summary[['id', 'trans_order']], on='id', how='inner')
    
    memory_tracker.measure("after_data_prep")
    
    min_start = plot_data['transcript_start'].min()
    max_end = plot_data['transcript_end'].max()
    y_max = len(transcript_summary) + 1
    
    fig = go.Figure()
    intron_color = '#85929E'
    
    intron_x = []
    intron_y = []
    intron_text = []
    
    for _, transcript in transcript_summary.iterrows():
        intron_x.extend([transcript['transcript_start'], transcript['transcript_end'], None])
        intron_y.extend([transcript['trans_order'], transcript['trans_order'], None])
        intron_text.extend([f"Isoform ID: {transcript['id']}<br>Transcript: {transcript['trans_id']}<br>Length: {transcript['transcript_length']:,} bp", "", ""])
    
    fig.add_trace(go.Scatter(
        x=intron_x,
        y=intron_y,
        mode='lines',
        line=dict(color=intron_color, width=2),
        showlegend=False,
        hovertemplate='%{text}<extra></extra>',
        text=intron_text,
        connectgaps=False
    ))
    
    shapes = []
    hover_traces_x = []
    hover_traces_y = []
    hover_traces_text = []

    if color_by_abundance and 'abundance_tpm' in transcript_summary.columns:
        tpm_values = transcript_summary['abundance_tpm'].fillna(0)
        tpm_min, tpm_max = tpm_values.min(), tpm_values.max()
        cmap = get_matplotlib_colormap(colorscale)

    for _, transcript in transcript_summary.iterrows():
        isoform_id = transcript['id']
        trans_id = transcript['trans_id']
        trans_order = transcript['trans_order']
        trans_exons = plot_data[plot_data['id'] == isoform_id].sort_values('exon_start')

        # Check for TPM color first (highest priority when enabled)
        if color_by_abundance and 'abundance_tpm' in transcript_summary.columns:
            tpm = transcript.get('abundance_tpm', 0)

            if tpm_max > tpm_min:
                normalized = (tpm - tpm_min) / (tpm_max - tpm_min)
            else:
                # When all values are equal, use 0.0 to show the minimum color on the scale
                normalized = 0.0

            rgba = cmap(normalized)
            exon_fill_color = f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{int(rgba[3]*255)})'

        # Check for individual transcript color (second priority, convert to string for comparison)
        elif individual_transcript_colors and str(isoform_id) in individual_transcript_colors:
            exon_fill_color = individual_transcript_colors[str(isoform_id)]

        else:
            exon_fill_color = exon_color

        for _, exon in trans_exons.iterrows():
            # Batch add shapes (way faster than individual add_shape calls)
            shapes.append({
                'type': "rect",
                'x0': exon['exon_start'], 'y0': trans_order - 0.3,
                'x1': exon['exon_end'], 'y1': trans_order + 0.3,
                'fillcolor': exon_fill_color,
                'line': {'color': exon_fill_color, 'width': 1},
                'opacity': 0.8
            })

            # Build hover text with optional TPM info
            hover_text = f"Isoform ID: {isoform_id}<br>Transcript ID: {trans_id}<br>Exon: {exon['exon_number']}<br>Size: {exon['exon_size']} bp<br>Coordinates: {exon['exon_start']:,} - {exon['exon_end']:,}"
            if color_by_abundance and 'abundance_tpm' in transcript_summary.columns:
                tpm = transcript.get('abundance_tpm', None)
                if tpm is not None:
                    if abundance_type == 'tissue' and tissue_name:
                        hover_text += f"<br>{tissue_name} Tissue TPM: {tpm:.2f}"

                    elif abundance_type == 'organ' and organ_name:
                        hover_text += f"<br>{organ_name.capitalize()} Organ TPM: {tpm:.2f}"

                    else:
                        hover_text += f"<br>Average TPM: {tpm:.2f}"

            # Create multiple invisible points across the entire exon block so user can click anywhere on it
            exon_length = exon['exon_end'] - exon['exon_start']
            num_points = max(10, int(exon_length / 1000))  # At least 10 points, more for longer exons
            for i in range(num_points + 1):
                x_point = exon['exon_start'] + (exon_length * i / num_points)
                hover_traces_x.append(x_point)
                hover_traces_y.append(trans_order)
                hover_traces_text.append(hover_text)
            # Add separator
            hover_traces_x.append(None)
            hover_traces_y.append(None)
            hover_traces_text.append("")
    
    fig.update_layout(shapes=shapes)

    if hover_traces_x:
        fig.add_trace(go.Scatter(
            x=hover_traces_x,
            y=hover_traces_y,
            mode='lines',
            line=dict(width=25, color='rgba(100,100,100,0)'),
            showlegend=False,
            hovertemplate='%{text}<extra></extra>',
            text=hover_traces_text,
            connectgaps=False  
        ))

    if color_by_abundance:
        if colorscale:
            selected_colorscale = colorscale
        else:
            selected_colorscale = 'Viridis'

        if not transcript_summary.empty and 'abundance_tpm' in transcript_summary.columns:
            tpm_values = transcript_summary['abundance_tpm'].dropna()
            if not tpm_values.empty:
                tpm_min, tpm_max = tpm_values.min(), tpm_values.max()

                if abundance_type == 'tissue' and tissue_name:
                    formatted_tissue = ' '.join(word.capitalize() for word in tissue_name.split())
                    colorbar_title = f"{formatted_tissue}<br>Tissue TPM"

                elif abundance_type == 'organ' and organ_name:
                    formatted_organ = ' '.join(word.capitalize() for word in organ_name.split())
                    colorbar_title = f"{formatted_organ}<br>Organ TPM"

                else:
                    colorbar_title = "Transcript<br>Average TPM"

                # case where all TPM values are 0 (tpm_min == tpm_max)
                if tpm_max > tpm_min:
                    cmin_val = tpm_min
                    cmax_val = tpm_max
                else:
                    # When all values are 0 or the same, need to use a small range for colorbar to still show up...
                    cmin_val = 0
                    cmax_val = 1

                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(
                        colorscale=selected_colorscale,
                        cmin=cmin_val,
                        cmax=cmax_val,
                        colorbar=dict(
                            title=dict(text=colorbar_title, font=dict(size=16)),
                            thickness=20,
                            len=0.6,
                            x=1.15,
                            y=0.78,
                            tickfont=dict(size=10)
                        ),
                        showscale=True,
                        size=0,
                        opacity=0
                    ),
                    showlegend=False,
                    hoverinfo='none',
                    name='TPM Scale'
                ))

    title_text = f"Transcripts for Gene {gene_name} ({gene_ensembl_id})<br>(ORF Perplexity: {orf_perplexity}, Coordinates: {min_start} - {max_end}, Strand: {strand})"

    if color_by_abundance:
        right_margin = 300
    else: 
        right_margin = 200

    fig.update_layout(
        title={
            'text': title_text,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis=dict(
            title="Genomic Position",
            range=[min_start - 1000, max_end + (max_end - min_start) * 0.3 + 1000],
            showgrid=False,
            tickformat=',',
            rangeslider=dict(visible=False, range=[min_start, max_end]),
            autorange=False,
            fixedrange=False
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[0, y_max],
            autorange=False,
            fixedrange=False
        ),
        height=height,
        margin=dict(
            l=100,
            r=right_margin,
            t=80,
            b=50
        ),
        hovermode='closest',
        dragmode='zoom',
        plot_bgcolor='white',
        autosize=True
    )
    
    for _, transcript in transcript_summary.iterrows():
        fig.add_annotation(
            x=max_end + (max_end - min_start) * 0.02,
            y=transcript['trans_order'],
            text=abbreviate_transcript_name(transcript['trans_id']),
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            font=dict(size=12)
        )
    
    return fig


def calculate_optimal_height(transcript_names, num_rows, show_tables, base_height):
    """Calculate height that prevents cutoff"""
    min_cell_height = 20
    optimal_cell_height = max(min_cell_height, 600 / max(num_rows, 1))
    data_height = num_rows * optimal_cell_height
    
    if transcript_names: 
        max_transcript_length = max([len(str(name)) for name in transcript_names])
    else: 
        max_transcript_length = 10

    bottom_margin_needed = max(100, min(180, max_transcript_length * 7))
    
    total_needed = data_height + bottom_margin_needed + 80  # top margin + title
    
    if show_tables == 'show':
        max_allowed = min(base_height, 450)
    else:
        max_allowed = min(base_height, 650)
    
    return min(total_needed, max_allowed)


def create_isoform_expression_heatmap(tpm_data: pd.DataFrame, 
                                      gene_name: str, 
                                      height: int = 600,
                                      colorscale: str = 'Viridis',
                                      data_type: str = 'TPM',
                                      show_tables: str = 'show',
                                      show_labels: bool = False,
                                      collapse_mode: str = 'tissue') -> go.Figure:
    """Creates isoform expression heatmap with proper height to prevent cutoff"""
    
    if tpm_data.empty:
        return create_empty_isoform_message(f"No data for gene: {gene_name}")
    
    try:
        if 'gene_name' not in tpm_data.columns:
            return create_empty_isoform_message("Data missing 'gene_name' column.")
        
    except Exception as e:
        print(f"Error filtering data: {e}")
        return create_empty_isoform_message(f"Error filtering data for gene: {gene_name}")
    
    if tpm_data.empty:
        return create_empty_isoform_message(f"No isoform data found for gene {gene_name}.")
    
    metadata_cols = ['id', 'trans_id', 'transcript', 'gene', 'tpm_average', 'tpm_sum', 'gene_name', 'max_ratio', 'min_ratio', 'prob']
    tissue_cols = [col for col in tpm_data.columns if col not in metadata_cols]
    
    transcript_names = tpm_data['transcript'].tolist() if 'transcript' in tpm_data.columns else tpm_data.index.tolist()
    transcript_names_abbreviated = abbreviate_transcript_names(transcript_names)

    if collapse_mode == 'tissue':
        heatmap_data, tissue_display_names, tissue_categories = average_lrs_by_tissue(tpm_data, tissue_cols)

    elif collapse_mode == 'replicate':
        heatmap_data, tissue_display_names, tissue_categories = average_lrs_by_replicates(tpm_data, tissue_cols)

    else:
        heatmap_data = tpm_data[tissue_cols].values.T
        tissue_display_names, tissue_categories = process_individual_tissues(tissue_cols)
    
    num_tissues = len(tissue_display_names)
    num_transcripts = len(transcript_names)
    
    calculated_height = calculate_optimal_height(transcript_names_abbreviated, num_tissues, show_tables, height)
    
    if show_labels and show_tables != 'show':
        max_tissue_name_length = max([len(name) for name in tissue_display_names]) if tissue_display_names else 10
        left_margin = min(40, min(50, max_tissue_name_length * 8))
    else:
        left_margin = 40
    
    max_transcript_length = max([len(str(name)) for name in transcript_names_abbreviated]) if transcript_names_abbreviated else 10
    bottom_margin = max(100, min(180, max_transcript_length * 7))
    
    right_margin = 100
    top_margin = 60
    
    tissue_colors = get_organ_colors()
    
    if show_tables == 'show':
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.06, 0.94],
            horizontal_spacing=0.005,
            shared_yaxes=True
        )
        
        tissue_color_values = []
        for category in tissue_categories:
            color = tissue_colors.get(category, '#CCCCCC')
            tissue_color_values.append([color])
        
        colorbar_title = "TPM" if data_type == "TPM" else "Ratio"
        fig.add_trace(
            go.Heatmap(
                z=tissue_color_values,
                y=tissue_display_names,
                x=[''],
                colorscale=[[0, '#CCCCCC'], [1, '#CCCCCC']],
                showscale=False,
                hovertemplate='Tissue: %{y}<br>Category: %{customdata}<extra></extra>',
                customdata=tissue_categories,
                colorbar=dict(
                    title=dict(text=colorbar_title, font=dict(size=16)),
                    tickfont=dict(size=10)
                )
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                y=tissue_display_names,
                x=transcript_names_abbreviated,
                colorscale=colorscale,
                hovertemplate=f'Transcript: %{{x}}<br>Tissue: %{{y}}<br>{data_type}: %{{z:.2f}}<extra></extra>',
                colorbar=dict(title=dict(text=data_type, font=dict(size=16)), x=1.02, tickfont=dict(size=10))
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(showticklabels=False, row=1, col=2)
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=1, col=2)
        
    else:
        fig = go.Figure()
        
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                y=tissue_display_names, 
                x=transcript_names_abbreviated,
                colorscale=colorscale,
                hovertemplate=f'Transcript: %{{x}}<br>Tissue: %{{y}}<br>{data_type}: %{{z:.2f}}<extra></extra>',
                colorbar=dict(title=dict(text=data_type, font=dict(size=16)), tickfont=dict(size=10))
            )
        )
        
        if show_labels: 
            fig.update_yaxes(
                showticklabels=True,
                tickmode='array',
                tickvals=list(range(len(tissue_display_names))),
                ticktext=tissue_display_names,
                tickfont=dict(size=9),
                automargin=True
            )
        else: 
            fig.update_yaxes(showticklabels=False)
        
        fig.update_xaxes(
            showticklabels=True,
            tickangle=90,
            tickfont=dict(size=8),
            automargin=True,
            tickmode='array',
            tickvals=list(range(len(transcript_names))),
            ticktext=transcript_names_abbreviated
        )
    
    if collapse_mode == 'tissue':
        heatmap_resolution_level = "averaged by tissue"
    elif collapse_mode == 'replicate':
        heatmap_resolution_level = "averaged by replicate"
    else:
        heatmap_resolution_level = "across all samples and tissues"

    fig.update_layout(
        height=calculated_height,
        margin=dict(l=left_margin, r=right_margin, t=top_margin, b=bottom_margin),
        title={
            'text': f'Isoform Expression for {gene_name} {heatmap_resolution_level} ({len(transcript_names)} isoforms)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14},
            'pad': {'b': 8}
        },
        font=dict(size=9),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        autosize=False
    )
    
    return fig


def calculate_legend_x_position(colorbar_x, offset=0.15):
    """Calculate organ legend x position based on colorbar position"""
    x_position = colorbar_x + offset

    return x_position


def calculate_bottom_margin(show_labels: bool, transcript_names: list, hide_tissue_labels: bool = False) -> int:
    """Calculate bottom margin based on label visibility and name lengths

    Args:
        show_labels: Whether to show transcript (row) labels
        transcript_names: List of transcript names for sizing
        hide_tissue_labels: Whether tissue (column) labels are hidden
    """
    # If tissue labels are hidden, use minimal bottom margin
    if hide_tissue_labels:
        return 50

    # Otherwise, calculate based on transcript labels
    if show_labels:
        max_name_length = max(len(str(name)) for name in transcript_names) if transcript_names else 10
        return max(100, min(200, max_name_length * 7))
    return 50


def create_isoform_expression_clustergram(tpm_data: pd.DataFrame,
                                          ratio_data: pd.DataFrame,
                                          log_tpm_data: pd.DataFrame,
                                          gene_name: str,
                                          height: int = 600,
                                          colorscale: str = 'Viridis',
                                          data_type: str = 'Ratio',
                                          show_tables: str = 'show',
                                          show_labels: bool = False,
                                          collapse_mode: str = 'tissue',
                                          distance_metric: str = 'euclidean',
                                          linkage_method: str = 'complete',
                                          show_gridlines: bool = False,
                                          gridline_color: str = '#ffffff',
                                          db_path: str = None,
                                          species: str = 'human') -> go.Figure:
    """Create responsive clustergram with transcripts ordered by average TPM (descending)"""
    if data_type == 'TPM' or data_type == 'tpm':
        expression_data = tpm_data
    elif data_type == 'Log TPM' or data_type == 'log_tpm':
        expression_data = log_tpm_data
    else:
        expression_data = ratio_data

    if expression_data.empty or tpm_data.empty or ratio_data.empty:
        return create_empty_isoform_message(f"No data for gene: {gene_name}")

    if 'gene_name' not in expression_data.columns:
        return create_empty_isoform_message("Data missing 'gene_name' column.")
    #expression_data = expression_data[expression_data['gene_name'] == gene_name].copy()

    if expression_data.empty:
        return create_empty_isoform_message(f"No isoform data found for gene {gene_name}.")

    metadata_cols = ['id', 'trans_id', 'transcript', 'gene', 'tpm_average', 'tpm_sum', 'gene_name', 'max_ratio', 'min_ratio', 'prob', 'isoform_average_tpm']
    tissue_cols = [col for col in expression_data.columns if col not in metadata_cols]

    if 'isoform_average_tpm' not in expression_data.columns and db_path:
        conn = sqlite3.connect(db_path)
        table_prefix = get_table_prefix(species)
        # Get gene_id for gene_name
        gene_id_for_tpm = get_gene_id_for_gene_name(db_path, gene_name, species)
        if gene_id_for_tpm:
            isoform_tpm_query = f"SELECT id, isoform_average_tpm FROM {table_prefix}isoforms WHERE gene_id LIKE ?"
            isoform_tpm = pd.read_sql_query(isoform_tpm_query, conn, params=[f"{gene_id_for_tpm}%"])
        else:
            isoform_tpm = pd.DataFrame()
        conn.close()

        if not isoform_tpm.empty:
            expression_data = expression_data.merge(isoform_tpm, on='id', how='left')
            tpm_data = tpm_data.merge(isoform_tpm, on='id', how='left')
            ratio_data = ratio_data.merge(isoform_tpm, on='id', how='left')
            log_tpm_data = log_tpm_data.merge(isoform_tpm, on='id', how='left')

    # Sort all data by isoform_average_tpm (ascending), treating NaNs as 0
    # Weirdly, Dash Bio displays first row at BOTTOM, so we have to sort in ascending order so highest TPM appears at top...
    if 'isoform_average_tpm' in expression_data.columns:
        sort_col = expression_data['isoform_average_tpm'].fillna(0)
        sort_indices = sort_col.argsort().values
        expression_data = expression_data.iloc[sort_indices].reset_index(drop=True)
        tpm_data = tpm_data.iloc[sort_indices].reset_index(drop=True)
        ratio_data = ratio_data.iloc[sort_indices].reset_index(drop=True)
        log_tpm_data = log_tpm_data.iloc[sort_indices].reset_index(drop=True)

    transcript_names = expression_data['transcript'].tolist() if 'transcript' in expression_data.columns else expression_data.index.tolist()
    transcript_names_abbreviated = abbreviate_transcript_names(transcript_names)
    num_transcripts = len(transcript_names)

    if height <= 700:
        min_required_height = calculate_clustergram_min_height(num_transcripts, base_height=600)
        height = max(height, min_required_height)

    if collapse_mode == 'tissue':
        heatmap_data, tissue_display_names, tissue_categories = average_lrs_by_tissue(expression_data, tissue_cols, species)

        # Mapping from tissue name to column indices
        tissue_name_to_indices = {}
        for col in tissue_cols:
            tissue_name = extract_tissue_name_from_column(col)
            if tissue_name not in tissue_name_to_indices:
                tissue_name_to_indices[tissue_name] = []
            tissue_name_to_indices[tissue_name].append(col)

        kept_tissue_cols = []
        for tissue_name in tissue_display_names:
            if tissue_name in tissue_name_to_indices:
                kept_tissue_cols.extend(tissue_name_to_indices[tissue_name])

        # Average TPM and Ratio using only kept tissues, forcing the same output tissues
        tpm_heatmap_data, _, _ = average_lrs_by_tissue(tpm_data, kept_tissue_cols, species, force_keep_tissues=tissue_display_names)
        ratio_heatmap_data, _, _ = average_lrs_by_tissue(ratio_data, kept_tissue_cols, species, force_keep_tissues=tissue_display_names)
        tissue_cols_for_organs = tissue_display_names

    elif collapse_mode == 'replicate':
        heatmap_data, tissue_display_names, tissue_categories = average_lrs_by_replicates(expression_data, tissue_cols, species)

        # Use the same tissue list for TPM and Ratio as used for the main data type
        tissue_name_to_indices = {}
        for col in tissue_cols:
            tissue_name = extract_tissue_name_from_column(col)
            if tissue_name not in tissue_name_to_indices:
                tissue_name_to_indices[tissue_name] = []
            tissue_name_to_indices[tissue_name].append(col)

        kept_tissue_cols = []
        for tissue_name in tissue_display_names:
            if tissue_name in tissue_name_to_indices:
                kept_tissue_cols.extend(tissue_name_to_indices[tissue_name])

        # Average TPM and Ratio using only the kept tissues, forcing the same output tissues
        tpm_heatmap_data, tpm_tissue_names, _ = average_lrs_by_replicates(tpm_data, kept_tissue_cols, species, force_keep_tissues=tissue_display_names)
        ratio_heatmap_data, ratio_tissue_names, _ = average_lrs_by_replicates(ratio_data, kept_tissue_cols, species, force_keep_tissues=tissue_display_names)
        tissue_cols_for_organs = tissue_display_names

    else:
        heatmap_data = expression_data[tissue_cols].values.T
        tpm_heatmap_data = tpm_data[tissue_cols].values.T
        ratio_heatmap_data = ratio_data[tissue_cols].values.T
        tissue_display_names, tissue_categories = process_individual_tissues(tissue_cols, species)
        tissue_cols_for_organs = tissue_cols

    organ_list, color_list = create_organ_annotation_bar(tissue_cols_for_organs, species=species)
    clean_tissue_names = [extract_tissue_name_from_column(col) for col in tissue_cols_for_organs]
    
    # Ensure proper data orientation: rows = transcripts, columns = tissues
    if heatmap_data.shape[0] == len(tissue_cols_for_organs):
        clustergram_data = heatmap_data.T
    else:
        clustergram_data = heatmap_data
    
    num_tissues = len(clean_tissue_names)
    # Hide tissue labels for mouse data by default due to many cell types
    hide_tissue_labels = (num_tissues > 30) or (show_tables == 'show') or (species == 'Mouse')
    #left_margin = min(40, int(height * 0.1))
    if not show_labels or (num_transcripts > 30): 
        left_margin = 2  # Minimum left margin, just enough to prevent figure overflow
    else:
        # Calculate pixel width for longest label
        max_label_len = max((len(str(name)) for name in transcript_names_abbreviated), default=10)
        # Approximate: 7px per character, add 10px for a buffer
        left_margin = min(70, max(20, 7 * max_label_len + 10))
    
    if hide_tissue_labels:
        bottom_margin = 50
        actual_clustergram_height = height
    else:
        bottom_margin = max(200, int(height * 0.35))
        actual_clustergram_height = height - 80

    clustergram_data_processed = pd.DataFrame(clustergram_data).copy()

    # For log TPM data, keep track of NaN values to show them instead of filling with 0
    show_nan_as_black = (data_type == "Log TPM" or data_type == "log_tpm")
    clustergram_data_with_nan = None

    if show_nan_as_black:
        # Replace -inf and inf values with NaN, keep NaN as NaN for proper handling
        clustergram_data_processed = clustergram_data_processed.replace([np.inf, -np.inf], np.nan)
        nan_count = clustergram_data_processed.isna().sum().sum()
        clustergram_data_with_nan = clustergram_data_processed.copy()

        for idx in clustergram_data_processed.index:
            row = clustergram_data_processed.loc[idx]
            row_median = row.median()  # median of non-NaN values
            if pd.notna(row_median):
                clustergram_data_processed.loc[idx, row.isna()] = row_median
            else:
                # If all values are NaN in this row, use 0
                clustergram_data_processed.loc[idx, row.isna()] = 0
    else:
        # Original behavior: replace -inf and inf with 0, fill NaN with 0
        clustergram_data_processed = clustergram_data_processed.replace([np.inf, -np.inf], 0)
        clustergram_data_processed = clustergram_data_processed.fillna(0)

    clustergram_data_processed = clustergram_data_processed.astype(float)

    if distance_metric in ['correlation', 'seuclidean', 'cosine']:
        clustergram_data_processed = apply_distance_preprocessing(clustergram_data_processed, distance_metric)
    
    if num_transcripts == 1:
        return create_single_transcript_heatmap(
            heatmap_data=heatmap_data,
            tpm_heatmap_data=tpm_heatmap_data,
            ratio_heatmap_data=ratio_heatmap_data,
            transcript_names=transcript_names_abbreviated,
            tissue_display_names=tissue_display_names,
            gene_name=gene_name,
            height=height,
            colorscale=colorscale,
            data_type=data_type
        )
    
    try:
        clustergram, computed_traces = dash_bio.Clustergram(
            data=clustergram_data_processed.values,
            column_labels=clean_tissue_names,
            row_labels=transcript_names_abbreviated,
            column_colors=color_list,
            height=actual_clustergram_height,
            color_threshold={'row': 0.7, 'col': 0.7},
            hidden_labels='col' if hide_tissue_labels else None,
            cluster='col', 
            color_list={
                'row': ['#636EFA', '#EF553B', '#00CC96', '#AB63FA'],
                'col': ['#FFA15A', '#19D3F3', '#FF6692', '#B6E880'],
                'bg': '#506784'
            },
            line_width=2,
            display_ratio=[0.12, 0.08] if not hide_tissue_labels else [0.08, 0.05],
            standardize='none',
            center_values=False,
            return_computed_traces=True,
            row_dist=distance_metric,
            col_dist=distance_metric,
            link_method=linkage_method
        )

        # Update heatmap trace with reordered organs and both TPM/ratio values
        column_ids = computed_traces['column_ids']
        row_ids = computed_traces['row_ids']
        reordered_organ_list = [organ_list[i] for i in column_ids]

        # Custom hover data with TPM, log10(TPM), and ratio values: need to reorder based on column clustering
        if tpm_heatmap_data.shape[0] == len(tissue_cols_for_organs):
            tpm_clustered = tpm_heatmap_data.T[row_ids][:, column_ids]
            ratio_clustered = ratio_heatmap_data.T[row_ids][:, column_ids]
        else:
            tpm_clustered = tpm_heatmap_data[row_ids][:, column_ids]
            ratio_clustered = ratio_heatmap_data[row_ids][:, column_ids]

        # Always get log10(TPM) data for tooltip, regardless of which data type is being displayed
        log_tpm_clustered = None
        if clustergram_data_with_nan is not None:
            if clustergram_data_with_nan.shape[0] == len(tissue_cols_for_organs):
                log_tpm_clustered = clustergram_data_with_nan.T.iloc[row_ids, :].iloc[:, column_ids].values
            else:
                log_tpm_clustered = clustergram_data_with_nan.iloc[row_ids, :].iloc[:, column_ids].values

        customdata = np.zeros((len(transcript_names), len(clean_tissue_names), 4), dtype=object)
        for i in range(len(transcript_names)):
            for j in range(len(clean_tissue_names)):
                customdata[i, j, 0] = reordered_organ_list[j]
                customdata[i, j, 1] = tpm_clustered[i, j]
                # Format log10(TPM) to show NaN as 'NaN', otherwise format to 2 decimal places
                if log_tpm_clustered is not None:
                    log_val = log_tpm_clustered[i, j]
                    customdata[i, j, 2] = 'NaN' if pd.isna(log_val) else f'{log_val:.2f}'
                else:
                    customdata[i, j, 2] = 'NaN'
                customdata[i, j, 3] = ratio_clustered[i, j]

        heatmap_trace = clustergram.data[-1]
        heatmap_trace.customdata = customdata
        heatmap_trace.hovertemplate = (
            "<b>Transcript:</b> %{y}<br>"
            "<b>Tissue:</b> %{x}<br>"
            "<b>Organ:</b> %{customdata[0]}<br>"
            "<b>TPM:</b> %{customdata[1]:.2f}<br>"
            "<b>Log10(TPM):</b> %{customdata[2]}<br>"
            "<b>Ratio:</b> %{customdata[3]:.2f}"
            "<extra></extra>"
        )

        # Log TPM NaN clustering workaround: replace the median-filled NaN values back with NaN in the heatmap display
        if show_nan_as_black and clustergram_data_with_nan is not None:
            nan_data_reordered = clustergram_data_with_nan.iloc[row_ids, :].iloc[:, column_ids]
            # Replace the heatmap trace data with the version containing NaN (show as transparent/white)
            heatmap_trace.z = nan_data_reordered.values
        
    except Exception as e2:
        print(f"Error creating clustergram: {e2}")
        return create_empty_isoform_message(f"Error creating visualization for {gene_name}")
    
    clustergram = apply_colorscale_to_clustergram(clustergram, colorscale, show_nan_as_black=show_nan_as_black)
    colorbar_x = -0.35
    
    try:
        if len(clustergram.data) > 0:
            heatmap_trace = clustergram.data[-1]
            
            if show_gridlines:
                heatmap_trace.xgap = 1
                heatmap_trace.ygap = 1

            if hasattr(heatmap_trace, 'colorbar'):
                heatmap_trace.colorbar.x = colorbar_x 
                heatmap_trace.colorbar.y = 1.0     
                heatmap_trace.colorbar.yanchor = 'top'
                heatmap_trace.colorbar.len = 0.3    
                heatmap_trace.colorbar.thickness = 20

    except Exception as e:
        print(f"Warning: Could not update colorbar position: {e}")

    organ_list, color_list = create_organ_annotation_bar(tissue_cols_for_organs, species=species)
    unique_organs = []
    unique_colors = []
    for organ, color in zip(organ_list, color_list):
        if organ not in unique_organs:
            unique_organs.append(organ)
            unique_colors.append(color)

    if show_labels and not (num_transcripts > 30):
        # Calculate approximate width needed for y-axis labels (transcript names)
        max_transcript_label_len = max((len(str(name)) for name in transcript_names_abbreviated), default=10)
        yaxis_label_width_approx = max_transcript_label_len * 7
    else:
        yaxis_label_width_approx = 0

    left_margin_paper = max(0.15, (yaxis_label_width_approx + 60) / 800)
    num_tissues = len(tissue_cols_for_organs) if 'tissue_cols_for_organs' in locals() else len(tissue_cols)
    
    # Scale positions based on number of tissues - more tissues = legends closer to plot
    #width_scale = min(1.0, 20 / max(num_tissues, 20))
    
    # Colorbar positioning - use paper coordinate at plot edge, then shift by pixels
    colorbar_x_paper = 1.0  
    colorbar_pixel_offset = 150 
    colorbar_y_position = 1.005

    # Use pixel-based positioning for legend items to keep spacing constant regardless of height
    # yshift works in pixels, not in paper coordinates, so spacing remains fixed
    pixels_between_items = 25  # Pixel distance between legend items
    vertical_offset_pixels = 7  # Pixel offset from top before first item
    legend_y_start = colorbar_y_position  # Fixed paper coordinate (top of plot)

    try:
        if len(clustergram.data) > 0:
            heatmap_trace = clustergram.data[-1]
            if hasattr(heatmap_trace, 'colorbar'):
                heatmap_trace.colorbar.x = colorbar_x_paper
                heatmap_trace.colorbar.xpad = colorbar_pixel_offset
                heatmap_trace.colorbar.y = colorbar_y_position
                heatmap_trace.colorbar.yanchor = 'top'
                heatmap_trace.colorbar.len = 0.3
                heatmap_trace.colorbar.thickness = 20
                heatmap_trace.colorbar.title = dict(text=data_type, font=dict(size=16))
    except Exception as e:
        print(f"Warning: Could not update colorbar position: {e}")

    legend_base_x = colorbar_x_paper
    # Offset from plot edge = colorbar offset + colorbar width + spacing between legends
    base_spacing = 30
    extra_spacing_log_tpm = 20 if (data_type == "Log TPM" or data_type == "log_tpm") else 0
    legend_pixel_offset = colorbar_pixel_offset + 20 + base_spacing + extra_spacing_log_tpm 

    clustergram.add_annotation(
        x=legend_base_x,
        y=legend_y_start,
        xref="paper",
        yref="paper",
        xshift=legend_pixel_offset,  # Shift by pixels instead of paper coords
        yshift=-vertical_offset_pixels,  # Negative shifts down in pixels
        text="Organ Legend",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=16, family="Open Sans, verdana, arial, sans-serif")
    )

    for i, (organ, color) in enumerate(zip(unique_organs, unique_colors)):
        clustergram.add_annotation(
            x=legend_base_x,
            y=legend_y_start,
            xref='paper',
            yref='paper',
            xshift=legend_pixel_offset,
            yshift=-(vertical_offset_pixels + 30 + (i * pixels_between_items)),  # Title height (30px) + item spacing
            text=f'<span style="color:{color}; font-size:16px">&#9632;</span> {organ}',
            showarrow=False,
            xanchor='left',
            yanchor='top',
            font=dict(size=13)
        )
    
    clustergram.update_layout(
        title={
            'text': f"Isoform Expression Clustergram for {gene_name} ({len(transcript_names)} isoforms, {data_type} data)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18 if hide_tissue_labels else 20}
        },
        margin=dict(
                l=min(20, left_margin + 50),
                r=350,
                t=90,
                b=calculate_bottom_margin(show_labels, transcript_names_abbreviated, hide_tissue_labels)
        ),
        autosize=True,
        width=None,
        height=height,
        uirevision='constant',
        yaxis=dict(
            automargin=True,
            tickangle=0,
            tickfont=dict(size=min(13, max(10, int(height/60)+2))),
            showgrid=True,
            gridcolor='white',
            gridwidth=1
        ),
        xaxis=dict(
            automargin=True,
            tickangle=90 if show_labels else 0,
            tickfont=dict(
                size=8 if show_labels else 1,
                color='rgba(0,0,0,0)' if not show_labels else None
            ),
            showticklabels=show_labels,
            showgrid=True,
            gridcolor='white',
            gridwidth=1
        ),
        plot_bgcolor=gridline_color if show_gridlines else 'rgba(0,0,0,0)',
        paper_bgcolor='white'
    )
    
    return clustergram


def create_single_transcript_heatmap(heatmap_data, tpm_heatmap_data, ratio_heatmap_data,
                                     transcript_names, tissue_display_names,
                                     gene_name, height, colorscale, data_type):
    """Create simple heatmap when only one transcript remains"""
    fig = go.Figure()

    # Customdata array for each cell with format [tpm_value, ratio_value] needed to show both expression types
    customdata = np.zeros((len(tissue_display_names), len(transcript_names), 2), dtype=object)
    for i in range(len(tissue_display_names)):
        for j in range(len(transcript_names)):
            customdata[i, j, 0] = tpm_heatmap_data[i, j]   
            customdata[i, j, 1] = ratio_heatmap_data[i, j]  

    fig.add_trace(go.Heatmap(
        z=heatmap_data,
        x=transcript_names,
        y=tissue_display_names,
        colorscale=colorscale,
        colorbar=dict(title=dict(text=data_type, font=dict(size=16)), tickfont=dict(size=10)),
        customdata=customdata,
        hovertemplate='<b>Transcript:</b> %{x}<br><b>Tissue:</b> %{y}<br><b>TPM:</b> %{customdata[0]:.2f}<br><b>Ratio:</b> %{customdata[1]:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"{gene_name} Expression for Transcript {transcript_names[0]}",
        height=height,
        xaxis=dict(title="Transcript"),
        yaxis=dict(title="Tissue"),
        margin=dict(l=100, r=50, t=80, b=100)
    )

    return fig


def apply_colorscale_to_clustergram(fig, colorscale, show_nan_as_black=False):
    """Apply colorscale to the heatmap portion of a clustergram"""
    try:
        if len(fig.data) > 0:
            heatmap_trace = fig.data[-1]
            heatmap_trace.colorscale = colorscale
            heatmap_trace.showscale = True
    except Exception as e:
        pass

    return fig


def average_lrs_by_tissue(tpm_data: pd.DataFrame,
                                tissue_cols: List[str],
                                species: str = 'human',
                                force_keep_tissues: List[str] = None) -> Tuple[np.ndarray, List[str], List[str]]:
    """Collapse multiple experiments per tissue by averaging values

    Args:
        tpm_data: DataFrame with expression data
        tissue_cols: List of column names to process
        species: Species identifier
        force_keep_tissues: If provided, ensures output includes these tissues in this order,
                           padding with NaN rows if tissue not found in data
    """
    tissue_to_organ_mapping = get_tissue_to_organ_mapping(species)
    tissue_mapping = {}

    for col in tissue_cols:
        tissue_mapping[col] = extract_tissue_name_from_column(col)

    # Group columns by tissue name
    tissue_groups = {}
    for col, tissue_name in tissue_mapping.items():
        if tissue_name not in tissue_groups:
            tissue_groups[tissue_name] = []
        tissue_groups[tissue_name].append(col)

    # First pass: compute averages for all tissues
    tissue_data_map = {}  # tissue_name -> averaged_values

    for tissue_name, columns in tissue_groups.items():
        tissue_data = tpm_data[columns].values

        # skip if all values in tissue are NaN (unless forced to keep)
        if np.isnan(tissue_data).all():
            if force_keep_tissues is None:
                continue
            else:
                # Keep as all NaN
                averaged_values = np.full(len(tpm_data), np.nan)
        else:
            # Use nanmean to properly handle NaN values
            if np.isnan(tissue_data).any():
                mask = ~np.isnan(tissue_data)
                with np.errstate(invalid='ignore', all='ignore'):
                    averaged_values = np.where(mask.any(axis=1), np.nanmean(tissue_data, axis=1), np.nan)
            else:
                averaged_values = np.mean(tissue_data, axis=1)

        tissue_data_map[tissue_name] = averaged_values

    # Second pass: build output in correct order
    if force_keep_tissues is not None:
        # Use forced tissue list, padding with NaN if tissue not in data
        averaged_data = []
        tissue_display_names = []
        tissue_categories = []

        for tissue_name in force_keep_tissues:
            if tissue_name in tissue_data_map:
                averaged_data.append(tissue_data_map[tissue_name])
            else:
                # Tissue not found in this data, pad with NaN
                averaged_data.append(np.full(len(tpm_data), np.nan))

            tissue_display_names.append(tissue_name)
            tissue_category = tissue_to_organ_mapping.get(tissue_name, 'unknown')
            tissue_categories.append(tissue_category)
    else:
        # Use natural order
        averaged_data = []
        tissue_display_names = []
        tissue_categories = []

        for tissue_name, averaged_values in tissue_data_map.items():
            averaged_data.append(averaged_values)
            tissue_display_names.append(tissue_name)
            tissue_category = tissue_to_organ_mapping.get(tissue_name, 'unknown')
            tissue_categories.append(tissue_category)

    heatmap_data = np.array(averaged_data)

    return heatmap_data, tissue_display_names, tissue_categories


def average_lrs_by_replicates(tpm_data: pd.DataFrame,
                              tissue_cols: List[str],
                              species: str = 'human',
                              force_keep_tissues: List[str] = None) -> Tuple[np.ndarray, List[str], List[str]]:
    """Collapse only replicates by averaging, keep unique samples separate

    Args:
        tpm_data: DataFrame with expression data
        tissue_cols: List of column names to process
        species: Species identifier
        force_keep_tissues: If provided, ensures output includes these tissues in this order,
                           padding with NaN rows if tissue not found in data
    """
    replicate_groups = get_sample_replicates_mapping(species)
    tissue_to_organ_mapping = get_tissue_to_organ_mapping(species)

    # First pass: collect data for all available tissues
    tissue_data_map = {}  # tissue_name -> averaged_values

    for group in replicate_groups:
        # Find which columns from tissue_cols are in this group
        group_columns = [col for col in tissue_cols if col in group]
        if not group_columns:
            continue

        tissue_name = extract_tissue_name_from_column(group_columns[0])

        # Case 1: multiple replicates, so need to average them
        if len(group_columns) > 1:
            tissue_data = tpm_data[group_columns].values

            if np.isnan(tissue_data).all():
                if force_keep_tissues is None:
                    continue  # Skip if not forced to keep
                else:
                    # Keep as all NaN
                    averaged_values = np.full(len(tpm_data), np.nan)
            else:
                # Use nanmean to properly preserve NaN if present
                if np.isnan(tissue_data).any():
                    mask = ~np.isnan(tissue_data)
                    with np.errstate(invalid='ignore', all='ignore'):
                        averaged_values = np.where(mask.any(axis=1), np.nanmean(tissue_data, axis=1), np.nan)
                else:
                    averaged_values = np.mean(tissue_data, axis=1)

        # Case 2: single replicate
        else:
            averaged_values = tpm_data[group_columns[0]].values

        tissue_data_map[tissue_name] = averaged_values

    # Second pass: build output in correct order
    if force_keep_tissues is not None:
        # Use forced tissue list, padding with NaN if tissue not in data
        averaged_data = []
        tissue_display_names = []
        tissue_categories = []

        for tissue_name in force_keep_tissues:
            if tissue_name in tissue_data_map:
                averaged_data.append(tissue_data_map[tissue_name])
            else:
                # Tissue not found in this data, pad with NaN
                averaged_data.append(np.full(len(tpm_data), np.nan))

            tissue_display_names.append(tissue_name)
            tissue_category = tissue_to_organ_mapping.get(tissue_name, 'unknown')
            tissue_categories.append(tissue_category)
    else:
        # Use natural order from replicate groups
        averaged_data = []
        tissue_display_names = []
        tissue_categories = []

        for group in replicate_groups:
            group_columns = [col for col in tissue_cols if col in group]
            if not group_columns:
                continue

            tissue_name = extract_tissue_name_from_column(group_columns[0])
            if tissue_name in tissue_data_map:
                averaged_data.append(tissue_data_map[tissue_name])
                tissue_display_names.append(tissue_name)
                tissue_category = tissue_to_organ_mapping.get(tissue_name, 'unknown')
                tissue_categories.append(tissue_category)

    heatmap_data = np.array(averaged_data)

    return heatmap_data, tissue_display_names, tissue_categories


def process_individual_tissues(tissue_cols: List[str], species: str = 'human') -> Tuple[List[str], List[str]]:
    """Process individual tissue experiments (current behavior)"""
    tissue_display_names = []
    tissue_categories = []
    tissue_to_organ_mapping = get_tissue_to_organ_mapping(species)
    
    for col in tissue_cols:
        display_name = extract_tissue_name_from_column(col)
        tissue_display_names.append(display_name)
        tissue_category = tissue_to_organ_mapping.get(display_name, 'unknown')
        tissue_categories.append(tissue_category)
    
    return tissue_display_names, tissue_categories


def get_mouse_tissue_to_organ_mapping():
    """Create mouse-specific tissue to organ mapping"""
    return {
        'left cerebral cortex': 'brain',
        'layer of hippocampus': 'brain',
        'gastrocnemius': 'muscle',
        'adrenal gland': 'adrenal gland',
        'heart': 'heart',
        'F121.9': 'cell line',
        'F121-9': 'cell line',
        'myotube': 'muscle',
        'C2C12': 'muscle cell line',
        'forelimb': 'limb'
    }


def get_human_tissue_to_organ_mapping():
    """Create human-specific tissue to organ mapping with complete heart anatomy"""
    return {
        'GM12878': 'blood',
        'HL-60': 'blood',
        'HL.60': 'blood',
        'K562': 'blood',
        'OCI-LY7': 'blood',
        'OCI.LY7': 'blood',
        'astrocyte': 'brain',
        'dorsolateral prefrontal cortex': 'brain',
        'glutamatergic neuron': 'brain',
        'chondrocyte': 'cartilage',
        'HFFc6': 'cartilage',
        'IMR-90': 'lung',
        'IMR.90': 'lung',
        'mesenteric fat pad': 'adipose',
        'osteocyte': 'bone',
        'WTC11': 'iPS',
        'endodermal cell': 'embryo',
        'endothelial cell': 'embryo',
        'H1': 'embryo',
        'H9': 'embryo',
        'neural crest cell': 'embryo',
        'endothelial cell of umbilical vein': 'epithelial',
        'HepG2': 'liver',
        'mammary epithelial cell': 'epithelial',
        'MCF 10A': 'breast',
        'MCF.7': 'breast',
        'MCF-7': 'breast',
        'Panc1': 'pancreas',
        'type B pancreatic cell': 'pancreas',
        'adrenal gland': 'adrenal gland',
        'PC-3': 'prostate',
        'PC.3': 'prostate',
        'progenitor cell of endocrine pancreas': 'pancreas',
        'aorta': 'vessels',
        'cardiac septum': 'heart',
        'heart left ventricle': 'heart',
        'heart right ventricle': 'heart',
        'left cardiac atrium': 'heart',
        'left ventricle myocardium inferior': 'heart',
        'left ventricle myocardium superior': 'heart',
        'posterior vena cava': 'vessels',
        'right cardiac atrium': 'heart',
        'Right ventricle myocardium inferior': 'heart',
        'Right ventricle myocardium superior': 'heart',
        'Caco-2': 'colon',
        'Caco.2': 'colon',
        'HCT116': 'colon',
        'left colon': 'colon',
        'mucosa of descending colon': 'colon',
        'GM23338': 'iPS',
        'kidney': 'kidney',
        'right lobe of liver': 'liver',
        'Calu3': 'lung',
        'left lung': 'lung',
        'lower lobe of left lung': 'lung',
        'lower lobe of right lung': 'lung',
        'PC-9': 'lung',
        'PC.9': 'lung',
        'upper lobe of right lung': 'lung',
        'A673': 'bone',
        'psoas muscle': 'muscle',
        'ovary': 'ovary'
    }


def get_tissue_to_organ_mapping(species='human'):
    """
    Get tissue to organ mapping for the specified species.

    Args:
        species: 'human' or 'mouse'

    Returns:
        Dictionary mapping tissue names to organ categories
    """
    if species.lower() == 'mouse':
        return get_mouse_tissue_to_organ_mapping()
    else:
        return get_human_tissue_to_organ_mapping()


def get_organ_colors():
    """Get organ color mapping based on R code"""
    organ_colors = {
        'blood': '#FF6B6B',        # coral
        'brain': '#4ECDC4',        # blue
        'cartilage': '#45B7D1',    # green
        'embryo': '#96CEB4',       # purple
        'epithelial': '#FFEAA7',   # burlywood
        'adrenal gland': '#DDA0DD', # pink
        'heart': '#FF7675',        # red
        'colon': '#A0522D',        # brown
        'kidney': '#00CED1',       # cyan
        'liver': '#FFD700',        # gold
        'lung': '#87CEEB',         # lightblue
        'muscle': '#00008B',       # darkblue
        'ovary': '#800000',        # maroon
        'iPS': '#808080',          # grey
        'adipose': '#FFA500',      # orange
        'bone': '#F5F5DC',         # beige
        'breast': '#EE82EE',       # violet
        'pancreas': '#90EE90',     # lightgreen
        'prostate': '#FFFF00',     # yellow
        'vessels': '#D2691E',      # chocolate
        'cell line': '#A9A9A9',    # gray
        'muscle cell line': '#A9A9A9',  # gray
        'limb': '#DEB887',         # light orange
        'unknown': '#CCCCCC'       # light gray for unmapped tissues
    }
    return organ_colors


def extract_tissue_name_from_column(column_name):
    """Extract tissue name from column name (remove ENCFF prefix)"""
    # Remove ENCFF prefix: ENCFF123ABC.tissue_name -> tissue_name
    if '.' in column_name and column_name.startswith('ENCFF'):
        tissue_name = '.'.join(column_name.split('.')[1:])
    else:
        tissue_name = column_name

    tissue_name = (
        tissue_name.replace('_', ' ')
                  .strip()
    )
    return tissue_name


def create_organ_annotation_bar(tissue_cols, height=20, species='human'):
    """Create organ color annotation bar for clustergram"""
    tissue_to_organ = get_tissue_to_organ_mapping(species)
    organ_colors = get_organ_colors()

    organ_list = []
    color_list = []

    for col in tissue_cols:
        tissue_name = extract_tissue_name_from_column(col)
        organ = tissue_to_organ.get(tissue_name, 'unknown')
        color = organ_colors.get(organ, '#CCCCCC')

        organ_list.append(organ)
        color_list.append(color)

    return organ_list, color_list


def get_tissue_tpm_for_isoforms(db_path: str, gene_name: str, tissue_name: str, species="Human") -> dict:
    """
    Get TPM values averaged by tissue for all isoforms of a gene.
    Returns a dictionary mapping isoform_id -> tissue_average_tpm
    """
    try:
        tpm_data = load_expression_data(db_path, gene_name, data_type='tpm', species=species)

        if tpm_data.empty:
            return {}

        tissue_cols = [col for col in tpm_data.columns
                      if col.startswith('ENCFF') and extract_tissue_name_from_column(col) == tissue_name]

        if not tissue_cols:
            return {}

        # Calculate average TPM across tissue replicates for each isoform
        tissue_tpm_dict = {}
        for _, row in tpm_data.iterrows():
            isoform_id = row['id']
            tissue_tpm_values = [row[col] for col in tissue_cols if col in tpm_data.columns]
            # Calculate mean, ignoring NaN values
            tissue_tpm_values = [v for v in tissue_tpm_values if pd.notna(v)]

            if tissue_tpm_values:
                tissue_tpm_dict[isoform_id] = sum(tissue_tpm_values) / len(tissue_tpm_values)

            else:
                tissue_tpm_dict[isoform_id] = 0

        return tissue_tpm_dict

    except Exception as e:
        print(f"Error calculating tissue TPM for {tissue_name}: {e}")
        return {}


def get_organ_tpm_for_isoforms(db_path: str, gene_name: str, organ_name: str, species="Human") -> dict:
    """
    Get TPM values averaged by organ for all isoforms of a gene.
    Returns a dictionary mapping isoform_id -> organ_average_tpm
    """
    try:
        tpm_data = load_expression_data(db_path, gene_name, data_type='tpm', species=species)

        if tpm_data.empty:
            return {}

        tissue_to_organ = get_tissue_to_organ_mapping(species)

        organ_cols = []
        for col in tpm_data.columns:
            if col.startswith('ENCFF'):
                tissue_name = extract_tissue_name_from_column(col)
                mapped_organ = tissue_to_organ.get(tissue_name, '')
                if mapped_organ.lower() == organ_name.lower():
                    organ_cols.append(col)

        if not organ_cols:
            return {}

        organ_tpm_dict = {}
        for _, row in tpm_data.iterrows():
            isoform_id = row['id']
            organ_tpm_values = [row[col] for col in organ_cols if col in tpm_data.columns]
            # Calculate mean, ignoring NaN values
            organ_tpm_values = [v for v in organ_tpm_values if pd.notna(v)]

            if organ_tpm_values:
                organ_tpm_dict[isoform_id] = sum(organ_tpm_values) / len(organ_tpm_values)

            else:
                organ_tpm_dict[isoform_id] = 0

        return organ_tpm_dict

    except Exception as e:
        print(f"Error calculating organ TPM for {organ_name}: {e}")
        return {}


def get_unique_tissues_for_gene(db_path: str, gene_name: str, species: str = 'Human') -> list:
    """Get list of unique tissues for a given gene"""
    try:
        tpm_data = load_expression_data(db_path, gene_name, data_type='tpm', species=species)

        if tpm_data.empty:
            return []

        tissues = set()
        exclude_cols = {'id', 'transcript', 'trans_id', 'gene', 'gene_name', 'isoform_average_tpm',
                       'index', 'tpm average', 'tpm sum'}
        tissue_cols = [col for col in tpm_data.columns
                      if col not in exclude_cols and col.startswith('ENCFF')]

        for col in tissue_cols:
            tissue_name = extract_tissue_name_from_column(col)
            tissues.add(tissue_name)

        return sorted(list(tissues))

    except Exception as e:
        print(f"Error getting unique tissues: {e}")
        traceback.print_exc()
        return []


def get_unique_organs_for_gene(db_path: str, gene_name: str, species: str = 'Human') -> list:
    """Get list of unique organs for a given gene"""
    try:
        tpm_data = load_expression_data(db_path, gene_name, data_type='tpm', species=species)

        if tpm_data.empty:
            return []

        tissue_to_organ = get_tissue_to_organ_mapping(species)

        organs = set()
        exclude_cols = {'id', 'transcript', 'trans_id', 'gene', 'gene_name', 'isoform_average_tpm',
                       'index', 'tpm average', 'tpm sum'}
        tissue_cols = [col for col in tpm_data.columns
                      if col not in exclude_cols and col.startswith('ENCFF')]

        for col in tissue_cols:
            tissue_name = extract_tissue_name_from_column(col)
            organ = tissue_to_organ.get(tissue_name, 'unknown')
            if organ != 'unknown':
                organs.add(organ)

        return sorted(list(organs))

    except Exception as e:
        print(f"Error getting unique organs: {e}")
        traceback.print_exc()
        return []


def create_empty_isoform_message(message: str) -> go.Figure:
    """Create empty figure with message for isoform plots"""
    fig = go.Figure()
    
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='white',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

##############################################################################
# Methods for isoform hashing 
##############################################################################

def convert_psl_ids(psl_file, out_file):
    """
    Convert read names in a PSL file to stable hash-based IDs 
    based on splice junction starts and ends.

    Parameters:
        psl_file (str): Input PSL file
        out_file (str): Output PSL file with updated IDs
    """
    with open(psl_file, "r") as infile, open(out_file, "w") as outfile:
        for line in infile:
            read = line.rstrip("\n").split("\t")
            name = read[9]

            # parse blocksizes, tstarts
            blocksizes = [int(x) for x in read[18].rstrip(",").split(",")]
            tstarts = [int(x) for x in read[20].rstrip(",").split(",")]

            # compute tends
            tends = [s + l for s, l in zip(tstarts, blocksizes)]

            # collapse junctions
            collapse_tstarts = tstarts[1:]
            collapse_tends = tends[:-1]

            # hash start junctions
            s = ",".join(map(str, collapse_tstarts))
            s_hashed = hashlib.shake_256(s.encode("utf-8")).hexdigest(8)
            s_id = "s" + s_hashed

            # hash end junctions
            e = ",".join(map(str, collapse_tends))
            e_hashed = hashlib.shake_256(e.encode("utf-8")).hexdigest(8)
            e_id = "e" + e_hashed

            # new read ID = startID:endID
            new_id = f"{s_id}:{e_id}"
            read[9] = new_id

            outfile.write("\t".join(read) + "\n")


def calculate_single_isoform_hash(tstarts: list, blocksizes: list) -> str:
    """
    Calculate hash ID for a single isoform based on its splice junction coordinates.

    Parameters:
        tstarts (list): List of target start positions for each exon
        blocksizes (list): List of block sizes for each exon

    Returns:
        str: Hash ID in format "s{start_hash}:e{end_hash}"

    Example:
        >>> tstarts = [1000, 2000, 3000]
        >>> blocksizes = [200, 300, 400]
        >>> calculate_single_isoform_hash(tstarts, blocksizes)
        's12345678:e87654321'
    """
    if not tstarts or not blocksizes:
        raise ValueError("tstarts and blocksizes cannot be empty")

    if len(tstarts) != len(blocksizes):
        raise ValueError("tstarts and blocksizes must have the same length")

    # Convert to int if str
    tstarts = [int(x) for x in tstarts]
    blocksizes = [int(x) for x in blocksizes]

    # Compute exon ends
    tends = [s + l for s, l in zip(tstarts, blocksizes)]

    # Extract junction coordinates (skip first start and last end)
    collapse_tstarts = tstarts[1:]    
    collapse_tends = tends[:-1]       

    # Hash start junctions
    s = ",".join(map(str, collapse_tstarts))
    s_hashed = hashlib.shake_256(s.encode("utf-8")).hexdigest(8)
    s_id = "s" + s_hashed

    # Hash end junctions
    e = ",".join(map(str, collapse_tends))
    e_hashed = hashlib.shake_256(e.encode("utf-8")).hexdigest(8)
    e_id = "e" + e_hashed

    # Return combined hash ID
    return f"{s_id}:{e_id}"


def parse_gtf_and_calculate_hashes(gtf_content: str) -> dict:
    """
    Parse GTF content and calculate hash IDs for all isoforms.
    Uses sequential parsing logic, i.e. assumes that exons contained
    within a transcript follow that transcript feature line.

    Transcripts with identical splice junctions (same internal structure) 
    are collapsed and share the same hash_id, but may have different TSS and TES.

    Parameters:
        gtf_content (str): Content of GTF file as string

    Returns:
        dict: {
            'results': List of dicts with gene_id, transcript_id, hash_id, exon_count,
            'merged_transcripts': Dict mapping hash_id to list of transcripts that were merged,
            'has_merges': Boolean indicating if any merges occurred
        }
    """
    results = []
    transcripts = []
    current_transcript = None

    ###################################################################################
    # First pass: sequentially parse and group exons with their preceding transcript
    ###################################################################################
    for line in gtf_content.strip().split('\n'):
        if line.startswith('#') or not line.strip():
            continue

        fields = line.split('\t')
        if len(fields) < 9:
            continue

        feature_type = fields[2]
        if feature_type not in ['transcript', 'exon']:
            continue

        start = int(fields[3])
        end = int(fields[4])
        attributes = fields[8]
        gene_id = None
        transcript_id = None
        for attr in attributes.split(';'):
            attr = attr.strip()
            if not attr:
                continue
            try:
                if attr.startswith('gene_id'):
                    gene_id = extract_gtf_attr_val(attr)
                elif attr.startswith('transcript_id'):
                    transcript_id = extract_gtf_attr_val(attr)
            except (ValueError, IndexError):
                continue

        if not gene_id:
            continue

        if feature_type == 'transcript':
            current_transcript = {
                'gene_id': gene_id,
                'transcript_id': transcript_id,
                'transcript_start': start,
                'transcript_end': end,
                'exons': [],
                'transcript_index': len(transcripts)
            }
            transcripts.append(current_transcript)

        # For exon features, add to current transcript IF same gene
        elif feature_type == 'exon' and current_transcript and current_transcript['gene_id'] == gene_id:
            current_transcript['exons'].append((start, end))

    ###########################################################################
    # Second pass: validate transcript coordinates and calculate hash IDs
    ###########################################################################
    # Map hash_id to list of transcripts with that hash
    hash_to_transcripts = {}

    for i, transcript_data in enumerate(transcripts):
        exons = sorted(transcript_data['exons'])

        if len(exons) >= 2:
            first_exon_start = exons[0][0]
            last_exon_end = exons[-1][1]

            if (transcript_data['transcript_start'] == first_exon_start and
                transcript_data['transcript_end'] == last_exon_end):

                # Fix for congruence with application data: GTF is 1-indexed, PSL is 0-indexed. 
                # We now subtract 1 from the start coords to ensure we match PSL-based coordinates.
                tstarts = [exon[0] - 1 for exon in exons]
                blocksizes = [exon[1] - (exon[0] - 1) for exon in exons]

                try:
                    hash_id = calculate_single_isoform_hash(tstarts, blocksizes)

                    result_entry = {
                        'transcript_id': transcript_data['transcript_id'],
                        'gene_id': transcript_data['gene_id'],
                        'hash_id': hash_id,
                        'exon_count': len(exons)
                    }
                    results.append(result_entry)

                    # Track which transcripts share the same hash
                    if hash_id not in hash_to_transcripts:
                        hash_to_transcripts[hash_id] = []
                    hash_to_transcripts[hash_id].append(transcript_data['transcript_id'])

                except Exception as e:
                    print(f"Error calculating hash for transcript {i}: {e}")
                    continue

    # Identify merged transcripts (same hash, different TSS/TES)
    merged_transcripts = {}
    for hash_id, transcript_ids in hash_to_transcripts.items():
        if len(transcript_ids) > 1:
            merged_transcripts[hash_id] = transcript_ids

    has_merges = len(merged_transcripts) > 0

    return {
        'results': results,
        'merged_transcripts': merged_transcripts,
        'has_merges': has_merges
    }


def generate_annotated_gtf(gtf_content: str) -> str:
    """
    Generate annotated GTF with transcript_id (hash IDs) and exon_number added to attributes.

    Assummptions: 
    - Exons following a transcript line belong to that transcript, until the next transcript feature is processed.
    - All transcript coordinates must align with first and last exon coordinates.

    Parameters:
        gtf_content (str): Original GTF content as string

    Returns:
        str: Annotated GTF content with transcript_id and exon_number attributes
    """
    lines = gtf_content.strip().split('\n')
    annotated_lines = []
    transcripts = []
    current_transcript = None

    ###################################################################################
    # First pass: parse sequentially and group exons with their preceding transcript
    ###################################################################################
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue

        fields = line.split('\t')
        if len(fields) < 9:
            continue

        feature_type = fields[2]
        if feature_type not in ['transcript', 'exon']:
            continue

        start = int(fields[3])
        end = int(fields[4])
        attributes = fields[8]

        gene_id = None
        for attr in attributes.split(';'):
            attr = attr.strip()
            if attr.startswith('gene_id'):
                gene_id = attr.split('"')[1]
                break

        if not gene_id:
            continue

        if feature_type == 'transcript':
            current_transcript = {
                'gene_id': gene_id,
                'transcript_start': start,
                'transcript_end': end,
                'exons': [],
                'hash_id': None
            }
            transcripts.append(current_transcript)

        elif feature_type == 'exon' and current_transcript and current_transcript['gene_id'] == gene_id:
            current_transcript['exons'].append({
                'start': start,
                'end': end
            })

    ############################################################################
    # Second pass: validate transcript coordinates and calculate hash IDs
    ############################################################################
    for transcript_data in transcripts:
        exons = sorted(transcript_data['exons'], key=lambda x: x['start'])

        if len(exons) >= 2:
            first_exon_start = exons[0]['start']
            last_exon_end = exons[-1]['end']

            if (transcript_data['transcript_start'] == first_exon_start and
                transcript_data['transcript_end'] == last_exon_end):

                # GTF is 1-indexed, PSL is 0-indexed. Convert to PSL coordinates.
                # Subtract 1 from both start and end, then calculate blocksize
                tstarts = [exon['start'] - 1 for exon in exons]
                psl_ends = [exon['end'] for exon in exons]
                blocksizes = [end - start for start, end in zip(tstarts, psl_ends)]
    
                try:
                    hash_id = calculate_single_isoform_hash(tstarts, blocksizes)
                    transcript_data['hash_id'] = hash_id
                except Exception as e:
                    print(f"Error calculating hash for transcript: {e}")
                    continue

    ############################################################################
    # Third pass: generate annotated output
    ############################################################################
    current_transcript_index = -1

    for line in lines:
        if line.startswith('#') or not line.strip():
            annotated_lines.append(line)
            continue

        fields = line.split('\t')
        if len(fields) < 9:
            annotated_lines.append(line)
            continue

        feature_type = fields[2]
        if feature_type not in ['transcript', 'exon']:
            annotated_lines.append(line)
            continue

        start = int(fields[3])
        end = int(fields[4])
        attributes = fields[8].rstrip()

        gene_id = None
        for attr in attributes.split(';'):
            attr = attr.strip()
            if attr.startswith('gene_id'):
                gene_id = attr.split('"')[1]
                break

        if not gene_id:
            annotated_lines.append(line)
            continue

        if feature_type == 'transcript':
            current_transcript_index += 1

        if (current_transcript_index >= 0 and
            current_transcript_index < len(transcripts) and
            transcripts[current_transcript_index]['hash_id'] and
            transcripts[current_transcript_index]['gene_id'] == gene_id):

            transcript_data = transcripts[current_transcript_index]
            hash_id = transcript_data['hash_id']

            if not attributes.endswith(';'):
                attributes += ';'
            attributes += f' transcript_id "{hash_id}";'

            # Add exon_number for exon features
            if feature_type == 'exon':
                exons = sorted(transcript_data['exons'], key=lambda x: x['start'])
                exon_number = None
                for i, exon in enumerate(exons):
                    if exon['start'] == start and exon['end'] == end:
                        exon_number = i + 1
                        break

                if exon_number:
                    attributes += f' exon_number "{exon_number}";'

            # Reconstruct full line for gtf final output
            fields[8] = attributes
            annotated_lines.append('\t'.join(fields))
        else:
            annotated_lines.append(line)

    return '\n'.join(annotated_lines)


def test_psl_hash_algorithm():
    """
    Test case to match actual PSL hash format.
    
    example PSL data: 
    - blocksizes: 2445,1632  
    - tstarts: 58351029,58353713

    Expected hash: s05ff6e0b43331bfc:e2111d58dcc6fc4ae
    """
    tstarts = [58351029, 58353713]
    blocksizes = [2445, 1632]
    expected_hash = "s05ff6e0b43331bfc:e2111d58dcc6fc4ae"

    tends = [s + l for s, l in zip(tstarts, blocksizes)]  # [58353474, 58355345]

    # Extract junction coordinates (skip first start and last end)
    collapse_tstarts = tstarts[1:]    # [58353713] - start of second exon
    collapse_tends = tends[:-1]       # [58353474] - end of first exon

    print(f"Original tstarts: {tstarts}")
    print(f"Original blocksizes: {blocksizes}")
    print(f"Calculated tends: {tends}")
    print(f"Junction starts (collapse_tstarts): {collapse_tstarts}")
    print(f"Junction ends (collapse_tends): {collapse_tends}")
    print(f"Expected hash: {expected_hash}")

    try:
        current_hash = calculate_single_isoform_hash(tstarts, blocksizes)
        print(f"Current algorithm result: {current_hash}")

        matches = current_hash == expected_hash
        print(f"Matches expected: {matches}")

        if matches:
            print("SUCCESS: Hash algorithm is correct!")
        else:
            print("ERROR: Hash algorithm needs adjustment!")

    except Exception as e:
        print(f"Current algorithm error: {e}")

    print(f"\nJunction coordinate strings:")
    
    s_string = ",".join(map(str, collapse_tstarts))
    e_string = ",".join(map(str, collapse_tends))

    print(f"Start string: '{s_string}'")
    print(f"End string: '{e_string}'")


# Example usage for hashing function:
# convert_psl_ids("input.psl", "output_with_hash_ids.psl")
# test_psl_hash_algorithm()  # Run this to test hash algorithm