import os
import math
import traceback
from tqdm import tqdm
import sqlite3
import scipy
import numpy as np
import pandas as pd
import base64
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import dash
from dash import html, dcc, dash_table, callback_context, no_update
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.graph_objs as go
import plotly.io as pio
import kaleido
from dash.exceptions import PreventUpdate
from colorama import Fore, Style, init
import traceback
import logging
logging.getLogger('dash.dash').setLevel(logging.WARNING)
from src.isoformgazer.export_client import get_export_client
from src.isoformgazer.db_config import initialize_database, get_db_config
from src.isoformgazer.gene_cache_redis import get_cached_gene_list
from src.isoformgazer.data_utils import (
    get_master_table_columns, parse_filter_query, query_master_table, get_gene_options,
    get_all_gene_options, create_custom_spinner, validate_filter_input,
    is_cache_valid, load_default_gene_cache, save_default_gene_cache,
    generate_default_gene_cache, get_default_gene_cache_path, extract_gtf_attr_val,
    get_table_prefix
)
from src.isoformgazer.junction_utils import (
    create_summary_clustergram, create_gene_clustergram,
    load_atse_data, process_gene_atse_data, create_empty_atse_message,
    create_junction_exon_visualization, create_empty_clustergram_message,
    filter_junctions_by_transcripts, filter_transcripts_by_junctions,
    get_gene_id_from_atse
)
from src.isoformgazer.isoform_utils import (
    load_expression_data, process_transcript_structure, create_transcript_structure_plot,
    create_isoform_expression_clustergram, create_empty_isoform_message,
    calculate_unified_plot_height, calculate_clustergram_min_height, calculate_single_isoform_hash,
    calculate_dynamic_structure_plot_height, parse_gtf_and_calculate_hashes, generate_annotated_gtf,
    get_unique_tissues_for_gene, get_unique_organs_for_gene, get_gene_id_for_gene_name
)

RANDOM_SEED = 18
np.random.seed(RANDOM_SEED)

###################################################################
# POSTGRESQL DATABASE SETUP
###################################################################
# All environment variables are loaded from .env file
from dotenv import load_dotenv
load_dotenv()

###################################################################
# SQLLITE DATABASE SETUP
###################################################################
def check_database_status():
    """
    Checks and reports database status once at app startup. 
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    db_path = os.path.join(data_dir, "isoformgazer.db")
    
    if Path(db_path).exists():
        print(f"Found existing master table database at {db_path} to use.")
        print()
        return True
    
    return False


def setup_local_database(data_dir=None, force_rebuild=False, include_mouse=True):
    """
    Sets up SQLite database from data files for human and optionally mouse data.

    Args:
        data_dir: Optional path to data directory. Defaults to src/isoformgazer/data
        force_rebuild: If True, rebuild the database even if it exists
        include_mouse: If True, load mouse data; if False, only load human data (default: True)
    """
    if data_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")

    os.makedirs(data_dir, exist_ok=True)

    db_path = os.path.join(data_dir, "isoformgazer.db")
    if Path(db_path).exists() and not force_rebuild:
        return db_path

    print()
    print(f"Creating new database at {db_path}.")
    if not include_mouse:
        print("NOTE: Mouse data will be excluded from this build.")
    print()

    if Path(db_path).exists():
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = OFF")  
    conn.execute("PRAGMA synchronous = OFF")  
    conn.execute("PRAGMA cache_size = -200000")  # 200MB cache (negative = KB)
    conn.execute("PRAGMA temp_store = MEMORY") 
    conn.execute("PRAGMA locking_mode = EXCLUSIVE") 
    conn.execute("PRAGMA page_size = 4096")  # Smaller page size for less waste

    human_data_dir = os.path.join(data_dir, "human")
    load_species_data(conn, human_data_dir, table_prefix="", species_name="Human")

    if include_mouse:
        mouse_data_dir = os.path.join(data_dir, "mouse")
        load_species_data(conn, mouse_data_dir, table_prefix="mouse_", species_name="Mouse")
    else:
        print("\n" + "="*80)
        print("Skipping mouse data (include_mouse=False)")
        print("="*80 + "\n")

    ######################################################################
    # Load human-mouse high-confidence conserved junctions mapping table
    ######################################################################
    if include_mouse:
        print(f"\n================================================================================")
        print(f"Loading human-mouse conserved junctions mapping")
        print(f"================================================================================\n")
        conserved_junctions_file = os.path.join(mouse_data_dir, "junction_mapping_mouse_human_with_annotations.csv")

        with tqdm(desc="Loading conserved junctions mapping data", unit=" rows") as pbar:
            df_conserved = pd.read_csv(conserved_junctions_file)
            pbar.update(len(df_conserved))

        with tqdm(desc="Writing conserved junctions mapping to local database", unit="rows", total=len(df_conserved)) as pbar:
            df_conserved.to_sql('human_mouse_conserved_junctions', conn, if_exists='replace', index=False)
            pbar.update(len(df_conserved))

        print(f" Processed all {len(df_conserved):,} rows from conserved junctions mapping!")
        print()
    else:
        print("\nSkipping human-mouse conserved junctions mapping (mouse data not included)")
        print()

    print("\nFinalizing database...")
    conn.commit()
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.commit()

    print("Optimizing database. This may take a few minutes.")
    conn.execute("VACUUM")

    conn.close()

    print("Database setup complete!")
    print()

    return db_path


def load_species_data(conn, species_data_dir, table_prefix="", species_name="Human"):
    """
    Load data files for a specific species into the database.

    Args:
        conn: SQLite connection object
        species_data_dir: Path to species-specific data directory
        table_prefix: Prefix to add to table names (e.g., "mouse_" for mouse tables)
        species_name: Name of species for logging purposes
    """
    print(f"\n{'='*80}")
    print(f"Loading {species_name} data from {species_data_dir}")
    print(f"{'='*80}\n")

    ########################################################
    # Load isoform master table data
    ########################################################
    # Determine the correct isoform file based on species
    if species_name == "Human":
        isoform_file = os.path.join(species_data_dir, "mt_isoform_gazers_250828.tsv")
    elif species_name == "Mouse":
        isoform_file = os.path.join(species_data_dir, "mt_isoform_gazers_mouse_250929.tsv")

    with tqdm(desc=f"Loading {species_name} isoform master table data", unit=" rows") as pbar:
        df_isoform = pd.read_csv(isoform_file, sep='\t')
        df_isoform['id'] = df_isoform.index # IMPORTANT: we need this to match PSL file 1-based indexing!
        df_isoform['gene_id'] = df_isoform['gene'].str.split('.').str[0]

        if 'gene_name' in df_isoform.columns:
            df_isoform['gene_name'] = df_isoform['gene_name'].astype(str).str.upper()

        columns_to_drop = ['gene_total_tpm', 'gene_gencode_v46_basic_transcript_counts']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df_isoform.columns]
        if existing_columns_to_drop:
            df_isoform = df_isoform.drop(columns=existing_columns_to_drop)

        pbar.update(len(df_isoform))

    isoforms_table = f"{table_prefix}isoforms"
    with tqdm(desc=f"Writing {species_name} isoform master table data to local database", unit="rows", total=len(df_isoform)) as pbar:
        df_isoform.to_sql(isoforms_table, conn, if_exists='replace', index=False)
        pbar.update(len(df_isoform))

    print(f" Processed all {len(df_isoform):,} rows from {species_name} isoform master table!")
    print()

    ########################################################
    # Load isoform PSL data
    ########################################################
    print(f"Processing {species_name} PSL data...")
    if species_name == "Human":
        psl_file = os.path.join(species_data_dir, "all_samples_sp_collapse_all_chr_no_treatment_hashid_isoform_full.psl")
    elif species_name == "Mouse":  
        psl_file = os.path.join(species_data_dir, "all_samples_sp_collapse_all_chr_full.psl")

    psl_columns = [
        'matches', 'misMatches', 'repMatches', 'nCount', 'qNumInsert', 'qBaseInsert',
        'tNumInsert', 'tBaseInsert', 'strand', 'qName', 'qSize', 'qStart', 'qEnd',
        'tName', 'tSize', 'tStart', 'tEnd', 'blockCount', 'blockSizes', 'qStarts', 'tStarts'
    ]

    # Get total PSL rows for progress tracking
    total_psl_rows = sum(1 for _ in open(psl_file, 'r'))
    chunk_size = 100000

    psl_table = f"{table_prefix}psl_data"
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {psl_table} (
            id INTEGER PRIMARY KEY,
            matches INTEGER,
            misMatches INTEGER,
            repMatches INTEGER,
            nCount INTEGER,
            qNumInsert INTEGER,
            qBaseInsert INTEGER,
            tNumInsert INTEGER,
            tBaseInsert INTEGER,
            strand TEXT,
            qName TEXT,
            qSize INTEGER,
            qStart INTEGER,
            qEnd INTEGER,
            tName TEXT,
            tSize INTEGER,
            tStart INTEGER,
            tEnd INTEGER,
            blockCount INTEGER,
            blockSizes TEXT,
            qStarts TEXT,
            tStarts TEXT,
            gene_id TEXT,
            trans_id TEXT,
            transcript_length INTEGER
        )
    """)

    with tqdm(total=total_psl_rows, desc="Loading PSL data", unit="rows") as pbar:
        psl_chunks = pd.read_csv(psl_file, sep='\t', names=psl_columns, chunksize=chunk_size)
        
        for chunk_idx, chunk in enumerate(psl_chunks):
            start_idx = (chunk_idx * chunk_size) + 1
            chunk['id'] = range(start_idx, start_idx + len(chunk))
            
            # Extract transcript/gene IDs from qName
            split_qname = chunk['qName'].str.split('_', n=1, expand=True)
            if split_qname.shape[1] >= 2:
                # Format: transcript_gene (e.g. s-6373022524934343100:e2226920842648636318_ENSG00000223972.5)
                chunk['trans_id'] = split_qname[0]  # Keep full transcript name
                chunk['gene_id'] = split_qname[1].str.split('.').str[0]
            else:
                split_result = chunk['qName'].str.split(':', n=1, expand=True)
                if split_result.shape[1] >= 2:
                    chunk['trans_id'] = split_result[0]
                    chunk['gene_id'] = split_result[1].str.split('.').str[0]
                else:
                    # fallback: use entire qName as transcript
                    chunk['trans_id'] = chunk['qName']
                    chunk['gene_id'] = 'unknown'
            chunk['transcript_length'] = chunk['tEnd'] - chunk['tStart']
            
            if 'index' in chunk.columns:
                chunk.drop(columns=['index'], inplace=True)

            chunk.to_sql(psl_table, conn, if_exists='append', index=False)
            pbar.update(len(chunk))

    print(f"Processing {species_name} isoform TPM data...")
    if species_name == "Human":
        tpm_file = os.path.join(species_data_dir, "all_tpm_250828.tsv")
    elif species_name == "Mouse": 
        tpm_file = os.path.join(species_data_dir, "all_tpm_mouse.tsv")

    tpm_df = pd.read_csv(tpm_file, sep='\t')
    tpm_df['id'] = tpm_df.index

    if 'gene_name' in tpm_df.columns:
        tpm_df['gene_name'] = tpm_df['gene_name'].astype(str).str.upper()
    if 'gene' in tpm_df.columns:
        tpm_df['gene'] = tpm_df['gene'].astype(str).str.upper()

    tpm_table = f"{table_prefix}tpm_data"
    tpm_df.to_sql(tpm_table, conn, if_exists='replace')

    print(f"Processing {species_name} isoform ratio data...")
    if species_name == "Human":
        ratio_file = os.path.join(species_data_dir, "all_quant_ratio_250828.tsv")
    elif species_name == "Mouse":
        ratio_file = os.path.join(species_data_dir, "all_quant_ratio_mouse.tsv")

    ratio_df = pd.read_csv(ratio_file, sep='\t')
    ratio_df['id'] = ratio_df.index

    if 'gene_name' in ratio_df.columns:
        ratio_df['gene_name'] = ratio_df['gene_name'].astype(str).str.upper()
    if 'gene' in ratio_df.columns:
        ratio_df['gene'] = ratio_df['gene'].astype(str).str.upper()

    ratio_table = f"{table_prefix}ratio_data"
    ratio_df.to_sql(ratio_table, conn, if_exists='replace')

    print(f"Processing {species_name} isoform log TPM data...")
    if species_name == "Human":
        log_tpm_file = os.path.join(species_data_dir, "all_tpm_log10_250828.tsv")
    elif species_name == "Mouse":
        log_tpm_file = os.path.join(species_data_dir, "all_tpm_log10_mouse.tsv")

    log_tpm_df = pd.read_csv(log_tpm_file, sep='\t')
    log_tpm_df['id'] = log_tpm_df.index
    
    if 'gene_name' in log_tpm_df.columns:
        log_tpm_df['gene_name'] = log_tpm_df['gene_name'].astype(str).str.upper()
    if 'gene' in log_tpm_df.columns:
        log_tpm_df['gene'] = log_tpm_df['gene'].astype(str).str.upper()

    log_tpm_table = f"{table_prefix}log_tpm_data"
    log_tpm_df.to_sql(log_tpm_table, conn, if_exists='replace')

    print(f"Updating {species_name} isoform transcript names with full names from PSL data...")
    conn.execute(f"""
        UPDATE {isoforms_table}
        SET transcript = (
            SELECT psl.trans_id
            FROM {psl_table} psl
            WHERE psl.id = {isoforms_table}.id
        )
        WHERE EXISTS (
            SELECT 1 FROM {psl_table} psl WHERE psl.id = {isoforms_table}.id
        )
    """)
    conn.commit()

    print(f"Creating {species_name} isoform indexes...")
    conn.execute(f"CREATE INDEX {table_prefix}idx_psl_gene ON {psl_table}(id)")
    conn.execute(f"CREATE INDEX {table_prefix}idx_iso_gene ON {isoforms_table}(gene_name, id)")
    conn.commit()

    ########################################################
    # Append GENCODE-only transcripts (not detected in expression data)
    ########################################################
    if species_name == "Human":
        gencode_csv = os.path.join(species_data_dir, "expected_transcripts_human_v46_gencode.csv")
        gencode_psl = os.path.join(species_data_dir, "expected_transcripts_human_v46_gencode_psl.tsv")
    elif species_name == "Mouse":
        gencode_csv = os.path.join(species_data_dir, "expected_transcripts_mouse_m25_gencode.csv")
        gencode_psl = os.path.join(species_data_dir, "expected_transcripts_mouse_m25_gencode_psl.tsv")
    else:
        gencode_csv = None
        gencode_psl = None

    if gencode_csv and gencode_psl and os.path.exists(gencode_csv) and os.path.exists(gencode_psl):
        print(f"\nAppending {species_name} GENCODE-only transcripts (not in expression data)...")
        with tqdm(desc=f"Loading {species_name} GENCODE transcripts", unit=" rows") as pbar:
            df_gencode_isoforms = pd.read_csv(gencode_csv)
            pbar.update(len(df_gencode_isoforms))

        # Get list of genes that already exist in the expression data (isoforms table): 
        # we only add reference non-detected transcripts for these currently
        existing_genes_query = f"SELECT DISTINCT gene_name FROM {isoforms_table}"
        existing_genes_df = pd.read_sql_query(existing_genes_query, conn)
        existing_genes = set(existing_genes_df['gene_name'].str.upper())

        print(f"  Found {len(existing_genes):,} genes in expression data")
        # Ensure gene_name is uppercase for matching
        if 'gene_name' in df_gencode_isoforms.columns:
            df_gencode_isoforms['gene_name'] = df_gencode_isoforms['gene_name'].astype(str).str.upper()
        original_count = len(df_gencode_isoforms)
        df_gencode_isoforms = df_gencode_isoforms[df_gencode_isoforms['gene_name'].isin(existing_genes)]
        filtered_count = len(df_gencode_isoforms)
        print(f"  Filtered GENCODE transcripts: {original_count:,} -> {filtered_count:,} (kept only genes with expression data)")

        # Get current max ID from isoforms table to continue numbering, assign sequential IDs starting after the last existing ID
        max_id_result = conn.execute(f"SELECT MAX(id) FROM {isoforms_table}").fetchone()
        max_id = max_id_result[0] if max_id_result[0] is not None else 0
        df_gencode_isoforms['id'] = range(max_id + 1, max_id + 1 + len(df_gencode_isoforms))

        # Ensure gene_id column exists (no version number)
        if 'gene_id' in df_gencode_isoforms.columns:
            df_gencode_isoforms['gene_id'] = df_gencode_isoforms['gene_id'].str.split('.').str[0]

        cursor = conn.execute(f"PRAGMA table_info({isoforms_table})")
        table_columns = {row[1] for row in cursor.fetchall()}  # row[1] is column name
        gencode_columns = set(df_gencode_isoforms.columns)
        columns_to_drop = gencode_columns - table_columns
        if columns_to_drop:
            df_gencode_isoforms = df_gencode_isoforms.drop(columns=list(columns_to_drop))
            print(f"  Dropped {len(columns_to_drop)} incompatible columns: {', '.join(sorted(columns_to_drop))}")

        # Verify all required table columns are present (fill with NaN if missing)
        for col in table_columns:
            if col not in df_gencode_isoforms.columns:
                df_gencode_isoforms[col] = None

        # Append to isoforms table
        with tqdm(desc=f"Writing {species_name} GENCODE transcripts to database", unit="rows", total=len(df_gencode_isoforms)) as pbar:
            df_gencode_isoforms.to_sql(isoforms_table, conn, if_exists='append', index=False)
            pbar.update(len(df_gencode_isoforms))

        print(f" Appended {len(df_gencode_isoforms):,} GENCODE-only transcripts to isoform table!")
        print(f"Loading {species_name} GENCODE PSL structures...")
        with tqdm(desc=f"Loading {species_name} GENCODE PSL data", unit=" rows") as pbar:
            df_gencode_psl = pd.read_csv(gencode_psl, sep='\t')
            pbar.update(len(df_gencode_psl))

        # Assign same IDs as the isoform table (matching by transcript ID), create mapping from transcript ID to the assigned database ID
        transcript_id_map = dict(zip(df_gencode_isoforms['transcript'], df_gencode_isoforms['id']))
        df_gencode_psl['id'] = df_gencode_psl['trans_id'].map(transcript_id_map)

        # Filter out PSL rows that don't have a matching transcript ID (gene was filtered out)
        psl_original_count = len(df_gencode_psl)
        df_gencode_psl = df_gencode_psl[df_gencode_psl['id'].notna()]
        psl_filtered_count = len(df_gencode_psl)
        print(f"  Filtered GENCODE PSL data: {psl_original_count:,} -> {psl_filtered_count:,} (kept only transcripts with expression data genes)")

        df_gencode_psl['gene_id'] = df_gencode_psl['gene_id'].str.split('.').str[0]

        # Add transcript_length column (sum of blockSizes)
        def calculate_transcript_length(block_sizes_str):
            if pd.isna(block_sizes_str):
                return 0
            sizes = [int(s) for s in str(block_sizes_str).rstrip(',').split(',') if s]
            return sum(sizes)

        df_gencode_psl['transcript_length'] = df_gencode_psl['blockSizes'].apply(calculate_transcript_length)

        # Convert relative tStarts to absolute genomic coordinates
        def convert_relative_to_absolute_starts(row):
            """Convert relative tStarts (0-based from transcript start) to absolute genomic coordinates"""
            if pd.isna(row['tStarts']) or pd.isna(row['tStart']):
                return row['tStarts']
            relative_starts = row['tStarts'].rstrip(',').split(',')
            absolute_starts = [str(int(start) + int(row['tStart'])) for start in relative_starts if start]
            return ', '.join(absolute_starts) + ','

        df_gencode_psl['tStarts_absolute'] = df_gencode_psl.apply(convert_relative_to_absolute_starts, axis=1)

        df_psl_to_append = pd.DataFrame()
        df_psl_to_append['id'] = df_gencode_psl['id']
        df_psl_to_append['qName'] = df_gencode_psl['trans_id']
        df_psl_to_append['tName'] = df_gencode_psl['tName']
        df_psl_to_append['strand'] = df_gencode_psl['strand']
        df_psl_to_append['tStart'] = df_gencode_psl['tStart']
        df_psl_to_append['tEnd'] = df_gencode_psl['tEnd']
        df_psl_to_append['qSize'] = df_gencode_psl['transcript_length']
        df_psl_to_append['blockCount'] = df_gencode_psl['blockCount']
        df_psl_to_append['blockSizes'] = df_gencode_psl['blockSizes']
        df_psl_to_append['tStarts'] = df_gencode_psl['tStarts_absolute']
        df_psl_to_append['gene_id'] = df_gencode_psl['gene_id']
        df_psl_to_append['trans_id'] = df_gencode_psl['trans_id']
        df_psl_to_append['transcript_length'] = df_gencode_psl['transcript_length']
        df_psl_to_append['matches'] = 0
        df_psl_to_append['misMatches'] = 0
        df_psl_to_append['repMatches'] = 0
        df_psl_to_append['nCount'] = 0
        df_psl_to_append['qNumInsert'] = 0
        df_psl_to_append['qBaseInsert'] = 0
        df_psl_to_append['tNumInsert'] = df_gencode_psl['blockCount'] - 1  # Number of introns
        df_psl_to_append['tBaseInsert'] = 0
        df_psl_to_append['qStart'] = 0
        df_psl_to_append['qEnd'] = df_gencode_psl['transcript_length']
        df_psl_to_append['tSize'] = 0
        df_psl_to_append['qStarts'] = '0,'

        with tqdm(desc=f"Writing {species_name} GENCODE PSL data to database", unit="rows", total=len(df_psl_to_append)) as pbar:
            df_psl_to_append.to_sql(psl_table, conn, if_exists='append', index=False)
            pbar.update(len(df_psl_to_append))

        print(f" Appended {len(df_psl_to_append):,} GENCODE PSL structures to PSL table!")
        print()

    else:
        if gencode_csv and not os.path.exists(gencode_csv):
            print(f"\nNote: GENCODE transcript file not found at {gencode_csv}")
            print(f"Skipping GENCODE-only transcript append for {species_name}")
        if gencode_psl and not os.path.exists(gencode_psl):
            print(f"\nNote: GENCODE PSL file not found at {gencode_psl}")
            print(f"Skipping GENCODE PSL structure append for {species_name}")
        print()

    ########################################################
    # Load junction master table data
    ########################################################
    if species_name == "Human":
        junction_file = os.path.join(species_data_dir, "pseudobulk_final_tissue_celltype_aligned_20260105_170740_withmappings.csv")
    if species_name == "Mouse":
        junction_file = os.path.join(species_data_dir, "pseudobulk_final_tissue_celltype_20260105_171037_withmappings.csv")

    # Need to count total lines (minus header) to estimate progress
    with open(junction_file, 'r') as f:
        total_lines = sum(1 for _ in f) - 1

    chunk_size = 100000
    estimated_chunks = (total_lines // chunk_size) + 1

    print(f"Loading {total_lines:,} rows of {species_name} junction data in groupings of {chunk_size:,} rows...")

    # Create separate tables for junction master data and PSI data
    junctions_table = f"{table_prefix}junctions"
    junction_psis_table = f"{table_prefix}junction_psis"

    # Temporary table to hold all raw data
    temp_table = f"{table_prefix}junctions_temp"

    first_chunk = True
    row_count = 0

    master_column_order = [
        'gene_symbol',
        'gene_id',
        'event_id',
        'junction_id',
        'junction_id_index',
        'atse_count',
        'junction_count',
        'junction_average_psi',
        'matched_transcript_ids'
    ]

    psi_column_order = [
        'junction_id',
        'junction_id_index',
        'cell_type',
        'n_cells',
        'psi'
    ]

    with tqdm(desc=f"Writing {species_name} junction data to temporary table",
              unit="chunk",
              total=estimated_chunks) as chunk_pbar:

        for i, chunk in enumerate(pd.read_csv(junction_file,
                                              chunksize=chunk_size,
                                              low_memory=False)):
            if 'gene_symbol' in chunk.columns:
                chunk['gene_symbol'] = chunk['gene_symbol'].astype(str).str.upper()

            if first_chunk:
                chunk.to_sql(temp_table, conn, if_exists='replace', index=False)
                first_chunk = False
            else:
                chunk.to_sql(temp_table, conn, if_exists='append', index=False)

            # Commit every 10 chunks to free up memory and reduce temp filesize
            if i % 10 == 0 and i > 0:
                conn.commit()

            row_count += len(chunk)
            chunk_pbar.update(1)

            chunk_pbar.set_postfix({
                'rows': f"{row_count:,}",
                'chunk': f"{i+1}/{estimated_chunks}"
            })

    print(f" Loaded all {row_count:,} rows from {species_name} junction file into temporary table!")

    ########################################################
    # Append GENCODE junctions (non-expressed junctions for structure plot)
    ########################################################
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(species_data_dir))))
    gencode_dir = os.path.join(repo_root, "adding_GENCODE_junctions")
    if species_name == "Human":
        gencode_junction_file = os.path.join(gencode_dir, "expected_junctions_human_043026.csv")
    elif species_name == "Mouse":
        gencode_junction_file = os.path.join(gencode_dir, "expected_junctions_mouse_043026.csv")
    else:
        gencode_junction_file = None

    if gencode_junction_file and os.path.exists(gencode_junction_file):
        print(f"\nAppending {species_name} GENCODE junctions to temporary table...")

        # Count total GENCODE junction rows for progress tracking
        with open(gencode_junction_file, 'r') as f:
            gencode_total_lines = sum(1 for _ in f) - 1

        gencode_chunk_size = 100000
        gencode_estimated_chunks = (gencode_total_lines // gencode_chunk_size) + 1
        gencode_row_count = 0

        print(f"Loading {gencode_total_lines:,} GENCODE junction rows in chunks of {gencode_chunk_size:,}...")

        with tqdm(desc=f"Appending {species_name} GENCODE junctions",
                  unit="chunk",
                  total=gencode_estimated_chunks) as gencode_pbar:

            for i, gencode_chunk in enumerate(pd.read_csv(gencode_junction_file,
                                                          chunksize=gencode_chunk_size,
                                                          low_memory=False)):
                if 'gene_symbol' in gencode_chunk.columns:
                    gencode_chunk['gene_symbol'] = gencode_chunk['gene_symbol'].astype(str).str.upper()

                gencode_chunk.to_sql(temp_table, conn, if_exists='append', index=False)

                # Commit every 10 chunks to free up memory and reduce temp file size
                if i % 10 == 0:
                    conn.commit()

                gencode_row_count += len(gencode_chunk)
                gencode_pbar.update(1)

                gencode_pbar.set_postfix({
                    'rows': f"{gencode_row_count:,}",
                    'chunk': f"{i+1}/{gencode_estimated_chunks}"
                })

        row_count += gencode_row_count
        print(f" Appended all {gencode_row_count:,} GENCODE junction rows!")
        print(f" Total rows in temporary table: {row_count:,}")
        
    else:
        if gencode_junction_file:
            print(f"\nWarning: GENCODE junction file not found at {gencode_junction_file}")
            print(f"Skipping GENCODE junction append for {species_name}")

    # Rename gene_symbol to gene_name
    conn.execute(f"ALTER TABLE {temp_table} RENAME COLUMN gene_symbol TO gene_name")

    # Create junction master table (one row per junction)
    print(f"Creating {species_name} junction master table...")
    conn.execute(f"""
        CREATE TABLE {junctions_table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gene_name TEXT,
            gene_id TEXT,
            event_id TEXT,
            junction_id TEXT,
            junction_id_index INTEGER,
            junction_average_psi REAL,
            matched_transcript_ids TEXT
        )
    """)

    conn.execute(f"""
        INSERT INTO {junctions_table} (
            gene_name,
            gene_id,
            event_id,
            junction_id,
            junction_id_index,
            junction_average_psi,
            matched_transcript_ids
        )
        SELECT DISTINCT
            gene_name,
            gene_id,
            event_id,
            junction_id,
            junction_id_index,
            junction_average_psi,
            matched_transcript_ids
        FROM {temp_table}
    """)

    master_count = conn.execute(f"SELECT COUNT(*) FROM {junctions_table}").fetchone()[0]
    print(f" Created junction master table with {master_count:,} unique junctions")

    # Create junction PSI table (one row per junction-cell_type pair)
    print(f"Creating {species_name} junction PSI table...")
    conn.execute(f"""
        CREATE TABLE {junction_psis_table} AS
        SELECT
            junction_id,
            junction_id_index,
            cell_type,
            n_cells,
            psi,
            atse_count,
            junction_count
        FROM {temp_table}
        WHERE cell_type IS NOT NULL
    """)

    psi_count = conn.execute(f"SELECT COUNT(*) FROM {junction_psis_table}").fetchone()[0]
    print(f" Created junction PSI table with {psi_count:,} junction-cell_type pairs")

    # Drop temporary table
    conn.execute(f"DROP TABLE {temp_table}")

    print(f" Processed all data from {species_name} junction file!")
    print()

    ########################################################
    # Load ATSE data into database
    ########################################################
    if species_name == "Human":
        atse_file = os.path.join(species_data_dir, "TMS_atse_file_unanno_also_2025-05-11_06-23-05.tsv")
    elif species_name == "Mouse":
        atse_file = os.path.join(species_data_dir, "MOUSE_FOUNDATION_ATSE_FILE_unanno_also_2025-10-01_21-36-40.tsv")
    
    if os.path.exists(atse_file):
        print(f"Processing {species_name} ATSE data...")
        atse_table = f"{table_prefix}atse_data"
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {atse_table} (
                event_id TEXT,
                gene_id TEXT,
                gene_name TEXT,
                gene_types TEXT,
                num_junctions INTEGER,
                event_type TEXT,
                chromosome TEXT,
                event_strand TEXT,
                atse_start INTEGER,
                atse_end INTEGER,
                atse_length INTEGER,
                atse_number INTEGER,
                total_atses_in_gene INTEGER,
                distance_to_previous INTEGER,
                distance_to_next INTEGER,
                transcripts TEXT,
                both_ends_transcripts TEXT,
                only_5_prime_transcripts TEXT,
                only_3_prime_transcripts TEXT,
                transcript_types TEXT,
                annotation_status TEXT,
                perfect_match_5_prime TEXT,
                perfect_match_3_prime TEXT,
                junction_id TEXT,
                chrom TEXT,
                start INTEGER,
                end INTEGER,
                junction_strand TEXT,
                cells INTEGER,
                total_score REAL,
                five_prime_usage REAL,
                three_prime_usage REAL,
                donor_usage REAL,
                acceptor_usage REAL,
                donor_total_reads INTEGER,
                acceptor_total_reads INTEGER,
                splice_motif TEXT,
                donor_seq TEXT,
                acceptor_seq TEXT,
                position_off_5_prime INTEGER,
                position_off_3_prime INTEGER
            )
        """)

        chunk_size = 50000
        total_lines = sum(1 for _ in open(atse_file, 'r')) - 1 
        
        with tqdm(total=total_lines, desc="Loading ATSE data", unit="rows") as pbar:
            for chunk in pd.read_csv(atse_file, sep='\t', chunksize=chunk_size, low_memory=False):

                chunk = chunk.rename(columns={
                    'strand': 'event_strand',
                    'strand.1': 'junction_strand'
                })

                if 'gene_name' in chunk.columns:
                    chunk['gene_name'] = chunk['gene_name'].astype(str).str.upper()

                required_columns = [
                    'event_id', 'gene_id', 'gene_name', 'gene_types', 'num_junctions',
                    'event_type', 'chromosome', 'event_strand', 'atse_start', 'atse_end',
                    'atse_length', 'atse_number', 'total_atses_in_gene',
                    'distance_to_previous', 'distance_to_next', 'transcripts',
                    'both_ends_transcripts', 'only_5_prime_transcripts',
                    'only_3_prime_transcripts', 'transcript_types', 'annotation_status',
                    'perfect_match_5_prime', 'perfect_match_3_prime', 'junction_id',
                    'chrom', 'start', 'end', 'junction_strand', 'cells', 'total_score',
                    'five_prime_usage', 'three_prime_usage', 'donor_usage',
                    'acceptor_usage', 'donor_total_reads', 'acceptor_total_reads',
                    'splice_motif', 'donor_seq', 'acceptor_seq',
                    'position_off_5_prime', 'position_off_3_prime'
                ]
                
                for col in required_columns:
                    if col not in chunk.columns:
                        chunk[col] = np.nan

                chunk[required_columns].to_sql(atse_table, conn, if_exists='append', index=False)
                pbar.update(len(chunk))


        print(f" Processed {total_lines:,} {species_name} ATSE records!")

    ###############################################################
    # Load Gencode GTF data for transcript hash ID cross-checking
    ###############################################################
    if species_name == "Human":
        gencode_gtf_file = os.path.join(species_data_dir, "gencode.v46.basic.annotation.gtf")
    elif species_name == "Mouse":
        gencode_gtf_file = os.path.join(species_data_dir, "gencode.vM25.annotation.gtf")

    print(f"Processing {species_name} Gencode GTF file...")
    gencode_table = f"{table_prefix}gencode_gtf"
    if os.path.exists(gencode_gtf_file):
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {gencode_table} (
            id INTEGER PRIMARY KEY,
            gene_id TEXT,
            transcript_id TEXT,
            gene_name TEXT,
            transcript_type TEXT,
            exon_number INTEGER,
            exon_id TEXT,
            chromosome TEXT,
            strand TEXT,
            exon_start INTEGER,
            exon_end INTEGER,
            transcript_start INTEGER,
            transcript_end INTEGER
        )
    """)

    gencode_records = []
    with open(gencode_gtf_file, 'r') as f:
        total_gtf_lines = sum(1 for _ in f)

    with tqdm(total=total_gtf_lines, desc="Loading Gencode GTF data", unit="lines") as pbar:
        with open(gencode_gtf_file, 'r') as f:
            for line in f:
                pbar.update(1)
                if line.startswith('#') or not line.strip():
                    continue

                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue

                feature_type = fields[2]
                if feature_type not in ['transcript', 'exon']:
                    continue

                chromosome = fields[0]
                strand = fields[6]
                start = int(fields[3])
                end = int(fields[4])
                attributes = fields[8]
                gene_id = None
                transcript_id = None
                gene_name = None
                transcript_type = None
                exon_number = None
                exon_id = None

                for attr in attributes.split(';'):
                    attr = attr.strip()
                    if not attr:
                        continue
                    try:
                        if attr.startswith('gene_id'):
                            gene_id = extract_gtf_attr_val(attr)
                        elif attr.startswith('transcript_id'):
                            transcript_id = extract_gtf_attr_val(attr)
                        elif attr.startswith('gene_name'):
                            gene_name = extract_gtf_attr_val(attr)
                        elif attr.startswith('transcript_type'):
                            transcript_type = extract_gtf_attr_val(attr)
                        elif attr.startswith('exon_number'):
                            value = extract_gtf_attr_val(attr)
                            if value:
                                exon_number = int(value)
                        elif attr.startswith('exon_id'):
                            exon_id = extract_gtf_attr_val(attr)
                    except (ValueError, IndexError):
                        continue

                if gene_id and transcript_id:
                    if feature_type == 'transcript':
                        record = {
                            'gene_id': gene_id,
                            'transcript_id': transcript_id,
                            'gene_name': gene_name,
                            'transcript_type': transcript_type,
                            'exon_number': None,
                            'exon_id': None,
                            'chromosome': chromosome,
                            'strand': strand,
                            'exon_start': start,
                            'exon_end': end,
                            'transcript_start': start,
                            'transcript_end': end
                        }
                    else:
                        record = {
                            'gene_id': gene_id,
                            'transcript_id': transcript_id,
                            'gene_name': gene_name,
                            'transcript_type': transcript_type,
                            'exon_number': exon_number,
                            'exon_id': exon_id,
                            'chromosome': chromosome,
                            'strand': strand,
                            'exon_start': start,
                            'exon_end': end,
                            'transcript_start': None,
                            'transcript_end': None
                        }

                    gencode_records.append(record)

        if gencode_records:
            gencode_df = pd.DataFrame(gencode_records)

            if 'gene_name' in gencode_df.columns:
                gencode_df['gene_name'] = gencode_df['gene_name'].astype(str).str.upper()

            gencode_df.to_sql(gencode_table, conn, if_exists='replace', index=False)
            print(f" Processed {len(gencode_records):,} {species_name} Gencode GTF records!")

    print(f"Creating optimized {species_name} database indices...")
    indices = [
        # Junction master table indexes
        (f"{table_prefix}idx_junctions_gene", junctions_table, "(gene_name, gene_id)"),
        (f"{table_prefix}idx_junctions_gene_name", junctions_table, "(gene_name)"),
        (f"{table_prefix}idx_junctions_gene_id", junctions_table, "(gene_id)"),
        (f"{table_prefix}idx_junctions_junction_id", junctions_table, "(junction_id)"),
        (f"{table_prefix}idx_junctions_avg_psi", junctions_table, "(junction_average_psi)"),
        # Junction PSI table indexes
        (f"{table_prefix}idx_junction_psis_junction_id", junction_psis_table, "(junction_id)"),
        (f"{table_prefix}idx_junction_psis_cell_type", junction_psis_table, "(cell_type)"),
        (f"{table_prefix}idx_junction_psis_psi", junction_psis_table, "(psi)"),
        (f"{table_prefix}idx_junction_psis_junction_cell", junction_psis_table, "(junction_id, cell_type)"),
        # Isoform table indexes
        (f"{table_prefix}idx_isoforms_gene", isoforms_table, "(gene_name, gene_id)"),
        (f"{table_prefix}idx_isoforms_gene_name", isoforms_table, "(gene_name)"),
        (f"{table_prefix}idx_isoforms_id", isoforms_table, "(id)"),
        # PSL table indexes
        (f"{table_prefix}idx_psl_gene", psl_table, "(gene_id, id)"),
        (f"{table_prefix}idx_psl_id", psl_table, "(id)"),
        # TPM and ratio table indexes
        (f"{table_prefix}idx_tpm_id", tpm_table, "(id)"),
        (f"{table_prefix}idx_ratio_id", ratio_table, "(id)")
    ]

    # Add gencode index only for human
    extra_indices = 4 if species_name == "Human" else 3

    with tqdm(desc=f"Creating {species_name} indices", total=len(indices) + extra_indices, unit="index") as idx_pbar:
        for idx_name, table_name, columns in indices:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name}{columns}")
            idx_pbar.update(1)

        conn.execute(f"ALTER TABLE {atse_table} ADD COLUMN gene_id_clean TEXT")
        conn.execute(f"UPDATE {atse_table} SET gene_id_clean = SUBSTR(gene_id, 1, INSTR(gene_id, '.') - 1) WHERE gene_id LIKE '%.%'")
        conn.execute(f"UPDATE {atse_table} SET gene_id_clean = gene_id WHERE gene_id_clean IS NULL")
        idx_pbar.update(1)
        conn.execute(f"CREATE INDEX IF NOT EXISTS {table_prefix}idx_atse_gene ON {atse_table}(gene_id_clean, gene_name)")
        idx_pbar.update(1)
        conn.execute(f"CREATE INDEX IF NOT EXISTS {table_prefix}idx_atse_coords ON {atse_table}(chromosome, start, end)")
        idx_pbar.update(1)
        if species_name == "Human":
            conn.execute(f"CREATE INDEX IF NOT EXISTS {table_prefix}idx_gencode_transcript ON {gencode_table}(gene_id, transcript_id)")
            idx_pbar.update(1)

    conn.commit()
    print(f" {species_name} data loading complete!")
    print()


def verify_database_schema(db_path):
    """Verify database schema has required columns"""
    conn = sqlite3.connect(db_path)
    
    iso_schema = pd.read_sql("PRAGMA table_info(isoforms)", conn)
    print("Isoforms table columns:", iso_schema['name'].tolist())
    
    psl_schema = pd.read_sql("PRAGMA table_info(psl_data)", conn)
    print("PSL data table columns:", psl_schema['name'].tolist())
    
    if 'id' not in iso_schema['name'].values:
        print("ERROR: Missing 'id' column in isoforms table")
    if 'id' not in psl_schema['name'].values:
        print("ERROR: Missing 'id' column in psl_data table")
    
    conn.close()


def verify_database_schema(db_path):
    """Verify database schema has required columns"""
    conn = sqlite3.connect(db_path)
    
    iso_schema = pd.read_sql("PRAGMA table_info(isoforms)", conn)
    print("Isoforms table columns:", iso_schema['name'].tolist())
    
    psl_schema = pd.read_sql("PRAGMA table_info(psl_data)", conn)
    print("PSL data table columns:", psl_schema['name'].tolist())
    
    if 'id' not in iso_schema['name'].values:
        print("ERROR: Missing 'id' column in isoforms table")
    if 'id' not in psl_schema['name'].values:
        print("ERROR: Missing 'id' column in psl_data table")
    
    conn.close()


###################################################################
# HELPER FUNCTIONS
###################################################################
def create_loading_progress_figure():
    """
    Setup for white loading progress circle around pulsing logo in loading screen
    """
    radius = 1.5
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        mode='lines',
        line=dict(color='rgba(255,255,255,0.5)', width=8),
        showlegend=False,
        hoverinfo='none'
    ))
    
    fig.update_layout(
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin={'l': 20, 'r': 20, 't': 20, 'b': 20},
        xaxis={
            'visible': False,
            'range': [-2.0, 2.0],
            'scaleanchor': 'y',
            'scaleratio': 1
        },
        yaxis={
            'visible': False,
            'range': [-2.0, 2.0]
        },
        width=450,
        height=450
    )
    
    return fig

###################################################################
# APPLICATION AND DB CONNECTION SETUP
###################################################################
base_dir = os.path.dirname(os.path.abspath(__file__))
db_type = os.getenv('DB_TYPE', 'sqlite').lower()

if db_type == 'postgresql':
    print(f"{Fore.CYAN}Initializing PostgreSQL database connection...{Style.RESET_ALL}")
    initialize_database(use_postgresql=True)
    db_config = get_db_config()
    print(f"{Fore.GREEN}Connected to IsoformGazer database{Style.RESET_ALL}")
    # db_path is None for PostgreSQL mode (we use db_config instead)
    db_path = None

else:
    print(f"{Fore.CYAN}Using SQLite database for local development...{Style.RESET_ALL}")
    db_path = setup_local_database()
    initialize_database(db_path=db_path, use_postgresql=False)
    db_config = get_db_config()
    print(f"{Fore.GREEN}Connected to SQLite database at {db_path}{Style.RESET_ALL}")

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Cache for default gene AACS
DEFAULT_GENE = 'AACS'
default_gene_cache = None
cache_loaded_from_disk = False

# Note: Cache validation logic still uses db_path for SQLite compatibility
if db_type == 'sqlite' and is_cache_valid(base_dir, db_path):
    default_gene_cache = load_default_gene_cache(base_dir)
    cache_loaded_from_disk = True

if not default_gene_cache and db_path:
    default_gene_cache = generate_default_gene_cache(db_path, DEFAULT_GENE)
    if default_gene_cache:
        save_default_gene_cache(base_dir, db_path, default_gene_cache)

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# CSS for styling components with responsive design
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Isoform Gazer</title>
        <link rel="icon" type="image/png" href="/assets/Isoform-Gazer-Logo.png">
        <link rel="shortcut icon" type="image/png" href="/assets/Isoform-Gazer-Logo.png">
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

###################################################################
# CACHE STATISTICS API ENDPOINT
###################################################################
@app.server.route('/api/cache-stats')
def api_cache_stats():
    """
    API endpoint to check cache statistics.
    Access: GET /api/cache-stats
    Returns cache hit/miss stats and Redis connection status
    """
    try:
        from src.isoformgazer.gene_cache_redis import get_cache_stats
        stats = get_cache_stats()

    except Exception as e:
        stats = {
            'error': str(e),
            'backend': 'unknown',
            'gene_list_cached': False,
            'default_gene_cached': False
        }

    return Response(
        json.dumps(stats, indent=2),
        mimetype='application/json'
    )

###################################################################
# APPLICATION TITLE (ISOFORM GAZER)
###################################################################
header = html.Div(className='app-header', children=[
    html.Img(src='/assets/Isoform-Gazer-telescope.png', className='app-header--logo'),
    html.Img(src='/assets/Isoform-Gazer-text.png', className='app-header--title'),
    html.A(
        href='https://github.com/daklab/IsoformGazer',
        target='_blank',
        className='github-button',
        children=[
            html.Img(src='/assets/Octicons-mark-github.svg', className='github-icon'),
            html.Span('VIEW ON GITHUB', className='github-text')
        ]
    )
])

###################################################################
# ISOFORM MASTER TABLE 
###################################################################
left_data_table = dash_table.DataTable(
    id='left_data_table',
    columns=get_master_table_columns(db_path, table_name='isoforms'),
    hidden_columns=['id', 'gene_id', 'gene_name', 'gene_potential', 'gene_perplexity', 
                    'gene_protein_category', 'gene_gencode_v46_basic_transcript_counts', 
                    'gene_average_tpm', 'gene_total_tpm', 'ptc_potential', 'ptc_perplexity', 
                    'ORF_potential', 'ORF_perplexity', 'ORF_expressed_samples', 'gene_expressed_samples'],
    data=[],
    editable=False,
    filter_action="custom",
    filter_options={'placeholder_text': 'Filter column...'},
    sort_action="custom",
    sort_mode="multi",
    page_action="custom",
    page_current=0,
    page_count=0,
    page_size=10,
    filter_query='',
    sort_by=[],
    style_cell={
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
        'minWidth': '150px',
        'padding': '10px 8px',
        'whiteSpace': 'normal',
        'font-family': '"Open Sans", sans-serif',
        'fontSize': '12px',
        'lineHeight': '1.4',
        'border': '1px solid #e1e5e9'
    },
    style_table={
        'overflowY': 'visible',
        'overflowX': 'auto', 
        'minWidth': '100%',
        'height': 'auto'     
    },
    style_header={
        'backgroundColor': '#301279',
        'color': 'white',
        'fontWeight': '600',
        'font-family': '"Open Sans", sans-serif',
        'fontSize': '13px',
        'whiteSpace': 'normal',
        'height': 'auto',        
        'lineHeight': '16px',    
        'padding': '12px 8px',        
        'textAlign': 'center',
        'border': '1px solid #622e9d',
        'borderRadius': '0px'
    },
    style_data={
        'font-family': '"Open Sans", sans-serif',
        'fontSize': '12px',
        'backgroundColor': 'white',
        'color': '#1C1C2C',
        'padding': '8px',
        'border': '1px solid #e1e5e9'
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'}, 
            'backgroundColor': '#f8f9fa'
        },
        {
            'if': {'state': 'selected'},
            'backgroundColor': '#622e9d',
            'color': 'white'
        }
    ],
    style_filter={
        'backgroundColor': '#622e9d',
        'color': 'white',
        'fontWeight': '500',
        'font-family': '"Open Sans", sans-serif',
        'fontSize': '12px',
        'padding': '10px 8px',
        'border': '1px solid #301279'
    },
    virtualization=True,
    fixed_rows={'headers': True},
    column_selectable=False,
    row_selectable=False,
    css=[{"selector": ".show-hide", "rule": "display: none"}]
)

###################################################################
# JUNCTION MASTER TABLE 
###################################################################
right_data_table = dash_table.DataTable(
    id='right_data_table',
    columns=get_master_table_columns(db_path, table_name='junctions'),
    hidden_columns=['id', 'matched_transcript_ids'],
    data=[],
    editable=False,
    filter_action="custom",
    filter_options={'placeholder_text': 'Filter column...'},
    sort_action="custom",
    sort_mode="multi",
    page_action="custom",
    page_current=0,
    page_count=0,
    page_size=10,
    filter_query='',
    sort_by=[],
    style_cell={
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
        'minWidth': '150px', 
        #'maxWidth': '220px',
        'padding': '10px 8px',
        'whiteSpace': 'normal',
        'font-family': '"Open Sans", sans-serif',
        'fontSize': '12px',
        'lineHeight': '1.4',
        'border': '1px solid #e1e5e9'
    },
    style_table={
        'overflowY': 'visible',
        'overflowX': 'auto', 
        'minWidth': '100%',
        'height': 'auto'     
    },
    style_header={
        'backgroundColor': '#301279',
        'color': 'white',
        'fontWeight': '600',
        'font-family': '"Open Sans", sans-serif',
        'fontSize': '13px',
        'whiteSpace': 'normal',
        'height': 'auto',        
        'lineHeight': '16px',    
        'padding': '12px 8px',        
        'textAlign': 'center',
        'border': '1px solid #622e9d',
        'borderRadius': '0px'
    },
    style_data={
        'font-family': '"Open Sans", sans-serif',
        'fontSize': '12px',
        'backgroundColor': 'white',
        'color': '#1C1C2C',
        'padding': '8px',
        'border': '1px solid #e1e5e9'
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'}, 
            'backgroundColor': '#f8f9fa'
        },
        {
            'if': {'state': 'selected'},
            'backgroundColor': '#622e9d',
            'color': 'white'
        }
    ],
    style_filter={
        'backgroundColor': '#622e9d',
        'color': 'white',
        'fontWeight': '500',
        'font-family': '"Open Sans", sans-serif',
        'fontSize': '12px',
        'padding': '10px 8px',
        'border': '1px solid #301279'
    },
    virtualization=True,
    fixed_rows={'headers': True},
    column_selectable=False,
    row_selectable=False,
    css=[{"selector": ".show-hide", "rule": "display: none"}]
)

###################################################################
# MAIN LAYOUT
###################################################################
app.layout = html.Div(className='app-layout', children=[
    #####################################
    # Initial loading screen overlay!
    #####################################
    html.Div(id='loading-overlay', className='loading-overlay', children=[
        html.Div(className='loading-content', children=[
            html.Div(className='logo-progress-container', children=[
                html.Img(src='/assets/Isoform-Gazer-Logo.png', className='loading-logo pulsing'),
                html.Div(className='progress-ring', children=[
                    html.Div([
                        dcc.Graph(
                            id='progress-circle',
                            figure=create_loading_progress_figure(),
                            config={'displayModeBar': False},
                            className='progress-graph'
                        )
                    ])
                ])
            ])
        ]),
        # Transcript constellations!
        html.Div(className='constellation', children=[
            html.Div(className='transcript-star transcript-1 exons-2'),
            html.Div(className='transcript-star transcript-2 exons-4'),
            html.Div(className='transcript-star transcript-3 exons-3'),
            html.Div(className='transcript-star transcript-4 exons-5'),
            html.Div(className='transcript-star transcript-5 exons-2'),
            html.Div(className='transcript-star transcript-6 exons-6'),
            html.Div(className='transcript-star transcript-7 exons-3'),
            html.Div(className='transcript-star transcript-8 exons-5'),
            html.Div(className='transcript-star transcript-9 exons-2'),
            html.Div(className='transcript-star transcript-10 exons-4'),
            html.Div(className='transcript-star transcript-11 exons-3'),
            html.Div(className='transcript-star transcript-12 exons-6'),
            html.Div(className='transcript-star transcript-13 exons-2'),
            html.Div(className='transcript-star transcript-14 exons-5'),
            html.Div(className='transcript-star transcript-15 exons-4')
        ])
    ]),
    header,
    html.Div(className='app-body', children=[
        #####################################
        # Control panel (left sidebar)
        #####################################
        html.Div(id='control-tabs', className='control-tabs control-tabs-container', children=[
            dcc.Tabs(id='tabs', value='tab-1', className='tabs-full-height', children=[
                dcc.Tab(label='About', className='tab-1', value='tab-1', children=[
                    html.Div(className='control-tab', children=[
                        html.Div(className='about-logo-header-container', children=[
                            html.H2('About', className='about-tab-header'),
                            #html.Img(src='/assets/Isoform-Gazer-Logo.png', 
                            #        style={'height': '100px', 'width': 'auto'}, className='about-logo')
                        ]),
                        html.Div(className='about-content', children=[
                            html.P([
                                'Isoform Gazer provides a unified view of RNA splicing across both short-read junction-level single-cell data and long-read transcript-level isoform data in GENCODEv46 (GRCh38.p14). ',
                                'Isoform Gazer is developed and maintained by the ',
                                html.A('Knowles Lab',
                                       href='https://daklab.github.io/',
                                       target='_blank',
                                       className='knowles-lab-link'),
                                '.'
                            ]),
                            html.P([
                                'Use the controls in the ',
                                html.Span('Query', className='about-keyword-purple'),
                                ' tab to dynamically query the master table data and generate visualizations.'
                            ]),
                            html.P([
                                'Use the controls in the ',
                                html.Span('Custom', className='about-keyword-gold'),
                                ' tab to customize the visualizations.'
                            ])
                        ])
                    ])
                ]),
                dcc.Tab(label='Query', className='tab-1', value='tab-2', children=[
                    html.Div(className='control-tab', children=[
                        html.H2('Species', className='alignment-settings-section'),
                        html.Div(className='app-controls-block', children=[
                            dcc.Dropdown(
                                id='species-dropdown',
                                options=[
                                    {'label': 'Human (GRCh38)', 'value': 'Human'},
                                    {'label': 'Mouse (GRCm38)', 'value': 'Mouse'}
                                ],
                                value='Human',
                                searchable=False,
                                clearable=False
                            ),
                            html.Div(className='app-controls-desc', children='Select species to visualize data from')
                        ]),
                        html.H2('Search by Gene', className='alignment-settings-section'),
                        html.Div(className='app-controls-block', children=[
                            dcc.Dropdown(
                                id='gene-search-dropdown',
                                options=[
                                    {'label': 'AACS', 'value': 'AACS'},
                                    {'label': 'RBFOX2 (RNA Binding Fox-1 Homolog 2)', 'value': 'RBFOX2'},
                                    {'label': 'EGFR (Epidermal growth factor receptor)', 'value': 'EGFR'},
                                    {'label': 'BRCA1 (Breast cancer type 1)', 'value': 'BRCA1'},
                                    {'label': 'TARDBP (TAR DNA Binding Protein)', 'value': 'TARDBP'},
                                    {'label': 'TP53 (Tumor protein p53)', 'value': 'TP53'}
                                ],
                                placeholder="Type to search for a gene...",
                                value='AACS',
                                searchable=True,
                                clearable=False
                            ),
                            html.Div(className='app-controls-desc', children='Select a gene identifier to query or type to search')
                        ]),
                        html.Div(id='gene-filter-status'),

                        html.H3('Gene-Level Summary', className='summary-section-header'),
                        html.Div(id='gene-level-summary', className='summary-block', children=[
                            html.P("Select a gene to view summary information", className='summary-placeholder')
                        ]),

                        html.H3('ORF-Level Summary', className='summary-section-header'),
                        html.Div(id='ORF-level-summary', className='summary-block', children=[
                            html.P("Select a gene to view summary information", className='summary-placeholder')
                        ]),

                        html.H2('Query by Value', className='alignment-settings-section'),
                        html.Div(className='query-content', children=[
                            html.P("You can query master tables to update plots using the query boxes below each column header:"),
                            html.Ul([
                                html.Li([html.Strong("Text columns: "), "Type text for exact matches"]),
                                html.Li([html.Strong("Numerical columns: "), "Use operators =, >, <, >=, and <= (e.g., '>5', '<=10')"])
                            ]),
                            html.P([
                                "You can apply multiple column filters simultaneously."
                            ]),
                            html.P([
                                "Delete your queries in the filter boxes and hit Enter to remove individual filters. "
                                "To clear all filters, use the Clear All button above the master table.",
                            ])
                        ]),

                        html.H2('Isoform Hash Lookup', className='alignment-settings-section'),
                        html.Div(className='hash-lookup-content', children=[
                            html.P("Generate hash IDs for isoforms by uploading a GTF file. A table with gene_id, transcript_id, and hash_id will be generated."),
                            html.H3('GTF File Upload', className='summary-section-header'),
                            html.Div(className='app-controls-block', children=[
                                dcc.Upload(
                                    id='gtf-upload',
                                    children=html.Div([
                                        'Drag and drop or',
                                        html.Br(),
                                        html.A('select GTF file')
                                    ]),
                                    multiple=False
                                ),
                                html.Div(id='gtf-upload-status', className='upload-status'),
                                html.Div(id='gtf-download-section', className='hidden', children=[
                                    html.Div(className='download-buttons-container', children=[
                                        html.Button('Download Hash Results', id='download-hashes-btn', className='control-button'),
                                        html.Button('Download Annotated GTF', id='download-annotated-gtf-btn', className='control-button')
                                    ]),
                                    dcc.Download(id='download-hashes'),
                                    dcc.Download(id='download-annotated-gtf')
                                ])
                            ])
                        ])
                    ])
                ]),
                dcc.Tab(label='Custom', className='tab-1', value='tab-3', children=[
                    html.Div(className='control-tab', children=[
                        #####################################
                        # Isoform-level Event Plot Section
                        #####################################
                        html.H2('Structure Plot', className='alignment-settings-section'),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Plot Height'),
                            dcc.Slider(
                                id='bar-height-slider',
                                className='control-slider',
                                min=600,
                                max=1600,
                                step=100,
                                value=600,
                                marks={600: '600', 700: '', 800: '800', 900: '', 1000: '1000', 1100: '', 1200: '1200', 1300: '', 1400: '1400', 1500: '', 1600: '1600'}
                            ),
                            html.Div(className='app-controls-desc', children='Adjust the height of the structure plots')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Exon Color'),
                            html.Div(className='color-picker-container', children=[
                                daq.ColorPicker(
                                    id='exon-color-picker',
                                    value={'hex': '#2E86C1'},
                                    size=240,
                                    theme=None,
                                    className='color-picker'
                                )
                            ]),
                            html.Div(className='app-controls-desc', children='Customize default color of exons in the structure plots')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Junction Color'),
                            html.Div(className='color-picker-container', children=[
                                daq.ColorPicker(
                                    id='junction-color-picker',
                                    value={'hex': '#85929E'},
                                    size=240,
                                    theme=None,
                                    className='color-picker'
                                )
                            ]),
                            html.Div(className='app-controls-desc', children='Customize default color of junctions in the ATSE structure plot')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='toggle-switch-row', children=[
                                html.Div('Hide Junctions', className='app-controls-name toggle-switch-label-wide'),
                                daq.ToggleSwitch(
                                    id='hide-junctions-toggle',
                                    value=False,  # False = show junctions, True = hide junctions (show transcript plot)
                                    label={'label': 'Show / Hide', 'style': {'fontSize': '12px', 'color': '#506784'}},
                                    labelPosition='left',
                                    className='toggle-switch-inline'
                                )
                            ]),
                            html.Div(className='app-controls-desc', children='Toggle to hide junctions from the structure plot')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='toggle-switch-row', children=[
                                html.Div('Color by Average PSI', className='app-controls-name toggle-switch-label-wide'),
                                daq.ToggleSwitch(
                                    id='color-junctions-by-psi-toggle',
                                    value=True,
                                    label={'label': 'Off / On', 'style': {'fontSize': '12px', 'color': '#506784'}},
                                    labelPosition='left',
                                    className='toggle-switch-inline'
                                )
                            ]),
                            html.Div(className='app-controls-desc', children='Color junctions by their average PSI (percent spliced in)')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='toggle-switch-row', children=[
                                html.Div('Color by Abundance', className='app-controls-name toggle-switch-label-wide'),
                                daq.ToggleSwitch(
                                    id='color-by-abundance-toggle',
                                    value=True,
                                    label={'label': 'Off / On', 'style': {'fontSize': '12px', 'color': '#506784'}},
                                    labelPosition='left',
                                    className='toggle-switch-inline'
                                )
                            ]),
                            html.Div(className='app-controls-desc', children='Color transcripts by their abundance (TPM)'),
                            html.Div(id='abundance-color-options-container', style={'display': 'none'}, children=[
                                html.Div(style={'marginTop': '15px'}, children=[
                                    dcc.RadioItems(
                                        id='abundance-color-type-radio',
                                        options=[
                                            {'label': ' Average Abundance', 'value': 'average'},
                                            {'label': ' Tissue Abundance', 'value': 'tissue'},
                                            {'label': ' Organ Abundance', 'value': 'organ'}
                                        ],
                                        value='average',
                                        className='radio-items-vertical',
                                        labelStyle={'display': 'block', 'marginBottom': '8px', 'fontSize': '12px', 'color': '#506784'}
                                    )
                                ]),
                                html.Div(id='tissue-dropdown-container', style={'marginTop': '10px', 'display': 'none'}, children=[
                                    html.Div(className='app-controls-block', children=[
                                        html.Div('Select Tissue', className='app-controls-name'),
                                        dcc.Dropdown(
                                            id='tissue-abundance-dropdown',
                                            className='app-controls-block-dropdown',
                                            options=[],
                                            value=None,
                                            clearable=False
                                        ),
                                        html.Div(className='app-controls-desc', children='Choose which tissue to color transcripts by')
                                    ])
                                ]),
                                html.Div(id='organ-dropdown-container', style={'marginTop': '10px', 'display': 'none'}, children=[
                                    html.Div(className='app-controls-block', children=[
                                        html.Div('Select Organ', className='app-controls-name'),
                                        dcc.Dropdown(
                                            id='organ-abundance-dropdown',
                                            className='app-controls-block-dropdown',
                                            options=[],
                                            value=None,
                                            clearable=False
                                        ),
                                        html.Div(className='app-controls-desc', children='Choose which organ to color transcripts by')
                                    ])
                                ])
                            ])
                        ]),
                        html.Div(id='structure-plot-colorscale-container', style={'display': 'none'}, children=[
                            html.Div(className='app-controls-block', children=[
                                html.Div(className='app-controls-name', children='Colorscale'),
                                dcc.Dropdown(
                                    id='structure-plot-colorscale-dropdown',
                                    className='app-controls-block-dropdown',
                                    options=[
                                        {'label': 'RdBu_r', 'value': 'RdBu_r'},
                                        {'label': 'Viridis', 'value': 'Viridis'},
                                        {'label': 'Plasma', 'value': 'Plasma'},
                                        {'label': 'Spectral', 'value': 'Spectral'},
                                        {'label': 'Turbo', 'value': 'Turbo'},
                                        {'label': 'Cividis', 'value': 'Cividis'},
                                        {'label': 'Blues', 'value': 'Blues'},
                                        {'label': 'Reds', 'value': 'Reds'},
                                        {'label': 'YlOrRd', 'value': 'YlOrRd'},
                                        {'label': 'RdYlBu', 'value': 'RdYlBu'},
                                        {'label': 'Inferno', 'value': 'Inferno'},
                                        {'label': 'Magma', 'value': 'Magma'}
                                    ],
                                    value='Viridis',
                                    clearable=False
                                ),
                                html.Div(className='app-controls-desc', children='Choose the color theme for the structure plot')
                            ])
                        ]),

                        #####################################
                        # Clustergrams Section
                        #####################################
                        html.H2('Clustergram Plots', className='alignment-settings-section'),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Clustergram Height'),
                            dcc.Slider(
                                id='clustergram-height-slider',
                                className='control-slider',
                                min=750,
                                max=2000,
                                step=125,
                                value=1012,
                                marks={750: '750', 875: '', 1000: '1000', 1125: '', 1250: '1250', 1375: '', 1500: '1500', 1625: '', 1750: '1750', 1875: '', 2000: '2000'}
                            ),
                            html.Div(className='app-controls-desc', children='Adjust the height of the clustergrams')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='toggle-switch-row', children=[
                                html.Div('Isoform Clustergram Unit', className='app-controls-name toggle-switch-label-narrow'),
                                dcc.RadioItems(
                                    id='isoform-data-type-switch',
                                    options=[
                                        {'label': ' Ratio', 'value': 'ratio'},
                                        {'label': ' TPM', 'value': 'tpm'},
                                        {'label': ' Log TPM', 'value': 'log_tpm'}
                                    ],
                                    value='ratio',
                                    className='radio-items-inline',
                                    labelStyle={'display': 'inline-block', 'marginRight': '20px', 'fontSize': '12px', 'color': '#506784'}
                                )
                            ]),
                            html.Div(className='app-controls-desc', children='Select whether to show Ratio, TPM, or log TPM values for the isoform clustergram')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='radio-items-row', children=[
                                html.Div('Isoform Clustermap Sample Averaging', className='app-controls-name radio-items-label'),
                                dcc.RadioItems(
                                    id='collapse-tissue-toggle',
                                    options=[
                                        {'label': 'By Replicate', 'value': 'replicate'},
                                        {'label': 'By Tissue', 'value': 'tissue'},
                                        {'label': 'None', 'value': 'all'}
                                    ],
                                    value='replicate',
                                    className='radio-items-inline'
                                )
                            ]),
                            html.Div(className='app-controls-desc', children='Select whether isoform clustergram samples are averaged across replicates, '
                            'averaged across tissues, or not averaged (shows all samples). Averaging across replicates is recommended to reduce technical ' \
                            'noise while preserving true biological variation')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='toggle-switch-row', children=[
                                html.Div('Tissue Labels', className='app-controls-name toggle-switch-label-medium'),
                                daq.ToggleSwitch(
                                    id='show-labels-toggle',
                                    value=False,
                                    label={'label': 'Hide / Show', 'style': {'fontSize': '12px', 'color': '#506784'}},
                                    labelPosition='left',
                                    className='toggle-switch-inline'
                                )
                            ]),
                            html.Div(className='app-controls-desc', children='Toggle tissue labels visibility on isoform clustergram')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='toggle-switch-row', children=[
                                html.Div('Cell Type Labels', className='app-controls-name toggle-switch-label-narrow-plus'),
                                daq.ToggleSwitch(
                                    id='show-celltype-labels-toggle',
                                    value=False,
                                    label={'label': 'Hide / Show', 'style': {'fontSize': '12px', 'color': '#506784'}},
                                    labelPosition='left',
                                    className='toggle-switch-inline'
                                )
                            ]),
                            html.Div(className='app-controls-desc', children='Toggle cell type labels visibility on junction clustergram')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Colorscale'),
                            dcc.Dropdown(
                                id='colorscale-dropdown',
                                className='app-controls-block-dropdown',
                                options=[
                                    {'label': 'RdBu_r', 'value': 'RdBu_r'},
                                    {'label': 'Viridis', 'value': 'Viridis'},
                                    {'label': 'Plasma', 'value': 'Plasma'},
                                    {'label': 'Spectral', 'value': 'Spectral'},
                                    {'label': 'Turbo', 'value': 'Turbo'},
                                    {'label': 'Cividis', 'value': 'Cividis'},
                                    {'label': 'Blues', 'value': 'Blues'},
                                    {'label': 'Reds', 'value': 'Reds'},
                                    {'label': 'YlOrRd', 'value': 'YlOrRd'},
                                    {'label': 'RdYlBu', 'value': 'RdYlBu'},
                                    {'label': 'Inferno', 'value': 'Inferno'},
                                    {'label': 'Magma', 'value': 'Magma'}
                                ],
                                value='Viridis',
                                clearable=False
                            ),
                            html.Div(className='app-controls-desc', children='Choose the color theme of the heatmaps')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='toggle-switch-row', children=[
                                html.Div('Clustergram Gridlines', className='app-controls-name toggle-switch-label-narrow-plus'),
                                daq.ToggleSwitch(
                                    id='gridlines-toggle',
                                    value=False,
                                    label={'label': 'Hide / Show', 'style': {'fontSize': '12px', 'color': '#506784'}},
                                    labelPosition='left',
                                    className='toggle-switch-inline'
                                )
                            ]),
                            html.Div(className='app-controls-desc', children='Toggle white gridlines visibility in clustergrams')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Distance Metric'),
                            dcc.Dropdown(
                                id='distance-metric-dropdown',
                                className='app-controls-block-dropdown',
                                options=[
                                    {'label': 'Euclidean', 'value': 'euclidean'},
                                    {'label': 'Standardized Euclidean', 'value': 'seuclidean'},
                                    {'label': 'Cosine', 'value': 'cosine'},
                                    {'label': 'Correlation', 'value': 'correlation'},
                                    {'label': 'Manhattan (L1)', 'value': 'cityblock'}
                                ],
                                value='euclidean',
                                clearable=False
                            ),
                            html.Div(className='app-controls-desc', children='Distance metric used for clustering in both clustergrams')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Clustering Algorithm'),
                            dcc.Dropdown(
                                id='linkage-method-dropdown',
                                className='app-controls-block-dropdown',
                                options=[
                                    {'label': 'Ward', 'value': 'ward'},
                                    {'label': 'UPGMA', 'value': 'average'},
                                    {'label': 'UPGMC', 'value': 'centroid'},
                                    {'label': 'WPGMC', 'value': 'median'},
                                    {'label': 'WPGMA', 'value': 'weighted'},
                                    {'label': 'Nearest Point', 'value': 'single'},
                                    {'label': 'Furthest Point', 'value': 'complete'}
                                ],
                                value='ward',
                                clearable=False
                            ),
                            html.Div(className='app-controls-desc', children='Hierarchical clustering algorithm used for both clustergrams')
                        ]),

                        #####################################
                        # Plot Exports Section
                        #####################################
                        html.H2('Export Plot', className='alignment-settings-section'),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Dimensions'),
                            html.Div(style={'marginTop': '15px'}, children=[
                                html.Label("Width:", style={'fontWeight': '600', 'fontSize': '14px', 'color': '#301279'}),
                                dcc.Input(
                                    id='export-width-value',
                                    type='number',
                                    placeholder='Enter width',
                                    value=800,
                                    style={'width': 'calc(100% - 20px)', 'padding': '8px', 'marginTop': '5px', 'marginRight': '10px', 'border': '1px solid #ddd', 'borderRadius': '4px'}
                                )
                            ]),
                            html.Div(style={'marginTop': '15px'}, children=[
                                html.Label("Height:", style={'fontWeight': '600', 'fontSize': '14px', 'color': '#301279'}),
                                dcc.Input(
                                    id='export-height-value',
                                    type='number',
                                    placeholder='Enter height',
                                    value=600,
                                    style={'width': 'calc(100% - 20px)', 'padding': '8px', 'marginTop': '5px', 'marginRight': '10px', 'border': '1px solid #ddd', 'borderRadius': '4px'}
                                )
                            ]),

                            # Unit section
                            html.Div(style={'marginTop': '15px'}, children=[
                                html.Div('Unit', className='app-controls-name', style={'marginBottom': '10px'}),
                                dcc.RadioItems(
                                    id='export-unit-toggle',
                                    options=[
                                        {'label': ' Pixels (px)', 'value': 'px'},
                                        {'label': ' Inches (in)', 'value': 'in'}
                                    ],
                                    value='px',
                                    className='radio-items-vertical',
                                    labelStyle={'display': 'block', 'marginLeft': '6px', 'marginBottom': '5px', 'fontSize': '14px', 'color': '#506784'}
                                )
                            ]),
                            html.Div(style={'marginTop': '15px'}, children=[
                                html.Label("Title Font Size:", style={'fontWeight': '600', 'fontSize': '14px', 'color': '#301279'}),
                                dcc.Input(
                                    id='export-title-legend-font-size',
                                    type='number',
                                    placeholder='Enter font size',
                                    value=16,
                                    style={'width': 'calc(100% - 20px)', 'padding': '8px', 'marginTop': '5px', 'marginRight': '10px', 'border': '1px solid #ddd', 'borderRadius': '4px'}
                                )
                            ]),
                            html.Div(style={'marginTop': '15px'}, children=[
                                html.Label("Axis Labels Font Size:", style={'fontWeight': '600', 'fontSize': '14px', 'color': '#301279'}),
                                dcc.Input(
                                    id='export-axis-labels-font-size',
                                    type='number',
                                    placeholder='Enter font size',
                                    value=12,
                                    style={'width': 'calc(100% - 20px)', 'padding': '8px', 'marginTop': '5px', 'marginRight': '10px', 'border': '1px solid #ddd', 'borderRadius': '4px'}
                                )
                            ]),
                            html.Div(style={'marginTop': '20px'}, children=[
                                html.Div('Plot to Export', className='app-controls-name toggle-switch-label-narrow', style={'marginBottom': '10px'}),
                                dcc.RadioItems(
                                    id='export-plot-selection',
                                    options=[
                                        {'label': ' Structure Plot', 'value': 'structure'},
                                        {'label': ' Isoform Clustergram', 'value': 'isoform'},
                                        {'label': ' Junction Clustergram', 'value': 'junction'}
                                    ],
                                    value='structure',
                                    className='radio-items-vertical',
                                    labelStyle={'display': 'block', 'marginLeft': '6px', 'marginBottom': '5px', 'fontSize': '12px', 'color': '#506784'}
                                )
                            ]),
                            html.Div(style={'marginTop': '20px'}, children=[
                                dbc.Button(
                                    "Download Plot",
                                    id='export-unified-btn',
                                    color='primary',
                                    style={
                                        'backgroundColor': '#301279',
                                        'color': 'white',
                                        'padding': '10px 24px',
                                        'borderRadius': '4px',
                                        'border': 'none',
                                        'fontWeight': '600',
                                        'fontSize': '14px',
                                        'transition': 'all 0.2s ease'
                                    }
                                ),
                                dcc.Download(id="download-structure-plot"),
                                dcc.Download(id="download-isoform-clustergram"),
                                dcc.Download(id="download-junction-clustergram"),
                                dcc.Interval(id='export-status-timer', interval=7500, n_intervals=0, disabled=True)
                            ]),
                            html.Div(
                                id='export-status-message',
                                style={
                                    'marginTop': '15px',
                                    'fontSize': '14px',
                                    'color': '#301279',
                                    'fontWeight': '600',
                                    'display': 'none'
                                },
                                children='Download in Progress...'
                            )
                        ])
                    ])
                ])
            ])
        ]),

        #####################################
        # Main content section
        #####################################
        html.Div(className='main-content', children=[
            #####################################
            # Panels Wrapper - Contains both top and bottom panels
            #####################################
            html.Div(className='panels-wrapper', children=[
                #####################################
                # Top Structure Plot Panel (Full Width)
                #####################################
                html.Div(className='top-panel-container', children=[
                    html.Div(className='top-panel-with-header', children=[
                        html.H2("Transcripts and Junctions Structure Plot", className='top-panel-header'),
                        html.Div(id='top-panel-body', className='top-panel', children=[
                            html.Div(className='graph-wrapper', children=[
                                # ATSE/Junction Structure Plot
                                html.Div(id='top-junction-structure-plot-container', className='atse-container', style={'position': 'relative'}, children=[
                                    html.Div(className='loading-container plot-container-full-height', id='top-structure-plot-container-style', style={'position': 'relative'}, children=[
                                        dcc.Loading(
                                            id="loading-top-structure-plot",
                                            type="default",
                                            color='#EDAE49',
                                            delay_show=0,
                                            delay_hide=200,
                                            children=[
                                                dcc.Graph(
                                                    id='atse-map',
                                                    figure=create_empty_atse_message("Select a gene to view splice junctions and exons"),
                                                    config={
                                                        'responsive': True,
                                                        'displayModeBar': True,
                                                        'scrollZoom': False,
                                                        'toImageButtonOptions': {
                                                            'format': 'svg',
                                                            'filename': 'isoformgazer_plot'
                                                        }
                                                    },
                                                    className='graph-full-size'
                                                )
                                            ]
                                        ),
                                        html.Div(id="atse-map-loading-message", className="custom-loading-message")
                                    ]),
                                    html.Div(
                                        id='junction-color-picker-popup',
                                        children=[
                                            html.Div(
                                                className='junction-header-container',
                                                children=[
                                                    html.Div(
                                                        className='junction-color-title',
                                                        children='Junction Color'
                                                    ),
                                                    html.Div(
                                                        className='junction-color-bar'
                                                    ),
                                                    html.Div(
                                                        id='junction-id-display',
                                                        children=''
                                                    )
                                                ]
                                            ),
                                            html.Div(
                                                style={
                                                    'marginBottom': '12px'
                                                }
                                            ),
                                            html.Div(
                                                className='junction-color-picker-container',
                                                children=[
                                                    daq.ColorPicker(
                                                        id='junction-individual-color-picker',
                                                        value={'hex': '#85929E'},
                                                        size=160,
                                                        theme=None,
                                                        className='color-picker'
                                                    )
                                                ]
                                            ),
                                            html.Div(
                                                className='junction-color-buttons-container',
                                                children=[
                                                    dbc.Button(
                                                        "Apply",
                                                        id='junction-color-apply-btn',
                                                        className='clear-filters-btn junction-color-btn'
                                                    ),
                                                    dbc.Button(
                                                        "Reset",
                                                        id='junction-color-reset-btn',
                                                        className='clear-filters-btn junction-color-btn'
                                                    )
                                                ]
                                            )
                                        ],
                                        style={'display': 'none'}
                                    ),
                                    # Store for currently selected junction
                                    dcc.Store(id='selected-junction-info', data={})
                                ]),
                                # Transcript Structure Plot (hidden by default)
                                html.Div(id='top-transcript-structure-plot-container', className='barplot-container hidden', children=[
                                    html.Div(className='loading-container plot-container-full-height', children=[
                                        dcc.Loading(
                                            id="loading-top-transcript-plot",
                                            type="default",
                                            color='#EDAE49',
                                            delay_show=0,
                                            delay_hide=200,
                                            children=[dcc.Graph(
                                                id='top-barplot',
                                                config={
                                                    'responsive': True,
                                                    'displayModeBar': True,
                                                    'scrollZoom': False,
                                                    'toImageButtonOptions': {
                                                        'format': 'svg',
                                                        'filename': 'isoformgazer_structure_plot'
                                                    }
                                                }
                                            )]
                                        ),
                                        html.Div(id="top-barplot-loading-message", className="custom-loading-message")
                                    ])
                                ]),
                                # Color picker popups (outside both plot containers so they're always visible)
                                html.Div(
                                    id='transcript-color-picker-popup',
                                    children=[
                                        html.Div(
                                            className='junction-header-container',
                                            children=[
                                                html.Div(
                                                    className='junction-color-title',
                                                    children='Transcript Color'
                                                ),
                                                html.Div(
                                                    className='junction-color-bar'
                                                ),
                                                html.Div(
                                                    id='transcript-id-display',
                                                    children=''
                                                )
                                            ]
                                        ),
                                        html.Div(
                                            style={
                                                'marginBottom': '12px'
                                            }
                                        ),
                                        html.Div(
                                            className='junction-color-picker-container',
                                            children=[
                                                daq.ColorPicker(
                                                    id='transcript-individual-color-picker',
                                                    value={'hex': '#2E86C1'},
                                                    size=160,
                                                    theme=None,
                                                    className='color-picker'
                                                )
                                            ]
                                        ),
                                        html.Div(
                                            className='junction-color-buttons-container',
                                            children=[
                                                dbc.Button(
                                                    "Apply",
                                                    id='transcript-color-apply-btn',
                                                    className='clear-filters-btn junction-color-btn'
                                                ),
                                                dbc.Button(
                                                    "Reset",
                                                    id='transcript-color-reset-btn',
                                                    className='clear-filters-btn junction-color-btn'
                                                )
                                            ]
                                        )
                                    ],
                                    style={'display': 'none'}
                                )
                            ])
                        ])
                    ])
                ]),
                #####################################
                # Bottom Panels Container (Original Side-by-Side Layout)
                #####################################
                html.Div(id='panels-container', className='panels-container', children=[
                #####################################
                # Data Panel 1: Isoform Data
                #####################################
                html.Div(className='panel-with-header', children=[
                    html.H2("ENCODE4 BULK RNA-SEQ LONG-READ DATA", className='panel-header'),
                    html.Div(id='left-panel', className='panel', children=[
                        html.Div(className='graph-wrapper', children=[
                        html.Div(id='isoform-clustergram-container', className='heatmap-container', children=[
                            html.Div(className='loading-container', children=[
                                dcc.Loading(
                                    id="loading-isoform-heatmap",
                                    type="default",
                                    color='#EDAE49',
                                    delay_show=0,
                                    delay_hide=200,
                                    children=[dcc.Graph(
                                        id='heatmap1',
                                        figure={
                                            'data': [],
                                            'layout': go.Layout(
                                                title={'text': 'Loading isoform expression data...', 'font': {'size': 14}},
                                                plot_bgcolor='white',
                                                margin=dict(l=40, r=40, t=40, b=40)
                                            )
                                        },
                                        className='graph-full-size',
                                        config={
                                            'responsive': True,
                                            'displayModeBar': True,
                                            'scrollZoom': False,
                                            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                            'toImageButtonOptions': {
                                                'format': 'svg',
                                                'filename': 'isoformgazer_long_read_clustergram'
                                            }
                                        }
                                    )]
                                ),
                                html.Div(id="heatmap1-loading-message", className="custom-loading-message")
                            ])
                        ])
                    ])
                ]),
                ]),

                #####################################
                # Data Panel 2: Junction Data
                #####################################
                html.Div(className='panel-with-header', children=[
                    html.H2("TABULA SAPIENS 2.0 PSEUDOBULKED SMART-SEQ2 SINGLE-CELL AND ALLEN BRAIN SINGLE NUCLEI FOR BRAIN DATA", className='panel-header'),
                    html.Div(id='right-panel', className='panel', children=[
                        html.Div(className='graph-wrapper', children=[
                        html.Div(id='junction-clustergram-container', className='heatmap-container', children=[
                            html.Div(className='loading-container', children=[
                                dcc.Loading(
                                    id="loading-junction-heatmap",
                                    type="default",
                                    color='#EDAE49',
                                    delay_show=0,
                                    delay_hide=200,
                                    children=[
                                        dcc.Graph(
                                            id='heatmap2',
                                            figure={
                                                'data': [],
                                                'layout': go.Layout(
                                                    title={'text': 'Loading junction usage data...', 'font': {'size': 14}},
                                                    plot_bgcolor='white',
                                                    margin=dict(l=40, r=40, t=40, b=40)
                                                )
                                            },
                                            config={
                                                'responsive': True,
                                                'displayModeBar': True,
                                                'scrollZoom': False,
                                                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                                'toImageButtonOptions': {
                                                    'format': 'svg',
                                                    'filename': 'isoformgazer_short_read_clustergram'
                                                }
                                            },
                                            className='graph-full-size'
                                        )
                                    ]
                                ),
                                html.Div(id="heatmap2-loading-message", className="custom-loading-message")
                            ])
                        ])
                    ])
                ]),
                ])
            ])
            ]),
            #####################################
            # Master Tables Section (Outside Panels)
            #####################################
            html.Div(className='tables-section', children=[
                html.Div(className='table-container', id='table1-container', children=[
                    html.Div(className='table-header-controls', children=[
                        html.Div(children=[
                            dbc.Button(
                                "Apply Filters",
                                id='apply-left-filters',
                                color="secondary",
                                size="sm",
                                className="clear-filters-btn",
                                disabled=True,
                                style={'marginRight': '8px'}
                            ),
                            dbc.Button(
                                "Clear Filters",
                                id='clear-left-filters',
                                color="secondary",
                                size="sm",
                                className="clear-filters-btn",
                                disabled=True
                            )
                        ]),
                        html.Div(children=[
                            dbc.Button(
                                "Download Expression Data",
                                id='download-left-expression-button',
                                color="secondary",
                                size="sm",
                                className="clear-filters-btn",
                                style={'marginRight': '8px'}
                            ),
                            dcc.Download(id='download-left-expression'),
                            dbc.Button(
                                "Download Master Table CSV",
                                id='download-left-table-button',
                                color="secondary",
                                size="sm",
                                className="clear-filters-btn"
                            ),
                            dcc.Download(id='download-left-table')
                        ], style={'marginLeft': 'auto', 'display': 'flex', 'gap': '8px'})
                    ]),
                    # Isoform table filter error popup
                    html.Div(
                        id='left-table-error-popup',
                        className='filter-error-popup hidden',
                        style={'display': 'none'},
                        children=[]
                    ),
                    dcc.Loading(
                        id="loading-left-table",
                        type="default",
                        color='#EDAE49',
                        delay_show=0,
                        delay_hide=100,
                        children=[left_data_table]
                    )
                ]),
                html.Div(className='table-container', id='table2-container', children=[
                    html.Div(className='table-header-controls', children=[
                        html.Div(children=[
                            dbc.Button(
                                "Apply Filters",
                                id='apply-right-filters',
                                color="secondary",
                                size="sm",
                                className="clear-filters-btn",
                                disabled=True,
                                style={'marginRight': '8px'}
                            ),
                            dbc.Button(
                                "Clear Filters",
                                id='clear-right-filters',
                                color="secondary",
                                size="sm",
                                className="clear-filters-btn",
                                disabled=True
                            )
                        ]),
                        html.Div(children=[
                            dbc.Button(
                                "Download PSI Data",
                                id='download-right-psi-button',
                                color="secondary",
                                size="sm",
                                className="clear-filters-btn",
                                style={'marginRight': '8px'}
                            ),
                            dcc.Download(id='download-right-psi'),
                            dbc.Button(
                                "Download Master Table CSV",
                                id='download-right-table-button',
                                color="secondary",
                                size="sm",
                                className="clear-filters-btn"
                            ),
                            dcc.Download(id='download-right-table')
                        ], style={'marginLeft': 'auto', 'display': 'flex', 'gap': '8px'})
                    ]),
                    # Junction table filter error popup
                    html.Div(
                        id='right-table-error-popup',
                        className='filter-error-popup hidden',
                        style={'display': 'none'},
                        children=[]
                    ),
                    dcc.Loading(
                        id="loading-right-table",
                        type="default",
                        color='#EDAE49',
                        delay_show=0,
                        delay_hide=100,
                        children=[right_data_table]
                    )
                ])
            ])
        ])
    ])
])

app.layout.children.extend([
    dcc.Store(
        id='filtered-isoform-store',
        data=default_gene_cache.get('filtered_isoform_ids', []) if default_gene_cache else []
    ),
    dcc.Store(
        id='filtered-junction-store',
        data=default_gene_cache.get('filtered_junction_ids', []) if default_gene_cache else []
    ),
    dcc.Store(
        id='isoform-full-data-store',
        data=default_gene_cache.get('isoform_full_data', []) if default_gene_cache else []
    ),
    dcc.Store(
        id='junction-full-data-store',
        data=default_gene_cache.get('junction_full_data', []) if default_gene_cache else []
    ),
    dcc.Store(id='table-callback-prevention', data=False),
    dcc.Store(id='initial-loading-complete', data=False),
    dcc.Store(id='exon-color-store', data='#2E86C1'),
    dcc.Store(id='junction-color-store', data='#85929E'),
    dcc.Store(id='individual-junction-colors', data={}),
    dcc.Store(id='individual-transcript-colors', data={}),
    dcc.Store(id='viewport-dimensions', data={'width': 1920, 'height': 1080}),
    dcc.Store(id='selected-transcript-info', data={}),
    dcc.Store(id='loading-progress-store', data=0),
    dcc.Store(id='left-table-validation-store', data={'valid': True, 'errors': {}}),
    dcc.Store(id='right-table-validation-store', data={'valid': True, 'errors': {}}),
    dcc.Store(id='left-table-applied-filter-store', data=''),
    dcc.Store(id='right-table-applied-filter-store', data=''),
    dcc.Store(id='gtf-hash-results-store', data=[]),
    dcc.Store(id='all-gene-options-store', data=get_all_gene_options(db_path)),
    dcc.Store(id='cache-used-store', data=cache_loaded_from_disk),
    dcc.Interval(
        id='loading-delay-interval',
        interval=1000,
        n_intervals=0,
        max_intervals=1,
        disabled=True
    ),
    dcc.Interval(id='progress-update-interval', interval=50, n_intervals=0, disabled=False)
])

#######################################################################
# CALLBACKS
#######################################################################
#######################################################################
# EXON COLOR PICKER CALLBACK
#######################################################################
@app.callback(
    dash.dependencies.Output('exon-color-store', 'data'),
    [dash.dependencies.Input('exon-color-picker', 'value')]
)
def update_exon_color_store(color_value):
    """Update exon color store when color picker changes"""
    if color_value and 'hex' in color_value:
        return color_value['hex']
    return '#2E86C1'

@app.callback(
    dash.dependencies.Output('junction-color-store', 'data'),
    [dash.dependencies.Input('junction-color-picker', 'value')]
)
def update_junction_color_store(color_value):
    """Update junction color store when color picker changes"""
    if color_value and 'hex' in color_value:
        return color_value['hex']
    return '#85929E'


#######################################################################
# INDIVIDUAL JUNCTION COLOR PICKER CALLBACKS
#######################################################################
@app.callback(
    [dash.dependencies.Output('selected-junction-info', 'data'),
     dash.dependencies.Output('junction-individual-color-picker', 'value'),
     dash.dependencies.Output('junction-color-picker-popup', 'style'),
     dash.dependencies.Output('transcript-color-picker-popup', 'style', allow_duplicate=True)],
    [dash.dependencies.Input('atse-map', 'clickData')],
    [dash.dependencies.State('individual-junction-colors', 'data'),
     dash.dependencies.State('junction-color-store', 'data'),
     dash.dependencies.State('viewport-dimensions', 'data')],
    prevent_initial_call=True
)
def handle_junction_click(clickData, individual_colors, global_junction_color, viewport_dims):
    """
    Handle junction clicks to open the color picker popup.
    Extract junction ID from the hover text.
    """
    if not clickData or 'points' not in clickData or len(clickData['points']) == 0:
        raise PreventUpdate

    point = clickData['points'][0]

    # Extract junction ID from the hover text
    text = point.get('text', '')
    if 'Junction ID:' not in text:
        raise PreventUpdate

    junction_id = None
    parts = text.split('<br>')
    for part in parts:
        if 'Junction ID:' in part:
            junction_id = part.replace('Junction ID:', '').strip()
            break

    if not junction_id:
        raise PreventUpdate

    # Get junction coordinates from point
    x = point.get('x')
    y = point.get('y')

    if x is None or y is None:
        raise PreventUpdate

    if individual_colors: 
        current_color_hex = individual_colors.get(junction_id, global_junction_color)
    else: 
        current_color_hex = global_junction_color

    current_color = {'hex': current_color_hex}

    selected_junction = {
        'id': junction_id,
        'x': x,
        'y': y
    }

    # need bounding box of clicked junction point to position the popup
    bbox = point.get('bbox', {})
    popup_width = 280
    popup_height = 300
    offset_x = 15
    offset_y = 5

    x_pos = bbox.get('x1', 0) + offset_x
    y_pos = bbox.get('y0', 0) + offset_y

    # Add viewport boundary checking to prevent popup from going off-screen
    viewport_width = viewport_dims.get('width', 1920) if viewport_dims else 1920
    viewport_height = viewport_dims.get('height', 1080) if viewport_dims else 1080

    if x_pos + popup_width > viewport_width:
        # Position to the left of the junction instead
        x_pos = max(10, bbox.get('x0', 0) - popup_width - offset_x)

    if y_pos + popup_height > viewport_height:
        y_pos = max(10, bbox.get('y1', 0) - popup_height - offset_y)

    x_pos = max(10, x_pos)
    y_pos = max(10, y_pos)

    popup_style = {
        'position': 'fixed',
        'zIndex': 10000,
        'backgroundColor': '#ffffff',
        'border': '2px solid #EDAE49',
        'borderRadius': '10px',
        'padding': '16px',
        'boxShadow': '0 8px 24px rgba(90, 42, 145, 0.4)',
        'display': 'block',
        'width': '280px',
        'left': f"{x_pos}px",
        'top': f"{y_pos}px"
    }

    # Hide transcript popup
    transcript_popup_style = {'display': 'none'}

    return selected_junction, current_color, popup_style, transcript_popup_style


@app.callback(
    dash.dependencies.Output('junction-id-display', 'children'),
    [dash.dependencies.Input('selected-junction-info', 'data')],
    prevent_initial_call=True
)
def update_junction_id_display(selected_junction):
    """
    Update the junction ID display in the popup.
    """
    if not selected_junction or 'id' not in selected_junction:
        return ''

    junction_id = selected_junction['id']
    return junction_id


@app.callback(
    [dash.dependencies.Output('individual-junction-colors', 'data'),
     dash.dependencies.Output('junction-color-picker-popup', 'style', allow_duplicate=True)],
    [dash.dependencies.Input('junction-color-apply-btn', 'n_clicks'),
     dash.dependencies.Input('junction-color-reset-btn', 'n_clicks'),
     dash.dependencies.Input('junction-color-store', 'data')],
    [dash.dependencies.State('selected-junction-info', 'data'),
     dash.dependencies.State('junction-individual-color-picker', 'value'),
     dash.dependencies.State('individual-junction-colors', 'data')],
    prevent_initial_call=True
)
def manage_individual_junction_colors(apply_clicks, reset_clicks, global_color,
                                      selected_junction, color_value, individual_colors):
    """
    Manage individual junction colors: apply custom colors, reset to global,
    or clear all when global color changes.
    """
    individual_colors = individual_colors or {}
    triggered_id = callback_context.triggered_id if callback_context.triggered else None
    hidden_style = {
        'position': 'fixed',
        'zIndex': 10000,
        'backgroundColor': '#ffffff',
        'border': '2px solid #EDAE49',
        'borderRadius': '10px',
        'padding': '16px',
        'boxShadow': '0 8px 24px rgba(90, 42, 145, 0.4)',
        'display': 'none',
        'width': '280px'
    }

    if triggered_id == 'junction-color-apply-btn':
        if selected_junction and 'id' in selected_junction:
            if color_value and 'hex' in color_value:
                individual_colors[selected_junction['id']] = color_value['hex']
            return individual_colors, hidden_style

    elif triggered_id == 'junction-color-reset-btn':
        if selected_junction and 'id' in selected_junction:
            individual_colors.pop(selected_junction['id'], None)
            return individual_colors, hidden_style

    elif triggered_id == 'junction-color-store':
        return {}, hidden_style

    raise PreventUpdate


@app.callback(
    dash.dependencies.Output('atse-map', 'clickData'),
    [dash.dependencies.Input('junction-color-apply-btn', 'n_clicks'),
     dash.dependencies.Input('junction-color-reset-btn', 'n_clicks')],
    prevent_initial_call=True
)
def reset_click_data(apply_clicks, reset_clicks):
    """
    Reset clickData after color operations to allow re-clicking the same junction.
    """
    return None


#######################################################################
# INDIVIDUAL TRANSCRIPT COLOR PICKER CALLBACKS
#######################################################################
@app.callback(
    [dash.dependencies.Output('selected-transcript-info', 'data'),
     dash.dependencies.Output('transcript-individual-color-picker', 'value'),
     dash.dependencies.Output('transcript-color-picker-popup', 'style'),
     dash.dependencies.Output('junction-color-picker-popup', 'style', allow_duplicate=True)],
    [dash.dependencies.Input('atse-map', 'clickData'),
     dash.dependencies.Input('top-barplot', 'clickData')],
    [dash.dependencies.State('individual-transcript-colors', 'data'),
     dash.dependencies.State('exon-color-store', 'data'),
     dash.dependencies.State('viewport-dimensions', 'data')],
    prevent_initial_call=True
)
def handle_transcript_click(atse_clickData, transcript_clickData, individual_colors, global_exon_color, viewport_dims):
    """
    Handle transcript/exon clicks to open the color picker popup.
    Extract transcript/isoform ID from the hover text.
    Works for both junction+transcript plot (atse-map) and transcript-only plot (top-barplot).
    """
    # Determine which plot was clicked
    if not callback_context.triggered:
        raise PreventUpdate

    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'atse-map':
        clickData = atse_clickData
    elif triggered_id == 'top-barplot':
        clickData = transcript_clickData
    else:
        raise PreventUpdate

    if not clickData or 'points' not in clickData or len(clickData['points']) == 0:
        raise PreventUpdate

    point = clickData['points'][0]

    # Extract isoform ID from the hover text
    text = point.get('text', '')
    if 'Isoform ID:' not in text:
        raise PreventUpdate

    isoform_id = None
    transcript_id = None
    parts = text.split('<br>')
    for part in parts:
        if 'Isoform ID:' in part:
            isoform_id = part.replace('Isoform ID:', '').strip()
        elif 'Transcript ID:' in part or 'Transcript:' in part:
            transcript_id = part.replace('Transcript ID:', '').replace('Transcript:', '').strip()

    if not isoform_id:
        raise PreventUpdate

    # Get coordinates from point
    x = point.get('x')
    y = point.get('y')

    if x is None or y is None:
        raise PreventUpdate

    if individual_colors:
        current_color_hex = individual_colors.get(isoform_id, global_exon_color)
    else:
        current_color_hex = global_exon_color

    current_color = {'hex': current_color_hex}

    selected_transcript = {
        'id': isoform_id,
        'transcript_id': transcript_id or isoform_id,
        'x': x,
        'y': y
    }

    # need bounding box of clicked point to position the popup
    bbox = point.get('bbox', {})

    # Position popup to the right of the top of the clicked element
    popup_width = 280
    popup_height = 300
    offset_x = 15
    offset_y = 5

    # Get desired position (to the right of the top)
    x_pos = bbox.get('x1', 0) + offset_x
    y_pos = bbox.get('y0', 0) + offset_y

    # Add viewport boundary checking
    viewport_width = viewport_dims.get('width', 1920) if viewport_dims else 1920
    viewport_height = viewport_dims.get('height', 1080) if viewport_dims else 1080

    # Ensure popup doesn't overflow right edge
    if x_pos + popup_width > viewport_width:
        x_pos = max(10, bbox.get('x0', 0) - popup_width - offset_x)

    # Ensure popup doesn't overflow bottom edge
    if y_pos + popup_height > viewport_height:
        y_pos = max(10, bbox.get('y1', 0) - popup_height - offset_y)

    # Ensure minimum distance from edges
    x_pos = max(10, x_pos)
    y_pos = max(10, y_pos)

    transcript_popup_style = {
        'position': 'fixed',
        'zIndex': 10000,
        'backgroundColor': '#ffffff',
        'border': '2px solid #EDAE49',
        'borderRadius': '10px',
        'padding': '16px',
        'boxShadow': '0 8px 24px rgba(90, 42, 145, 0.4)',
        'display': 'block',
        'width': '280px',
        'left': f"{x_pos}px",
        'top': f"{y_pos}px"
    }

    # Hide junction popup
    junction_popup_style = {'display': 'none'}

    return selected_transcript, current_color, transcript_popup_style, junction_popup_style


@app.callback(
    dash.dependencies.Output('transcript-id-display', 'children'),
    [dash.dependencies.Input('selected-transcript-info', 'data')],
    prevent_initial_call=True
)
def update_transcript_id_display(selected_transcript):
    """
    Update the transcript ID display in the popup.
    """
    if not selected_transcript or 'transcript_id' not in selected_transcript:
        return ''

    transcript_id = selected_transcript['transcript_id']
    return transcript_id


@app.callback(
    [dash.dependencies.Output('individual-transcript-colors', 'data'),
     dash.dependencies.Output('transcript-color-picker-popup', 'style', allow_duplicate=True)],
    [dash.dependencies.Input('transcript-color-apply-btn', 'n_clicks'),
     dash.dependencies.Input('transcript-color-reset-btn', 'n_clicks'),
     dash.dependencies.Input('exon-color-store', 'data')],
    [dash.dependencies.State('selected-transcript-info', 'data'),
     dash.dependencies.State('transcript-individual-color-picker', 'value'),
     dash.dependencies.State('individual-transcript-colors', 'data')],
    prevent_initial_call=True
)
def manage_individual_transcript_colors(apply_clicks, reset_clicks, global_color,
                                       selected_transcript, color_value, individual_colors):
    """
    Manage individual transcript colors (apply, reset, or clear all on global color change).
    """
    individual_colors = individual_colors or {}
    triggered_id = callback_context.triggered_id if callback_context.triggered else None

    hidden_style = {
        'position': 'fixed',
        'zIndex': 10000,
        'backgroundColor': '#ffffff',
        'border': '2px solid #EDAE49',
        'borderRadius': '10px',
        'padding': '16px',
        'boxShadow': '0 8px 24px rgba(90, 42, 145, 0.4)',
        'display': 'none',
        'width': '280px'
    }

    if triggered_id == 'transcript-color-apply-btn':
        if selected_transcript and 'id' in selected_transcript:
            if color_value and 'hex' in color_value:
                individual_colors[selected_transcript['id']] = color_value['hex']
            return individual_colors, hidden_style

    elif triggered_id == 'transcript-color-reset-btn':
        if selected_transcript and 'id' in selected_transcript:
            individual_colors.pop(selected_transcript['id'], None)
            return individual_colors, hidden_style

    elif triggered_id == 'exon-color-store':
        return {}, hidden_style

    raise PreventUpdate


@app.callback(
    [dash.dependencies.Output('atse-map', 'clickData', allow_duplicate=True),
     dash.dependencies.Output('top-barplot', 'clickData', allow_duplicate=True)],
    [dash.dependencies.Input('transcript-color-apply-btn', 'n_clicks'),
     dash.dependencies.Input('transcript-color-reset-btn', 'n_clicks')],
    prevent_initial_call=True
)
def reset_transcript_click_data(apply_clicks, reset_clicks):
    """
    Reset clickData after transcript color operations to allow re-clicking the same transcript.
    Resets both junction+transcript plot and transcript-only plot.
    """
    return None, None


#######################################################################
# LOADING SCREEN PROGRESS BAR CALLBACKS
#######################################################################
@app.callback(
    [dash.dependencies.Output('loading-progress-store', 'data'),
     dash.dependencies.Output('progress-update-interval', 'disabled')],
    [dash.dependencies.Input('progress-update-interval', 'n_intervals')],
    [dash.dependencies.State('initial-loading-complete', 'data')]
)
def update_loading_progress(progress_intervals, loading_complete):
    """
    Updates white loading progress circle for loading screen with steady time-based progression.
    (with cache actual load time is ~50ms, without cache actual load time is 1-2 seconds)
    """
    if loading_complete:
        return 100, True

    progress_rate = 1.667

    new_progress = min(progress_intervals * progress_rate, 100)
    disable_interval = (new_progress >= 100)

    return new_progress, disable_interval


@app.callback(
    dash.dependencies.Output('progress-circle', 'figure'),
    [dash.dependencies.Input('loading-progress-store', 'data')]
)
def update_progress_bar(progress):
    """Updates loading screen circular progress bar based on
    timesteps defined in update_loading_progress() callback"""
    radius = 1.5

    fig = go.Figure()

    theta = np.linspace(0, 2 * np.pi, 200)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)

    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        mode='lines',
        line=dict(color='rgba(255,255,255,0.1)', width=4), 
        showlegend=False,
        hoverinfo='none'
    ))

    x_progress = []
    y_progress = []

    # Only render progress arc if progress is greater than 1%
    if progress > 1.0:
        num_points = max(int(progress * 1.5), 5)  
        progress_theta = np.linspace(-np.pi/2, -np.pi/2 + 2 * np.pi * (progress / 100), num_points)
        x_progress = radius * np.cos(progress_theta)
        y_progress = radius * np.sin(progress_theta)

    if len(x_progress) > 0 and progress > 1.0:
        fig.add_trace(go.Scatter(
            x=x_progress, y=y_progress,
            mode='lines',
            line=dict(
                color='rgba(255,255,255,0.15)',
                width=32
            ),
            showlegend=False,
            hoverinfo='none'
        ))

        fig.add_trace(go.Scatter(
            x=x_progress, y=y_progress,
            mode='lines',
            line=dict(
                color='rgba(255,255,255,0.3)',
                width=20
            ),
            showlegend=False,
            hoverinfo='none'
        ))

        fig.add_trace(go.Scatter(
            x=x_progress, y=y_progress,
            mode='lines',
            line=dict(
                color='rgba(255,255,255,1.0)',
                width=10
            ),
            showlegend=False,
            hoverinfo='none'
        ))

    fig.update_layout(
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin={'l': 20, 'r': 20, 't': 20, 'b': 20},
        xaxis={
            'visible': False,
            'range': [-2.0, 2.0],
            'scaleanchor': 'y',
            'scaleratio': 1
        },
        yaxis={
            'visible': False,
            'range': [-2.0, 2.0]
        },
        width=450,
        height=450,
        hovermode=False
    )

    return fig

#######################################################################
# FILTER VALIDATION CALLBACKS
#######################################################################
@app.callback(
    [dash.dependencies.Output('left-table-error-popup', 'style'),
     dash.dependencies.Output('left-table-error-popup', 'children'),
     dash.dependencies.Output('left-table-validation-store', 'data')],
    [dash.dependencies.Input('left_data_table', 'filter_query')],
    prevent_initial_call=True
)
def validate_left_table_filters(current_filter_query):
    """Validate isoform table filters and store validation results"""
    if not current_filter_query:
        return {'display': 'none'}, [], {'valid': True, 'errors': {}, 'query': ''}
    
    is_valid, errors = validate_filter_input(db_path, 'isoforms', current_filter_query)
    
    if not is_valid:
        error_messages = []
        for column, message in errors.items():
            error_messages.append(html.Div([
                html.Strong(f"{column.replace('_', ' ').title()}: "),
                message
            ], className='error-message'))
        
        return {'display': 'block'}, error_messages, {'valid': False, 'errors': errors, 'query': current_filter_query}
    
    else:
        return {'display': 'none'}, [], {'valid': True, 'errors': {}, 'query': current_filter_query}


@app.callback(
    [dash.dependencies.Output('right-table-error-popup', 'style'),
     dash.dependencies.Output('right-table-error-popup', 'children'),
     dash.dependencies.Output('right-table-validation-store', 'data')],
    [dash.dependencies.Input('right_data_table', 'filter_query')],
    prevent_initial_call=True
)
def validate_right_table_filters(current_filter_query):
    """Validate junction table filters and store validation results"""
    if not current_filter_query:
        return {'display': 'none'}, [], {'valid': True, 'errors': {}, 'query': ''}
    
    is_valid, errors = validate_filter_input(db_path, 'junctions', current_filter_query)
    
    if not is_valid:
        error_messages = []
        for column, message in errors.items():
            error_messages.append(html.Div([
                html.Strong(f"{column.replace('_', ' ').title()}: "),
                message
            ], className='error-message'))
        
        return {'display': 'block'}, error_messages, {'valid': False, 'errors': errors, 'query': current_filter_query}
    
    else:
        return {'display': 'none'}, [], {'valid': True, 'errors': {}, 'query': current_filter_query}


#######################################################################
# FILTER APPLICATION CALLBACKS (Apply Filters Button)
#######################################################################
@app.callback(
    dash.dependencies.Output('left-table-applied-filter-store', 'data'),
    [dash.dependencies.Input('apply-left-filters', 'n_clicks'),
     dash.dependencies.Input('clear-left-filters', 'n_clicks'),
     dash.dependencies.Input('gene-search-dropdown', 'value')],
    [dash.dependencies.State('left_data_table', 'filter_query'),
     dash.dependencies.State('left-table-validation-store', 'data')]
)
def apply_left_filter_store(apply_clicks, clear_clicks, selected_gene, current_filter, validation_data):
    """Apply or clear filters for left table - updates the applied filter store"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return ''

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Clear filters when gene changes or clear button clicked
    if trigger_id == 'gene-search-dropdown' or trigger_id == 'clear-left-filters':
        return ''

    # Apply current filter when apply button clicked (only if valid)
    if trigger_id == 'apply-left-filters':
        if validation_data and validation_data.get('valid', True):
            return current_filter or ''
        else:
            raise PreventUpdate

    return ''


@app.callback(
    dash.dependencies.Output('right-table-applied-filter-store', 'data'),
    [dash.dependencies.Input('apply-right-filters', 'n_clicks'),
     dash.dependencies.Input('clear-right-filters', 'n_clicks'),
     dash.dependencies.Input('gene-search-dropdown', 'value')],
    [dash.dependencies.State('right_data_table', 'filter_query'),
     dash.dependencies.State('right-table-validation-store', 'data')]
)
def apply_right_filter_store(apply_clicks, clear_clicks, selected_gene, current_filter, validation_data):
    """Apply or clear filters for right table - updates the applied filter store"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return ''

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'gene-search-dropdown' or trigger_id == 'clear-right-filters':
        return ''

    if trigger_id == 'apply-right-filters':
        if validation_data and validation_data.get('valid', True):
            return current_filter or ''
        else:
            raise PreventUpdate

    return ''


#######################################################################
# INITIAL LOADING SCREEN CALLBACKS
#######################################################################
@app.callback(
    [dash.dependencies.Output('loading-overlay', 'className'),
     dash.dependencies.Output('initial-loading-complete', 'data'),
     dash.dependencies.Output('loading-delay-interval', 'disabled')],
    [dash.dependencies.Input('isoform-full-data-store', 'data'),
     dash.dependencies.Input('junction-full-data-store', 'data'),
     dash.dependencies.Input('loading-delay-interval', 'n_intervals')],
    [dash.dependencies.State('initial-loading-complete', 'data')]
)
def hide_loading_screen(isoform_data, junction_data, timer_intervals, loading_complete):
    """Hide the initial loading screen with a 2-second delay after initial data has loaded"""
    if loading_complete:
        return 'loading-overlay hidden', True, True

    data_loaded = (isoform_data is not None) and (junction_data is not None)
    if data_loaded and timer_intervals == 0:
        return 'loading-overlay', False, False

    # CRUCIAL: moves position of loading overlay to back to prevent interaction issues with master tables once app layout loaded
    if data_loaded and timer_intervals > 0:
        return 'loading-overlay hidden', True, True

    # Data not loaded yet so keep showing loading screen
    return 'loading-overlay', False, True

#######################################################################
# MASTER TABLE FILTERING CALLBACKS
#######################################################################
@app.callback(
    [dash.dependencies.Output('filtered-isoform-store', 'data'),
     dash.dependencies.Output('filtered-junction-store', 'data')],
    [dash.dependencies.Input('isoform-full-data-store', 'data'),
     dash.dependencies.Input('junction-full-data-store', 'data'),
     dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('left-table-applied-filter-store', 'data'),
     dash.dependencies.Input('right-table-applied-filter-store', 'data'),
     dash.dependencies.Input('species-dropdown', 'value')]
)
def update_filtered_data_stores(isoform_full_data, junction_full_data, selected_gene, isoform_filter_query, junction_filter_query, species):
    """Store ALL filtered transcript/junction IDs from FULL datasets with transcript-based junction filtering"""

    try:
        has_isoform_filters = bool(isoform_filter_query and isoform_filter_query.strip())
        has_junction_filters = bool(junction_filter_query and junction_filter_query.strip())

        filtered_transcript_ids = []
        if isoform_full_data:
            filtered_transcript_ids = [row.get('id', '') for row in isoform_full_data if row.get('id')]

        filtered_junction_ids = []
        if junction_full_data:
            filtered_junction_ids = [row.get('junction_id', '') for row in junction_full_data if row.get('junction_id')]

        # Handle bidirectional filtering between transcripts and junctions when filters are applied
        # Start with the filtered IDs from the full data stores, apply bidirectional filtering (intersection of filters focusing on coords)
        final_transcript_ids = filtered_transcript_ids
        final_junction_ids = filtered_junction_ids

        if has_isoform_filters or has_junction_filters:
            # Isoform filtering → Junction filtering
            if selected_gene and has_isoform_filters and filtered_transcript_ids:
                try:
                    transcript_based_junction_ids = filter_junctions_by_transcripts(
                        db_path, selected_gene, filtered_transcript_ids, species
                    )
                    # If we also have junction filters, intersect them
                    if has_junction_filters and transcript_based_junction_ids:
                        final_junction_ids = list(set(filtered_junction_ids) & set(transcript_based_junction_ids))
                    elif transcript_based_junction_ids:
                        # Only isoform filter active
                        final_junction_ids = transcript_based_junction_ids
                except Exception as e:
                    print(f"Error in transcript-based junction filtering: {e}")

            # Junction filtering → Isoform filtering
            if has_junction_filters and filtered_junction_ids:
                try:
                    junction_based_transcript_ids = filter_transcripts_by_junctions(
                        db_path, filtered_junction_ids, species
                    )
                    # If we also have isoform filters, intersect them
                    if has_isoform_filters and junction_based_transcript_ids:
                        final_transcript_ids = list(set(filtered_transcript_ids) & set(junction_based_transcript_ids))
                    elif junction_based_transcript_ids:
                        # Only junction filter active
                        final_transcript_ids = junction_based_transcript_ids
                except Exception as e:
                    print(f"Error in junction-based transcript filtering: {e}")

        filtered_transcript_ids = final_transcript_ids
        filtered_junction_ids = final_junction_ids
        
        return filtered_transcript_ids, filtered_junction_ids
    
    except Exception as e:
        print(f"Error updating filtered data stores: {e}")
        return [], []


@app.callback(
    [dash.dependencies.Output('left_data_table', 'filter_query', allow_duplicate=True),
     dash.dependencies.Output('right_data_table', 'filter_query', allow_duplicate=True)],
    [dash.dependencies.Input('clear-left-filters', 'n_clicks'),
     dash.dependencies.Input('clear-right-filters', 'n_clicks'),
     dash.dependencies.Input('gene-search-dropdown', 'value')],
    [dash.dependencies.State('left_data_table', 'filter_query'),
     dash.dependencies.State('right_data_table', 'filter_query')],
    prevent_initial_call=True
)
def clear_filter_inputs(left_clicks, right_clicks, selected_gene, left_filter, right_filter):
    """Clear the filter input fields when clear buttons are clicked or gene changes"""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'clear-left-filters':
        return '', right_filter

    elif button_id == 'clear-right-filters':
        return left_filter, ''

    elif button_id == 'gene-search-dropdown':
        return '', ''

    return left_filter, right_filter


@app.callback(
    [dash.dependencies.Output('apply-left-filters', 'disabled'),
     dash.dependencies.Output('apply-right-filters', 'disabled')],
    [dash.dependencies.Input('left_data_table', 'filter_query'),
     dash.dependencies.Input('right_data_table', 'filter_query'),
     dash.dependencies.Input('left-table-applied-filter-store', 'data'),
     dash.dependencies.Input('right-table-applied-filter-store', 'data')]
)
def update_apply_button_states(left_filter, right_filter, left_applied, right_applied):
    """Enable Apply Filters buttons only when there's a pending filter that differs from applied"""
    # Enable Apply button if there's a filter typed AND it's different from what's currently applied
    left_disabled = not left_filter or left_filter.strip() == '' or left_filter == left_applied
    right_disabled = not right_filter or right_filter.strip() == '' or right_filter == right_applied
    return left_disabled, right_disabled


@app.callback(
    [dash.dependencies.Output('clear-left-filters', 'disabled'),
     dash.dependencies.Output('clear-right-filters', 'disabled')],
    [dash.dependencies.Input('left-table-applied-filter-store', 'data'),
     dash.dependencies.Input('right-table-applied-filter-store', 'data')]
)
def update_clear_button_states(left_applied_filter, right_applied_filter):
    """Enable Clear Filters buttons only when filters are actually applied"""
    left_disabled = not left_applied_filter or left_applied_filter.strip() == ''
    right_disabled = not right_applied_filter or right_applied_filter.strip() == ''
    return left_disabled, right_disabled


##############################################################################################
# Update all-gene-options-store when species changes
##############################################################################################
@app.callback(
    dash.dependencies.Output('all-gene-options-store', 'data'),
    [dash.dependencies.Input('species-dropdown', 'value')]
)
def update_all_gene_options_on_species_change(species):
    """Reload all gene options when species changes"""
    return get_all_gene_options(db_path, species)


##############################################################################################
# Reset gene selection when species changes
##############################################################################################
@app.callback(
    dash.dependencies.Output('gene-search-dropdown', 'value', allow_duplicate=True),
    [dash.dependencies.Input('species-dropdown', 'value')],
    [dash.dependencies.State('gene-search-dropdown', 'value')],
    prevent_initial_call=True
)
def reset_gene_on_species_change(species, current_gene):
    """Reset gene selection to default when species changes"""
    # AACS has data in both human and mouse
    return "AACS"


##############################################################################################
# CALLBACK FOR QUERYING BY GENE IN CONTROL PANEL ('Query' tab): if no search is performed, we
# show the first five gene names, but otherwise filter by the top ten matches to the current
# search string.
##############################################################################################
@app.callback(
    [dash.dependencies.Output('gene-search-dropdown', 'options'),
     dash.dependencies.Output('gene-search-dropdown', 'value')],
    [dash.dependencies.Input('gene-search-dropdown', 'search_value'),
     dash.dependencies.Input('species-dropdown', 'value')],
    [dash.dependencies.State('gene-search-dropdown', 'value'),
     dash.dependencies.State('all-gene-options-store', 'data')]
)
def update_gene_options(search_value, species, current_value, all_gene_options):
    """Update gene options using client-side filtering from cached data"""
    # If no cached options available, fall back to database query (shouldn't happen)
    if not all_gene_options:
        all_gene_options = get_all_gene_options(db_path, species)

    if not search_value:
        options = all_gene_options[:10]
        aacs_option = {'label': 'AACS', 'value': 'AACS', 'search': 'aacs'}
        if not any(opt['value'] == 'AACS' for opt in options):
            options.insert(0, aacs_option)
    else:
        # Client-side filtering using the pre-computed search string
        search_lower = search_value.lower()
        filtered = [opt for opt in all_gene_options if search_lower in opt.get('search', '')]
        options = filtered[:50]

    # Bug fix for figures refreshing every time user interacts with gene dropdown: 
    # - If species changed, preserve current value if it exists in new options
    # - If user is just typing (search_value changed), don't update the value
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if current_value is None:
        return options, 'AACS'
    
    if triggered_id == 'species-dropdown':
        option_values = [opt['value'] for opt in options]
        if current_value in option_values:
            return options, current_value
        else:
            return options, 'AACS'

    option_values = [opt['value'] for opt in options]

    if current_value in option_values:
        return options, dash.no_update
    else:
        # Find the current value in all options and add it to the list
        current_option = next((opt for opt in all_gene_options if opt['value'] == current_value), None)
        if current_option:
            options = [current_option] + [opt for opt in options if opt['value'] != current_value]
        return options, dash.no_update


@app.callback(
    [dash.dependencies.Output('hide-junctions-toggle', 'value'),
     dash.dependencies.Output('color-junctions-by-psi-toggle', 'value'),
     dash.dependencies.Output('color-by-abundance-toggle', 'value'),
     dash.dependencies.Output('abundance-color-type-radio', 'value'),
     dash.dependencies.Output('tissue-abundance-dropdown', 'value', allow_duplicate=True),
     dash.dependencies.Output('organ-abundance-dropdown', 'value', allow_duplicate=True),
     dash.dependencies.Output('gridlines-toggle', 'value')],
    [dash.dependencies.Input('gene-search-dropdown', 'value')],
    prevent_initial_call=True
)
def reset_custom_settings_on_gene_change(selected_gene):
    """Reset all custom settings to defaults when a new gene is selected (except colorscales)"""
    return (
        False,      # hide-junctions-toggle: show junctions by default
        True,       # color-junctions-by-psi-toggle: on by default
        True,       # color-by-abundance-toggle: on by default
        'average',  # abundance-color-type-radio: average by default
        None,       # tissue-abundance-dropdown: no tissue selected
        None,       # organ-abundance-dropdown: no organ selected
        False       # gridlines-toggle: off by default
    )


######################################################################
# SUMMARY BLOCK CALLBACKS
######################################################################
def format_protein_category(val):
    if val is None:
        return html.Span("N/A", className='summary-value')
    categories = [cat.strip() for cat in str(val).split(';') if cat.strip()]
    if len(categories) <= 1:
        return html.Span(format_value(val), className='summary-value')

    # Create a div with multiple lines for multiple categories
    return html.Div([
        html.Div(cat, className='summary-line')
        for cat in categories
    ], className='summary-value summary-multiline')


def format_value(val, field_name=None):
    if val is None:
        return "N/A"
    
    if field_name:
        if val is not None: 
            return str(int(val))
        else:
            return "N/A"
    
    if isinstance(val, float):
        return f"{val:.2f}" if val else "0.00"
    
    return str(val)


@app.callback(
    [dash.dependencies.Output('gene-level-summary', 'children'),
     dash.dependencies.Output('ORF-level-summary', 'children')],
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('species-dropdown', 'value')]
)
def update_summary_blocks(selected_gene, species):
    """Update Gene-Level and ORF-Level summary blocks based on selected gene"""
    if not selected_gene:
        return (
            [html.P("Select a gene to view summary information", className='summary-placeholder')],
            [html.P("Select a gene to view summary information", className='summary-placeholder')]
        )

    try:
        # Get the first row data for the selected gene, since these columns have the same values for all rows
        table_prefix = get_table_prefix(species)
        db_config = get_db_config()

        # Mouse data doesn't have gene_protein_category column
        if species == "Mouse":
            query = f"""
            SELECT
                gene_potential,
                gene_perplexity,
                ptc_potential,
                ptc_perplexity,
                gene_average_tpm,
                gene_expressed_samples,
                "ORF_potential",
                "ORF_perplexity",
                "ORF_expressed_samples"
            FROM {table_prefix}isoforms
            WHERE gene_name = :gene_name
            LIMIT 1
            """
        else:
            query = f"""
            SELECT
                gene_protein_category,
                gene_potential,
                gene_perplexity,
                ptc_potential,
                ptc_perplexity,
                gene_average_tpm,
                gene_expressed_samples,
                "ORF_potential",
                "ORF_perplexity",
                "ORF_expressed_samples"
            FROM {table_prefix}isoforms
            WHERE gene_name = :gene_name
            LIMIT 1
            """

        result = db_config.execute_query(query, params={'gene_name': selected_gene})

        if result.empty:
            row = None
        else:
            row = tuple(result.iloc[0])

        if not row:
            return (
                [html.P("No data found for selected gene", className='summary-placeholder')],
                [html.P("No data found for selected gene", className='summary-placeholder')]
            )

        if species == "Mouse":
            (gene_potential, gene_perplexity, ptc_potential,
             ptc_perplexity, gene_average_tpm, gene_expressed_samples,
             ORF_potential, ORF_perplexity, ORF_expressed_samples) = row
            gene_protein_category = None
        else:
            (gene_protein_category, gene_potential, gene_perplexity, ptc_potential,
             ptc_perplexity, gene_average_tpm, gene_expressed_samples,
             ORF_potential, ORF_perplexity, ORF_expressed_samples) = row

        gene_summary = []

        # Only show protein category for human data
        if species != "Mouse" and gene_protein_category is not None:
            gene_summary.append(
                html.Div(className='summary-item', children=[
                    html.Span('Gene Protein Category:', className='summary-label'),
                    format_protein_category(gene_protein_category)
                ])
            )

        gene_summary.extend([
            html.Div(className='summary-item', children=[
                html.Span('Number of detected transcripts:', className='summary-label'),
                html.Span(format_value(gene_potential, 'gene_potential'), className='summary-value')
            ]),
            html.Div(className='summary-item', children=[
                html.Span('Number of detected protein-coding transcripts:', className='summary-label'),
                html.Span(format_value(ptc_potential, 'ptc_potential'), className='summary-value')
            ]),
            html.Div(className='summary-item', children=[
                html.Span('Gene Perplexity:', className='summary-label'),
                html.Span(format_value(gene_perplexity), className='summary-value')
            ]),
            html.Div(className='summary-item', children=[
                html.Span('PTC Perplexity:', className='summary-label'),
                html.Span(format_value(ptc_perplexity), className='summary-value')
            ]),
            html.Div(className='summary-item', children=[
                html.Span('Gene Average TPM:', className='summary-label'),
                html.Span(format_value(gene_average_tpm), className='summary-value')
            ]),
            html.Div(className='summary-item', children=[
                html.Span('Gene Expressed Samples:', className='summary-label'),
                html.Span(format_value(gene_expressed_samples), className='summary-value')
            ])
        ])

        ORF_summary = [
            html.Div(className='summary-item', children=[
                html.Span('Number of detected ORFs:', className='summary-label'),
                html.Span(format_value(ORF_potential, 'ORF_potential'), className='summary-value')
            ]),
            html.Div(className='summary-item', children=[
                html.Span('ORF Perplexity:', className='summary-label'),
                html.Span(format_value(ORF_perplexity), className='summary-value')
            ]),
            html.Div(className='summary-item', children=[
                html.Span('ORF Expressed Samples:', className='summary-label'),
                html.Span(format_value(ORF_expressed_samples, 'ORF_expressed_samples'), className='summary-value')
            ])
        ]

        return gene_summary, ORF_summary

    except Exception as e:
        print(f"Error updating summary blocks: {e}")
        return (
            [html.P("Error loading summary data", className='summary-placeholder')],
            [html.P("Error loading summary data", className='summary-placeholder')]
        )


######################################################################
# SQLLITE MASTER TABLE PROCESSING CALLBACKS
######################################################################
# Callback 1: Update isoform-full-data-store (does NOT depend on filtered stores)
@app.callback(
    dash.dependencies.Output('isoform-full-data-store', 'data'),
    [dash.dependencies.Input('left_data_table', 'sort_by'),
     dash.dependencies.Input('left-table-applied-filter-store', 'data'),
     dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('species-dropdown', 'value')]
)
def update_isoform_full_data_store(sort_by, filter_query, selected_gene, species):
    """Query database and store full isoform data (filtered by applied filter only)"""

    table_prefix = get_table_prefix(species)
    table_name = f'{table_prefix}isoforms'
    filters = parse_filter_query(db_path, filter_query, table_name=table_name)

    # Convert gene_name to gene_id for filtering
    gene_filter = selected_gene
    if selected_gene:
        gene_id = get_gene_id_for_gene_name(db_path, selected_gene, species)
        if gene_id:
            gene_filter = gene_id

    _, total_count = query_master_table(
        db_path,
        table_name=table_name,
        page=0,
        page_size=0,
        sort_by=None,
        filters=filters,
        gene_filter=gene_filter
    )

    full_data, _ = query_master_table(
        db_path,
        table_name=table_name,
        page=0,
        page_size=total_count,
        sort_by=sort_by,
        filters=filters,
        gene_filter=gene_filter
    )

    return full_data


# Callback 2: Update table display (depends on both stores, applies bidirectional filtering)
@app.callback(
    [dash.dependencies.Output('left_data_table', 'data'),
     dash.dependencies.Output('left_data_table', 'page_count')],
    [dash.dependencies.Input('left_data_table', 'page_current'),
     dash.dependencies.Input('left_data_table', 'page_size'),
     dash.dependencies.Input('isoform-full-data-store', 'data'),
     dash.dependencies.Input('filtered-isoform-store', 'data')],
    [dash.dependencies.State('right_data_table', 'filter_query')]
)
def update_isoform_table_display(page_current, page_size, isoform_full_data, filtered_isoform_ids, junction_filter_query):
    """Display paginated isoform data with bidirectional filtering applied"""
    if not isoform_full_data:
        return [], 0

    full_data = list(isoform_full_data)  # Make a copy

    # Apply bidirectional filtering: use filtered_isoform_ids if it represents a filtered subset
    if filtered_isoform_ids and isinstance(filtered_isoform_ids, list) and len(filtered_isoform_ids) > 0:
        # Get all IDs from full data
        all_ids_in_full_data = set(row.get('id') for row in full_data if row.get('id'))
        filtered_ids_set = set(filtered_isoform_ids)

        # Apply filtering if filtered set is different from full set
        if filtered_ids_set != all_ids_in_full_data:
            full_data = [row for row in full_data if row.get('id') in filtered_ids_set]

    total_count = len(full_data)
    page_current = page_current or 0
    page_size = page_size or 10

    start_idx = page_current * page_size
    end_idx = (page_current + 1) * page_size
    paginated_data = full_data[start_idx:end_idx]
    page_count = math.ceil(total_count / page_size) if page_size else 1

    return paginated_data, page_count


# Callback 3: Update junction-full-data-store (does NOT depend on filtered stores)
@app.callback(
    dash.dependencies.Output('junction-full-data-store', 'data'),
    [dash.dependencies.Input('right_data_table', 'sort_by'),
     dash.dependencies.Input('right-table-applied-filter-store', 'data'),
     dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('species-dropdown', 'value')]
)
def update_junction_full_data_store(sort_by, filter_query, selected_gene, species):
    """Query database and store full junction data (filtered by applied filter only)"""

    table_prefix = get_table_prefix(species)
    table_name = f'{table_prefix}junctions'
    filters = parse_filter_query(db_path, filter_query, table_name=table_name)

    # Convert gene_name to gene_id for filtering
    gene_filter = selected_gene
    if selected_gene:
        gene_id = get_gene_id_from_atse(db_path, selected_gene, species)
        if gene_id:
            gene_filter = gene_id

    _, total_count = query_master_table(
        db_path,
        table_name=table_name,
        page=0,
        page_size=0,
        sort_by=None,
        filters=filters,
        gene_filter=gene_filter
    )

    full_data, _ = query_master_table(
        db_path,
        table_name=table_name,
        page=0,
        page_size=total_count,
        sort_by=sort_by,
        filters=filters,
        gene_filter=gene_filter
    )

    return full_data


# Callback 4: Update table display (depends on both stores, applies bidirectional filtering)
@app.callback(
    [dash.dependencies.Output('right_data_table', 'data'),
     dash.dependencies.Output('right_data_table', 'page_count')],
    [dash.dependencies.Input('right_data_table', 'page_current'),
     dash.dependencies.Input('right_data_table', 'page_size'),
     dash.dependencies.Input('junction-full-data-store', 'data'),
     dash.dependencies.Input('filtered-junction-store', 'data')],
    [dash.dependencies.State('left_data_table', 'filter_query')]
)
def update_junction_table_display(page_current, page_size, junction_full_data, filtered_junction_ids, isoform_filter_query):
    """Display paginated junction data with bidirectional filtering applied"""
    if not junction_full_data:
        return [], 0

    full_data = list(junction_full_data)  # Make a copy

    # Apply bidirectional filtering: use filtered_junction_ids if it represents a filtered subset
    if filtered_junction_ids and isinstance(filtered_junction_ids, list) and len(filtered_junction_ids) > 0:
        # Get all IDs from full data
        all_ids_in_full_data = set(row.get('junction_id') for row in full_data if row.get('junction_id'))
        filtered_ids_set = set(filtered_junction_ids)

        # Apply filtering if filtered set is different from full set
        if filtered_ids_set != all_ids_in_full_data:
            full_data = [row for row in full_data if row.get('junction_id') in filtered_ids_set]

    total_count = len(full_data)
    page_current = page_current or 0
    page_size = page_size or 10

    start_idx = page_current * page_size
    end_idx = (page_current + 1) * page_size
    paginated_data = full_data[start_idx:end_idx]
    page_count = math.ceil(total_count / page_size) if page_size else 1

    return paginated_data, page_count



######################################################################
# DYNAMIC HEIGHT CALCULATION AND PANEL ADJUSTMENT
######################################################################
@app.callback(
    [dash.dependencies.Output('bar-height-slider', 'value'),
     dash.dependencies.Output('left-panel', 'style', allow_duplicate=True),
     dash.dependencies.Output('right-panel', 'style', allow_duplicate=True),
     dash.dependencies.Output('isoform-clustergram-container', 'style', allow_duplicate=True),
     dash.dependencies.Output('junction-clustergram-container', 'style', allow_duplicate=True)],
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('filtered-isoform-store', 'data'),
     dash.dependencies.Input('filtered-junction-store', 'data'),
     dash.dependencies.Input('clustergram-height-slider', 'value'),
     dash.dependencies.Input('species-dropdown', 'value')],
    [dash.dependencies.State('bar-height-slider', 'value')],
    prevent_initial_call=True
)
def update_dynamic_height_and_panels(selected_gene, filtered_isoform_ids, filtered_junction_ids, clustergram_height, species, current_height):
    """Calculate unified height for both plots and update slider and panels when gene changes"""
    if not selected_gene:
        panel_height = max(clustergram_height + 108, 1075)
        left_panel_style = {'height': f'{panel_height}px', 'minHeight': f'{panel_height}px', 'flex': '1'}
        right_panel_style = {'height': f'{panel_height}px', 'minHeight': f'{panel_height}px', 'flex': '1.2'}
        container_style = {'height': f'{clustergram_height}px', 'minHeight': f'{clustergram_height}px'}
        return current_height, left_panel_style, right_panel_style, container_style, container_style

    try:
        filtered_ids = [int(id) for id in filtered_isoform_ids] if filtered_isoform_ids else []
        transcript_data = process_transcript_structure(db_path, selected_gene, filtered_ids, species)
        gene_data = process_gene_atse_data(selected_gene, db_path, filtered_junction_ids, species)

        # Calculate unified height for structure plots based on both transcript and junction data: use max height from either
        calculated_height = calculate_unified_plot_height(transcript_data, gene_data)

        if abs(calculated_height - current_height) < 100:
            calculated_height = current_height

        num_transcripts = transcript_data['id'].nunique() if not transcript_data.empty else 0
        num_junctions = len(gene_data.get('junctions', [])) if gene_data and not gene_data.get('error') else 0

        min_isoform_height = calculate_clustergram_min_height(num_transcripts, base_height=750) if num_transcripts > 0 else 750
        min_junction_height = calculate_clustergram_min_height(num_junctions, base_height=750) if num_junctions > 0 else 750
        min_clustergram_height = max(min_isoform_height, min_junction_height)

        actual_clustergram_height = max(clustergram_height, min_clustergram_height)

        # Panel height = clustergram height + margins
        panel_height = actual_clustergram_height + 108
        panel_height = max(panel_height, 860)

        left_panel_style = {
            'height': f'{panel_height}px',
            'minHeight': f'{panel_height}px',
            'flex': '1'
        }

        right_panel_style = {
            'height': f'{panel_height}px',
            'minHeight': f'{panel_height}px',
            'flex': '1.2'
        }

        container_style = {
            'height': f'{actual_clustergram_height}px',
            'minHeight': f'{actual_clustergram_height}px'
        }

        return calculated_height, left_panel_style, right_panel_style, container_style, container_style

    except Exception as e:
        print(f"Error calculating dynamic height: {e}")
        panel_height = max(clustergram_height + 108, 1075) if clustergram_height else 860
        left_panel_style = {'height': f'{panel_height}px', 'minHeight': f'{panel_height}px', 'flex': '1'}
        right_panel_style = {'height': f'{panel_height}px', 'minHeight': f'{panel_height}px', 'flex': '1.2'}
        container_style = {'height': f'{clustergram_height}px', 'minHeight': f'{clustergram_height}px'} if clustergram_height else {'height': '1012px', 'minHeight': '1012px'}

        return current_height, left_panel_style, right_panel_style, container_style, container_style


######################################################################
# DYNAMIC PANEL HEIGHT ADJUSTMENT (Manual Slider Changes)
######################################################################
@app.callback(
    [dash.dependencies.Output('left-panel', 'style'),
     dash.dependencies.Output('right-panel', 'style'),
     dash.dependencies.Output('isoform-clustergram-container', 'style'),
     dash.dependencies.Output('junction-clustergram-container', 'style')],
    [dash.dependencies.Input('clustergram-height-slider', 'value'),
     dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('filtered-isoform-store', 'data'),
     dash.dependencies.Input('filtered-junction-store', 'data'),
     dash.dependencies.Input('species-dropdown', 'value')],
    prevent_initial_call=True
)
def adjust_panel_heights(clustergram_height, selected_gene, filtered_isoform_ids, filtered_junction_ids, species):
    """Adjust panel heights based on clustergram height slider and gene data"""

    if not selected_gene:
        panel_height = clustergram_height + 108
        panel_height = max(panel_height, 860)
        container_height = clustergram_height
    else:
        try:
            filtered_ids = [int(id) for id in filtered_isoform_ids] if filtered_isoform_ids else []
            transcript_data = process_transcript_structure(db_path, selected_gene, filtered_ids, species)
            gene_data = process_gene_atse_data(selected_gene, db_path, filtered_junction_ids, species)

            num_transcripts = transcript_data['id'].nunique() if not transcript_data.empty else 0
            num_junctions = len(gene_data.get('junctions', [])) if gene_data and not gene_data.get('error') else 0

            min_isoform_height = calculate_clustergram_min_height(num_transcripts, base_height=750) if num_transcripts > 0 else 750
            min_junction_height = calculate_clustergram_min_height(num_junctions, base_height=750) if num_junctions > 0 else 750

            min_clustergram_height = max(min_isoform_height, min_junction_height)
            actual_clustergram_height = max(clustergram_height, min_clustergram_height)

            # panel height = clustergram height + margins
            panel_height = actual_clustergram_height + 108
            panel_height = max(panel_height, 1075)

            container_height = actual_clustergram_height

        except Exception as e:
            print(f"Error calculating panel height: {e}")
            panel_height = max(clustergram_height + 108, 1075)
            container_height = clustergram_height

    left_panel_style = {
        'height': f'{panel_height}px',
        'minHeight': f'{panel_height}px',
        'flex': '1'
    }

    right_panel_style = {
        'height': f'{panel_height}px',
        'minHeight': f'{panel_height}px',
        'flex': '1.2'
    }

    container_style = {
        'height': f'{container_height}px',
        'minHeight': f'{container_height}px'
    }

    return left_panel_style, right_panel_style, container_style, container_style


######################################################################
# HEATMAP PROCESSING CALLBACKS
######################################################################
@app.callback(
    [dash.dependencies.Output('heatmap2', 'figure'),
     dash.dependencies.Output('heatmap2', 'config')],
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('colorscale-dropdown', 'value'),
     dash.dependencies.Input('filtered-junction-store', 'data'),
     dash.dependencies.Input('show-celltype-labels-toggle', 'value'),
     dash.dependencies.Input('clustergram-height-slider', 'value'),
     dash.dependencies.Input('distance-metric-dropdown', 'value'),
     dash.dependencies.Input('linkage-method-dropdown', 'value'),
     dash.dependencies.Input('filtered-isoform-store', 'data'),
     dash.dependencies.Input('gridlines-toggle', 'value'),
     dash.dependencies.Input('species-dropdown', 'value')]
)
def update_junction_clustergram(selected_gene, colorscale,
                                filtered_junction_ids, show_celltype_labels, clustergram_height,
                                distance_metric, linkage_method, filtered_isoform_ids, show_gridlines, species):
    """Update junction visualization based on gene selection and filtering"""

    if selected_gene:
        try:
            filtered_ids = [int(id) for id in filtered_isoform_ids] if filtered_isoform_ids else []
            transcript_data = process_transcript_structure(db_path, selected_gene, filtered_ids, species)
            gene_data = process_gene_atse_data(selected_gene, db_path, filtered_junction_ids, species)

            num_transcripts = transcript_data['id'].nunique() if not transcript_data.empty else 0
            num_junctions = len(gene_data.get('junctions', [])) if gene_data and not gene_data.get('error') else 0

            min_isoform_height = calculate_clustergram_min_height(num_transcripts, base_height=750) if num_transcripts > 0 else 750
            min_junction_height = calculate_clustergram_min_height(num_junctions, base_height=750) if num_junctions > 0 else 750

            min_clustergram_height = max(min_isoform_height, min_junction_height)
            heatmap_height = max(clustergram_height, min_clustergram_height)
        except:
            heatmap_height = clustergram_height
    else:
        heatmap_height = clustergram_height

    if not selected_gene:
        try:
            fig = create_summary_clustergram(db_path,
                                             height=heatmap_height,
                                             colorscale=colorscale,
                                             show_celltype_labels=show_celltype_labels,
                                             distance_metric=distance_metric,
                                             linkage_method=linkage_method,
                                             show_gridlines=show_gridlines,
                                             species=species)
            fig.update_layout(
                autosize=True,
                width=None,
                transition_duration=200
            )
            default_config = {
                'responsive': True,
                'displayModeBar': True,
                'scrollZoom': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'svg',
                    'filename': 'isoformgazer_short_read_clustergram'
                }
            }
            return fig, default_config
        
        except Exception as e:
            print(f"Error creating summary clustergram: {e}")
            empty_fig = create_empty_clustergram_message("Error loading summary data")
            default_config = {
                'responsive': True,
                'displayModeBar': True,
                'scrollZoom': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'svg',
                    'filename': 'isoformgazer_short_read_clustergram'
                }
            }
            return empty_fig, default_config

    try:
        fig = create_gene_clustergram(
            db_path,
            selected_gene,
            height=heatmap_height,
            colorscale=colorscale,
            filtered_junction_ids=filtered_junction_ids,
            show_celltype_labels=show_celltype_labels,
            distance_metric=distance_metric,
            linkage_method=linkage_method,
            show_gridlines=show_gridlines,
            species=species
        )
        fig.update_layout(
            autosize=True,
            width=None,
            transition_duration=200
        )
        # Create config with gene name in filename
        gene_clean = str(selected_gene).replace(' ', '_').replace('/', '_')
        config = {
            'responsive': True,
            'displayModeBar': True,
            'scrollZoom': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'svg',
                'filename': f'{gene_clean}_isoformgazer_short_read_clustergram'
            }
        }
        return fig, config

    except Exception as e:
        print(f"Error creating gene clustergram: {e}")
        empty_fig = create_empty_clustergram_message(f"Error loading data for {selected_gene}")
        default_config = {
            'responsive': True,
            'displayModeBar': True,
            'scrollZoom': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'svg',
                'filename': 'isoformgazer_short_read_clustergram'
            }
        }
        return empty_fig, default_config
    

@app.callback(
    [dash.dependencies.Output('heatmap1', 'figure'),
     dash.dependencies.Output('heatmap1', 'config')],
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('colorscale-dropdown', 'value'),
     dash.dependencies.Input('isoform-data-type-switch', 'value'),
     dash.dependencies.Input('show-labels-toggle', 'value'),
     dash.dependencies.Input('collapse-tissue-toggle', 'value'),
     dash.dependencies.Input('filtered-isoform-store', 'data'),
     dash.dependencies.Input('clustergram-height-slider', 'value'),
     dash.dependencies.Input('distance-metric-dropdown', 'value'),
     dash.dependencies.Input('linkage-method-dropdown', 'value'),
     dash.dependencies.Input('filtered-junction-store', 'data'),
     dash.dependencies.Input('gridlines-toggle', 'value'),
     dash.dependencies.Input('species-dropdown', 'value')]
)
def update_isoform_heatmap(selected_gene, colorscale, data_type_selection,
                          show_labels, collapse_mode, filtered_transcript_ids,
                          clustergram_height, distance_metric, linkage_method, filtered_junction_ids,
                          show_gridlines, species):
    """Update isoform clustergram with unified height based on both isoform and junction data"""
    # Return empty figure if no gene selected
    if not selected_gene:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title={'text': 'Select a gene to view isoform expression data', 'font': {'size': 14}},
            plot_bgcolor='white',
            margin=dict(l=40, r=40, t=40, b=40),
            height=710
        )
        default_config = {
            'responsive': True,
            'displayModeBar': True,
            'scrollZoom': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'svg',
                'filename': 'isoformgazer_long_read_clustergram'
            }
        }
        return empty_fig, default_config

    gridline_color = '#ffffff'

    ratio_data = load_expression_data(db_path=db_path,
                                      gene_name=selected_gene,
                                      data_type='ratio',
                                      species=species)

    tpm_data = load_expression_data(db_path=db_path,
                                    gene_name=selected_gene,
                                    data_type='tpm',
                                    species=species)

    log_tpm_data = load_expression_data(db_path=db_path,
                                        gene_name=selected_gene,
                                        data_type='log_tpm',
                                        species=species)

    if data_type_selection == 'ratio':
        data_type = "Ratio"
    elif data_type_selection == 'log_tpm':
        data_type = "Log TPM"
    else: 
        data_type = "TPM"

    if selected_gene:
        try:
            filtered_ids = [int(id) for id in filtered_transcript_ids] if filtered_transcript_ids else []
            transcript_data = process_transcript_structure(db_path, selected_gene, filtered_ids, species)
            gene_data = process_gene_atse_data(selected_gene, db_path, filtered_junction_ids, species)

            num_transcripts = transcript_data['id'].nunique() if not transcript_data.empty else 0
            num_junctions = len(gene_data.get('junctions', [])) if gene_data and not gene_data.get('error') else 0

            min_isoform_height = calculate_clustergram_min_height(num_transcripts, base_height=750) if num_transcripts > 0 else 750
            min_junction_height = calculate_clustergram_min_height(num_junctions, base_height=750) if num_junctions > 0 else 750

            min_clustergram_height = max(min_isoform_height, min_junction_height)
            heatmap_height = max(clustergram_height, min_clustergram_height)
        except:
            heatmap_height = clustergram_height
    else:
        heatmap_height = clustergram_height
    
    try:
        if data_type == "Ratio":
            current_data = ratio_data
        else: 
            current_data = tpm_data

        filtered_ids = [int(id) for id in filtered_transcript_ids] if filtered_transcript_ids else []

        if filtered_ids:
            # Filter while preserving the sorted order from the database
            filtered_ratio_data = ratio_data[ratio_data['id'].isin(filtered_ids)].copy()
            filtered_tpm_data = tpm_data[tpm_data['id'].isin(filtered_ids)].copy()
            filtered_log_tpm_data = log_tpm_data[log_tpm_data['id'].isin(filtered_ids)].copy()
        else:
            filtered_ratio_data = ratio_data.copy()
            filtered_tpm_data = tpm_data.copy()
            filtered_log_tpm_data = log_tpm_data.copy()

        fig = create_isoform_expression_clustergram(
            tpm_data=filtered_tpm_data,
            ratio_data=filtered_ratio_data,
            log_tpm_data=filtered_log_tpm_data,
            gene_name=selected_gene,
            height=heatmap_height,
            colorscale=colorscale,
            data_type=data_type,
            show_labels=show_labels,
            collapse_mode=collapse_mode,
            distance_metric=distance_metric,
            linkage_method=linkage_method,
            show_gridlines=show_gridlines,
            gridline_color=gridline_color,
            db_path=db_path,
            species=species
        )
        fig.update_layout(
            autosize=True,
            width=None
        )

        gene_clean = str(selected_gene).replace(' ', '_').replace('/', '_')
        config = {
            'responsive': True,
            'displayModeBar': True,
            'scrollZoom': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'svg',
                'filename': f'{gene_clean}_isoformgazer_long_read_clustergram'
            }
        }
        return fig, config

    except Exception as e:
        print(f"Error creating isoform clustergram: {e}")
        traceback.print_exc()
        empty_fig = create_empty_isoform_message(f"Error loading {data_type.lower()} data for {selected_gene}")
        error_config = {
            'responsive': True,
            'displayModeBar': True,
            'scrollZoom': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'svg',
                'filename': 'isoformgazer_long_read_clustergram'
            }
        }
        return empty_fig, error_config


######################################################################
# STRUCTURE-LEVEL VISUALIZATIONS CALLBACKS
######################################################################
@app.callback(
    [dash.dependencies.Output('atse-map', 'figure'),
     dash.dependencies.Output('atse-map', 'config')],
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('filtered-junction-store', 'data'),
     dash.dependencies.Input('filtered-isoform-store', 'data'),
     dash.dependencies.Input('bar-height-slider', 'value'),
     dash.dependencies.Input('exon-color-store', 'data'),
     dash.dependencies.Input('junction-color-store', 'data'),
     dash.dependencies.Input('left_data_table', 'filter_query'),
     dash.dependencies.Input('left-table-validation-store', 'data'),
     dash.dependencies.Input('color-junctions-by-psi-toggle', 'value'),
     dash.dependencies.Input('color-by-abundance-toggle', 'value'),
     dash.dependencies.Input('structure-plot-colorscale-dropdown', 'value'),
     dash.dependencies.Input('abundance-color-type-radio', 'value'),
     dash.dependencies.Input('tissue-abundance-dropdown', 'value'),
     dash.dependencies.Input('organ-abundance-dropdown', 'value'),
     dash.dependencies.Input('individual-junction-colors', 'data'),
     dash.dependencies.Input('individual-transcript-colors', 'data'),
     dash.dependencies.Input('species-dropdown', 'value')]
)
def update_atse_visualization(selected_gene, filtered_junction_ids, filtered_transcript_ids,
                              plot_height, exon_color, junction_color, isoform_filter_query,
                              validation_data, color_junctions_by_psi, color_by_abundance,
                              structure_colorscale, abundance_type, tissue_name, organ_name,
                              individual_junction_colors, individual_transcript_colors, species):
    """Update ATSE splice junction visualization with filtered data"""
    # Check if current filter is valid: if not, don't update plot
    if isoform_filter_query and validation_data and not validation_data.get('valid', True):
        raise PreventUpdate

    # Determine if actual filtering is happening by checking if filtered IDs is a subset: both direct isoform filtering AND bidirectional filtering from junctions
    actual_filtered_transcript_ids = None
    if filtered_transcript_ids and isinstance(filtered_transcript_ids, list) and len(filtered_transcript_ids) > 0 and selected_gene:
        try:
            # Get total transcript count for this gene
            db_config = get_db_config()
            table_prefix = '' if species == "Human" else 'mouse_'
            count_query = f'SELECT COUNT(*) FROM "{table_prefix}isoforms" WHERE gene_name = :gene_name'
            result = db_config.execute_query(count_query, params={'gene_name': selected_gene})
            total_count = result.iloc[0, 0]

            # Only apply filtering if filtered list is smaller than total (actual filtering is happening)
            if len(filtered_transcript_ids) < total_count:
                actual_filtered_transcript_ids = filtered_transcript_ids
            # If filtered_transcript_ids has same count as total, pass None (no filtering)
            else:
                actual_filtered_transcript_ids = None
        except Exception as e:
            print(f"Error checking filter status: {e}")
            actual_filtered_transcript_ids = None

    if not selected_gene:
        empty_fig = create_empty_atse_message("Select a gene to view splice junctions and exons")
        default_config = {
            'responsive': True,
            'displayModeBar': True,
            'scrollZoom': False,
            'toImageButtonOptions': {'format': 'svg', 'filename': 'isoformgazer_plot'}
        }
        return empty_fig, default_config

    try:
        gene_data = process_gene_atse_data(
            selected_gene,
            db_path,
            filtered_junction_ids=filtered_junction_ids,
            species=species
        )

        # If gene not found in junction data, check if it exists in isoform data
        if 'error' in gene_data:
            filtered_ids = [int(id) for id in actual_filtered_transcript_ids] if actual_filtered_transcript_ids else []
            transcript_data = process_transcript_structure(db_path, selected_gene, filtered_ids, species)

            # Case 1: gene exists in isoform data but not junction data - show transcript-only plot
            if not transcript_data.empty:
                
                if plot_height == 600:
                    height_to_use = None
                else:
                    height_to_use = plot_height

                fig = create_transcript_structure_plot(
                    db_path,
                    transcript_data,
                    gene_name=selected_gene,
                    height=height_to_use,
                    show_y_labels=True,
                    exon_color=exon_color,
                    color_by_abundance=color_by_abundance,
                    colorscale=structure_colorscale,
                    abundance_type=abundance_type,
                    tissue_name=tissue_name,
                    organ_name=organ_name,
                    individual_transcript_colors=individual_transcript_colors,
                    species=species
                )

                gene_clean = str(selected_gene).replace(' ', '_').replace('/', '_')
                config = {
                    'responsive': True,
                    'displayModeBar': True,
                    'scrollZoom': False,
                    'toImageButtonOptions': {
                        'format': 'svg',
                        'filename': f'{gene_clean}_isoformgazer_structure_plot'
                    }
                }
                return fig, config
            
            # Case 2: gene doesn't exist in either dataset
            else:
                empty_fig = create_empty_atse_message(gene_data['error'])
                default_config = {
                    'responsive': True,
                    'displayModeBar': True,
                    'scrollZoom': False,
                    'toImageButtonOptions': {'format': 'svg', 'filename': 'isoformgazer_plot'}
                }
                return empty_fig, default_config

        show_labels = False

        fig = create_junction_exon_visualization(
            gene_data,
            height=plot_height,
            show_y_labels=show_labels,
            exon_color=exon_color,
            junction_color=junction_color,
            filtered_transcript_ids=actual_filtered_transcript_ids,
            color_by_abundance=color_by_abundance,
            color_junctions_by_psi=color_junctions_by_psi,
            db_path=db_path,
            colorscale=structure_colorscale,
            abundance_type=abundance_type,
            tissue_name=tissue_name,
            organ_name=organ_name,
            individual_junction_colors=individual_junction_colors,
            individual_transcript_colors=individual_transcript_colors,
            species=species
        )

        # Create config with gene name in filename
        gene_clean = str(selected_gene).replace(' ', '_').replace('/', '_')
        config = {
            'responsive': True,
            'displayModeBar': True,
            'scrollZoom': False,
            'toImageButtonOptions': {
                'format': 'svg',
                'filename': f'{gene_clean}_isoformgazer_structure_plot'
            }
        }
        return fig, config

    except Exception as e:
        print(f"Error creating ATSE visualization: {e}")
        empty_fig = create_empty_atse_message(f"Error loading ATSE data for {selected_gene}: {str(e)}")
        default_config = {
            'responsive': True,
            'displayModeBar': True,
            'scrollZoom': False,
            'toImageButtonOptions': {'format': 'svg', 'filename': 'isoformgazer_plot'}
        }
        return empty_fig, default_config


def empty_fig(height=200):
    fig = go.Figure()
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


######################################################################
# TOP PANEL TOGGLE AND STRUCTURE PLOT CALLBACKS
######################################################################
@app.callback(
    [dash.dependencies.Output('top-junction-structure-plot-container', 'style'),
     dash.dependencies.Output('top-transcript-structure-plot-container', 'style')],
    [dash.dependencies.Input('hide-junctions-toggle', 'value')]
)
def toggle_top_panel_plots(hide_junctions):
    """Toggle between junction and transcript structure plots in top panel"""
    if hide_junctions:
        # Hide junctions, show transcript plot with height overrides to prevent CSS clipping
        junction_style = {'display': 'none'}
        transcript_style = {
            'display': 'block',
            'height': 'auto',
            'minHeight': 'auto',
            'maxHeight': 'none',
            'margin-bottom': '15px'
        }
    else:
        # Show junctions, hide transcript plot with height overrides
        junction_style = {
            'display': 'block',
            'height': 'auto',
            'minHeight': 'auto',
            'maxHeight': 'none',
            'margin-bottom': '15px'
        }
        transcript_style = {'display': 'none'}

    return junction_style, transcript_style


@app.callback(
    [dash.dependencies.Output('top-barplot', 'figure'),
     dash.dependencies.Output('top-barplot', 'config')],
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('bar-height-slider', 'value'),
     dash.dependencies.Input('filtered-isoform-store', 'data'),
     dash.dependencies.Input('exon-color-store', 'data'),
     dash.dependencies.Input('hide-junctions-toggle', 'value'),
     dash.dependencies.Input('left_data_table', 'filter_query'),
     dash.dependencies.Input('left-table-validation-store', 'data'),
     dash.dependencies.Input('color-by-abundance-toggle', 'value'),
     dash.dependencies.Input('structure-plot-colorscale-dropdown', 'value'),
     dash.dependencies.Input('abundance-color-type-radio', 'value'),
     dash.dependencies.Input('tissue-abundance-dropdown', 'value'),
     dash.dependencies.Input('organ-abundance-dropdown', 'value'),
     dash.dependencies.Input('individual-transcript-colors', 'data'),
     dash.dependencies.Input('species-dropdown', 'value')]
)
def update_top_transcript_structure(selected_gene, plot_height, filtered_ids, exon_color, hide_junctions, filter_query, validation_data, color_by_abundance, colorscale, abundance_type, tissue_name, organ_name, individual_transcript_colors, species):
    """Update transcript structure plot in top panel when toggle is activated"""
    # Only update if junctions are hidden (transcript plot should be shown)
    if not hide_junctions:
        raise PreventUpdate

    # Check if current filter is valid: if not, don't update plot
    if filter_query and validation_data and not validation_data.get('valid', True):
        raise PreventUpdate

    if not selected_gene:
        empty_fig = create_empty_isoform_message("Select a gene to view transcript structures")
        default_config = {
            'responsive': True,
            'displayModeBar': True,
            'scrollZoom': False,
            'toImageButtonOptions': {'format': 'svg', 'filename': 'isoformgazer_structure_plot'}
        }
        return empty_fig, default_config

    try:
        filtered_ids = [int(id) for id in filtered_ids] if filtered_ids else []
        transcript_data = process_transcript_structure(db_path, selected_gene, filtered_ids, species)

        # transcript plot function handles its own height calculation when using default: only overriden when user manually sets a specific height
        if plot_height == 600:
            height_to_use = None
        else:
            height_to_use = plot_height

        fig = create_transcript_structure_plot(
            db_path,
            transcript_data,
            gene_name=selected_gene,
            height=height_to_use,
            show_y_labels=True,
            exon_color=exon_color,
            color_by_abundance=color_by_abundance,
            colorscale=colorscale,
            abundance_type=abundance_type,
            tissue_name=tissue_name,
            organ_name=organ_name,
            individual_transcript_colors=individual_transcript_colors,
            species=species
        )

        # Create config with gene name in filename
        gene_clean = str(selected_gene).replace(' ', '_').replace('/', '_')
        config = {
            'responsive': True,
            'displayModeBar': True,
            'scrollZoom': False,
            'toImageButtonOptions': {
                'format': 'svg',
                'filename': f'{gene_clean}_isoformgazer_structure_plot'
            }
        }
        return fig, config

    except Exception as e:
        print(f"Error creating top transcript plot: {e}")
        empty_fig = create_empty_isoform_message(f"Error loading transcript data for {selected_gene}")
        default_config = {
            'responsive': True,
            'displayModeBar': True,
            'scrollZoom': False,
            'toImageButtonOptions': {'format': 'svg', 'filename': 'isoformgazer_structure_plot'}
        }
        return empty_fig, default_config


@app.callback(
    dash.dependencies.Output('top-barplot-loading-message', 'children'),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('filtered-isoform-store', 'data')]
)
def update_top_barplot_loading_message(selected_gene, filtered_ids):
    if not selected_gene:
        selected_gene = 'AACS'
    T = len(filtered_ids) if filtered_ids else 0
    return f"Loading data for {T} transcript structures for {selected_gene}"


######################################################################
# TOP PANEL MANUAL HEIGHT ADJUSTMENT (Manual Slider Changes)
######################################################################
@app.callback(
    [dash.dependencies.Output('top-structure-plot-container-style', 'style', allow_duplicate=True),
     dash.dependencies.Output('top-panel-body', 'style', allow_duplicate=True)],
    [dash.dependencies.Input('bar-height-slider', 'value')],
    prevent_initial_call=True
)
def adjust_top_panel_height(plot_height):
    """Manually adjust top panel height when user changes the height slider"""

    container_style = {
        'height': f'{plot_height}px',
        'min-height': f'{plot_height}px',
        'margin-bottom': '15px'
    }

    panel_body_style = {
        'width': '100%',
        'background-color': 'white',
        'padding': '15px 15px 30px 15px',
        'border-radius': '0',
        'min-height': 'auto',
        'height': 'auto',
        'color': '#1C1C2C',
        'transition': 'height 0.3s ease, min-height 0.3s ease',
        'box-sizing': 'border-box'
    }

    return container_style, panel_body_style


@app.callback(
    [dash.dependencies.Output('top-structure-plot-container-style', 'style'),
     dash.dependencies.Output('top-panel-body', 'style')],
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('filtered-isoform-store', 'data'),
     dash.dependencies.Input('filtered-junction-store', 'data'),
     dash.dependencies.Input('hide-junctions-toggle', 'value'),
     dash.dependencies.Input('species-dropdown', 'value')],
    [dash.dependencies.State('bar-height-slider', 'value')]
)
def update_top_panel_height(selected_gene, filtered_transcript_ids, filtered_junction_ids, hide_junctions, species, current_height):
    """Calculate unified height for top panel using the same system as old structure plots"""
    if not selected_gene:
        container_style = {'height': '400px', 'min-height': '400px', 'margin-bottom': '15px'}
        panel_body_style = {
            'width': '100%',
            'background-color': 'white',
            'padding': '15px 15px 165px 15px',
            'border-radius': '0',
            'min-height': '450px',
            'height': 'auto',
            'color': '#1C1C2C',
            'transition': 'height 0.3s ease, min-height 0.3s ease',
            'box-sizing': 'border-box'
        }
        return container_style, panel_body_style

    try:
        filtered_ids = [int(id) for id in filtered_transcript_ids] if filtered_transcript_ids else []
        transcript_data = process_transcript_structure(db_path, selected_gene, filtered_ids, species)

        if hide_junctions:
            num_transcripts = len(transcript_data['id'].unique()) if not transcript_data.empty else 0
            calculated_height = calculate_dynamic_structure_plot_height(num_transcripts)
        else:
            gene_data = process_gene_atse_data(selected_gene, db_path, filtered_junction_ids, species)
            calculated_height = calculate_unified_plot_height(transcript_data, gene_data)

        if abs(calculated_height - current_height) < 100:
            calculated_height = current_height

        container_style = {
            'height': f'{calculated_height}px',
            'min-height': f'{calculated_height}px'
        }

        panel_body_style = {
            'width': '100%',
            'background-color': 'white',
            'padding': '15px',
            'border-radius': '0',
            'min-height': 'auto',
            'height': 'auto',
            'color': '#1C1C2C',
            'box-sizing': 'border-box'
        }

        return container_style, panel_body_style

    except Exception as e:
        print(f"Error calculating top panel height: {e}")
        container_style = {'height': '400px', 'min-height': '400px', 'margin-bottom': '15px'}
        panel_body_style = {
            'width': '100%',
            'background-color': 'white',
            'padding': '15px 15px 165px 15px',
            'border-radius': '0',
            'min-height': '450px',
            'height': 'auto',
            'color': '#1C1C2C',
            'transition': 'height 0.3s ease, min-height 0.3s ease',
            'box-sizing': 'border-box'
        }
        return container_style, panel_body_style


@app.callback(
    dash.dependencies.Output('heatmap1-loading-message', 'children'),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('filtered-isoform-store', 'data')]
)
def update_heatmap1_loading_message(selected_gene, filtered_ids):
    if not selected_gene:
        selected_gene = 'AACS'
    T = len(filtered_ids) if filtered_ids else 0
    return f"Loading data for {T} isoform transcripts for {selected_gene}"


@app.callback(
    dash.dependencies.Output('atse-map-loading-message', 'children'),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('filtered-junction-store', 'data')]
)
def update_atse_map_loading_message(selected_gene, filtered_ids):
    if not selected_gene:
        selected_gene = 'AACS'
    N = len(filtered_ids) if filtered_ids else 0
    return f"Loading data for {N} junctions for {selected_gene}"


@app.callback(
    dash.dependencies.Output('heatmap2-loading-message', 'children'),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('filtered-junction-store', 'data')]
)
def update_heatmap2_loading_message(selected_gene, filtered_ids):
    if not selected_gene:
        selected_gene = 'AACS'
    N = len(filtered_ids) if filtered_ids else 0
    return f"Loading data for {N} junctions for {selected_gene}"


###################################################################
# ISOFORM HASH LOOKUP CALLBACKS
###################################################################
@app.callback(
    [dash.dependencies.Output('gtf-upload-status', 'children'),
     dash.dependencies.Output('gtf-download-section', 'style'),
     dash.dependencies.Output('gtf-hash-results-store', 'data')],
    [dash.dependencies.Input('gtf-upload', 'contents')],
    [dash.dependencies.State('gtf-upload', 'filename')]
)
def handle_gtf_upload(contents, filename):
    """Handle GTF file upload and hash calculation"""
    if not contents:
        return html.Div("No file uploaded", className='app-controls-desc'), {'display': 'none'}, []

    try:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        gtf_content = decoded.decode('utf-8')

        parse_results = parse_gtf_and_calculate_hashes(gtf_content)
        hash_results = parse_results['results']
        merged_transcripts = parse_results['merged_transcripts']
        has_merges = parse_results['has_merges']

        if not hash_results:
            return (
                html.Div([
                    html.P(f"File '{filename}' uploaded successfully", className='success-message'),
                    html.P("Warning: No valid transcripts found in GTF file. " \
                    "Please refer to the documentation to ensure your GTF adheres to the required format.", className='warning-message')
                ]),
                {'display': 'none'},
                []
            )

        annotated_gtf = generate_annotated_gtf(gtf_content)
        encoded_gtf = base64.b64encode(annotated_gtf.encode('utf-8')).decode('utf-8')

        status_children = [
            html.P(f"Loaded '{filename}' successfully.", className='success-message success-message-bold'),
            html.P(f"Number of transcripts processed: {len(hash_results)}")
        ]

        # Show warning if transcripts were merged
        if has_merges:
            merge_info_text = "Transcript Merging Detected: The following transcripts share identical internal splice junction structure and have been assigned the same hash ID, but may differ in TSS/TES:\n\n"
            for hash_id, transcript_ids in merged_transcripts.items():
                merge_info_text += f"Hash ID {hash_id}:\n"
                for tid in transcript_ids:
                    merge_info_text += f"  • {tid}\n"
                merge_info_text += "\n"

            status_children.append(
                html.Div([
                    html.P("Warning: Transcript Merging Detected", className='warning-message warning-message-bold'),
                    html.P(merge_info_text, className='warning-message', style={'white-space': 'pre-wrap', 'font-family': 'monospace', 'font-size': '12px'})
                ])
            )

        upload_status = html.Div(status_children)

        combined_data = {
            'hash_results': hash_results,
            'annotated_gtf': encoded_gtf,
            'original_filename': filename,
            'merged_transcripts': merged_transcripts,
            'has_merges': has_merges
        }

        return upload_status, {'display': 'block'}, combined_data

    except Exception as e:
        return (
            html.Div([
                html.P(f"Error processing file '{filename}':", className='error-message'),
                html.P(str(e), className='error-message-light')
            ]),
            {'display': 'none'},
            []
        )


@app.callback(
    dash.dependencies.Output('download-hashes', 'data'),
    [dash.dependencies.Input('download-hashes-btn', 'n_clicks')],
    [dash.dependencies.State('gtf-hash-results-store', 'data')]
)
def download_hash_results(download_clicks, stored_data):
    """Handle download of hash results as TSV with gene_id, transcript_id, gencode_transcript_id, hash_id"""
    if not download_clicks or not stored_data:
        return dash.no_update

    try:
        if isinstance(stored_data, dict) and 'hash_results' in stored_data:
            hash_results = stored_data['hash_results']
        else:
            hash_results = stored_data

        db_config = get_db_config()

        # Results TSV format: gene_id, transcript_id, gencode_transcript_id, hash_id
        lines = ["gene_id\ttranscript_id\tgencode_transcript_id\thash_id"]

        for result in hash_results:
            gene_id = result['gene_id']
            transcript_id = result['transcript_id']
            hash_id = result['hash_id']

            # Look up gencode_transcript_id from gencode_gtf table...
            # Match by transcript_id, version-agnostic (e.g., ENST00000456328.2 matches ENST00000456328.1)...
            # For novel transcripts (e.g., ENSG00000100320.24.novel10), there won't be a match, so we output 'N/A'.
            transcript_base = transcript_id.split('.')[0] if '.' in transcript_id else transcript_id

            gencode_query = """
                SELECT DISTINCT transcript_id FROM gencode_gtf
                WHERE transcript_id LIKE :transcript_id
                ORDER BY transcript_id
                LIMIT 1
            """
            gencode_result = db_config.execute_query(gencode_query, params={'transcript_id': f"{transcript_base}.%"})

            gencode_transcript_id = gencode_result.iloc[0]['transcript_id'] if not gencode_result.empty else "N/A"

            lines.append(f"{gene_id}\t{transcript_id}\t{gencode_transcript_id}\t{hash_id}")

        download_content = "\n".join(lines)

        return dict(content=download_content, filename="isoform_hashes.tsv")

    except Exception:
        return dash.no_update


@app.callback(
    dash.dependencies.Output('download-annotated-gtf', 'data'),
    [dash.dependencies.Input('download-annotated-gtf-btn', 'n_clicks')],
    [dash.dependencies.State('gtf-hash-results-store', 'data')]
)
def download_annotated_gtf(download_clicks, stored_data):
    """Handle download of annotated GTF file"""
    if not download_clicks or not stored_data:
        return dash.no_update

    try:
        if isinstance(stored_data, dict) and 'annotated_gtf' in stored_data:
            encoded_gtf = stored_data['annotated_gtf']
            original_filename = stored_data.get('original_filename', 'annotated.gtf')

            gtf_content = base64.b64decode(encoded_gtf).decode('utf-8')
            base_name = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename
            output_filename = f"{base_name}_annotated.gtf"

            return dict(content=gtf_content, filename=output_filename)
        else:
            return dash.no_update

    except Exception as e:
        return dash.no_update


######################################################################
# MASTER TABLE CSV DOWNLOAD CALLBACKS
######################################################################
@app.callback(
    dash.dependencies.Output('download-left-table', 'data'),
    dash.dependencies.Input('download-left-table-button', 'n_clicks'),
    dash.dependencies.State('isoform-full-data-store', 'data'),
    dash.dependencies.State('gene-search-dropdown', 'value'),
    prevent_initial_call=True
)
def download_isoform_table(n_clicks, full_data, selected_gene):
    """Download current isoform master table view as CSV"""
    if not n_clicks or not full_data or not selected_gene:
        raise PreventUpdate

    try:
        df = pd.DataFrame(full_data)
        filename = f"{selected_gene}_isoforms_master_table.csv"
        return dcc.send_data_frame(df.to_csv, filename, index=False)
    
    except Exception as e:
        print(f"Error downloading isoform table: {e}")
        raise PreventUpdate


@app.callback(
    dash.dependencies.Output('download-right-table', 'data'),
    dash.dependencies.Input('download-right-table-button', 'n_clicks'),
    dash.dependencies.State('junction-full-data-store', 'data'),
    dash.dependencies.State('gene-search-dropdown', 'value'),
    prevent_initial_call=True
)
def download_junction_table(n_clicks, full_data, selected_gene):
    """Download current junction master table view as CSV"""
    if not n_clicks or not full_data or not selected_gene:
        raise PreventUpdate

    try:
        df = pd.DataFrame(full_data)
        filename = f"{selected_gene}_junctions_master_table.csv"
        return dcc.send_data_frame(df.to_csv, filename, index=False)

    except Exception as e:
        print(f"Error downloading junction table: {e}")
        raise PreventUpdate


@app.callback(
    dash.dependencies.Output('download-left-expression', 'data'),
    dash.dependencies.Input('download-left-expression-button', 'n_clicks'),
    dash.dependencies.State('isoform-full-data-store', 'data'),
    dash.dependencies.State('gene-search-dropdown', 'value'),
    dash.dependencies.State('isoform-data-type-switch', 'value'),
    dash.dependencies.State('species-dropdown', 'value'),
    prevent_initial_call=True
)
def download_isoform_expression(n_clicks, full_data, selected_gene, data_type_selection, species):
    """Download expression data (TPM, logTPM, or ratio) for current isoform table view"""
    if not n_clicks or not full_data or not selected_gene:
        raise PreventUpdate

    try:
        if data_type_selection == 'ratio':
            data_type = 'ratio'
            data_label = 'ratio'

        elif data_type_selection == 'log_tpm':
            data_type = 'log_tpm'
            data_label = 'logTPM'

        else:
            data_type = 'tpm'
            data_label = 'TPM'

        expression_data = load_expression_data(
            db_path=db_path,
            gene_name=selected_gene,
            data_type=data_type,
            species=species
        )

        if expression_data.empty:
            raise PreventUpdate

        table_df = pd.DataFrame(full_data)
        if 'id' not in table_df.columns:
            raise PreventUpdate

        filtered_expression = expression_data[expression_data['id'].isin(table_df['id'])].copy()
        if 'transcript' in table_df.columns:
            id_to_transcript = dict(zip(table_df['id'], table_df['transcript']))
            filtered_expression.insert(0, 'transcript_id', filtered_expression['id'].map(id_to_transcript))

        # Remove trans_id col from export (duplicate of transcript_id)
        if 'trans_id' in filtered_expression.columns:
            filtered_expression = filtered_expression.drop(columns=['trans_id'])

        filename = f"{selected_gene}_isoforms_{data_label}_expression.csv"
        return dcc.send_data_frame(filtered_expression.to_csv, filename, index=False)

    except Exception as e:
        print(f"Error downloading isoform expression data: {e}")
        raise PreventUpdate


@app.callback(
    dash.dependencies.Output('download-right-psi', 'data'),
    dash.dependencies.Input('download-right-psi-button', 'n_clicks'),
    dash.dependencies.State('junction-full-data-store', 'data'),
    dash.dependencies.State('gene-search-dropdown', 'value'),
    dash.dependencies.State('species-dropdown', 'value'),
    prevent_initial_call=True
)
def download_junction_psi(n_clicks, full_data, selected_gene, species):
    """Download PSI data for current junction table view"""
    if not n_clicks or not full_data or not selected_gene:
        raise PreventUpdate

    try:
        # Get the junction IDs from the current table view only
        table_df = pd.DataFrame(full_data)
        if 'junction_id' not in table_df.columns:
            raise PreventUpdate

        junction_ids = table_df['junction_id'].unique().tolist()
        db_config = get_db_config()
        table_prefix = get_table_prefix(species)

        placeholders = ','.join([f':jid_{i}' for i in range(len(junction_ids))])
        query = f"""
        SELECT junction_id, junction_id_index, cell_type, n_cells, psi, atse_count, junction_count
        FROM {table_prefix}junction_psis
        WHERE junction_id IN ({placeholders})
        ORDER BY junction_id, cell_type
        """
        params = {f'jid_{i}': junction_ids[i] for i in range(len(junction_ids))}
        psi_data = db_config.execute_query(query, params=params)

        if psi_data.empty:
            raise PreventUpdate

        filename = f"{selected_gene}_junctions_psi_data.csv"
        return dcc.send_data_frame(psi_data.to_csv, filename, index=False)

    except Exception as e:
        print(f"Error downloading junction PSI data: {e}")
        raise PreventUpdate


######################################################################
# FIGURE EXPORT OPTIONS CALLBACKS
######################################################################
def convert_dimensions(width, height, from_unit, to_unit='px', dpi=96):
    """
    Convert dimensions between pixels and inches.
    DPI is assumed to be 96 for screen displays (TBD: better way to 
    fetch this automatically for each user's screen?)
    """
    if from_unit == to_unit:
        return width, height

    if from_unit == 'px' and to_unit == 'in':
        return width / dpi, height / dpi
    
    elif from_unit == 'in' and to_unit == 'px':
        return width * dpi, height * dpi

    return width, height


@app.callback(
    [dash.dependencies.Output('export-status-message', 'style'),
     dash.dependencies.Output('export-status-timer', 'disabled')],
    [dash.dependencies.Input('export-unified-btn', 'n_clicks'),
     dash.dependencies.Input('export-status-timer', 'n_intervals')],
    [dash.dependencies.State('export-width-value', 'value'),
     dash.dependencies.State('export-height-value', 'value'),
     dash.dependencies.State('gene-search-dropdown', 'value')],
    prevent_initial_call=True
)
def manage_download_status(n_clicks, n_intervals, width, height, selected_gene):
    """Manage download status message display and timer"""
    if not callback_context.triggered:
        raise PreventUpdate

    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'export-unified-btn':
        if not n_clicks or not width or not height or not selected_gene:
            raise PreventUpdate

        return (
            {
                'marginTop': '15px',
                'fontSize': '12px',
                'color': '#301279',
                'fontWeight': '600',
                'display': 'block'
            },
            False
        )

    elif triggered_id == 'export-status-timer':
        if not n_intervals:
            raise PreventUpdate

        return (
            {
                'marginTop': '15px',
                'fontSize': '12px',
                'color': '#301279',
                'fontWeight': '600',
                'display': 'none'
            },
            True
        )


@app.callback(
    [dash.dependencies.Output("download-structure-plot", "data"),
     dash.dependencies.Output("download-isoform-clustergram", "data"),
     dash.dependencies.Output("download-junction-clustergram", "data")],
    dash.dependencies.Input('export-unified-btn', 'n_clicks'),
    [dash.dependencies.State('export-plot-selection', 'value'),
     dash.dependencies.State('heatmap1', 'figure'),
     dash.dependencies.State('heatmap2', 'figure'),
     dash.dependencies.State('export-width-value', 'value'),
     dash.dependencies.State('export-height-value', 'value'),
     dash.dependencies.State('export-unit-toggle', 'value'),
     dash.dependencies.State('export-title-legend-font-size', 'value'),
     dash.dependencies.State('export-axis-labels-font-size', 'value'),
     dash.dependencies.State('gene-search-dropdown', 'value'),
     dash.dependencies.State('filtered-isoform-store', 'data'),
     dash.dependencies.State('exon-color-store', 'data'),
     dash.dependencies.State('color-by-abundance-toggle', 'value'),
     dash.dependencies.State('structure-plot-colorscale-dropdown', 'value'),
     dash.dependencies.State('abundance-color-type-radio', 'value'),
     dash.dependencies.State('tissue-abundance-dropdown', 'value'),
     dash.dependencies.State('organ-abundance-dropdown', 'value'),
     dash.dependencies.State('species-dropdown', 'value')],
     prevent_initial_call=True
)
def export_plot(n_clicks, plot_selection, isoform_fig, junction_fig, width, height, unit, title_legend_font_size, axis_labels_font_size, selected_gene, filtered_ids, exon_color, color_by_abundance, structure_colorscale, abundance_type, tissue_name, organ_name, species):
    """Export selected plot with custom dimensions as SVG"""
    print(f"EXPORT CALLBACK TRIGGERED")
    print(f"n_clicks={n_clicks} (type={type(n_clicks)}), plot_selection={plot_selection}, width={width}, height={height}, selected_gene={selected_gene}")
    if not width or not height or not selected_gene:
        print(f"ERROR: Missing required parameters - width={width}, height={height}, selected_gene={selected_gene}")
        return None, None, None

    try:
        if unit == 'in':
            width_px, height_px = convert_dimensions(width, height, 'in', 'px')
        else:
            width_px, height_px = int(width), int(height)

        download_data = None
        filename = ""

        if plot_selection == 'structure':
            filtered_ids = [int(id) for id in filtered_ids] if filtered_ids else []
            transcript_data = process_transcript_structure(db_path, selected_gene, filtered_ids, species)

            fig = create_transcript_structure_plot(
                db_path,
                transcript_data,
                gene_name=selected_gene,
                height=None,
                show_y_labels=True,
                exon_color=exon_color,
                color_by_abundance=color_by_abundance,
                colorscale=structure_colorscale,
                abundance_type=abundance_type,
                tissue_name=tissue_name,
                organ_name=organ_name,
                individual_transcript_colors={},  # Export uses default colors
                species=species
            )
            filename = f"{selected_gene}_structure_plot.svg"

        elif plot_selection == 'isoform':
            if not isoform_fig:
                print("No figure data available for isoform clustergram - please select a gene first")
                raise PreventUpdate
            fig = go.Figure(isoform_fig)
            filename = f"{selected_gene}_isoform_clustergram.svg"

        elif plot_selection == 'junction':
            if not junction_fig:
                print("No figure data available for junction clustergram - please select a gene first")
                raise PreventUpdate
            fig = go.Figure(junction_fig)
            filename = f"{selected_gene}_junction_clustergram.svg"

        fig.update_layout(width=int(width_px), height=int(height_px), autosize=False)

        if title_legend_font_size:
            fig.update_layout(title={'font': {'size': title_legend_font_size}})

        if axis_labels_font_size:
            fig.update_xaxes(tickfont={'size': axis_labels_font_size})
            fig.update_yaxes(tickfont={'size': axis_labels_font_size})

        # export client call to generate image (supports remote service with fallback to local)
        export_client = get_export_client()
        print(f"Exporting {plot_selection} plot for {selected_gene}...")
        svg_bytes = export_client.export_figure(fig, format='svg', width=int(width_px), height=int(height_px))

        # TO DO: case where export fails: issue with how image bytes are being sent back to application VM?
        if svg_bytes is None:
            print("ERROR: Failed to export figure - export service returned None")
            return None, None, None
        print(f"Export successful! Received {len(svg_bytes)} bytes")
        svg_str = svg_bytes.decode('utf-8')

        download_data = dict(
            content=svg_str,
            filename=filename
        )
        print(f"Returning download data: filename={filename}, content_length={len(svg_str)}")

        if plot_selection == 'structure':
            return download_data, None, None
        
        elif plot_selection == 'isoform':
            return None, download_data, None
        
        else:
            return None, None, download_data

    except Exception as e:
        print(f"ERROR: Exception during export: {e}")
        traceback.print_exc()
        return None, None, None


@app.callback(
    dash.dependencies.Output('abundance-color-options-container', 'style'),
    dash.dependencies.Input('color-by-abundance-toggle', 'value')
)
def toggle_abundance_options(toggle_value):
    """Show/hide abundance options when toggle is turned on/off"""
    if toggle_value:
        return {'display': 'block'}

    else:
        return {'display': 'none'}


@app.callback(
    dash.dependencies.Output('structure-plot-colorscale-container', 'style'),
    [dash.dependencies.Input('color-junctions-by-psi-toggle', 'value'),
     dash.dependencies.Input('color-by-abundance-toggle', 'value')]
)
def toggle_structure_plot_colorscale(color_junctions_by_psi, color_by_abundance):
    """Show/hide structure plot colorscale when either coloring option is enabled"""
    if color_junctions_by_psi or color_by_abundance:
        return {'display': 'block'}
    
    else:
        return {'display': 'none'}


@app.callback(
    [dash.dependencies.Output('tissue-abundance-dropdown', 'options'),
     dash.dependencies.Output('tissue-abundance-dropdown', 'value'),
     dash.dependencies.Output('tissue-dropdown-container', 'style'),
     dash.dependencies.Output('organ-abundance-dropdown', 'options'),
     dash.dependencies.Output('organ-abundance-dropdown', 'value'),
     dash.dependencies.Output('organ-dropdown-container', 'style')],
    [dash.dependencies.Input('abundance-color-type-radio', 'value'),
     dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('species-dropdown', 'value')],
    prevent_initial_call=False
)
def update_abundance_dropdowns(color_type, selected_gene, species):
    """Update tissue and organ dropdown options and visibility based on selected coloring type"""
    tissue_options = []
    tissue_value = None
    tissue_style = {'display': 'none', 'marginTop': '10px'}
    organ_options = []
    organ_value = None
    organ_style = {'display': 'none', 'marginTop': '10px'}

    if color_type == 'tissue' and selected_gene:
        try:
            tissues = get_unique_tissues_for_gene(db_path, selected_gene, species)
            if tissues:
                tissue_options = [{'label': f' {tissue}', 'value': tissue} for tissue in tissues]
                tissue_value = tissue_options[0]['value'] if tissue_options else None
                tissue_style = {'display': 'block', 'marginTop': '10px'}

        except Exception as e:
            traceback.print_exc()

    elif color_type == 'organ' and selected_gene:
        try:
            organs = get_unique_organs_for_gene(db_path, selected_gene, species)
            if organs:
                organ_options = [{'label': f' {organ}', 'value': organ} for organ in organs]
                organ_value = organ_options[0]['value'] if organ_options else None
                organ_style = {'display': 'block', 'marginTop': '10px'}

        except Exception as e:
            traceback.print_exc()

    return tissue_options, tissue_value, tissue_style, organ_options, organ_value, organ_style


###################################################################
# VIEWPORT DIMENSIONS TRACKING
###################################################################
# neede for individual junction coloring popup positioning
app.clientside_callback(
    """
    function(n_intervals) {
        return {
            width: window.innerWidth,
            height: window.innerHeight
        };
    }
    """,
    dash.dependencies.Output('viewport-dimensions', 'data'),
    [dash.dependencies.Input('loading-delay-interval', 'n_intervals')]
)


###################################################################
# INTRO BANNER
###################################################################
def display_ascii_banner():
    """
    Displays ASCII art banner for Isoform Gazer!
    """
    init(autoreset=True)
    
    PURPLE = Fore.MAGENTA
    GOLD = Fore.YELLOW
    CYAN = Fore.CYAN
    WHITE = Fore.WHITE
    BLUE = Fore.BLUE
    RESET = Style.RESET_ALL
    
    banner = f"""
    {WHITE}    *       .  +     *           .    +    *        .        *    +         .   {GOLD}*{WHITE}    +       * *       .  +    
    {WHITE}       .    *         +    .      *         .  +       *      .    {GOLD}*{WHITE}        +         .    *     .    *      
    {CYAN}  +      *      .        *    +        .       *    +      .      *         {GOLD}*{WHITE}    .       ++      *      .    
    {WHITE}    .        +    *         .     +  {CYAN}   *{WHITE}     .    *     +   {GOLD}        *{CYAN}      .         * .        +    *    
    {GOLD} *     .          +    *       .    {CYAN}  *   *{WHITE}    *        .      +    *       {GOLD}*{CYAN}     +      .*     .          +  
    {WHITE}       +    *        .      *    {CYAN}   *     *{WHITE}   .    *         +        .       {GOLD}*{CYAN}     *       *        .      *
    {PURPLE}   ██╗███████╗ ██████╗ ███████╗ ██████╗ ██████╗ ███╗   ███╗  -*{GOLD}  ██████╗  █████╗ ███████╗███████╗██████╗ 
    {PURPLE} * ██║██╔════╝██╔═══██╗██╔════╝██╔═══██╗██╔══██╗████╗ ████║    {GOLD} ██╔════╝ ██╔══██╗╚══███╔╝██╔════╝██╔══██╗*+
    {PURPLE}-  ██║███████╗██║   ██║█████╗  ██║   ██║██████╔╝██╔████╔██║ *  {GOLD} ██║  ███╗███████║  ███╔╝ █████╗  ██████╔╝  .
    {PURPLE}  .██║╚════██║██║   ██║██╔══╝  ██║   ██║██╔══██╗██║╚██╔╝██║ ** {GOLD} ██║   ██║██╔══██║ ███╔╝  ██╔══╝  ██╔══██╗
    {PURPLE} * ██║███████║╚██████╔╝██║ *   ╚██████╔╝██║  ██║██║ ╚═╝ ██║  * {GOLD} ╚██████╔╝██║  ██║███████╗███████╗██║  ██║.    *-
    {PURPLE}   ╚═╝╚══════╝ ╚═════╝ ╚═╝     -╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝+   {GOLD}  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝                                                                                                
    {WHITE}       *    +         .    *      +         .    *        +     .   {GOLD}*{CYAN}      +         *       *    +         .  
    {CYAN}  +        .    *     {GOLD}   +    .  {WHITE}    *         +     .        *           .       {GOLD}*{WHITE}    + +        .    *     
    {WHITE}    .    *    {CYAN}   *{GOLD}    +    .         *      +    .        *    +        .       {GOLD}*{WHITE}     *.    *    {CYAN}  
    {WHITE} *     +   {CYAN}     *   *{GOLD}    .  {WHITE}    *    +        .       *         +      .    *       {GOLD}*{WHITE}    .*     +   {WHITE}    *  
    {CYAN}    .    {CYAN}    * *{WHITE}      +        .      *       +     .    *        +     {GOLD}*{WHITE}         .     *    .    {WHITE}   * * *{WHITE} +. 
    {WHITE}  +    *    {WHITE}     *{WHITE}    .    +       *        .      +     *       .         {GOLD}*{CYAN}    +       *  +    *    {WHITE}     *{WHITE} .
    {RESET}"""
    
    print(banner)


def warm_up_caches():
    """
    Warm up critical caches on application startup.
    Preloads gene list and default gene data to Redis if available.
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        if get_cached_gene_list() is None:
            logger.info("Warming up gene list cache...")
            genes = get_all_gene_options(db_path)
            logger.info(f"Cached {len(genes)} genes to Redis")
        else:
            logger.info("Gene list already cached in Redis")

    except Exception as e:
        logger.warning(f"Cache warm-up skipped (will cache on first request): {e}")


if __name__ == '__main__':
    database_exists = check_database_status()
    if not database_exists:
        print("Database initialization completed.")
        #verify_database_schema(db_path)

    display_ascii_banner()

    warm_up_caches()

    app.run(debug=False, port=8050, use_reloader=False)