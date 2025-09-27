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
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate
from colorama import Fore, Style, init
import logging
logging.getLogger('dash.dash').setLevel(logging.WARNING)
from data_utils import get_master_table_columns, parse_filter_query, query_master_table, get_gene_options, create_custom_spinner, validate_filter_input
from junction_utils import (
    create_summary_clustergram, create_gene_clustergram,
    load_atse_data, process_gene_atse_data, create_empty_atse_message, 
    create_junction_exon_visualization, create_empty_clustergram_message, 
    filter_junctions_by_transcripts, filter_transcripts_by_junctions
)
from isoform_utils import (
    load_expression_data, process_transcript_structure, create_transcript_structure_plot, 
    create_isoform_expression_clustergram, create_empty_isoform_message, 
    calculate_unified_plot_height, calculate_single_isoform_hash, 
    parse_gtf_and_calculate_hashes, generate_annotated_gtf
)

RANDOM_SEED = 18
np.random.seed(RANDOM_SEED)

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


def setup_local_database(force_rebuild=False):
    """
    Sets up SQLite database from data files.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    db_path = os.path.join(data_dir, "isoformgazer.db")
    if Path(db_path).exists() and not force_rebuild:
        return db_path
    
    print()
    print(f"Creating new database at {db_path}.")
    print()

    if Path(db_path).exists():
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL") 
    conn.execute("PRAGMA cache_size = 50000") 
    conn.execute("PRAGMA temp_store = MEMORY")

    ########################################################
    # Load isoform master table data
    ########################################################
    isoform_file = os.path.join(data_dir, "mt_isoform_gazers_250828.tsv")
    
    with tqdm(desc="Loading isoform master table data", unit=" rows") as pbar:
        df_isoform = pd.read_csv(isoform_file, sep='\t')
        df_isoform['id'] = df_isoform.index # IMPORTANT: we need this to match PSL file 1-based indexing!
        df_isoform['gene_id'] = df_isoform['gene'].str.split('.').str[0]

        columns_to_drop = ['gene_total_tpm', 'gene_gencode_v46_basic_transcript_counts']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df_isoform.columns]
        if existing_columns_to_drop:
            df_isoform = df_isoform.drop(columns=existing_columns_to_drop)

        pbar.update(len(df_isoform))
    
    with tqdm(desc="Writing isoform master table data to local database", unit="rows", total=len(df_isoform)) as pbar:
        df_isoform.to_sql('isoforms', conn, if_exists='replace', index=False)
        pbar.update(len(df_isoform))
    
    print(f"✓ Processed all {len(df_isoform):,} rows from isoform master table!")
    print()

    ########################################################
    # Load isoform PSL data
    ########################################################
    print("Processing PSL data...")
    psl_file = os.path.join(data_dir, "all_samples_sp_collapse_all_chr_no_treatment_hashid_isoform_full.psl")
    psl_columns = [
        'matches', 'misMatches', 'repMatches', 'nCount', 'qNumInsert', 'qBaseInsert',
        'tNumInsert', 'tBaseInsert', 'strand', 'qName', 'qSize', 'qStart', 'qEnd',
        'tName', 'tSize', 'tStart', 'tEnd', 'blockCount', 'blockSizes', 'qStarts', 'tStarts'
    ]

    # Get total PSL rows for progress tracking
    total_psl_rows = sum(1 for _ in open(psl_file, 'r')) 
    chunk_size = 100000

    conn.execute("""
        CREATE TABLE IF NOT EXISTS psl_data (
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
                
            chunk.to_sql('psl_data', conn, if_exists='append', index=False)
            pbar.update(len(chunk))

    print("Processing isoform TPM data...")
    tpm_file = os.path.join(data_dir, "all_tpm_250828.tsv")
    tpm_df = pd.read_csv(tpm_file, sep='\t')
    tpm_df['id'] = tpm_df.index
    tpm_df.to_sql('tpm_data', conn, if_exists='replace')

    print("Processing isoform ratio data...")
    ratio_file = os.path.join(data_dir, "all_quant_ratio_250828.tsv")
    ratio_df = pd.read_csv(ratio_file, sep='\t')
    ratio_df['id'] = ratio_df.index
    ratio_df.to_sql('ratio_data', conn, if_exists='replace')

    print("Updating isoform transcript names with full names from PSL data...")
    conn.execute("""
        UPDATE isoforms 
        SET transcript = (
            SELECT psl.trans_id 
            FROM psl_data psl 
            WHERE psl.id = isoforms.id
        )
        WHERE EXISTS (
            SELECT 1 FROM psl_data psl WHERE psl.id = isoforms.id
        )
    """)
    conn.commit()

    print("Creating isoform indexes...")
    conn.execute("CREATE INDEX idx_psl_gene ON psl_data(id)")
    conn.execute("CREATE INDEX idx_iso_gene ON isoforms(gene_name, id)")
    conn.commit()

    ########################################################
    # Load junction master table data
    ########################################################
    junction_file = os.path.join(data_dir, "pseudobulk_final_broad_cell_type_20250623_171456_withmappings.csv")
    
    # Need to count total lines (minus header) to estimate progress
    with open(junction_file, 'r') as f:
        total_lines = sum(1 for _ in f) - 1 
    
    chunk_size = 100000
    estimated_chunks = (total_lines // chunk_size) + 1
    
    print(f"Loading {total_lines:,} rows of junction data in groupings of {chunk_size:,} rows...")
    first_chunk = True
    row_count = 0

    column_order = [
        'gene_symbol',
        'gene_id',
        'event_id', 
        'junction_id',
        'junction_id_index',
        'atse_count',
        'junction_count',
        'cell_type',
        'n_cells',
        'psi',
        'matched_transcript_ids'
    ]
    
    with tqdm(desc="Writing junction master table data to local database", 
              unit="chunk", 
              total=estimated_chunks) as chunk_pbar:
            
        for i, chunk in enumerate(pd.read_csv(junction_file, 
                                              chunksize=chunk_size,
                                              low_memory=False)):
            available_columns = [col for col in column_order if col in chunk.columns]
            remaining_columns = [col for col in chunk.columns if col not in column_order]
            
            final_column_order = available_columns + remaining_columns
            chunk = chunk[final_column_order]


            if first_chunk:
                chunk.to_sql('junctions', conn, if_exists='replace', index=False)
                first_chunk = False
            else:
                chunk.to_sql('junctions', conn, if_exists='append', index=False)
            
            row_count += len(chunk)
            chunk_pbar.update(1)
            
            chunk_pbar.set_postfix({
                'rows': f"{row_count:,}",
                'chunk': f"{i+1}/{estimated_chunks}"
            })
    
    print(f"✓ Processed all {row_count:,} rows from junction master table!")
    # rename 'gene_symbol' to 'gene_name' for more consistency in junctions table (match isoforms table)
    conn.execute("ALTER TABLE junctions RENAME COLUMN gene_symbol TO gene_name")
    print()

    ########################################################
    # Load ATSE data into database
    ########################################################
    atse_file = os.path.join(data_dir, "TMS_atse_file_unanno_also_2025-05-11_06-23-05.tsv")
    
    if os.path.exists(atse_file):
        print("Processing ATSE data...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS atse_data (
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
                
                chunk[required_columns].to_sql('atse_data', conn, if_exists='append', index=False)
                pbar.update(len(chunk))

        
        print(f"✓ Processed {total_lines:,} ATSE records!")
    
        print("Creating optimized database indices...")
        indices = [
            ("idx_junctions_gene", "junctions", "(gene_name, gene_id)"),
            ("idx_junctions_gene_name", "junctions", "(gene_name)"),  
            ("idx_isoforms_gene", "isoforms", "(gene_name, gene_id)"),
            ("idx_isoforms_gene_name", "isoforms", "(gene_name)"), 
            ("idx_isoforms_id", "isoforms", "(id)"), 
            ("idx_psl_gene", "psl_data", "(gene_id, id)"),
            ("idx_psl_id", "psl_data", "(id)"),  
            ("idx_junctions_junction_id", "junctions", "(junction_id)"),
            ("idx_junctions_psi", "junctions", "(gene_name, psi)"),
            ("idx_junctions_cell_type", "junctions", "(cell_type)"),  
            ("idx_tpm_id", "tpm_data", "(id)"), 
            ("idx_ratio_id", "ratio_data", "(id)") 
        ]
        
        with tqdm(desc="Creating indices", total=len(indices) + 3, unit="index") as idx_pbar:
            for idx_name, table_name, columns in indices:
                conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name}{columns}")
                idx_pbar.update(1)
            
            conn.execute("ALTER TABLE atse_data ADD COLUMN gene_id_clean TEXT")
            conn.execute("UPDATE atse_data SET gene_id_clean = SUBSTR(gene_id, 1, INSTR(gene_id, '.') - 1) WHERE gene_id LIKE '%.%'")
            conn.execute("UPDATE atse_data SET gene_id_clean = gene_id WHERE gene_id_clean IS NULL")
            idx_pbar.update(1)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_atse_gene ON atse_data(gene_id_clean, gene_name)")
            idx_pbar.update(1)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_atse_coords ON atse_data(chromosome, start, end)")
            idx_pbar.update(1)
    
    conn.commit()
    conn.close()
    
    print("✓ Database setup complete!")
    print() 

    return db_path


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
# APPLICATION SETUP
###################################################################
db_path = setup_local_database()
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
        'minWidth': '100px', 
        'maxWidth': '220px',
        'padding': '10px 8px',
        'whiteSpace': 'normal',
        'font-family': '"Open Sans", sans-serif',
        'fontSize': '12px',
        'lineHeight': '1.4',
        'border': '1px solid #e1e5e9'
    },
    style_table={
        'overflowY': 'visible',
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
        'minWidth': '100px', 
        'maxWidth': '220px',
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
                                'Isoform Gazer provides a unified view of RNA splicing across both short-read junction-level single-cell data and long-read transcript-level isoform data in GENCODEv46 (GRCh38.p14).'
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
                        html.H2('Search by Gene', className='alignment-settings-section'),
                        html.Div(className='app-controls-block', children=[
                            dcc.Dropdown(
                                id='gene-search-dropdown',
                                options=[
                                    {'label': 'A1BG-AS1', 'value': 'A1BG-AS1'},
                                    {'label': 'RBFOX2 (RNA Binding Fox-1 Homolog 2)', 'value': 'RBFOX2'},
                                    {'label': 'EGFR (Epidermal growth factor receptor)', 'value': 'EGFR'},
                                    {'label': 'BRCA1 (Breast cancer type 1)', 'value': 'BRCA1'},
                                    {'label': 'TARDBP (TAR DNA Binding Protein)', 'value': 'TARDBP'},
                                    {'label': 'TP53 (Tumor protein p53)', 'value': 'TP53'}
                                ],
                                placeholder="Type to search for a gene...",
                                value='A1BG-AS1',
                                searchable=True,
                                clearable=True
                            ),
                            html.Div(className='app-controls-desc', children='Select a gene identifier to query or type to search')
                        ]),
                        html.Div(id='gene-filter-status'),

                        html.H3('Gene-Level Summary', className='summary-section-header'),
                        html.Div(id='gene-level-summary', className='summary-block', children=[
                            html.P("Select a gene to view summary information", className='summary-placeholder')
                        ]),

                        html.H3('ORF-Level Summary', className='summary-section-header'),
                        html.Div(id='orf-level-summary', className='summary-block', children=[
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
                            html.P("Generate hash IDs for isoforms based on their junction coordinates."),

                            html.H3('Coordinate Entry', className='summary-section-header'),
                            html.Div(className='app-controls-block', children=[
                                html.Div(className='coordinate-mode-selector', children=[
                                    html.Label('Input Mode:', className='coordinate-section-label'),
                                    dcc.RadioItems(
                                        id='coordinate-input-mode',
                                        options=[
                                            {'label': 'Start positions +\nBlock sizes', 'value': 'start_block'},
                                            {'label': 'Start positions +\nEnd positions', 'value': 'start_end'}
                                        ],
                                        value='start_block',
                                        className='coordinate-radio-items',
                                        inline=True
                                    )
                                ]),

                                html.Div(className='coordinate-input-group', children=[
                                    html.Label('Exon Start Positions (comma-separated):', className='coordinate-section-label'),
                                    dcc.Input(
                                        id='exon-starts-input',
                                        type='text',
                                        placeholder='e.g., 1000,2000,3000',
                                        className='coordinate-input'
                                    )
                                ]),

                                html.Div(id='second-input-container', children=[
                                    html.Div(className='coordinate-input-group', children=[
                                        html.Label('Block Sizes (comma-separated):', className='coordinate-section-label', id='second-input-label'),
                                        dcc.Input(
                                            id='second-coordinate-input',
                                            type='text',
                                            placeholder='e.g., 200,300,400',
                                            className='coordinate-input'
                                        )
                                    ])
                                ]),

                                html.Button('Calculate Hash', id='calculate-hash-btn', className='control-button'),
                                html.Div(id='hash-result')
                            ]),

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
                                        html.Button('Download Annotated GTF', id='download-annotated-gtf-btn', className='control-button'),
                                        html.Button('Download Hash Results', id='download-hashes-btn', className='control-button')
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
                        # General Controls Section
                        #####################################
                        html.H2('General', className='alignment-settings-section'),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Plots to Show'),
                            dcc.Dropdown(
                                id='overview-dropdown',
                                className='app-controls-block-dropdown',
                                options=[
                                    {'label': 'Structure Plots', 'value': 'event-level'},
                                    {'label': 'Clustergrams', 'value': 'clustergram'},
                                    {'label': 'Both', 'value': 'both'}
                                ],
                                value='both'
                            ),
                            html.Div(className='app-controls-desc', children='Select which visualizations to show')
                        ]),

                        #####################################
                        # Isoform-level Event Plot Section
                        #####################################
                        html.H2('Transcript Plots', className='alignment-settings-section'),
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
                            html.Div(className='app-controls-desc', children='Toggle to hide junctions from the structure plot and show transcript structure instead')
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
                                min=600,
                                max=1600,
                                step=100,
                                value=710,
                                marks={600: '600', 700: '', 800: '800', 900: '', 1000: '1000', 1100: '', 1200: '1200', 1300: '', 1400: '1400', 1500: '', 1600: '1600'}
                            ),
                            html.Div(className='app-controls-desc', children='Adjust the height of the clustergrams')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='toggle-switch-row', children=[
                                html.Div('Isoform Clustergram Unit', className='app-controls-name toggle-switch-label-narrow'),
                                daq.ToggleSwitch(
                                    id='isoform-data-type-switch',
                                    value=False,  # False = TPM, True = Ratio
                                    label={'label': 'TPM / Ratio', 'style': {'fontSize': '12px', 'color': '#506784'}},
                                    labelPosition='right',
                                    className='toggle-switch-inline'
                                )
                            ]),
                            html.Div(className='app-controls-desc', children='Toggle whether isoform clustergram shows TPM values or ratio values across all tissues')
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
                                    value=True,
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
                                value='Viridis'
                            ),
                            html.Div(className='app-controls-desc', children='Choose the color theme of the heatmaps')
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
                                value='euclidean'
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
                                value='ward'
                            ),
                            html.Div(className='app-controls-desc', children='Hierarchical clustering algorithm used for both clustergrams')
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
                                html.Div(id='top-junction-structure-plot-container', className='atse-container', children=[
                                    html.Div(className='loading-container plot-container-full-height', id='top-structure-plot-container-style', children=[
                                        dcc.Loading(
                                            id="loading-top-structure-plot",
                                            type="default",
                                            color='#EDAE49',
                                            delay_hide=500,
                                            children=[
                                                dcc.Graph(
                                                    id='atse-map',
                                                    figure=create_empty_atse_message("Select a gene to view splice junctions and exons"),
                                                    config={
                                                        'responsive': True,
                                                        'displayModeBar': True,
                                                        'scrollZoom': True
                                                    },
                                                    className='graph-full-size'
                                                )
                                            ]
                                        ),
                                        html.Div(id="atse-map-loading-message", className="custom-loading-message")
                                    ])
                                ]),
                                # Transcript Structure Plot (hidden by default)
                                html.Div(id='top-transcript-structure-plot-container', className='barplot-container hidden', children=[
                                    html.Div(className='loading-container plot-container-full-height', children=[
                                        dcc.Loading(
                                            id="loading-top-transcript-plot",
                                            type="default",
                                            color='#EDAE49',
                                            delay_show=500,
                                            delay_hide=200,
                                            children=[dcc.Graph(id='top-barplot')]
                                        ),
                                        html.Div(id="top-barplot-loading-message", className="custom-loading-message")
                                    ])
                                ])
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
                                    delay_show=500,
                                    delay_hide=200,
                                    children=[dcc.Graph(
                                        id='heatmap1',
                                        className='graph-full-size',
                                        config={'responsive': True}
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
                                    delay_hide=500,
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
                                            config={'responsive': True},
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
                        dbc.Button(
                            "Clear Filters",
                            id='clear-left-filters',
                            color="secondary",
                            size="sm",
                            className="clear-filters-btn",
                            disabled=True
                        )
                    ]),
                    # Isoform table filter error popup
                    html.Div(
                        id='left-table-error-popup',
                        className='filter-error-popup hidden',
                        children=[]
                    ),
                    left_data_table
                ]),
                html.Div(className='table-container', id='table2-container', children=[
                    html.Div(className='table-header-controls', children=[
                        dbc.Button(
                            "Clear Filters",
                            id='clear-right-filters',
                            color="secondary",
                            size="sm",
                            className="clear-filters-btn",
                            disabled=True
                        )
                    ]),
                    # Junction table filter error popup
                    html.Div(
                        id='right-table-error-popup',
                        className='filter-error-popup hidden',
                        children=[]
                    ),
                    right_data_table
                ])
            ])
        ])
    ])
])

app.layout.children.extend([
    dcc.Store(id='filtered-isoform-store', data=[]),
    dcc.Store(id='filtered-junction-store', data=[]),
    dcc.Store(id='isoform-full-data-store', data=[]),
    dcc.Store(id='junction-full-data-store', data=[]),
    dcc.Store(id='table-callback-prevention', data=False),
    dcc.Store(id='initial-loading-complete', data=False),
    dcc.Store(id='exon-color-store', data='#2E86C1'),
    dcc.Store(id='junction-color-store', data='#85929E'),
    dcc.Store(id='loading-progress-store', data=0),
    dcc.Store(id='left-table-validation-store', data={'valid': True, 'errors': {}}),
    dcc.Store(id='right-table-validation-store', data={'valid': True, 'errors': {}}),
    dcc.Store(id='gtf-hash-results-store', data=[]),
    dcc.Interval(id='loading-delay-interval', interval=1000, n_intervals=0, max_intervals=1, disabled=True),
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
    Notes: 
    - We are assuming it takes ~5.65 seconds to load the initial data based on testing
    - Currently using 50ms interval updates, meaning 113 steps to reach 100% (~0.885% per timestep)
    """
    if loading_complete:
        return 100, True
    
    new_progress = min(progress_intervals * 0.885, 100)
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
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    
    # Progress arc (clockwise)
    progress_theta = np.linspace(-np.pi/2, -np.pi/2 + 2 * np.pi * (progress / 100), max(int(progress), 2))
    if len(progress_theta) > 0:
        x_progress = radius * np.cos(progress_theta)
        y_progress = radius * np.sin(progress_theta)
    else:
        x_progress = []
        y_progress = []
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_circle, y=y_circle,
        mode='lines',
        line=dict(color='rgba(255,255,255,0.5)', width=8),
        showlegend=False,
        hoverinfo='none'
    ))
    
    if len(x_progress) > 0:
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
        
        # Glow effect using slightly larger, more transparent line around original
        fig.add_trace(go.Scatter(
            x=x_progress, y=y_progress,
            mode='lines',
            line=dict(
                color='rgba(255,255,255,0.4)', 
                width=16
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
        height=450
    )
    
    return fig

#######################################################################
# FILTER VALIDATION CALLBACKS
#######################################################################
@app.callback(
    [dash.dependencies.Output('left-table-error-popup', 'style'),
     dash.dependencies.Output('left-table-error-popup', 'children'),
     dash.dependencies.Output('left-table-validation-store', 'data')],
    [dash.dependencies.Input('left_data_table', 'filter_query')]
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
    [dash.dependencies.Input('right_data_table', 'filter_query')]
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
# INITIAL LOADING SCREEN CALLBACK
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
    
    data_loaded = bool(isoform_data and junction_data)
    if data_loaded and timer_intervals == 0:
        return 'loading-overlay', False, False 
    
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
     dash.dependencies.Input('left_data_table', 'filter_query'),
     dash.dependencies.Input('right_data_table', 'filter_query'),
     dash.dependencies.Input('left-table-validation-store', 'data'),
     dash.dependencies.Input('right-table-validation-store', 'data')]
)
def update_filtered_data_stores(isoform_full_data, junction_full_data, selected_gene, isoform_filter_query, junction_filter_query, left_validation, right_validation):
    """Store ALL filtered transcript/junction IDs from FULL datasets with transcript-based junction filtering"""
    # Check if either filter is invalid: if so, don't update filtered stores since user will need to fix errors before query proceeds
    if ((isoform_filter_query and left_validation and not left_validation.get('valid', True)) or
        (junction_filter_query and right_validation and not right_validation.get('valid', True))):
        raise PreventUpdate
    
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
        if has_isoform_filters or has_junction_filters:
            transcript_based_junction_ids = []
            junction_based_transcript_ids = []
            
            # Isoform filtering → Junction filtering
            if selected_gene and has_isoform_filters and filtered_transcript_ids:
                try:
                    transcript_based_junction_ids = filter_junctions_by_transcripts(
                        db_path, selected_gene, filtered_transcript_ids
                    )
                except Exception as e:
                    print(f"Error in transcript-based junction filtering: {e}")
            
            # Junction filtering → Isoform filtering
            if has_junction_filters and filtered_junction_ids:
                try:
                    junction_based_transcript_ids = filter_transcripts_by_junctions(
                        db_path, filtered_junction_ids
                    )
                except Exception as e:
                    print(f"Error in junction-based transcript filtering: {e}")
            
            # Intersect filters from both master tables for joint filtering
            if has_isoform_filters and has_junction_filters:
                final_transcript_ids = list(set(filtered_transcript_ids) & set(junction_based_transcript_ids)) if junction_based_transcript_ids else filtered_transcript_ids
                final_junction_ids = list(set(filtered_junction_ids) & set(transcript_based_junction_ids)) if transcript_based_junction_ids else filtered_junction_ids

            elif has_isoform_filters:
                final_transcript_ids = filtered_transcript_ids
                final_junction_ids = transcript_based_junction_ids

            elif has_junction_filters:
                final_transcript_ids = junction_based_transcript_ids
                final_junction_ids = filtered_junction_ids

            else:
                final_transcript_ids = filtered_transcript_ids
                final_junction_ids = filtered_junction_ids
            
            filtered_transcript_ids = final_transcript_ids
            filtered_junction_ids = final_junction_ids
        
        return filtered_transcript_ids, filtered_junction_ids
    
    except Exception as e:
        print(f"Error updating filtered data stores: {e}")
        return [], []


@app.callback(
    [dash.dependencies.Output('left_data_table', 'filter_query'),
     dash.dependencies.Output('right_data_table', 'filter_query')],
    [dash.dependencies.Input('clear-left-filters', 'n_clicks'),
     dash.dependencies.Input('clear-right-filters', 'n_clicks'),
     dash.dependencies.Input('gene-search-dropdown', 'value')],
    [dash.dependencies.State('left_data_table', 'filter_query'),
     dash.dependencies.State('right_data_table', 'filter_query')]
)
def clear_filters(left_clicks, right_clicks, selected_gene, left_filter, right_filter):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Clear filters if Clear All buttons are clicked
    if button_id == 'clear-left-filters':
        return '', right_filter
    
    elif button_id == 'clear-right-filters':
        return left_filter, ''
    
    # Also clear all filters when gene changes
    elif button_id == 'gene-search-dropdown':
        return '', ''

    return left_filter, right_filter


@app.callback(
    [dash.dependencies.Output('clear-left-filters', 'disabled'),
     dash.dependencies.Output('clear-right-filters', 'disabled')],
    [dash.dependencies.Input('left_data_table', 'filter_query'),
     dash.dependencies.Input('right_data_table', 'filter_query')]
)
def update_button_states(left_filter, right_filter):
    left_disabled = not left_filter or left_filter.strip() == ''
    right_disabled = not right_filter or right_filter.strip() == ''
    return left_disabled, right_disabled


##############################################################################################
# CALLBACK FOR QUERYING BY GENE IN CONTROL PANEL ('Query' tab): if no search is performed, we 
# show the first five gene names, but otherwise filter by the top ten matches to the current 
# search string. 
##############################################################################################
@app.callback(
    [dash.dependencies.Output('gene-search-dropdown', 'options'),
     dash.dependencies.Output('gene-search-dropdown', 'value')],
    [dash.dependencies.Input('gene-search-dropdown', 'search_value')],
    [dash.dependencies.State('gene-search-dropdown', 'value')]
)
def update_gene_options(search_value, current_value):
    """Update gene options while preserving current selection"""
    
    if not search_value:
        options = get_gene_options(db_path, limit=5)
        a1bg_option = {'label': 'A1BG-AS1', 'value': 'A1BG-AS1'}
        if not any(opt['value'] == 'A1BG-AS1' for opt in options):
            options.insert(0, a1bg_option)
    else:
        options = get_gene_options(db_path, search_term=search_value, limit=10)
    
    if current_value is None:
        return options, 'A1BG-AS1'
    
    option_values = [opt['value'] for opt in options]
    if current_value in option_values:
        return options, current_value
    else:
        current_options = get_gene_options(db_path, search_term=current_value, limit=1)
        if current_options:
            options = current_options + [opt for opt in options if opt['value'] != current_value]
        return options, current_value


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
     dash.dependencies.Output('orf-level-summary', 'children')],
    [dash.dependencies.Input('gene-search-dropdown', 'value')]
)
def update_summary_blocks(selected_gene):
    """Update Gene-Level and ORF-Level summary blocks based on selected gene"""
    if not selected_gene:
        return (
            [html.P("Select a gene to view summary information", className='summary-placeholder')],
            [html.P("Select a gene to view summary information", className='summary-placeholder')]
        )

    try:
        # Get the first row data for the selected gene, since these columns have the same values for all rows
        conn = sqlite3.connect(db_path)
        query = """
        SELECT
            gene_protein_category,
            gene_potential,
            gene_perplexity,
            ptc_potential,
            ptc_perplexity,
            gene_average_tpm,
            gene_expressed_samples,
            orf_potential,
            orf_perplexity,
            orf_expressed_samples
        FROM isoforms
        WHERE gene_name = ?
        LIMIT 1
        """
        cursor = conn.cursor()
        cursor.execute(query, (selected_gene,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return (
                [html.P("No data found for selected gene", className='summary-placeholder')],
                [html.P("No data found for selected gene", className='summary-placeholder')]
            )

        (gene_protein_category, gene_potential, gene_perplexity, ptc_potential, 
         ptc_perplexity, gene_average_tpm, gene_expressed_samples, 
         orf_potential, orf_perplexity, orf_expressed_samples) = row

        gene_summary = [
            html.Div(className='summary-item', children=[
                html.Span('Gene Protein Category:', className='summary-label'),
                format_protein_category(gene_protein_category)
            ]),
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
        ]

        orf_summary = [
            html.Div(className='summary-item', children=[
                html.Span('Number of detected ORFs:', className='summary-label'),
                html.Span(format_value(orf_potential, 'orf_potential'), className='summary-value')
            ]),
            html.Div(className='summary-item', children=[
                html.Span('ORF Perplexity:', className='summary-label'),
                html.Span(format_value(orf_perplexity), className='summary-value')
            ]),
            html.Div(className='summary-item', children=[
                html.Span('ORF Expressed Samples:', className='summary-label'),
                html.Span(format_value(orf_expressed_samples, 'ORF_expressed_samples'), className='summary-value')
            ])
        ]

        return gene_summary, orf_summary

    except Exception as e:
        print(f"Error updating summary blocks: {e}")
        return (
            [html.P("Error loading summary data", className='summary-placeholder')],
            [html.P("Error loading summary data", className='summary-placeholder')]
        )


######################################################################
# SQLLITE MASTER TABLE PROCESSING CALLBACKS
######################################################################
@app.callback(
    [dash.dependencies.Output('left_data_table', 'data'),
     dash.dependencies.Output('left_data_table', 'page_count'),
     dash.dependencies.Output('isoform-full-data-store', 'data')],
    [dash.dependencies.Input('left_data_table', 'page_current'),
     dash.dependencies.Input('left_data_table', 'page_size'),
     dash.dependencies.Input('left_data_table', 'sort_by'),
     dash.dependencies.Input('left_data_table', 'filter_query'),
     dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('left-table-validation-store', 'data')]
)
def update_isoform_table(page_current, page_size, sort_by, filter_query, selected_gene, validation_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    # Check if current filter is valid before processing
    if filter_query and validation_data and not validation_data.get('valid', True):
        raise PreventUpdate
    
    filters = parse_filter_query(db_path, filter_query, table_name='isoforms')
    
    _, total_count = query_master_table(
        db_path,
        table_name='isoforms',
        page=0,
        page_size=0,
        sort_by=None,
        filters=filters,
        gene_filter=selected_gene
    )
    
    full_data, _ = query_master_table(
        db_path,
        table_name='isoforms',
        page=0,
        page_size=total_count,
        sort_by=sort_by,
        filters=filters,
        gene_filter=selected_gene
    )
    
    # Handle None values for pagination params
    page_current = page_current or 0
    page_size = page_size or 10
    
    start_idx = page_current * page_size
    end_idx = (page_current + 1) * page_size
    paginated_data = full_data[start_idx:end_idx]
    page_count = math.ceil(total_count / page_size) if page_size else 1
    
    return paginated_data, page_count, full_data


@app.callback(
    [dash.dependencies.Output('right_data_table', 'data'),
     dash.dependencies.Output('right_data_table', 'page_count'),
     dash.dependencies.Output('junction-full-data-store', 'data')],
    [dash.dependencies.Input('right_data_table', 'page_current'),
     dash.dependencies.Input('right_data_table', 'page_size'),
     dash.dependencies.Input('right_data_table', 'sort_by'),
     dash.dependencies.Input('right_data_table', 'filter_query'),
     dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('right-table-validation-store', 'data')]
)
def update_junction_table(page_current, page_size, sort_by, filter_query, selected_gene, validation_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    # Check if current filter is valid before processing
    if filter_query and validation_data and not validation_data.get('valid', True):
        raise PreventUpdate
    
    filters = parse_filter_query(db_path, filter_query, table_name='junctions')
    
    _, total_count = query_master_table(
        db_path,
        table_name="junctions",
        page=0,
        page_size=0,
        sort_by=None,
        filters=filters,
        gene_filter=selected_gene
    )
    
    full_data, _ = query_master_table(
        db_path,
        table_name="junctions",
        page=0,
        page_size=total_count,
        sort_by=sort_by,
        filters=filters,
        gene_filter=selected_gene
    )
    
    # Handle None values for pagination params
    page_current = page_current or 0
    page_size = page_size or 10
    
    start_idx = page_current * page_size
    end_idx = (page_current + 1) * page_size
    paginated_data = full_data[start_idx:end_idx]
    page_count = math.ceil(total_count / page_size) if page_size else 1
    
    return paginated_data, page_count, full_data


######################################################################
# CALLBACK CONTROLLING WHICH PLOTS ARE VISIBLE (DROPDOWN OPTION)
######################################################################
@app.callback(
    [
        dash.dependencies.Output('isoform-clustergram-container', 'style'),
        dash.dependencies.Output('junction-clustergram-container', 'style'),
        dash.dependencies.Output('panels-container', 'className'),
        dash.dependencies.Output('left-panel', 'className'),
        dash.dependencies.Output('right-panel', 'className'),
    ],
    [dash.dependencies.Input('overview-dropdown', 'value')]
)
def toggle_plot_visibility(overview_value):
    """Enhanced plot visibility toggle with proper container sizing"""
    show_full = {'display': 'block', 'height': '100%', 'flex': '1 1 auto'}
    hide = {'display': 'none', 'height': '0', 'flex': '0 0 0'}
    show_normal = {'display': 'block', 'height': '100%'}

    panels_class = 'panels-container'
    left_panel_class = 'panel'
    right_panel_class = 'panel'

    if overview_value == 'both':
        return show_normal, show_normal, panels_class, left_panel_class, right_panel_class

    # Show only structure plots, hide clustergrams
    elif overview_value == 'event-level':
        left_panel_class = 'panel full-panel-event'
        right_panel_class = 'panel full-panel-event'
        return hide, hide, panels_class, left_panel_class, right_panel_class

    # Show only clustergrams, hide structure plots
    elif overview_value == 'clustergram':
        left_panel_class = 'panel full-panel-clustergram'
        right_panel_class = 'panel full-panel-clustergram'
        return show_full, show_full, panels_class, left_panel_class, right_panel_class

    else:
        return show_normal, show_normal, panels_class, left_panel_class, right_panel_class


######################################################################
# DYNAMIC HEIGHT CALCULATION AND PANEL ADJUSTMENT
######################################################################
@app.callback(
    [dash.dependencies.Output('bar-height-slider', 'value'),
     dash.dependencies.Output('left-panel', 'style', allow_duplicate=True),
     dash.dependencies.Output('right-panel', 'style', allow_duplicate=True)],
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('filtered-isoform-store', 'data'),
     dash.dependencies.Input('filtered-junction-store', 'data')],
    [dash.dependencies.State('bar-height-slider', 'value')],
    prevent_initial_call=True
)
def update_dynamic_height_and_panels(selected_gene, filtered_isoform_ids, filtered_junction_ids, current_height):
    """Calculate unified height for both plots and update slider and panels when gene changes"""
    if not selected_gene:
        # Return current values if no gene selected
        panel_style = {'height': '760px', 'minHeight': '760px'}
        return current_height, panel_style, panel_style
    
    try:
        filtered_ids = [int(id) for id in filtered_isoform_ids] if filtered_isoform_ids else []
        transcript_data = process_transcript_structure(db_path, selected_gene, filtered_ids)
        gene_data = process_gene_atse_data(selected_gene, db_path, filtered_junction_ids)
        
        # Calculate unified height for structure plots based on both transcript and junction data: use max height from either
        calculated_height = calculate_unified_plot_height(transcript_data, gene_data)
        
        if abs(calculated_height - current_height) < 100:
            calculated_height = current_height
        
        # Calculate panel heights (only clustergrams now, no transcript plot in panels)
        base_panel_height = 710 + 50  # clustergram height + margins
        panel_height = max(base_panel_height, 760)
        
        panel_style = {
            'height': f'{panel_height}px',
            'minHeight': f'{panel_height}px'
        }
        
        plot_container_style = {
            'height': f'{calculated_height}px',
            'minHeight': f'{max(calculated_height - 50, 500)}px',
            'maxHeight': f'{calculated_height + 100}px'
        }
        
        return calculated_height, panel_style, panel_style
        
    except Exception as e:
        print(f"Error calculating dynamic height: {e}")

        # Return current values on error to avoid breaking layout
        panel_style = {'height': '760px', 'minHeight': '760px'}

        return current_height, panel_style, panel_style


######################################################################
# DYNAMIC PANEL HEIGHT ADJUSTMENT (Manual Slider Changes)
######################################################################
@app.callback(
    [dash.dependencies.Output('left-panel', 'style'),
     dash.dependencies.Output('right-panel', 'style')],
    [dash.dependencies.Input('bar-height-slider', 'value')],
    prevent_initial_call=True
)
def adjust_panel_heights(plot_height):
    """Panel heights are now fixed since only clustergrams remain in panels"""

    # Fixed panel height for clustergrams only (no transcript plots in panels)
    base_panel_height = 710 + 50  # clustergram height + margins
    panel_height = max(base_panel_height, 760)

    panel_style = {
        'height': f'{panel_height}px',
        'minHeight': f'{panel_height}px'
    }

    return panel_style, panel_style


######################################################################
# HEATMAP PROCESSING CALLBACKS
######################################################################
@app.callback(
    dash.dependencies.Output('heatmap2', 'figure'),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('colorscale-dropdown', 'value'),
     dash.dependencies.Input('filtered-junction-store', 'data'),
     dash.dependencies.Input('overview-dropdown', 'value'),
     dash.dependencies.Input('show-celltype-labels-toggle', 'value'),
     dash.dependencies.Input('clustergram-height-slider', 'value'),
     dash.dependencies.Input('distance-metric-dropdown', 'value'),
     dash.dependencies.Input('linkage-method-dropdown', 'value')]
)
def update_junction_clustergram(selected_gene, colorscale, 
                                filtered_junction_ids, plots_dropdown_value, show_celltype_labels, clustergram_height,
                                distance_metric, linkage_method):
    """Update junction visualization based on gene selection and filtering"""
    if plots_dropdown_value == 'event-level':                       
        return empty_fig() 
    
    heatmap_height = clustergram_height
    
    if not selected_gene:
        try:
            fig = create_summary_clustergram(db_path, 
                                             height=heatmap_height, 
                                             colorscale=colorscale, 
                                             show_celltype_labels=show_celltype_labels, 
                                             distance_metric=distance_metric, 
                                             linkage_method=linkage_method)
            fig.update_layout(
                autosize=True, 
                height=heatmap_height,
                width=None,
                transition_duration=200
            )
            return fig
        except Exception as e:
            print(f"Error creating summary clustergram: {e}")
            return create_empty_clustergram_message("Error loading summary data")
    
    try:
        fig = create_gene_clustergram(
            db_path, 
            selected_gene, 
            height=heatmap_height, 
            colorscale=colorscale, 
            filtered_junction_ids=filtered_junction_ids,
            show_celltype_labels=show_celltype_labels,
            distance_metric=distance_metric,
            linkage_method=linkage_method
        )
        fig.update_layout(
            autosize=True, 
            height=heatmap_height,
            width=None,
            transition_duration=200
        )
        return fig
    
    except Exception as e:
        print(f"Error creating gene clustergram: {e}")
        return create_empty_clustergram_message(f"Error loading data for {selected_gene}")
    

@app.callback(
    dash.dependencies.Output('heatmap1', 'figure'),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('colorscale-dropdown', 'value'),
     dash.dependencies.Input('isoform-data-type-switch', 'value'),
     dash.dependencies.Input('show-labels-toggle', 'value'),
     dash.dependencies.Input('collapse-tissue-toggle', 'value'),
     dash.dependencies.Input('filtered-isoform-store', 'data'),
     dash.dependencies.Input('overview-dropdown', 'value'),
     dash.dependencies.Input('clustergram-height-slider', 'value'),
     dash.dependencies.Input('distance-metric-dropdown', 'value'),
     dash.dependencies.Input('linkage-method-dropdown', 'value')]
)
def update_isoform_heatmap(selected_gene, colorscale, use_ratio_data, 
                          show_labels, collapse_mode, filtered_transcript_ids, 
                          plots_dropdown_value, clustergram_height, distance_metric, linkage_method):
    """Update isoform clustergram with junction clustergram heights"""
    ratio_data = load_expression_data(db_path=db_path, 
                                      gene_name=selected_gene, 
                                      data_type='ratio')
    
    tpm_data = load_expression_data(db_path=db_path, 
                                    gene_name=selected_gene, 
                                    data_type='tpm')
    
    if use_ratio_data:
        data_type = "Ratio"
    else: 
        data_type = "TPM"

    if plots_dropdown_value == 'event-level':
        return empty_fig() 
    
    heatmap_height = clustergram_height
    
    try:
        if data_type == "Ratio":
            current_data = ratio_data
        else: 
            current_data = tpm_data

        filtered_ids = [int(id) for id in filtered_transcript_ids] if filtered_transcript_ids else []

        filtered_ratio_data = ratio_data.copy()
        filtered_tpm_data = tpm_data.copy()

        filtered_ratio_data = filtered_ratio_data[filtered_ratio_data['id'].isin(filtered_ids)] if filtered_ids else filtered_ratio_data
        filtered_tpm_data = filtered_tpm_data[filtered_tpm_data['id'].isin(filtered_ids)] if filtered_ids else filtered_tpm_data
        
        fig = create_isoform_expression_clustergram(
            tpm_data=filtered_tpm_data,
            ratio_data=filtered_ratio_data,
            gene_name=selected_gene,
            height=heatmap_height,  
            colorscale=colorscale,
            data_type=data_type,
            show_labels=show_labels,
            collapse_mode=collapse_mode,
            distance_metric=distance_metric,
            linkage_method=linkage_method
        )
        fig.update_layout(
            autosize=True, 
            height=heatmap_height,
            width=None
        )

        return fig
    
    except Exception as e:
        print(f"Error creating isoform clustergram: {e}")
        return create_empty_isoform_message(f"Error loading {data_type.lower()} data for {selected_gene}")


######################################################################
# STRUCTURE-LEVEL VISUALIZATIONS CALLBACKS
######################################################################
@app.callback(
    dash.dependencies.Output('atse-map', 'figure'),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('filtered-junction-store', 'data'),
     dash.dependencies.Input('filtered-isoform-store', 'data'),
     dash.dependencies.Input('bar-height-slider', 'value'),
     dash.dependencies.Input('overview-dropdown', 'value'),
     dash.dependencies.Input('exon-color-store', 'data'),
     dash.dependencies.Input('junction-color-store', 'data'),
     dash.dependencies.Input('left_data_table', 'filter_query'),
     dash.dependencies.Input('left-table-validation-store', 'data')]
)
def update_atse_visualization(selected_gene, filtered_junction_ids, filtered_transcript_ids, plot_height, plots_dropdown_value, exon_color, junction_color, isoform_filter_query, validation_data):
    """Update ATSE splice junction visualization with filtered data"""
    # Check if current filter is valid: if not, don't update plot
    if isoform_filter_query and validation_data and not validation_data.get('valid', True):
        raise PreventUpdate

    has_isoform_filter = bool(isoform_filter_query and isoform_filter_query.strip())
    actual_filtered_transcript_ids = filtered_transcript_ids if has_isoform_filter else None

    if not selected_gene:
        return create_empty_atse_message("Select a gene to view splice junctions and exons")
    
    try:
        gene_data = process_gene_atse_data(
            selected_gene, 
            db_path,
            filtered_junction_ids=filtered_junction_ids
        )

        if plots_dropdown_value == 'event-level': 
            plot_height = min(1500, int(0.8 * 1500))
        
        show_labels = (plots_dropdown_value == 'event-level')
        
        fig = create_junction_exon_visualization(
            gene_data, 
            height=plot_height,
            show_y_labels=show_labels,
            exon_color=exon_color,
            junction_color=junction_color,
            filtered_transcript_ids=actual_filtered_transcript_ids
        )
        return fig
    
    except Exception as e:
        print(f"Error creating ATSE visualization: {e}")
        return create_empty_atse_message(f"Error loading ATSE data for {selected_gene}: {str(e)}")


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
    dash.dependencies.Output('top-barplot', 'figure'),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('bar-height-slider', 'value'),
     dash.dependencies.Input('filtered-isoform-store', 'data'),
     dash.dependencies.Input('overview-dropdown', 'value'),
     dash.dependencies.Input('exon-color-store', 'data'),
     dash.dependencies.Input('hide-junctions-toggle', 'value'),
     dash.dependencies.Input('left_data_table', 'filter_query'),
     dash.dependencies.Input('left-table-validation-store', 'data')]
)
def update_top_transcript_structure(selected_gene, plot_height, filtered_ids, plots_dropdown_value, exon_color, hide_junctions, filter_query, validation_data):
    """Update transcript structure plot in top panel when toggle is activated"""
    # Only update if junctions are hidden (transcript plot should be shown)
    if not hide_junctions:
        raise PreventUpdate

    # Check if current filter is valid: if not, don't update plot
    if filter_query and validation_data and not validation_data.get('valid', True):
        raise PreventUpdate

    if not selected_gene:
        return create_empty_isoform_message("Select a gene to view transcript structures")

    try:
        # Convert filtered IDs to integers
        filtered_ids = [int(id) for id in filtered_ids] if filtered_ids else []

        # Process transcript structure data
        transcript_data = process_transcript_structure(db_path, selected_gene, filtered_ids)

        # Let the transcript plot function handle its own height calculation when using default
        # Only override when user manually sets a specific height
        if plot_height == 600:
            # Use None to let the function calculate its own dynamic height
            height_to_use = None
        else:
            # Use the manually set height from the slider
            height_to_use = plot_height

        # Create transcript structure plot
        fig = create_transcript_structure_plot(
            db_path,
            transcript_data,
            gene_name=selected_gene,
            height=height_to_use,
            show_y_labels=True,
            exon_color=exon_color
        )
        return fig

    except Exception as e:
        print(f"Error creating top transcript plot: {e}")
        return create_empty_isoform_message(f"Error loading transcript data for {selected_gene}")


@app.callback(
    dash.dependencies.Output('top-barplot-loading-message', 'children'),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('filtered-isoform-store', 'data')]
)
def update_top_barplot_loading_message(selected_gene, filtered_ids):
    if not selected_gene:
        selected_gene = 'A1BG-AS1'
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

    # With auto-sizing containers, keep panel minimal and content-driven
    container_style = {
        'height': 'auto',
        'min-height': '200px',
        'margin-bottom': '15px'
    }

    panel_body_style = {
        'width': '100%',
        'background-color': 'white',
        'padding': '15px 15px 30px 15px',  # minimal bottom padding
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
     dash.dependencies.Input('hide-junctions-toggle', 'value')],
    [dash.dependencies.State('bar-height-slider', 'value')]
)
def update_top_panel_height(selected_gene, filtered_transcript_ids, filtered_junction_ids, hide_junctions, current_height):
    """Calculate unified height for top panel using the same system as old structure plots"""
    if not selected_gene:
        container_style = {'height': '100%', 'min-height': '400px', 'margin-bottom': '15px'}
        panel_body_style = {
            'width': '100%',
            'background-color': 'white',
            'padding': '15px 15px 165px 15px',  # extra bottom padding for x-axis labels and margin
            'border-radius': '0',
            'min-height': '450px',  # 300px + 150px for padding
            'height': 'auto',
            'color': '#1C1C2C',
            'transition': 'height 0.3s ease, min-height 0.3s ease',
            'box-sizing': 'border-box'
        }
        return container_style, panel_body_style

    try:
        filtered_ids = [int(id) for id in filtered_transcript_ids] if filtered_transcript_ids else []
        transcript_data = process_transcript_structure(db_path, selected_gene, filtered_ids)

        # If showing transcript plot, use transcript-specific height calculation
        if hide_junctions:
            from isoform_utils import calculate_dynamic_structure_plot_height
            num_transcripts = len(transcript_data['id'].unique()) if not transcript_data.empty else 0
            calculated_height = calculate_dynamic_structure_plot_height(num_transcripts)
        else:
            # For junction plot, use unified height calculation
            gene_data = process_gene_atse_data(selected_gene, db_path, filtered_junction_ids)
            calculated_height = calculate_unified_plot_height(transcript_data, gene_data)

        # Only update if the difference is significant (same logic as old system)
        if abs(calculated_height - current_height) < 100:
            calculated_height = current_height

        container_style = {
            'height': 'auto',
            'min-height': '200px'
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
        container_style = {'height': '100%', 'min-height': '400px', 'margin-bottom': '15px'}
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
        selected_gene = 'A1BG-AS1'
    T = len(filtered_ids) if filtered_ids else 0
    return f"Loading data for {T} isoform transcripts for {selected_gene}"


@app.callback(
    dash.dependencies.Output('atse-map-loading-message', 'children'),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('filtered-junction-store', 'data')]
)
def update_atse_map_loading_message(selected_gene, filtered_ids):
    if not selected_gene:
        selected_gene = 'A1BG-AS1'
    N = len(filtered_ids) if filtered_ids else 0
    return f"Loading data for {N} junctions for {selected_gene}"


@app.callback(
    dash.dependencies.Output('heatmap2-loading-message', 'children'),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('filtered-junction-store', 'data')]
)
def update_heatmap2_loading_message(selected_gene, filtered_ids):
    if not selected_gene:
        selected_gene = 'A1BG-AS1'
    N = len(filtered_ids) if filtered_ids else 0
    return f"Loading data for {N} junctions for {selected_gene}"


###################################################################
# ISOFORM HASH LOOKUP CALLBACKS
###################################################################

@app.callback(
    [dash.dependencies.Output('second-input-label', 'children'),
     dash.dependencies.Output('second-coordinate-input', 'placeholder')],
    [dash.dependencies.Input('coordinate-input-mode', 'value')]
)
def update_second_input_labels(input_mode):
    """Update the second input field label and placeholder based on selected mode"""
    if input_mode == 'start_end':
        return 'Exon End Positions (comma-separated):', 'e.g., 1200,2300,3400'
    else:  # start_block
        return 'Block Sizes (comma-separated):', 'e.g., 200,300,400'


@app.callback(
    dash.dependencies.Output('hash-result', 'children'),
    [dash.dependencies.Input('calculate-hash-btn', 'n_clicks')],
    [dash.dependencies.State('coordinate-input-mode', 'value'),
     dash.dependencies.State('exon-starts-input', 'value'),
     dash.dependencies.State('second-coordinate-input', 'value')]
)
def calculate_hash_from_coordinates(n_clicks, input_mode, starts_input, second_input):
    """Calculate hash ID from user-entered coordinates"""
    if not n_clicks or not starts_input or not second_input:
        return html.Div()

    try:
        # Parse comma-separated inputs
        tstarts = [int(x.strip()) for x in starts_input.split(',') if x.strip()]
        second_values = [int(x.strip()) for x in second_input.split(',') if x.strip()]

        if len(tstarts) != len(second_values):
            error_msg = ("Error: Number of start positions must match number of " +
                        ("end positions" if input_mode == 'start_end' else "block sizes"))
            return html.Div([
                html.P(error_msg, className='error-message')
            ])

        if len(tstarts) < 2:
            return html.Div([
                html.P("Error: At least 2 exons required for hash calculation",
                       className='error-message')
            ])

        # Convert end positions to block sizes if needed
        if input_mode == 'start_end':
            # second_values are end positions, convert to block sizes
            blocksizes = [end - start for start, end in zip(tstarts, second_values)]
            # Validate that all block sizes are positive
            if any(size <= 0 for size in blocksizes):
                return html.Div([
                    html.P("Error: End positions must be greater than start positions",
                           className='error-message error-message-spaced')
                ])
        else:
            # second_values are already block sizes
            blocksizes = second_values

        hash_id = calculate_single_isoform_hash(tstarts, blocksizes)

        return html.Div([
            html.Div("Isoform Hash ID:", className='hash-result-label'),
            html.Div(className='hash-output', children=[
                html.Div(className='hash-result-with-copy', children=[
                    html.Div(hash_id, className='hash-result-box', id='hash-result-value'),
                    dcc.Clipboard(
                        target_id="hash-result-value",
                        title="Copy hash ID to clipboard",
                        className='hash-copy-button'
                    )
                ])
            ])
        ])

    except ValueError as e:
        return html.Div([
            html.P(f"Error: Invalid input - {str(e)}",
                   className='error-message')
        ])
    except Exception as e:
        return html.Div([
            html.P(f"Error calculating hash: {str(e)}",
                   className='error-message')
        ])


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
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        gtf_content = decoded.decode('utf-8')

        hash_results = parse_gtf_and_calculate_hashes(gtf_content)

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

        upload_status = html.Div([
            html.P(f"Loaded '{filename}' successfully.", className='success-message success-message-bold'),
            html.P(f"Number of transcripts processed: {len(hash_results)}")
        ])

        combined_data = {
            'hash_results': hash_results,
            'annotated_gtf': encoded_gtf,
            'original_filename': filename
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
    """Handle download of hash results"""
    if not download_clicks or not stored_data:
        return dash.no_update

    try:
        if isinstance(stored_data, dict) and 'hash_results' in stored_data:
            hash_results = stored_data['hash_results']
        else:
            hash_results = stored_data  # backwards compatibility

        lines = ["transcript_id\tgene_id\thash_id\texon_count"]
        for result in hash_results:
            lines.append(f"{result['transcript_id']}\t{result['gene_id']}\t{result['hash_id']}\t{result['exon_count']}")

        download_content = "\n".join(lines)

        return dict(content=download_content, filename="isoform_hashes.tsv")

    except Exception as e:
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


if __name__ == '__main__':
    database_exists = check_database_status()
    if not database_exists:
        print("Database initialization completed.")
        #verify_database_schema(db_path)

    display_ascii_banner()

    app.run(debug=True, port=8050, use_reloader=False)