import os 
import math
import traceback
from tqdm import tqdm
import sqlite3
import scipy
import numpy as np
import pandas as pd
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
from data_utils import get_master_table_columns, parse_filter_query, query_master_table, get_gene_options, create_custom_spinner
from junction_utils import (
    create_summary_clustergram, create_gene_clustergram,
    load_atse_data, process_gene_atse_data, create_junction_exon_visualization,
    create_empty_atse_message, create_empty_clustergram_message
)
from isoform_utils import (
    load_expression_data, process_transcript_structure,
    create_transcript_structure_plot, create_isoform_expression_clustergram, 
    create_empty_isoform_message
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

    ########################################################
    # Load isoform master table data
    ########################################################
    isoform_file = os.path.join(data_dir, "mt_isoform_gazers_250616.tsv")
    
    with tqdm(desc="Loading isoform master table data", unit=" rows") as pbar:
        df_isoform = pd.read_csv(isoform_file, sep='\t')
        df_isoform['id'] = df_isoform.index # IMPORTANT: we need this to match PSL file 1-based indexing!
        df_isoform['gene_id'] = df_isoform['gene'].str.split('.').str[0]
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
    psl_file = os.path.join(data_dir, "all_samples_sp_collapse_all_chr_no_treatment_full.psl")
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
            #split_qname = chunk['qName'].str.split('_')
            #chunk['trans_id'] = split_qname.str[0]
            #chunk['gene_id'] = split_qname.str[1].split('.').str[0]
            split_result = chunk['qName'].str.split(r'[:_]', n=1, expand=True)  # Split on : or _
            chunk['trans_id'] = split_result[0].str.split('.').str[0]
            chunk['gene_id'] = split_result[1].str.split('.').str[0]
            chunk['transcript_length'] = chunk['tEnd'] - chunk['tStart']
            
            if 'index' in chunk.columns:
                chunk.drop(columns=['index'], inplace=True)
                
            chunk.to_sql('psl_data', conn, if_exists='append', index=False)
            pbar.update(len(chunk))

    print("Processing isoform TPM data...")
    tpm_file = os.path.join(data_dir, "all_tpm.tsv")
    tpm_df = pd.read_csv(tpm_file, sep='\t')
    tpm_df['id'] = tpm_df.index
    tpm_df.to_sql('tpm_data', conn, if_exists='replace')

    print("Processing isoform ratio data...")
    ratio_file = os.path.join(data_dir, "all_quant_ratio.tsv")
    ratio_df = pd.read_csv(ratio_file, sep='\t')
    ratio_df['id'] = ratio_df.index
    ratio_df.to_sql('ratio_data', conn, if_exists='replace')

    print("Creating isoform indexes...")
    conn.execute("CREATE INDEX idx_psl_gene ON psl_data(id)")
    conn.execute("CREATE INDEX idx_iso_gene ON isoforms(gene_name, id)")
    conn.commit()

    ########################################################
    # Load junction master table data
    ########################################################
    junction_file = os.path.join(data_dir, "pseudobulk_final_broad_cell_type_20250623_171456.csv")
    
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
        'psi'#,
#        'matched_transcript_ids'
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
    with tqdm(desc="Creating indices", total=8, unit="index") as idx_pbar:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_junctions_gene ON junctions(gene_name, gene_id)")
        idx_pbar.update(1)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_isoforms_gene ON isoforms(gene_name, gene_id)")
        idx_pbar.update(1)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_psl_gene ON psl_data(gene_id, id)")
        idx_pbar.update(1)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_junctions_junction_id ON junctions(junction_id)")
        idx_pbar.update(1)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_junctions_psi ON junctions(gene_name, psi)")
        idx_pbar.update(1)
        conn.execute("ALTER TABLE atse_data ADD COLUMN gene_id_clean TEXT")
        conn.execute("UPDATE atse_data SET gene_id_clean = SUBSTR(gene_id, 1, INSTR(gene_id, '.') - 1) WHERE gene_id LIKE '%.%'")
        conn.execute("UPDATE atse_data SET gene_id_clean = gene_id WHERE gene_id_clean IS NULL")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_atse_gene ON atse_data(gene_id_clean, gene_name)")
        idx_pbar.update(1)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_atse_coords ON atse_data(chromosome, start, end)")
        idx_pbar.update(1)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_psl_coords ON psl_data(gene_id, tStart, tEnd)")
        idx_pbar.update(1)
        #conn.execute("CREATE INDEX IF NOT EXISTS idx_junctions_transcript_mapping ON junctions(matched_transcript_ids)")
        #idx_pbar.update(1)
    
    conn.commit()
    conn.close()
    
    print("✓ Database setup complete!")
    print() 

    return db_path


def verify_database_schema(db_path):
    """Verify database schema has required columns"""
    conn = sqlite3.connect(db_path)
    
    # Check isoforms table
    iso_schema = pd.read_sql("PRAGMA table_info(isoforms)", conn)
    print("Isoforms table columns:", iso_schema['name'].tolist())
    
    # Check psl_data table
    psl_schema = pd.read_sql("PRAGMA table_info(psl_data)", conn)
    print("PSL data table columns:", psl_schema['name'].tolist())
    
    # Verify id columns exist
    if 'id' not in iso_schema['name'].values:
        print("ERROR: Missing 'id' column in isoforms table")
    if 'id' not in psl_schema['name'].values:
        print("ERROR: Missing 'id' column in psl_data table")
    
    conn.close()


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
    html.Img(src='/assets/Isoform-Gazer-text.png', className='app-header--title')
])

###################################################################
# ISOFORM MASTER TABLE 
###################################################################
left_data_table = dash_table.DataTable(
    id='left_data_table',
    columns=get_master_table_columns(db_path, table_name='isoforms'),
    hidden_columns=['id'],
    data=[],
    editable=False,
    filter_action="custom",
    filter_options={'placeholder_text': 'Filter column...'},
    sort_action="custom",
    sort_mode="multi",
    page_action="custom",
    page_current=0,
    page_size=10,
    page_count=0,
    filter_query='',
    sort_by=[],
    style_cell={
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
        'minWidth': '100px', 
        'maxWidth': '220px',
        'padding': '5px',
        'whiteSpace': 'normal'
    },
    style_table={
        'height': '100%', 
        'overflowY': 'auto',
        'overflowX': 'scroll', 
        'minWidth': '100%'     
    },
    style_header={
        'backgroundColor': 'white',
        'fontWeight': 'bold',
        'font-family': 'sans-serif',
        'whiteSpace': 'normal',
        'height': 'auto',        
        'lineHeight': '15px',    
        'padding': '8px',        
        'textAlign': 'center'
    },
    style_data={'font-family': 'sans-serif'},
    style_data_conditional=[
        {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}
    ],
    style_filter={
        'backgroundColor': '#f8f9fa',
        'fontWeight': 'bold',
        'padding': '8px 0px'
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
    #hidden_columns=['id', 'matched_transcript_ids'],
    hidden_columns=['id'],
    data=[],
    editable=False,
    filter_action="custom",
    filter_options={'placeholder_text': 'Filter column...'},
    sort_action="custom",
    sort_mode="multi",
    page_action="custom",
    page_current=0,
    page_size=10,
    page_count=0,
    filter_query='',
    sort_by=[],
    style_cell={
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
        'minWidth': '100px', 
        'maxWidth': '220px',
        'padding': '5px',
        'whiteSpace': 'normal'
    },
    style_table={
        'height': '100%', 
        'overflowY': 'auto',
        'overflowX': 'scroll',
        'minWidth': '100%'     
    },
    style_header={
        'backgroundColor': 'white',
        'fontWeight': 'bold',
        'font-family': 'sans-serif',
        'whiteSpace': 'normal',
        'height': 'auto',        
        'lineHeight': '15px',    
        'padding': '8px',        
        'textAlign': 'center'
    },
    style_data={'font-family': 'sans-serif'},
    style_data_conditional=[
        {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}
    ],
    style_filter={
        'backgroundColor': '#f8f9fa',
        'fontWeight': 'bold',
        'padding': '8px 0px'
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
app.layout = html.Div(style={'height': '100vh', 'width': '100%', 
                             'display': 'flex', 'flexDirection': 'column'}, 
                             children=[
    header,
    html.Div(className='app-body', children=[
        #####################################
        # Control panel (left sidebar)
        #####################################
        html.Div(id='control-tabs', className='control-tabs', style={
            'width': '340px', 
            'backgroundColor': 'white', 
            'borderRight': '1px solid #e1e1e1'
        }, children=[
            dcc.Tabs(id='tabs', value='tab-1', style={'height': '100%'}, children=[
                dcc.Tab(label='About', value='tab-1', children=[
                    html.Div(className='control-tab', children=[
                        html.H2('About'),
                        html.P('Isoform Gazer allows for a unified view of RNA splicing across ' \
                        'both single-cell junction usage and long-read isoform data in GENCODEv46 (GRCh38.p14).'),
                        html.P('Use the controls in the "Query" tab to dynamically query the master table data and generate visualizations.'),
                        html.P('Use the controls in the "Custom" tab to customize the visualizations.')
                    ])
                ]),
                dcc.Tab(label='Query', value='tab-2', children=[
                    html.Div(className='control-tab', children=[
                        html.Div(className='app-controls-block', children=[
                        html.Div(className='app-controls-query', children='Search by Gene'),
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
                        html.Div(className="app-controls-block", children=[
                            html.Div(className='app-controls-query', children='Query by Value'),
                            html.P("You can use the filter boxes below each master table column header to search and filter the data:"),
                            html.Ul([
                                html.Li([html.Strong("Text columns: "), "Simply type text to find rows with exact matches."]),
                                html.Li([html.Strong("Numeric columns: "), "Use operators for precise filtering:"]),
                                html.Ul([
                                    html.Li("Type a number (e.g., '4') to show only entries with exact matches."),
                                    html.Li("Use > for greater than (e.g., '>5')."),
                                    html.Li("Use < for less than (e.g., '<10')."),
                                    html.Li("Use >= or <= for inclusive ranges.")
                                ]),
                                html.Li([html.Strong("Multiple filters: "), "You can apply filters to multiple columns simultaneously."]),
                                html.Li([html.Strong("Clear filters: "), "Delete your queries in the filter boxes and hit Enter anytime to reset."])
                            ]),
                            html.P([
                                "Select a gene from the dropdown above or directly filter the tables to explore the data.",
                                html.Br(),
                                "Visualizations will update automatically."
                            ], className="app-controls-desc"),
                        ])
                    ])
                ]),
                dcc.Tab(label='Custom', value='tab-3', children=[
                    html.Div(className='control-tab', children=[
                        #####################################
                        # General Controls Section
                        #####################################
                        html.H3('General', className='alignment-settings-section'),
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
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Display Master Tables'),
                            dcc.RadioItems(
                                id='show-table-radio',
                                className='alignment-radio',
                                options=[
                                    {'label': 'Show', 'value': 'show'},
                                    {'label': 'Hide', 'value': 'hide'},
                                ],
                                value='show',
                                labelStyle={'display': 'inline-block', 'marginRight': '8px'}
                            ),
                            html.Div(className='app-controls-desc', children='Toggle the data tables display')
                        ]),
                        html.Hr(),

                        #####################################
                        # Isoform-level Event Plot Section
                        #####################################
                        html.H3('Isoform Transcripts Plot', className='alignment-settings-section'),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Plot Height'),
                            dcc.Slider(
                                id='bar-height-slider',
                                className='control-slider',
                                min=100,
                                max=300,
                                step=25,
                                value=250,
                                marks={str(i): str(i) for i in range(100, 301, 50)}
                            ),
                            html.Div(className='app-controls-desc', children='Adjust the height of both the isoform-level and junction-level event visualizations')
                        ]),
                        html.Hr(),

                        #####################################
                        # Isoform-level Heatmap Section
                        #####################################
                        html.H3('Heatmap', className='alignment-settings-section'),
                        html.Div(className='app-controls-block', children=[
                            html.Div([
                                html.Div('Isoform Heatmap Unit', className='app-controls-name', 
                                        style={'display': 'inline-block', 'marginRight': '15px'}),
                                daq.ToggleSwitch(
                                    id='isoform-data-type-switch',
                                    value=False,  # False = TPM, True = Ratio
                                    label={'label': 'TPM / Ratio', 'style': {'fontSize': '12px', 'color': '#506784'}},
                                    labelPosition='right',
                                    style={'display': 'inline-block'}
                                )
                            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                            html.Div(className='app-controls-desc', children='Toggle whether isoform heatmap shows TPM values or ratio values across all tissues')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div([
                                html.Div('Average by Tissue', className='app-controls-name', 
                                        style={'display': 'inline-block', 'marginRight': '15px'}),
                                daq.ToggleSwitch(
                                    id='collapse-tissue-toggle',
                                    value=True,  # Default to average by tissue (a lot cleaner looking for most genes)
                                    label={'label': 'Show All / Average', 'style': {'fontSize': '12px', 'color': '#506784'}},
                                    labelPosition='right',
                                    style={'display': 'inline-block'}
                                )
                            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                            html.Div(className='app-controls-desc', children='Toggle whether heatmap shows average values across experiments grouped by tissue')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div([
                                html.Div('Show Tissue Labels', className='app-controls-name', 
                                        style={'display': 'inline-block', 'marginRight': '15px'}),
                                daq.ToggleSwitch(
                                    id='show-labels-toggle',
                                    value=False,
                                    label={'label': 'Hide / Show', 'style': {'fontSize': '12px', 'color': '#506784'}},
                                    labelPosition='left',
                                    style={'display': 'inline-block'}
                                )
                            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                            html.Div(className='app-controls-desc', children='Toggle tissue name visibility on isoform heatmap when master tables are hidden')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Colorscale'),
                            dcc.Dropdown(
                                id='colorscale-dropdown',
                                className='app-controls-block-dropdown',
                                options=[
                                    {'label': 'RdBu_r', 'value': 'RdBu_r'},
                                    {'label': 'Viridis', 'value': 'Viridis'},
                                    {'label': 'Plasma', 'value': 'Plasma'}
                                ],
                                value='RdBu_r'
                            ),
                            html.Div(className='app-controls-desc', children='Choose the color theme of the heatmaps')
                        ])
                    ])
                ])
            ])
        ]),
        
        #####################################
        # Main content section
        #####################################
        html.Div(className='main-content', children=[
            html.Div(id='panels-container', className='panels-container', children=[
                #####################################
                # Data Panel 1: Isoform Data
                #####################################
                html.Div(id='left-panel', className='panel', children=[
                    html.H2("ENCODE4 Bulk RNA-seq Long-Read Data"),
                    html.Div(className='graph-wrapper', children=[
                        html.Div(id='isoform-event-plot-container', className='barplot-container', children=[
                            html.Div(className='loading-container', children=[
                                dcc.Loading(
                                    id="loading-transcript-plot",
                                    type="default",
                                    color='#EDAE49',
                                    delay_show=500,
                                    delay_hide=200,
                                    children=[dcc.Graph(id='barplot1')]
                                ),
                                html.Div(id="barplot1-loading-message", className="custom-loading-message")
                            ])
                        ]),
                        html.Div(id='isoform-clustergram-container', className='heatmap-container', children=[
                            html.Div(className='loading-container', children=[
                                dcc.Loading(
                                    id="loading-isoform-heatmap",
                                    type="default",
                                    color='#EDAE49',
                                    delay_show=500,
                                    delay_hide=200,
                                    children=[dcc.Graph(id='heatmap1')]
                                ),
                                html.Div(id="heatmap1-loading-message", className="custom-loading-message")
                            ])
                        ]),
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
                            left_data_table
                        ])
                    ])
                ]),
                
                #####################################
                # Data Panel 2: Junction Data
                #####################################
                html.Div(id='right-panel', className='panel', children=[
                    html.H2("Tabula Sapiens 2.0 Pseudobulked Smart-seq2 Single-cell and Allen Brain Single Nuclei for Brain Data"),
                    html.Div(className='graph-wrapper', children=[
                        html.Div(id='junction-event-plot-container', className='atse-container', children=[
                            html.Div(className='loading-container', children=[
                                dcc.Loading(
                                    id="loading-atse-plot",
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
                                            style={'height': '100%', 'width': '100%'}
                                        )
                                    ]
                                ),
                                html.Div(id="atse-map-loading-message", className="custom-loading-message")
                            ], style={'height': '25%', 'min-height': '200px', 'margin-bottom': '15px'})
                        ]),
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
                                            style={'height': '100%', 'width': '100%'}
                                        )
                                    ]
                                ),
                                html.Div(id="heatmap2-loading-message", className="custom-loading-message")
                            ])
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
                            right_data_table
                        ])
                    ])
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
    dcc.Store(id='table-callback-prevention', data=False)
])

#######################################################################
# CALLBACKS
#######################################################################
#######################################################################
# MASTER TABLE FILTERING CALLBACKS
#######################################################################
# Avoid master table updates when the tables are hidden (unlikely, but possible)
app.clientside_callback(
    """
    function(show_tables) {
        // Return true to prevent callback execution when tables are hidden
        if (show_tables === 'hide') {
            return true;
        }
        return false;
    }
    """,
    dash.dependencies.Output('table-callback-prevention', 'data'),
    [dash.dependencies.Input('show-table-radio', 'value')],
    prevent_initial_call=True
)


@app.callback(
    [dash.dependencies.Output('filtered-isoform-store', 'data'),
     dash.dependencies.Output('filtered-junction-store', 'data')],
    [dash.dependencies.Input('isoform-full-data-store', 'data'),
     dash.dependencies.Input('junction-full-data-store', 'data')]
)
def update_filtered_data_stores(isoform_full_data, junction_full_data):
    """Store ALL filtered transcript/junction IDs from FULL datasets with error handling"""
    try:
        filtered_transcript_ids = []
        if isoform_full_data:
            filtered_transcript_ids = [row.get('id', '') for row in isoform_full_data if row.get('id')]
        
        filtered_junction_ids = []
        if junction_full_data:
            filtered_junction_ids = [row.get('junction_id', '') for row in junction_full_data if row.get('junction_id')]
        
        return filtered_transcript_ids, filtered_junction_ids
    
    except Exception as e:
        print(f"Error updating filtered data stores: {e}")
        return [], []


@app.callback(
    [dash.dependencies.Output('left_data_table', 'filter_query'),
     dash.dependencies.Output('right_data_table', 'filter_query')],
    [dash.dependencies.Input('clear-left-filters', 'n_clicks'),
     dash.dependencies.Input('clear-right-filters', 'n_clicks')],
    [dash.dependencies.State('left_data_table', 'filter_query'),
     dash.dependencies.State('right_data_table', 'filter_query')]
)
def clear_filters(left_clicks, right_clicks, left_filter, right_filter):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'clear-left-filters':
        return '', right_filter
    elif button_id == 'clear-right-filters':
        return left_filter, ''
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
     dash.dependencies.Input('show-table-radio', 'value')]
)
def update_isoform_table(page_current, page_size, sort_by, filter_query, selected_gene, show_tables):
    if show_tables == 'hide':
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
        
        return [], 0, full_data
    
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
     dash.dependencies.Input('show-table-radio', 'value')]
)
def update_junction_table(page_current, page_size, sort_by, filter_query, selected_gene, show_tables):
    if show_tables == 'hide':
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
        
        return [], 0, full_data
    
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
        dash.dependencies.Output('isoform-event-plot-container', 'style'),
        dash.dependencies.Output('isoform-clustergram-container', 'style'),
        dash.dependencies.Output('junction-event-plot-container', 'style'),
        dash.dependencies.Output('junction-clustergram-container', 'style'),
        dash.dependencies.Output('panels-container', 'className'),
        dash.dependencies.Output('left-panel', 'className'),
        dash.dependencies.Output('right-panel', 'className'),
    ],
    [dash.dependencies.Input('overview-dropdown', 'value')]
)
def toggle_plot_visibility(overview_value):
    """Enhanced plot visibility toggle with proper container sizing"""
    
    # Base styles
    show_full = {'display': 'block', 'height': '100%', 'flex': '1 1 auto'}
    hide = {'display': 'none', 'height': '0', 'flex': '0 0 0'}
    show_normal = {'display': 'block', 'height': '100%'}
    
    # Default panel classes
    panels_class = 'panels-container'
    left_panel_class = 'panel'
    right_panel_class = 'panel'
    
    if overview_value == 'both':
        return show_normal, show_normal, show_normal, show_normal, panels_class, left_panel_class, right_panel_class
    
    # Show only event plots, hide clustergrams
    elif overview_value == 'event-level':
        left_panel_class = 'panel full-panel-event'
        right_panel_class = 'panel full-panel-event'
        return show_full, hide, show_full, hide, panels_class, left_panel_class, right_panel_class
    
    # Show only clustergrams, hide event plots
    elif overview_value == 'clustergram':
        left_panel_class = 'panel full-panel-clustergram'
        right_panel_class = 'panel full-panel-clustergram'
        return hide, show_full, hide, show_full, panels_class, left_panel_class, right_panel_class
    
    # Default: styles for both
    else:
        return show_normal, show_normal, show_normal, show_normal, panels_class, left_panel_class, right_panel_class


######################################################################
# HEATMAP PROCESSING CALLBACKS
######################################################################
@app.callback(
    dash.dependencies.Output('heatmap2', 'figure'),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('colorscale-dropdown', 'value'),
     dash.dependencies.Input('show-table-radio', 'value'),
     dash.dependencies.Input('filtered-junction-store', 'data'),
     dash.dependencies.Input('overview-dropdown', 'value')]
)
def update_junction_clustergram(selected_gene, colorscale, show_tables, 
                                filtered_junction_ids, plots_dropdown_value):
    """Update junction visualization based on gene selection and filtering"""
    if plots_dropdown_value == 'event-level':                       
        return empty_fig() 
    elif plots_dropdown_value == 'clustergram': 
        heatmap_height = min(800, int(0.85 * 800))
    elif show_tables == 'show':
        heatmap_height = 450
    else:
        heatmap_height = 650
    
    if not selected_gene:
        try:
            fig = create_summary_clustergram(db_path, height=heatmap_height, colorscale=colorscale, show_tables=show_tables)
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
            show_tables=show_tables,
            filtered_junction_ids=filtered_junction_ids
        )

        return fig
    
    except Exception as e:
        print(f"Error creating gene clustergram: {e}")
        return create_empty_clustergram_message(f"Error loading data for {selected_gene}")
    

@app.callback(
    [dash.dependencies.Output('table1-container', 'style'), 
     dash.dependencies.Output('table2-container', 'style'),
     dash.dependencies.Output('heatmap1', 'style'),
     dash.dependencies.Output('heatmap2', 'style')],
    [dash.dependencies.Input('show-table-radio', 'value')]
)
def toggle_tables(show_tables):
    if show_tables == 'show':
        table_style = {
            'display': 'block', 
            'height': '40vh',  
            'min-height': '250px', 
            'overflow': 'auto',
            'flex-shrink': 0,
            'visibility': 'visible'
        }
        heatmap1_style = {'height': '30vh', 'width': '100%'} 
        heatmap2_style = {'height': '35vh', 'width': '100%'} 
    else:
        table_style = {
            'display': 'block',  
            'height': '0',
            'min-height': '0',
            'max-height': '0',
            'overflow': 'hidden',
            'visibility': 'hidden', 
            'margin': '0',
            'padding': '0'
        }
        heatmap1_style = {'height': '45vh', 'width': '100%'}  
        heatmap2_style = {'height': '60vh', 'width': '100%'}
    
    return table_style, table_style, heatmap1_style, heatmap2_style


@app.callback(
    [dash.dependencies.Output('left-panel-graph-wrapper', 'className'),
     dash.dependencies.Output('right-panel-graph-wrapper', 'className')],
    [dash.dependencies.Input('show-table-radio', 'value')]
)
def update_container_classes(show_tables):
    """Update container classes based on table visibility"""
    if show_tables == 'show':
        return 'graph-wrapper', 'graph-wrapper'
    else:
        return 'graph-wrapper tables-hidden', 'graph-wrapper tables-hidden'


app.clientside_callback(
    """
    function(n_clicks, show_tables) {
        setTimeout(function() {
            // Force resize of all Plotly graphs
            var graphs = document.querySelectorAll('.js-plotly-plot');
            graphs.forEach(function(graph) {
                if (graph && graph.layout) {
                    // Force Plotly to recalculate size
                    Plotly.relayout(graph, {autosize: true});
                    Plotly.Plots.resize(graph);
                }
            });
        }, 200);
        return window.dash_clientside.no_update;
    }
    """,
    dash.dependencies.Output('heatmap1', 'className', allow_duplicate=True),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('show-table-radio', 'value')],
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(show_tables) {
        setTimeout(function() {
            // Use the correct Plotly method for resizing
            if (window.Plotly) {
                var graphs = document.querySelectorAll('.js-plotly-plot');
                graphs.forEach(function(graph) {
                    if (graph && graph.data) {
                        try {
                            // Use Plotly.relayout instead of Plots.resize
                            window.Plotly.relayout(graph, {autosize: true});
                        } catch (e) {
                            console.log('Plotly relayout error:', e);
                        }
                    }
                });
            }
        }, 200);
        return window.dash_clientside.no_update;
    }
    """,
    dash.dependencies.Output('gene-search-dropdown', 'style', allow_duplicate=True),
    [dash.dependencies.Input('show-table-radio', 'value')],
    prevent_initial_call=True
)


@app.callback(
    dash.dependencies.Output('heatmap1', 'figure'),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('colorscale-dropdown', 'value'),
     dash.dependencies.Input('isoform-data-type-switch', 'value'),
     dash.dependencies.Input('show-table-radio', 'value'),
     dash.dependencies.Input('show-labels-toggle', 'value'),
     dash.dependencies.Input('collapse-tissue-toggle', 'value'),
     dash.dependencies.Input('filtered-isoform-store', 'data'),
     dash.dependencies.Input('overview-dropdown', 'value')]
)
def update_isoform_heatmap(selected_gene, colorscale, use_ratio_data, show_tables, 
                          show_labels, collapse_tissues, filtered_transcript_ids, 
                          plots_dropdown_value):
    """Update isoform clustergram with junction clustergram heights"""
    if use_ratio_data: 
        current_data = load_expression_data(db_path=db_path, 
                                            gene_name=selected_gene, 
                                            data_type='ratio')
        data_type = "Ratio"
    else: 
        current_data = load_expression_data(db_path=db_path, 
                                            gene_name=selected_gene, 
                                            data_type='tpm')
        data_type = "TPM"

    if plots_dropdown_value == 'event-level':
        return empty_fig() 
    if plots_dropdown_value == 'clustergram': 
        heatmap_height = min(800, int(0.85 * 800))
    elif show_tables == 'show':
        heatmap_height = 450
    else:
        heatmap_height = 650
    
    try:
        filtered_data = current_data.copy()
        filtered_ids = [int(id) for id in filtered_transcript_ids] if filtered_transcript_ids else []
        filtered_data = current_data[current_data['id'].isin(filtered_ids)] if filtered_ids else current_data
        
        fig = create_isoform_expression_clustergram(
            tpm_data=filtered_data,
            gene_name=selected_gene,
            height=heatmap_height,  
            colorscale=colorscale,
            data_type=data_type,
            show_tables=show_tables,
            show_labels=show_labels,
            collapse_tissues=collapse_tissues
        )
        fig.update_layout(autosize=True)

        return fig
    
    except Exception as e:
        print(f"Error creating isoform clustergram: {e}")
        return create_empty_isoform_message(f"Error loading {data_type.lower()} data for {selected_gene}")


@app.callback(
    dash.dependencies.Output('heatmap1', 'className'),
    [dash.dependencies.Input('show-table-radio', 'value')]
)
def update_heatmap_container_class(show_tables):
    """Add CSS class to manage container behavior"""
    if show_tables == 'show':
        return 'with-tables'
    else:
        return ''


######################################################################
# EVENT-LEVEL VISUALIZATIONS CALLBACKS
######################################################################
app.clientside_callback(
    """
    function(overview_value) {
        setTimeout(function() {
            // Force plots to reset to default zoom when switching to event-level mode
            var graphs = document.querySelectorAll('.js-plotly-plot');
            graphs.forEach(function(graph) {
                if (graph && graph.layout) {
                    try {
                        // Check if ranges exist before using them
                        var xRange = graph.layout.xaxis && graph.layout.xaxis.range ? graph.layout.xaxis.range : null;
                        var yRange = graph.layout.yaxis && graph.layout.yaxis.range ? graph.layout.yaxis.range : null;
                        
                        var relayoutData = {};
                        
                        if (xRange && Array.isArray(xRange) && xRange.length === 2) {
                            relayoutData['xaxis.autorange'] = false;
                            relayoutData['xaxis.range'] = xRange;
                        } else {
                            relayoutData['xaxis.autorange'] = true;
                        }
                        
                        if (yRange && Array.isArray(yRange) && yRange.length === 2) {
                            relayoutData['yaxis.autorange'] = false;
                            relayoutData['yaxis.range'] = yRange;
                        } else {
                            relayoutData['yaxis.autorange'] = true;
                        }
                        
                        // Only call relayout if we have valid data
                        if (Object.keys(relayoutData).length > 0) {
                            Plotly.relayout(graph, relayoutData);
                        }
                    } catch (e) {
                        console.log('Plotly relayout error:', e);
                    }
                }
            });
        }, 500);
        return window.dash_clientside.no_update;
    }
    """,
    dash.dependencies.Output('gene-search-dropdown', 'style', allow_duplicate=True),
    [dash.dependencies.Input('overview-dropdown', 'value')],
    prevent_initial_call=True
)


@app.callback(
    dash.dependencies.Output('barplot1', 'figure'),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('bar-height-slider', 'value'),
     dash.dependencies.Input('filtered-isoform-store', 'data'),
     dash.dependencies.Input('overview-dropdown', 'value')]
)
def update_transcript_structure(selected_gene, plot_height, filtered_ids, plots_dropdown_value):
    """Update transcript structure plot based on gene selection"""
    if not selected_gene:
        fig = go.Figure()
        fig.add_annotation(
            text="Select a gene to view transcript structure",
            xref="paper", yref="paper", x=0.5, y=0.5,
            xanchor='center', yanchor='middle', showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor='white', height=plot_height,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        return fig

    try:
        filtered_ids = [int(id) for id in filtered_ids] if filtered_ids else []
        transcript_data = process_transcript_structure(db_path, selected_gene, filtered_ids)

        if filtered_ids and not transcript_data.empty:
            transcript_data = transcript_data[transcript_data['id'].isin(filtered_ids)]

        if plots_dropdown_value == 'clustergram':
            return empty_fig() 
        elif plots_dropdown_value == 'event-level': 
            plot_height = min(800, int(0.8 * 800))
        
        show_labels = (plots_dropdown_value == 'event-level')
        
        fig = create_transcript_structure_plot(
            db_path, 
            transcript_data, 
            selected_gene, 
            height=plot_height,
            show_y_labels=show_labels 
        )
        
        return fig
    
    except Exception as e:
        print(f"Error creating transcript plot: {e}")
        return create_empty_isoform_message(f"Error loading transcript data for {selected_gene}")
    

@app.callback(
    dash.dependencies.Output('atse-map', 'figure'),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('filtered-junction-store', 'data'),
     dash.dependencies.Input('bar-height-slider', 'value'),
     dash.dependencies.Input('overview-dropdown', 'value')]
)
def update_atse_visualization(selected_gene, filtered_junction_ids, plot_height, plots_dropdown_value):
    """Update ATSE splice junction visualization with filtered data"""
    if not selected_gene:
        return create_empty_atse_message("Select a gene to view splice junctions and exons")
    
    try:
        gene_data = process_gene_atse_data(
            selected_gene, 
            db_path,
            filtered_junction_ids=filtered_junction_ids
        )

        if plots_dropdown_value == 'event-level': 
            plot_height = min(800, int(0.8 * 800))
        
        show_labels = (plots_dropdown_value == 'event-level')
        
        fig = create_junction_exon_visualization(
            gene_data, 
            height=plot_height,
            show_y_labels=show_labels
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


@app.callback(
    dash.dependencies.Output('barplot1-loading-message', 'children'),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('filtered-isoform-store', 'data')]
)
def update_barplot1_loading_message(selected_gene, filtered_ids):
    if not selected_gene:
        selected_gene = 'A1BG-AS1'
    T = len(filtered_ids) if filtered_ids else 0
    return f"Loading data for {T} isoform transcripts for {selected_gene}"


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
