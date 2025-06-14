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
import dash_daq as daq
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate
from colorama import Fore, Style, init
from data_utils import generate_mock_data, get_master_table_columns, parse_filter_query, query_master_table, get_gene_options
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
# PROTOTYPE MOCK DATA
###################################################################
mock_data = generate_mock_data(RANDOM_SEED)
data1 = mock_data['data1']
data2 = mock_data['data2']
df1 = mock_data['df1']
df2 = mock_data['df2']
atse_data = mock_data['atse_data']
atse_fig = mock_data['atse_fig']

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
    isoform_file = os.path.join(data_dir, "mt_isoform_gazers_250514.tsv")
    
    with tqdm(desc="Loading isoform master table data", unit=" rows") as pbar:
        df_isoform = pd.read_csv(isoform_file, sep='\t')
        df_isoform['id'] = df_isoform.index # IMPORTANT: we need this to match PSL file 1-based indexing!
        df_isoform['gene_id'] = df_isoform['gene_id'].str.split('.').str[0]
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
    junction_file = os.path.join(data_dir, "pseudobulk_final_broad_cell_type_20250514_072922.csv")
    
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
        'psi'
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
    
    print("Creating database indices...")
    with tqdm(desc="Creating junction-level index", total=2, unit="index") as idx_pbar:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_junctions_gene ON junctions(gene_name, gene_id)")
        idx_pbar.update(1)
        idx_pbar.set_description("Creating isoform-level index")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_isoforms_gene ON isoforms(gene_name, gene_id)")
        idx_pbar.update(1)
    
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
# ATSE DATA SETUP
###################################################################
def setup_atse_data():
    """Load ATSE data for junction visualization"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    atse_file = os.path.join(base_dir, "data", "TMS_atse_file_unanno_also_2025-05-11_06-23-05.tsv")
    
    print("Loading ATSE data for junction visualization...")
    if os.path.exists(atse_file):
        atse_df = load_atse_data(atse_file)
        print(f"✓ Loaded ATSE data with {len(atse_df)} records")
        return atse_df
    else:
        print(f"ATSE file not found at {atse_file}")
        return pd.DataFrame()

atse_data = setup_atse_data()

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
    html.Div('daklab ---', style={'fontWeight': 'bold'}),
    html.Div('Isoform Gazer', className='app-header--title')
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
    },
    style_table={'height': '100%', 'overflowY': 'auto'},
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
)

###################################################################
# JUNCTION MASTER TABLE 
###################################################################
right_data_table = dash_table.DataTable(
    id='right_data_table',
    columns=get_master_table_columns(db_path, table_name='junctions'),
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
    },
    style_table={'height': '100%', 'overflowY': 'auto'},
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
                        html.H4('About'),
                        html.P('Isoform Gazers allows for a unified view of RNA splicing across ' \
                        'both single-cell junction usage and long-read isoform data in GENCODEv46 (GRCh38.p14).'),
                        html.P('Use the controls in the "Query" tab to dynamically query the master table data and generate visualizations.'),
                        html.P('Use the controls in the "Custom" tab to customize the visualizations.')
                    ])
                ]),
                dcc.Tab(label='Query', value='tab-2', children=[
                    html.Div(className='control-tab', children=[
                        html.Div(className='app-controls-block', children=[
                        html.Div(className='app-controls-name', children='Search by Gene'),
                        dcc.Dropdown(
                            id='gene-search-dropdown',
                            options=[
                                {'label': 'RBFOX2 (RNA Binding Fox-1 Homolog 2)', 'value': 'RBFOX2'},
                                {'label': 'EGFR (Epidermal growth factor receptor)', 'value': 'EGFR'},
                                {'label': 'BRCA1 (Breast cancer type 1)', 'value': 'BRCA1'},
                                {'label': 'TARDBP (TAR DNA Binding Protein)', 'value': 'TARDBP'},
                                {'label': 'TP53 (Tumor protein p53)', 'value': 'TP53'}
                            ],
                            placeholder="Type to search for a gene...",
                            searchable=True,
                            clearable=True
                        ),
                        html.Div(className='app-controls-desc', children='Select a gene identifier to query or type to search')
                        ]),
                        html.Div(id='gene-filter-status'),
                        html.Div(className="app-controls-block", children=[
                            html.Div(className='app-controls-name', children='Query by Value'),
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
                                    {'label': 'Event-level', 'value': 'event-level'},
                                    {'label': 'Heatmaps', 'value': 'heatmap'},
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
                                value=150,
                                marks={str(i): str(i) for i in range(100, 301, 50)}
                            ),
                            html.Div(className='app-controls-desc', children='Adjust the height of the isoform-level event visualization')
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
            html.Div(className='panels-container', children=[
                #####################################
                # Data Panel 1: Isoform Data
                #####################################
                html.Div(className='panel', children=[
                    html.H2("ENCODE4 Bulk RNA-seq Long-Read Data"),
                    html.Div(className='graph-wrapper', children=[
                        html.Div(className='barplot-container', children=[
                            dcc.Graph(
                                id='barplot1',
                                figure={
                                    'data': [go.Bar(x=list(range(10)), y=data1.sum(axis=1), marker_color='blue')],
                                    'layout': go.Layout(
                                        margin=dict(l=40, r=40, t=20, b=20),
                                        title={'text': 'Isoform Expression by Tissue', 'font': {'size': 14}, },
                                        xaxis={'title': {'text': 'Tissue', 'font': {'size': 12}}, 'title_standoff': 40, 'ticksuffix': ' '},
                                        yaxis={'title': {'text': 'Count', 'font': {'size': 12}}},
                                        autosize=True
                                    )
                                },
                                config={'responsive': True},
                                style={'height': '100%', 'width': '100%'}
                            )
                        ]),
                        html.Div(className='heatmap-container', children=[
                            dcc.Graph(
                                id='heatmap1',
                                figure={
                                    'data': [go.Heatmap(z=data1, colorscale='Viridis')],
                                    'layout': go.Layout(
                                        margin=dict(l=40, r=40, t=40, b=40),
                                        title={'text': 'Isoform Expression by Tissue', 'font': {'size': 14}},
                                        autosize=True
                                    )
                                },
                                config={'responsive': True, 'displayModeBar': True},
                                style={'height': '100%', 'width': '100%'}
                            )
                        ]),
                        html.Div(className='table-container', id='table1-container', children=[
                            left_data_table
                        ])
                    ])
                ]),
                
                #####################################
                # Data Panel 2: Junction Data
                #####################################
                html.Div(className='panel', children=[
                    html.H2("Tabula Sapiens 2.0 Pseudobulked Smart-seq2 Single-cell and Allen Brain Single Nuclei for Brain Data"),
                    html.Div(className='graph-wrapper', children=[
                        html.Div(className='atse-container', children=[
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
                        ], style={'height': '25%', 'min-height': '200px', 'margin-bottom': '15px'}),
                        html.Div(className='heatmap-container', children=[
                            dcc.Graph(
                                id='heatmap2',
                                figure={
                                    'data': [go.Heatmap(z=data2, colorscale='Plasma')],
                                    'layout': go.Layout(
                                        margin=dict(l=40, r=40, t=40, b=40),
                                        title={'text': 'Junction Usage by Cell Type', 'font': {'size': 14}},
                                        autosize=True
                                    ),
                                    
                                },
                                config={'responsive': True},
                                style={'height': '100%', 'width': '100%'}
                            )
                        ]),
                        html.Div(className='table-container', id='table2-container', children=[
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
    dcc.Store(id='filtered-junction-store', data=[])
])

#######################################################################
# CALLBACKS
#######################################################################
#######################################################################
# MASTER TABLE VISUALIZATION FILTERING CALLBACKS
#######################################################################
@app.callback(
    [dash.dependencies.Output('filtered-isoform-store', 'data'),
     dash.dependencies.Output('filtered-junction-store', 'data')],
    [dash.dependencies.Input('left_data_table', 'derived_virtual_data'),
     dash.dependencies.Input('right_data_table', 'derived_virtual_data')],
    [dash.dependencies.State('left_data_table', 'data'),
     dash.dependencies.State('right_data_table', 'data')]
)
def update_filtered_data_stores(isoform_filtered_data, junction_filtered_data, 
                               isoform_full_data, junction_full_data):
    """Store filtered transcript and junction IDs from master tables"""
    # Extract IDs using new 'id' column
    isoform_all_ids = [row.get('id', '') for row in isoform_full_data]
    junction_all_ids = [row.get('id', '') for row in junction_full_data]
    
    filtered_transcript_ids = []
    if isoform_filtered_data:
        filtered_transcript_ids = [row.get('id', '') 
                                  for row in isoform_filtered_data 
                                  if row.get('id') in isoform_all_ids]
    
    filtered_junction_ids = []
    if junction_filtered_data:
        filtered_junction_ids = [row.get('id', '') 
                                for row in junction_filtered_data 
                                if row.get('id') in junction_all_ids]
    
    return filtered_transcript_ids, filtered_junction_ids


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
    else:
        options = get_gene_options(db_path, search_term=search_value, limit=10)
    
    # Preserve current selection if it exists in the new options
    option_values = [opt['value'] for opt in options]
    
    if current_value and current_value in option_values:
        return options, current_value
    elif current_value and current_value not in option_values:
        if current_value:
            current_options = get_gene_options(db_path, search_term=current_value, limit=1)
            if current_options:
                options = current_options + [opt for opt in options if opt['value'] != current_value]
        return options, current_value
    else:
        return options, None

######################################################################
# SQLLITE MASTER TABLE PROCESSING CALLBACKS
######################################################################
@app.callback(
    [dash.dependencies.Output('left_data_table', 'data'),
     dash.dependencies.Output('left_data_table', 'page_count')],
    [dash.dependencies.Input('left_data_table', 'page_current'),
     dash.dependencies.Input('left_data_table', 'page_size'),
     dash.dependencies.Input('left_data_table', 'sort_by'),
     dash.dependencies.Input('left_data_table', 'filter_query'),
     dash.dependencies.Input('gene-search-dropdown', 'value')]
)
def update_isoform_table(page_current, page_size, sort_by, filter_query, selected_gene):
    filters = parse_filter_query(db_path, filter_query, table_name='isoforms')
    data, total_count = query_master_table(
        db_path,
        table_name='isoforms',
        page=page_current if page_current is not None else 0,
        page_size=page_size if page_size is not None else 10,
        sort_by=sort_by,
        filters=filters,
        gene_filter=selected_gene
    )
    page_count = math.ceil(total_count / page_size) if page_size else 1
    return data, page_count


@app.callback(
    [dash.dependencies.Output('right_data_table', 'data'),
     dash.dependencies.Output('right_data_table', 'page_count')],
    [dash.dependencies.Input('right_data_table', 'page_current'),
     dash.dependencies.Input('right_data_table', 'page_size'),
     dash.dependencies.Input('right_data_table', 'sort_by'),
     dash.dependencies.Input('right_data_table', 'filter_query'),
     dash.dependencies.Input('gene-search-dropdown', 'value')]
)
def update_junction_table(page_current, page_size, sort_by, filter_query, selected_gene):
    filters = parse_filter_query(db_path, filter_query, table_name='junctions')
    data, total_count = query_master_table(
        db_path,
        table_name="junctions",
        page=page_current if page_current is not None else 0,
        page_size=page_size if page_size is not None else 10,
        sort_by=sort_by,
        filters=filters,
        gene_filter=selected_gene
    )
    page_count = math.ceil(total_count / page_size) if page_size else 1
    return data, page_count


######################################################################
# HEATMAP PROCESSING CALLBACKS
######################################################################
@app.callback(
    dash.dependencies.Output('heatmap2', 'figure'),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('colorscale-dropdown', 'value'),
     dash.dependencies.Input('show-table-radio', 'value'),
     dash.dependencies.Input('filtered-junction-store', 'data')]
)
def update_junction_clustergram(selected_gene, colorscale, show_tables, filtered_junction_ids):
    """Update junction visualization based on gene selection"""
    if show_tables == 'show':
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
    [
        dash.dependencies.Output('table1-container', 'style'),
        dash.dependencies.Output('table2-container', 'style'),
        dash.dependencies.Output('heatmap1', 'style'),
        dash.dependencies.Output('heatmap2', 'style')
    ],
    [dash.dependencies.Input('show-table-radio', 'value')]
)
def toggle_tables(show_tables):
    if show_tables == 'show':
        table_style = {
            'display': 'block', 
            'height': '40vh',  
            'min-height': '250px', 
            'overflow': 'auto',
            'flex-shrink': 0
        }
        heatmap1_style = {'height': '30vh', 'width': '100%'} 
        heatmap2_style = {'height': '35vh', 'width': '100%'} 
    else:
        table_style = {'display': 'none', 'height': '0'}
        heatmap1_style = {'height': '45vh', 'width': '100%'}  
        heatmap2_style = {'height': '60vh', 'width': '100%'}
    
    return table_style, table_style, heatmap1_style, heatmap2_style


# Add this callback to manage container classes
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
     dash.dependencies.Input('filtered-isoform-store', 'data')]
)
def update_isoform_heatmap(selected_gene, colorscale, use_ratio_data, show_tables, 
                          show_labels, collapse_tissues, filtered_transcript_ids):
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

    if show_tables == 'show':
        heatmap_height = 450
    else:
        heatmap_height = 650

    if not selected_gene or current_data.empty:
        return {
            'data': [go.Heatmap(z=data1, colorscale=colorscale)],
            'layout': go.Layout(
                margin=dict(l=40, r=40, t=40, b=40),
                title={'text': f'Isoform Expression by Tissue ({data_type})', 'font': {'size': 14}},
                autosize=True,
                height=heatmap_height
            )
        }
    
    try:
        filtered_data = current_data.copy()
        filtered_ids = [int(id) for id in filtered_transcript_ids] if filtered_transcript_ids else []
        filtered_data = current_data[current_data['id'].isin(filtered_ids)] if filtered_ids else current_data
        #if filtered_transcript_ids:
        #    filtered_data = filtered_data[filtered_data['transcript'].isin(filtered_transcript_ids)]
        
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
@app.callback(
    dash.dependencies.Output('barplot1', 'figure'),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('bar-height-slider', 'value'),
     dash.dependencies.Input('filtered-isoform-store', 'data')]
)
def update_transcript_structure(selected_gene, plot_height, filtered_ids):
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

        # Filter transcript data based on master table querying by user
        if filtered_ids and not transcript_data.empty:
            transcript_data = transcript_data[transcript_data['id'].isin(filtered_ids)]
        
        fig = create_transcript_structure_plot(db_path, transcript_data, selected_gene, height=plot_height)
        
        return fig
    
    except Exception as e:
        print(f"Error creating transcript plot: {e}")
        return create_empty_isoform_message(f"Error loading transcript data for {selected_gene}")
    

@app.callback(
    dash.dependencies.Output('atse-map', 'figure'),
    [dash.dependencies.Input('gene-search-dropdown', 'value'),
     dash.dependencies.Input('filtered-junction-store', 'data')]
)
def update_atse_visualization(selected_gene, filtered_junction_ids):
    """Update ATSE splice junction visualization with filtered data"""
    
    if not selected_gene:
        return create_empty_atse_message("Select a gene to view splice junctions and exons")
    
    if atse_data.empty:
        return create_empty_atse_message("ATSE data not loaded - check file path")
    
    try:
        gene_data = process_gene_atse_data(atse_data, selected_gene, db_path)
        
        if filtered_junction_ids and 'junctions' in gene_data:
            original_junctions = gene_data['junctions']
            filtered_junctions = []
            
            for junction in original_junctions:
                junction_id = f"chr{gene_data.get('chromosome', '').replace('chr', '')}_{junction['start']}_{junction['end']}_{gene_data.get('strand', '+')}"
                if junction_id in filtered_junction_ids:
                    filtered_junctions.append(junction)
            
            gene_data['junctions'] = filtered_junctions
        
        fig = create_junction_exon_visualization(gene_data, height=300)
        
        return fig
    
    except Exception as e:
        print(f"Error creating ATSE visualization: {e}")
        traceback.print_exc()

        return create_empty_atse_message(f"Error loading ATSE data for {selected_gene}: {str(e)}")


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
