import os 
import math
from tqdm import tqdm
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import dash
from dash import html, dcc, dash_table
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate
from colorama import Fore, Style, init
from data_utils import generate_mock_data, get_isoform_columns, get_junction_columns, parse_filter_query, query_isoforms, query_junctions, get_gene_options

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
    isoform_file = os.path.join(data_dir, 
                                "mt_isoform_gazers_250514.tsv")
    
    with tqdm(desc="Loading isoform master table data", unit=" rows") as pbar:
        df_isoform = pd.read_csv(isoform_file, sep='\t')
        pbar.update(len(df_isoform))
    
    with tqdm(desc="Writing isoform master table data to local database", unit="rows", total=len(df_isoform)) as pbar:
        df_isoform.to_sql('isoforms', conn, if_exists='replace', index=False)
        pbar.update(len(df_isoform))
    
    print(f"✓ Processed all {len(df_isoform):,} rows from isoform master table!")
    print()

    ########################################################
    # Load junction master table data
    ########################################################
    junction_file = os.path.join(data_dir, 
                                 "pseudobulk_final_broad_cell_type_20250514_072922.csv")
    
    # Need to count total lines (minus header) to estimate progress
    with open(junction_file, 'r') as f:
        total_lines = sum(1 for _ in f) - 1 
    
    chunk_size = 100000
    estimated_chunks = (total_lines // chunk_size) + 1
    
    print(f"Loading {total_lines:,} rows of junction data in groupings of {chunk_size:,} rows...")
    first_chunk = True
    row_count = 0
    
    with tqdm(desc="Writing junction master table data to local database", 
              unit="chunk", 
              total=estimated_chunks) as chunk_pbar:
            
        for i, chunk in enumerate(pd.read_csv(junction_file, 
                                              chunksize=chunk_size,
                                              low_memory=False)):
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
    print()
    
    print("Creating database indices...")
    with tqdm(desc="Creating junction-level index", total=2, unit="index") as idx_pbar:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_junctions_gene ON junctions(gene_symbol, gene_id)")
        idx_pbar.update(1)
        idx_pbar.set_description("Creating isoform-level index")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_isoforms_gene ON isoforms(gene_name, gene_id)")
        idx_pbar.update(1)
    
    conn.commit()
    conn.close()
    
    print("✓ Database setup complete!")
    print() 

    return db_path


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
        <title>IsoformGazers</title>
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
# APPLICATION TITLE (ISOFORM GAZERS)
###################################################################
header = html.Div(className='app-header', children=[
    html.Div('daklab ---', style={'fontWeight': 'bold'}),
    html.Div('Isoform Gazers', className='app-header--title')
])

###################################################################
# ISOFORM MASTER TABLE 
###################################################################
left_data_table = dash_table.DataTable(
    id='left_data_table',
    columns=get_isoform_columns(db_path),
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
    columns=get_junction_columns(db_path),
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
                        html.Div(className="app-controls-block", children=[
                            html.Div(className='app-controls-name', children='Query by Value'),
                            html.P("You can use the filter boxes below each master table column header to search and filter the data:"),
                            html.Ul([
                                html.Li([html.Strong("Text columns: "), "Simply type text to find matching rows"]),
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
                                "Results will update automatically as you type."
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
                            html.Div(className='app-controls-name', children='Overview'),
                            dcc.Dropdown(
                                id='overview-dropdown',
                                className='app-controls-block-dropdown',
                                options=[
                                    {'label': 'Heatmap', 'value': 'heatmap'},
                                    {'label': 'Barplot', 'value': 'barplot'},
                                    {'label': 'Both', 'value': 'both'},
                                ],
                                value='both'
                            ),
                            html.Div(className='app-controls-desc', children='Select which visualizations to show')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Display Data Table'),
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
                        # Barplot Section
                        #####################################
                        html.H3('Barplot', className='alignment-settings-section'),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Bar Color'),
                            dcc.Dropdown(
                                id='bar-color-dropdown',
                                className='app-controls-block-dropdown',
                                options=[
                                    {'label': 'Blue', 'value': 'blue'},
                                    {'label': 'Green', 'value': 'green'},
                                    {'label': 'BLUE', 'value': 'BLUE'},
                                    {'label': 'Purple', 'value': 'purple'},
                                    {'label': 'Orange', 'value': 'orange'},
                                ],
                                value='blue'
                            ),
                            html.Div(className='app-controls-desc', children='Select the color for the barplots')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Bar Height'),
                            dcc.Slider(
                                id='bar-height-slider',
                                className='control-slider',
                                min=100,
                                max=300,
                                step=25,
                                value=150,
                                marks={str(i): str(i) for i in range(100, 301, 50)}
                            ),
                            html.Div(className='app-controls-desc', children='Adjust the height of the barplots')
                        ]),
                        html.Hr(),

                        #####################################
                        # Heatmap Section
                        #####################################
                        html.H3('Heatmap', className='alignment-settings-section'),
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
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Heatmap Height'),
                            dcc.Slider(
                                id='heatmap-height-slider',
                                className='control-slider',
                                min=200,
                                max=500,
                                step=50,
                                value=300,
                                marks={str(i): str(i) for i in range(200, 501, 100)}
                            ),
                            html.Div(className='app-controls-desc', children='Adjust the height of the heatmap')
                        ]),
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
                                config={'responsive': True},
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
                    html.Div(className='graph-wrapper', children=[
                        html.Div(className='atse-container', children=[
                            dcc.Graph(
                                id='atse-map',
                                figure=atse_fig,
                                config={
                                    'responsive': True, 
                                    'displayModeBar': True,
                                    'scrollZoom': True
                                },
                                style={'height': '100%', 'width': '100%'}
                            )
                        ], style={'height': '25%', 'min-height': '0', 'margin-bottom': '15px'}),
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

#######################################################################
# CALLBACKS
#######################################################################
##############################################################################################
# CALLBACK FOR QUERYING BY GENE IN CONTROL PANNEL 'Query' TAB: if no search is performed, we 
# show the first five gene names, but otherwise filter by the top ten matches to the current 
# search string. 
##############################################################################################
@app.callback(
    dash.dependencies.Output('gene-search-dropdown', 'options'),
    [dash.dependencies.Input('gene-search-dropdown', 'search_value')]
)
def update_gene_options(search_value):
    if not search_value:
        return get_gene_options(db_path, limit=5)
    
    return get_gene_options(db_path, search_term=search_value, limit=10)


@app.callback(
    dash.dependencies.Output('gene-search-dropdown', 'value'),
    [dash.dependencies.Input('gene-search-dropdown', 'options')]
)
def set_default_value(available_options):
    if len(available_options) > 0:
        return None 
    raise PreventUpdate

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
    data, total_count = query_isoforms(
        db_path,
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
    data, total_count = query_junctions(
        db_path,
        page=page_current if page_current is not None else 0,
        page_size=page_size if page_size is not None else 10,
        sort_by=sort_by,
        filters=filters,
        gene_filter=selected_gene
    )
    page_count = math.ceil(total_count / page_size) if page_size else 1
    return data, page_count


@app.callback(
    [
        dash.dependencies.Output('heatmap1', 'figure'),
        dash.dependencies.Output('heatmap2', 'figure')
    ],
    [dash.dependencies.Input('colorscale-dropdown', 'value')]
)
def update_colorscale(colorscale):
    heatmap1_fig = {
        'data': [go.Heatmap(z=data1, colorscale=colorscale)],
        'layout': go.Layout(
            margin=dict(l=40, r=40, t=40, b=40),
            title={'text': 'Isoform Expression by Tissue', 'font': {'size': 14}},
            autosize=True
        )
    }
    
    heatmap2_fig = {
        'data': [go.Heatmap(z=data2, colorscale=colorscale)],
        'layout': go.Layout(
            margin=dict(l=40, r=40, t=40, b=40),
            title={'text': 'Junction Usage by Cell Type', 'font': {'size': 14}},
            autosize=True
        )
    }
    
    return heatmap1_fig, heatmap2_fig


@app.callback(
    dash.dependencies.Output('barplot1', 'figure'),
    [
        dash.dependencies.Input('bar-height-slider', 'value'),
        dash.dependencies.Input('bar-color-dropdown', 'value')
    ]
)
def update_barplots(height, color):
    barplot1_fig = {
        'data': [go.Bar(x=list(range(10)), y=data1.sum(axis=1), marker_color=color)],
        'layout': go.Layout(
            margin=dict(l=40, r=40, t=20, b=20),
            title={'text': 'Isoform Expression by Tissue', 'font': {'size': 14}},
            xaxis={'title': {'text': 'Tissue', 'font': {'size': 12}}, 'title_standoff': 40, 'ticksuffix': ' '},
            yaxis={'title': {'text': 'Count', 'font': {'size': 12}}},
            autosize=True
        )
    }
    
    return barplot1_fig


@app.callback(
    [
        dash.dependencies.Output('table1-container', 'style'),
        dash.dependencies.Output('table2-container', 'style')
    ],
    [dash.dependencies.Input('show-table-radio', 'value')]
)
def toggle_tables(show_tables):
    if show_tables == 'show':
        style = {'display': 'block', 'height': '35%', 'min-height': '0', 'overflow': 'auto'}
    else:
        style = {'display': 'none', 'height': '0'}
    
    return style, style


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
    {PURPLE}   ██╗███████╗ ██████╗ ███████╗ ██████╗ ██████╗ ███╗   ███╗    {GOLD}  ██████╗  █████╗ ███████╗███████╗██████╗ 
    {PURPLE}   ██║██╔════╝██╔═══██╗██╔════╝██╔═══██╗██╔══██╗████╗ ████║    {GOLD} ██╔════╝ ██╔══██╗╚══███╔╝██╔════╝██╔══██╗
    {PURPLE}   ██║███████╗██║   ██║█████╗  ██║   ██║██████╔╝██╔████╔██║    {GOLD} ██║  ███╗███████║  ███╔╝ █████╗  ██████╔╝
    {PURPLE}   ██║╚════██║██║   ██║██╔══╝  ██║   ██║██╔══██╗██║╚██╔╝██║    {GOLD} ██║   ██║██╔══██║ ███╔╝  ██╔══╝  ██╔══██╗
    {PURPLE}   ██║███████║╚██████╔╝██║     ╚██████╔╝██║  ██║██║ ╚═╝ ██║    {GOLD} ╚██████╔╝██║  ██║███████╗███████╗██║  ██║
    {PURPLE}   ╚═╝╚══════╝ ╚═════╝ ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝    {GOLD}  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝
    {BLUE}                                                                                                                         
    {WHITE}       *    +         .    *      +         .    *        +     .   {GOLD}*{CYAN}      +         *       *    +         .  
    {CYAN}  +        .    *     {GOLD}   +    .  {WHITE}    *         +     .        *           .       {GOLD}*{WHITE}    + +        .    *     
    {WHITE}    .    *    {CYAN}   *{GOLD}    +    .         *      +    .        *    +        .       {GOLD}*{WHITE}     *.    *    {CYAN}  
    {WHITE} *     +   {CYAN}     *   *{GOLD}    .  {WHITE}    *    +        .       *         +      .    *       {GOLD}*{WHITE}    .*     +   {WHITE}    *  
    {CYAN}    .    {CYAN}    * *{WHITE}      +        .      *       +     .    *        +     {GOLD}*{WHITE}         .     *    .    {WHITE}   * * *{WHITE} +. 
    {WHITE}  +    *    {WHITE}     *{WHITE}    .    +       *        .      +     *       .         {GOLD}*{CYAN}    +       *  +    *    {WHITE}     *{WHITE} .
    {RESET}"""
    
    print(banner)
    

if __name__ == '__main__':
    display_ascii_banner()

    database_exists = check_database_status()
    if not database_exists:
        print("Database initialization completed.")
        
    app.run(debug=True, port=8050, use_reloader=False)
