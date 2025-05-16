import dash
from dash import html, dcc, dash_table
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from dash.dash_table.Format import Format, Scheme
from dash.exceptions import PreventUpdate
from utils import generate_mock_data

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
# APPLICATION SETUP
###################################################################
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
    columns=[
        dict(id='Isoform_Id', name='Isoform ID'),
        dict(id='Number of Effective Isoforms', name='Num Effective Isoforms', type='numeric',
             format=Format(precision=0, scheme=Scheme.fixed)),
        dict(id='Nuclear Localization', name='Nuclear Localization'),
        dict(id='Coordinates', name='Coordinates'),
        dict(id='Annotation', name='Annotation')
    ],
    data=df1.to_dict('records'),
    editable=False,
    filter_action="native",
    sort_action="native",
    sort_mode="multi",
    column_selectable="single",
    row_selectable="multi",
    selected_columns=[],
    selected_rows=[],
    page_action="native",
    page_current=0,
    page_size=10,
    style_cell={
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
        'minWidth': '100px', 
        'width': '120px', 
        'maxWidth': '120px',
        'padding': '5px',
    },
    style_table={'height': '100%', 'overflowY': 'auto'},
    style_header={
        'backgroundColor': 'white',
        'fontWeight': 'bold',
        'fontsize': 8,
        'font-family': 'sans-serif'
    },
    style_data={'fontsize': 6, 'font-family': 'sans-serif'},
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        }
    ],
)

###################################################################
# JUNCTION MASTER TABLE 
###################################################################
right_data_table = dash_table.DataTable(
    id='right_data_table',
    columns=[
        dict(id='Gene', name='Gene'),
        dict(id='Splice_Junction_ID', name='Splice Junction ID'),
        dict(id='ATSE_ID', name='ATSE ID'),
        dict(id='Strand', name='Strand'),
        dict(id='Annotation', name='Annotation')
    ],
    data=df2.to_dict('records'),
    editable=False,
    filter_action="native",
    sort_action="native",
    sort_mode="multi",
    column_selectable="single",
    row_selectable="multi",
    selected_columns=[],
    selected_rows=[],
    page_action="native",
    page_current=0,
    page_size=10,
    style_cell={
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
        'minWidth': '100px', 
        'width': '120px', 
        'maxWidth': '120px',
        'padding': '5px',
    },
    style_table={'height': '100%', 'overflowY': 'auto'},
    style_header={
        'backgroundColor': 'white',
        'fontWeight': 'bold',
        'fontsize': 8,
        'font-family': 'sans-serif'
    },
    style_data={'fontsize': 6, 'font-family': 'sans-serif'},
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        }
    ],
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
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Data Source'),
                            dcc.Dropdown(
                                id='data-dropdown',
                                options=[
                                    {'label': 'Dataset 1', 'value': 'dataset1'},
                                    {'label': 'Dataset 2', 'value': 'dataset2'},
                                ],
                                value='dataset1'
                            ),
                            html.Div(className='app-controls-desc', children='Select a dataset to visualize')
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Upload Data'),
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div(['Drag and drop or click to select files']),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px 0'
                                }
                            )
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
                                    {'label': 'Red', 'value': 'red'},
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
# TEMP MOCK GENE VALS: will add logic to fetch & format these directly from master table(s)!
mock_gene_options = [
    {'label': 'BRCA1 - Breast cancer type 1', 'value': 'BRCA1'},
    {'label': 'BRCA2 - Breast cancer type 2', 'value': 'BRCA2'},
    {'label': 'TP53 - Tumor protein p53', 'value': 'TP53'},
    {'label': 'EGFR - Epidermal growth factor receptor', 'value': 'EGFR'},
    {'label': 'KRAS - KRAS proto-oncogene', 'value': 'KRAS'},
    {'label': 'PTEN - Phosphatase and tensin homolog', 'value': 'PTEN'},
    {'label': 'TNF - Tumor necrosis factor', 'value': 'TNF'},
    {'label': 'APOE - Apolipoprotein E', 'value': 'APOE'},
    {'label': 'APP - Amyloid beta precursor protein', 'value': 'APP'},
    {'label': 'RBFOX2 (RNA Binding Fox-1 Homolog 2)', 'value': 'RBFOX2'},
    {'label': 'TARDBP (TAR DNA Binding Protein)', 'value': 'TARDBP'},
    {'label': 'FUS (Fused in Sarcoma)', 'value': 'FUS'}
]

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
        return mock_gene_options[:5]
    
    # Filter options based on search_value (note: case insensitive)
    filtered = [option for option in mock_gene_options 
               if search_value.lower() in option['label'].lower()]
    return filtered[:10]


@app.callback(
    dash.dependencies.Output('gene-search-dropdown', 'value'),
    [dash.dependencies.Input('gene-search-dropdown', 'options')]
)
def set_default_value(available_options):
    if len(available_options) > 0:
        return None 
    raise PreventUpdate


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


if __name__ == '__main__':
    app.run(debug=True, port=8050)
