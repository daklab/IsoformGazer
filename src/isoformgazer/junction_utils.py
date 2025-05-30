import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from matplotlib.patches import Patch
import dash_bio
from dash import dcc

###################################################################
# VISUALIZATION METHODS
###################################################################
def create_summary_clustergram(db_path, height=600, colorscale='Viridis', show_tables='show'):
    """Create summary-level clustergram across all cell types and top junctions"""
    conn = sqlite3.connect(db_path)
    
    # Get a representative sample of junction data across all cell types
    query = """
    SELECT cell_type, junction_id, AVG(psi) as avg_psi, COUNT(*) as n_observations
    FROM junctions 
    WHERE psi IS NOT NULL 
    GROUP BY cell_type, junction_id
    HAVING n_observations >= 5
    ORDER BY n_observations DESC
    LIMIT 2000
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) == 0:
        return create_empty_clustergram_message("No junction data available for summary view")
    
    psi_matrix = df.pivot(index='cell_type', columns='junction_id', values='avg_psi')
    psi_matrix = psi_matrix.fillna(0)
    
    orig_matrix = df.pivot(index='cell_type', columns='junction_id', values='avg_psi')
    valid_columns = orig_matrix.columns[~orig_matrix.isna().all()]
    psi_matrix = psi_matrix[valid_columns]
    
    junction_variance = psi_matrix.var(axis=0).sort_values(ascending=False)
    top_junctions = junction_variance.head(30).index
    psi_matrix_filtered = psi_matrix[top_junctions]
    
    if psi_matrix_filtered.empty:
        return create_empty_clustergram_message("No variable junction data available")
    
    left_margin = max(150, int(height * 0.25))
    hide_junction_labels = (show_tables == 'show')
    
    clustergram = dash_bio.Clustergram(
        data=psi_matrix_filtered.values,
        column_labels=list(psi_matrix_filtered.columns),
        row_labels=list(psi_matrix_filtered.index),
        height=height,
        width=min(900, int(height * 1.4)),  
        color_threshold={
            'row': 0.5,
            'col': 0.5
        },
        hidden_labels='col' if hide_junction_labels else None, 
        cluster='all',
        color_list={
            'row': ['#636EFA', '#EF553B', '#00CC96'],
            'col': ['#AB63FA', '#FFA15A', '#19D3F3'],
            'bg': '#506784'
        },
        line_width=2,
        display_ratio=[0.2, 0.1] 
    )
    
    clustergram = apply_colorscale_to_clustergram(clustergram, colorscale)
    
    bottom_margin = 60 if hide_junction_labels else 120
    
    clustergram.update_layout(
        title={
            'text': "Summary: Junction Usage Across All Cell Types",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        margin=dict(l=left_margin, r=50, t=80, b=bottom_margin),
        autosize=True,
        height=height,
        yaxis=dict(automargin=True),  
        xaxis=dict(automargin=True)   
    )
    
    return clustergram


def create_gene_clustergram(db_path, gene_name, height=600, colorscale='Viridis', show_tables='show'):
    """Create ATSE-level clustergram for a specific gene with all junctions"""
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT cell_type, junction_id, psi, event_id, gene_name
    FROM junctions 
    WHERE gene_name = ? AND psi IS NOT NULL
    """
    
    gene_vals = pd.read_sql_query(query, conn, params=[gene_name])
    conn.close()
    
    if len(gene_vals) == 0:
        return create_empty_clustergram_message(f"No junction data found for gene: {gene_name}")
    
    psi_matrix = gene_vals.pivot(index='cell_type', columns='junction_id', values='psi')
    psi_matrix = psi_matrix.fillna(0)
    
    orig_matrix = gene_vals.pivot(index='cell_type', columns='junction_id', values='psi')
    valid_columns = orig_matrix.columns[~orig_matrix.isna().all()]
    psi_matrix = psi_matrix[valid_columns]
    psi_matrix = psi_matrix.loc[(psi_matrix != 0).any(axis=1), :]
    
    if psi_matrix.empty:
        return create_empty_clustergram_message(f"No PSI data available for gene: {gene_name}")
    
    print(f"Showing {psi_matrix.shape[1]} junctions for gene {gene_name}")
    
    junction_labels = [column for column in psi_matrix.columns]
    
    num_junctions = len(junction_labels)
    hide_junction_labels = (num_junctions > 30) or (show_tables == 'show')
    left_margin = max(120, int(height * 0.2))
    
    if hide_junction_labels:
        bottom_margin = 50 
        actual_clustergram_height = height
    else:
        bottom_margin = max(200, int(height * 0.35)) 
        actual_clustergram_height = height - 80
    
    width = min(1000, max(800, len(junction_labels) * 12, int(height * 1.2)))
    
    # Create clustergram
    clustergram = dash_bio.Clustergram(
        data=psi_matrix.values,
        column_labels=junction_labels,
        row_labels=list(psi_matrix.index),
        height=actual_clustergram_height, 
        width=width,
        color_threshold={
            'row': 0.7,
            'col': 0.7
        },
        # Hide junction labels when master tables are showing: can look at junction name by hovering with cursor
        hidden_labels='col' if hide_junction_labels else None,
        cluster='all',
        color_list={
            'row': ['#636EFA', '#EF553B', '#00CC96', '#AB63FA'],
            'col': ['#FFA15A', '#19D3F3', '#FF6692', '#B6E880'],
            'bg': '#506784'
        },
        line_width=2,
        display_ratio=[0.12, 0.08] if not hide_junction_labels else [0.08, 0.05] 
    )
    clustergram = apply_colorscale_to_clustergram(clustergram, colorscale)
    clustergram.update_layout(
        title={
            'text': f"ATSE-level Splicing Analysis: {gene_name} ({psi_matrix.shape[1]} junctions)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14 if hide_junction_labels else 16}  
        },
        margin=dict(l=left_margin, r=50, t=90, b=bottom_margin),  
        autosize=True,
        height=height,  
        yaxis=dict(
            automargin=True,
            tickangle=0,
            tickfont=dict(size=min(11, max(8, int(height/60))))
        ),
        xaxis=dict(
            automargin=True,
            tickangle=45 if not hide_junction_labels else 0,
            tickfont=dict(size=8) if not hide_junction_labels else dict(size=10)
        )
    )
    
    return clustergram



def create_empty_clustergram_message(message):
    """Create an empty figure with a message"""
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
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig


def apply_colorscale_to_clustergram(fig, colorscale):
    """Apply colorscale to the heatmap portion of a clustergram"""
    try:
        if len(fig.data) > 0:
            heatmap_trace = fig.data[-1]
            heatmap_trace.colorscale = colorscale
            heatmap_trace.showscale = True

    except Exception as e:
        print(f"Warning: Could not apply colorscale: {e}")
    return fig
