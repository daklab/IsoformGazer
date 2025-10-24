import os
import sqlite3
import numpy as np
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
from data_utils import apply_distance_preprocessing
from isoform_utils import load_psl_data, get_gene_id_for_gene_name, abbreviate_transcript_name, calculate_dynamic_structure_plot_height, calculate_clustergram_min_height
from isoformgazer.performance_utils import cached, cached_transcript_structure_processing, plot_optimizer

###################################################################
# MARGIN PRESETS FOR FIGURES
###################################################################
MIN_MARGIN = 8 
MAX_MARGIN = 55
MAX_MARGIN_LABELS = 65

###################################################################
# VISUALIZATION METHODS
###################################################################
def create_summary_clustergram(db_path, height=600, colorscale='Viridis', show_tables='show', show_celltype_labels=False, distance_metric='euclidean', linkage_method='complete', show_gridlines=False):
    """Create summary-level clustergram across all cell types and top junctions"""
    conn = sqlite3.connect(db_path)
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
    
    psi_matrix = df.pivot(index='junction_id', columns='cell_type', values='avg_psi')

    orig_matrix = df.pivot(index='junction_id', columns='cell_type', values='avg_psi')
    valid_columns = orig_matrix.columns[~orig_matrix.isna().all()]

    psi_matrix = psi_matrix[valid_columns]
    psi_matrix_with_nan = psi_matrix.copy()
    
    # For variance calculation, use 0-filled data
    psi_matrix_for_variance = psi_matrix.fillna(0)
    junction_variance = psi_matrix_for_variance.var(axis=0).sort_values(ascending=False)
    top_junctions = junction_variance.head(30).index
    psi_matrix_filtered = psi_matrix[top_junctions]

    psi_matrix_filtered_with_nan = psi_matrix_filtered.copy()

    if psi_matrix_filtered.empty:
        return create_empty_clustergram_message("No variable junction data available")

    # For clustering, fill NaN with 0
    psi_matrix_filtered = psi_matrix_filtered.fillna(0)
    psi_matrix_filtered = psi_matrix_filtered.clip(lower=0, upper=1)
    
    # Check for constant rows/columns that cause issues
    row_std = psi_matrix_filtered.std(axis=1)
    col_std = psi_matrix_filtered.std(axis=0)
    
    # Apply preprocessing based on distance metric
    if distance_metric in ['correlation', 'seuclidean', 'cosine']:
        psi_matrix_filtered = apply_distance_preprocessing(psi_matrix_filtered, distance_metric)
    
    left_margin = max(150, int(height * 0.25))
    hide_junction_labels = (show_tables == 'show')
    
    clustergram = dash_bio.Clustergram(
        data=psi_matrix_filtered.values,
        column_labels=list(psi_matrix_filtered.columns),
        row_labels=list(psi_matrix_filtered.index),
        height=height,
        color_threshold={
            'row': 0.5,
            'col': 0.5
        },
        hidden_labels='col' if (hide_junction_labels or not show_celltype_labels) else None, 
        cluster='all',
        color_list={
            'row': ['#636EFA', '#EF553B', '#00CC96'],
            'col': ['#AB63FA', '#FFA15A', '#19D3F3'],
            'bg': '#506784'
        },
        row_dist=distance_metric,
        col_dist=distance_metric,
        link_method=linkage_method,
        line_width=2,
        display_ratio=[0.2, 0.1],
        standardize='none'
    )
    
    clustergram = apply_colorscale_to_clustergram(clustergram, colorscale)

    # Replace heatmap data with NaN values (shown as white)
    if len(clustergram.data) > 0:
        heatmap_trace = clustergram.data[-1]
        heatmap_trace.z = psi_matrix_filtered_with_nan.values

    if show_gridlines:
        if len(clustergram.data) > 0:
            heatmap_trace = clustergram.data[-1]
            heatmap_trace.xgap = 1
            heatmap_trace.ygap = 1

    bottom_margin = 60 if hide_junction_labels else 120

    clustergram.update_layout(
        title={
            'text': "Summary: Junction Usage Across All Cell Types",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        margin=dict(l=MIN_MARGIN,
                    r=MIN_MARGIN,
                    t=MAX_MARGIN,
                    b=bottom_margin),
        autosize=True,
        width=None,
        height=height,
        plot_bgcolor='white',
        yaxis=dict(automargin=True),
        xaxis=dict(automargin=True)
    )

    if len(clustergram.data) > 0:
        heatmap_trace = clustergram.data[-1]
        customdata = []
        for i in range(len(psi_matrix_filtered_with_nan)):
            row_data = []
            for j in range(len(psi_matrix_filtered_with_nan.columns)):
                psi_val = psi_matrix_filtered_with_nan.iloc[i, j]
                if pd.isna(psi_val):
                    row_data.append('NaN')
                else:
                    row_data.append(f'{psi_val:.2f}')
            customdata.append(row_data)

        heatmap_trace.customdata = customdata
        heatmap_trace.hovertemplate = (
            '<b>Junction ID</b>: %{y}<br>'
            '<b>Cell Type</b>: %{x}<br>'
            '<b>PSI</b>: %{customdata}<br>'
            '<extra></extra>'
        )

    return clustergram


def create_single_junction_heatmap(gene_vals, gene_name, height, colorscale):
    """Create simple heatmap when only one junction remains"""
    junction_id = gene_vals['junction_id'].iloc[0]
    heatmap_data = gene_vals.pivot(index='junction_id', columns='cell_type', values='psi')
    n_cells_data = gene_vals.pivot(index='junction_id', columns='cell_type', values='n_cells')

    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns.tolist(),
        y=heatmap_data.index.tolist(),
        colorscale=colorscale,
        zmin=0,
        zmax=1,
        customdata=n_cells_data.values,
        text=n_cells_data.values.astype(str),
        hovertemplate=(
            '<b>Junction ID</b>: %{y}<br>'
            '<b>Cell Type</b>: %{x}<br>'
            '<b>PSI</b>: %{z:.2f}<br>'
            '<b>Number of Cells</b>: %{text}<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title={
            'text': f"Splicing PSI Heatmap for Junction {junction_id} in {gene_name}",
            'font': {'size': 14},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=height,
        xaxis=dict(title="Junction"),
        yaxis=dict(title="Cell Type"),
        margin=dict(l=MIN_MARGIN, 
                    r=MIN_MARGIN, 
                    t=MAX_MARGIN, 
                    b=MAX_MARGIN)
    )
    
    return fig


def create_gene_clustergram(db_path, gene_name, height=600, colorscale='Viridis', show_tables='show', filtered_junction_ids=None, show_celltype_labels=False, distance_metric='euclidean', linkage_method='complete', show_gridlines=False):
    """Create ATSE-level clustergram with correct junction ID matching for tooltips"""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT cell_type, junction_id, psi, n_cells, event_id, gene_name
    FROM junctions 
    WHERE gene_name = ? AND psi IS NOT NULL
    """
    if filtered_junction_ids:
        filtered_ids_str = [str(jid) for jid in filtered_junction_ids]
        placeholders = ','.join(['?'] * len(filtered_ids_str))
        query += f" AND junction_id IN ({placeholders})"
        params = [gene_name] + filtered_ids_str
    else:
        params = [gene_name]
    
    gene_vals = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if len(gene_vals) == 0:
        return create_empty_clustergram_message(f"No junction data found for gene: {gene_name}")
    
    psi_matrix = gene_vals.pivot(index='junction_id', columns='cell_type', values='psi')

    orig_matrix = gene_vals.pivot(index='junction_id', columns='cell_type', values='psi')
    valid_columns = orig_matrix.columns[~orig_matrix.isna().all()]
    psi_matrix = psi_matrix[valid_columns]
    psi_matrix_with_nan = psi_matrix.copy()

    # n_cells matrix to allow us to show cell count in tooltip on hover in dashboard
    n_cells_matrix = gene_vals.pivot(index='junction_id', columns='cell_type', values='n_cells')
    n_cells_matrix = n_cells_matrix.reindex(index=psi_matrix.index, columns=psi_matrix.columns).fillna(0)
    
    n_cells_values = n_cells_matrix.values.astype(int)
    junction_labels = list(psi_matrix.index)
    cell_type_labels = list(psi_matrix.columns)

    if psi_matrix.empty:
        return create_empty_clustergram_message(f"No PSI data available for gene: {gene_name}")

    if height <= 700:
        num_junctions = len(junction_labels)
        min_required_height = calculate_clustergram_min_height(num_junctions, base_height=600)
        height = max(height, min_required_height)
    
    if len(gene_vals['junction_id'].unique()) == 1:
        return create_single_junction_heatmap(
            gene_vals, gene_name, height, colorscale
        )
    
    psi_matrix_processed = psi_matrix.copy()
    psi_matrix_processed = psi_matrix_processed.replace([np.inf, -np.inf], np.nan)
    psi_matrix_processed = psi_matrix_processed.astype(float)
    psi_matrix_processed = psi_matrix_processed.clip(lower=0, upper=1)

    # For clustering, use row-median filled data (NaN -> median value per row)
    for idx in psi_matrix_processed.index:
        row = psi_matrix_processed.loc[idx]
        row_median = row.median()
        if pd.notna(row_median):
            psi_matrix_processed.loc[idx, row.isna()] = row_median
        else:
            psi_matrix_processed.loc[idx, row.isna()] = 0
    
    # Apply preprocessing for problematic distance metrics
    if distance_metric in ['correlation', 'seuclidean', 'cosine']:
        psi_matrix_processed = apply_distance_preprocessing(psi_matrix_processed, distance_metric)
    
    num_junctions = len(junction_labels)
    hide_junction_labels = (num_junctions > 30) or (show_tables == 'show')
    left_margin = max(120, int(height * 0.2))
    
    if hide_junction_labels:
        bottom_margin = MAX_MARGIN
        actual_clustergram_height = height
    else:
        bottom_margin = max(200, int(height * 0.35)) 
        actual_clustergram_height = height - 80
    
    try:
        clustergram, computed_traces = dash_bio.Clustergram(
            data=psi_matrix_processed.values,
            row_labels=junction_labels,        
            column_labels=cell_type_labels,
            height=actual_clustergram_height,
            color_threshold={'row': 0.7, 'col': 0.7},
            hidden_labels='col' if not show_celltype_labels else None,
            cluster='all',
            color_list={
                'row': ['#636EFA', '#EF553B', '#00CC96', '#AB63FA'],
                'col': ['#FFA15A', '#19D3F3', '#FF6692', '#B6E880'],
                'bg': '#506784'
            },
            line_width=2,
            display_ratio=[0.12, 0.08] if not hide_junction_labels else [0.08, 0.05],
            standardize='none', 
            center_values=False,
            return_computed_traces=True,
            row_dist=distance_metric,
            col_dist=distance_metric,
            link_method=linkage_method
        )
        
        row_ids = computed_traces['row_ids']
        col_ids = computed_traces['column_ids']
        reordered_n_cells = n_cells_values[row_ids, :][:, col_ids]
        reordered_psi = psi_matrix_with_nan.iloc[row_ids, :].iloc[:, col_ids]

        heatmap_trace = clustergram.data[-1]

        customdata = []
        for i in range(len(row_ids)):
            row_data = []
            for j in range(len(col_ids)):
                psi_val = reordered_psi.iloc[i, j]
                if pd.isna(psi_val):
                    row_data.append('NaN')
                else:
                    row_data.append(f'{psi_val:.2f}')
            customdata.append(row_data)

        heatmap_trace.customdata = customdata
        heatmap_trace.hovertemplate = (
            '<b>Junction ID</b>: %{y}<br>'
            '<b>Cell Type</b>: %{x}<br>'
            '<b>PSI</b>: %{customdata}<br>'
            '<b>Number of Cells</b>: %{text}<extra></extra>'
        )
        heatmap_trace.text = reordered_n_cells.astype(str).tolist()
        
    except Exception as e:
        print(f"Error creating clustergram: {e}")
        return create_empty_clustergram_message(f"Error creating visualization for {gene_name}")

    clustergram = apply_colorscale_to_clustergram(clustergram, colorscale)

    if len(clustergram.data) > 0:
        nan_data_reordered = psi_matrix_with_nan.iloc[row_ids, :].iloc[:, col_ids]
        heatmap_trace = clustergram.data[-1]
        heatmap_trace.z = nan_data_reordered.values

    if show_gridlines:
        if len(clustergram.data) > 0:
            heatmap_trace = clustergram.data[-1]
            heatmap_trace.xgap = 1
            heatmap_trace.ygap = 1

    try:
        if len(clustergram.data) > 0:
            heatmap_trace = clustergram.data[-1]
            if hasattr(heatmap_trace, 'colorbar'):
                heatmap_trace.colorbar.x = 1.0 
                heatmap_trace.colorbar.xpad = 160
                heatmap_trace.colorbar.y = 1.005
                heatmap_trace.colorbar.yanchor = 'top'
                heatmap_trace.colorbar.len = 0.3
                heatmap_trace.colorbar.thickness = 20
                heatmap_trace.colorbar.title = 'PSI'

    except Exception as e:
        print(f"Warning: Could not update colorbar position: {e}")

    clustergram.update_layout(
        title={
            'text': f"Splicing PSI Clustermap for {gene_name} ({len(junction_labels)} junctions)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14 if hide_junction_labels else 16}
        },
        margin=dict(
            l=MIN_MARGIN,
            r=MIN_MARGIN,
            t=MAX_MARGIN,
            b=bottom_margin
        ),
        autosize=True,
        width=None,
        height=height,
        plot_bgcolor='white',
        yaxis=dict(
            automargin=True,
            tickangle=0,
            tickfont=dict(size=min(11, max(8, int(height/60))))
        ),
        xaxis=dict(
            automargin=False,
            tickangle=90,
            tickfont=dict(size=8) if not hide_junction_labels else dict(size=10),
            side='bottom'
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
        margin=dict(l=MIN_MARGIN, r=MIN_MARGIN, t=MIN_MARGIN, b=MIN_MARGIN)
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


def load_atse_data(atse_file_path: str) -> pd.DataFrame:
    """Load ATSE data for junction visualization"""
    try:
        atse_df = pd.read_csv(atse_file_path, sep='\t')
        print(f"Loaded ATSE data with {len(atse_df)} records")
        return atse_df
    except Exception as e:
        print(f"Error loading ATSE file: {e}")
        return pd.DataFrame()


def get_gene_id_from_atse(db_path: str, gene_name: str) -> str:
    """Get gene_id for a given gene_name from the database"""
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT gene_id FROM junctions WHERE gene_name = ? LIMIT 1"
    result = pd.read_sql_query(query, conn, params=[gene_name])
    conn.close()
    
    if len(result) > 0:
        # Remove version number
        return result.iloc[0]['gene_id'].split('.')[0]
    return None


def filter_junctions_by_transcripts(db_path: str, gene_name: str, filtered_transcript_ids: list) -> list:
    """
    Filter junction IDs based on transcript overlap using perfect_match_5_prime and perfect_match_3_prime columns.
    Returns a list of junction IDs that align with any of the filtered transcripts.
    """
    if not filtered_transcript_ids:
        return []
    
    conn = sqlite3.connect(db_path)
    transcript_id_placeholders = ','.join(['?'] * len(filtered_transcript_ids))
    transcript_query = f"""
    SELECT DISTINCT transcript 
    FROM isoforms 
    WHERE id IN ({transcript_id_placeholders})
    """
    transcript_result = pd.read_sql_query(transcript_query, conn, params=filtered_transcript_ids)
    
    if transcript_result.empty:
        conn.close()
        return []
    
    transcript_names = transcript_result['transcript'].tolist()
    
    # Get gene_id for the gene_name
    gene_id_query = "SELECT DISTINCT gene_id FROM junctions WHERE gene_name = ? LIMIT 1"
    gene_result = pd.read_sql_query(gene_id_query, conn, params=[gene_name])
    
    if gene_result.empty:
        conn.close()
        return []
    
    gene_id_base = gene_result.iloc[0]['gene_id'].split('.')[0]
    
    # Query ATSE data for junctions that match the transcripts
    atse_query = """
    SELECT junction_id, perfect_match_3_prime, perfect_match_5_prime
    FROM atse_data 
    WHERE (gene_id_clean = ? OR gene_name = ?)
    AND junction_id IS NOT NULL
    """
    atse_data = pd.read_sql_query(atse_query, conn, params=[gene_id_base, gene_name])
    conn.close()
    
    if atse_data.empty:
        return []
    
    matching_junction_ids = []
    
    for _, row in atse_data.iterrows():
        junction_id = row['junction_id']
        perfect_match_5_prime = row['perfect_match_5_prime'] 
        perfect_match_3_prime = row['perfect_match_3_prime']
        junction_matches = False
        
        transcript_names_full = set(transcript_names)
        transcript_names_abbreviated = set()
        for name in transcript_names:
            if ':' in name:
                abbreviated = abbreviate_transcript_name(name)
                transcript_names_abbreviated.add(abbreviated)

            transcript_names_abbreviated.add(name)
        
        # 5' perfect matches
        if pd.notna(perfect_match_5_prime):
            transcripts_5_prime = str(perfect_match_5_prime).split(',')
            transcripts_5_prime = [t.strip() for t in transcripts_5_prime if t.strip()]
            
            for transcript_in_junction in transcripts_5_prime:
                # Exact match
                if transcript_in_junction in transcript_names_full:
                    junction_matches = True
                    break
                # Abbreviated transcript name match 
                if transcript_in_junction in transcript_names_abbreviated:
                    junction_matches = True
                    break
                # Check if junction transcript matches any filtered transcript
                for filtered_transcript in transcript_names_full:
                    if (':' in filtered_transcript and abbreviate_transcript_name(filtered_transcript) == transcript_in_junction):
                        junction_matches = True
                        break

                if junction_matches:
                    break
        
        # 3' perfect matches
        if not junction_matches and pd.notna(perfect_match_3_prime):
            transcripts_3_prime = str(perfect_match_3_prime).split(',')
            transcripts_3_prime = [t.strip() for t in transcripts_3_prime if t.strip()]
            
            for transcript_in_junction in transcripts_3_prime:
                # Exact match
                if transcript_in_junction in transcript_names_full:
                    junction_matches = True
                    break

                # Abbreviated transcript name match 
                if transcript_in_junction in transcript_names_abbreviated:
                    junction_matches = True
                    break

                # Check if junction transcript matches any filtered transcript
                for filtered_transcript in transcript_names_full:
                    if (':' in filtered_transcript and 
                        abbreviate_transcript_name(filtered_transcript) == transcript_in_junction):
                        junction_matches = True
                        break

                if junction_matches:
                    break
        
        if junction_matches:
            matching_junction_ids.append(junction_id)
    
    return list(set(matching_junction_ids))


def filter_transcripts_by_junctions(db_path: str, filtered_junction_ids: list) -> list:
    """
    Filter transcript IDs based on junction overlap using matched_transcript_ids column.
    Returns a list of transcript IDs (from isoforms table) that align with the filtered junctions.
    """
    if not filtered_junction_ids:
        return []
    
    conn = sqlite3.connect(db_path)
    junction_id_placeholders = ','.join(['?'] * len(filtered_junction_ids))
    junction_query = f"""
    SELECT DISTINCT matched_transcript_ids 
    FROM junctions 
    WHERE junction_id IN ({junction_id_placeholders})
    AND matched_transcript_ids IS NOT NULL
    """
    junction_result = pd.read_sql_query(junction_query, conn, params=filtered_junction_ids)
    
    if junction_result.empty:
        conn.close()
        return []
    
    # Get all transcript names
    all_transcript_names = set()
    for _, row in junction_result.iterrows():
        transcript_ids_str = str(row['matched_transcript_ids'])

        if pd.notna(transcript_ids_str) and transcript_ids_str not in ('nan', 'None', ''):
            transcript_names = transcript_ids_str.split(',')

            for name in transcript_names:
                name = name.strip()

                if name:
                    all_transcript_names.add(name)
    
    if not all_transcript_names:
        conn.close()
        return []
    
    # Get transcript IDs that match the transcript names
    transcript_names_list = list(all_transcript_names)
    name_placeholders = ','.join(['?'] * len(transcript_names_list))
    transcript_query = f"""
    SELECT DISTINCT id 
    FROM isoforms 
    WHERE transcript IN ({name_placeholders})
    """
    transcript_result = pd.read_sql_query(transcript_query, conn, params=transcript_names_list)
    conn.close()
    
    if transcript_result.empty:
        return []
    
    return transcript_result['id'].tolist()


@cached(cache_timeout=300)
def process_gene_atse_data(gene_name: str, db_path: str, filtered_junction_ids=None) -> dict:
    """Process ATSE data for a specific gene with caching and performance optimization"""
    #with ProfilerContext(f"process_atse_data_{gene_name}"):
    gene_id_with_version = None
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size = 10000")
    conn.execute("PRAGMA temp_store = MEMORY")

    # Get gene_id from db
    gene_id_query = "SELECT DISTINCT gene_id FROM junctions WHERE gene_name = ? LIMIT 1"
    gene_result = pd.read_sql_query(gene_id_query, conn, params=[gene_name])
    
    if gene_result.empty:
        conn.close()
        return {'error': f"No gene_id found for {gene_name}"}
    
    gene_id_with_version = gene_result.iloc[0]['gene_id']
    gene_id_base = gene_id_with_version.split('.')[0]
    #memory_tracker.measure(f"after_gene_lookup_{gene_name}")
    
    atse_query = """
    SELECT event_id, gene_id, gene_name, event_strand, chromosome, 
            start, end, junction_id, transcripts, perfect_match_3_prime, 
            perfect_match_5_prime, both_ends_transcripts, 
            only_5_prime_transcripts, only_3_prime_transcripts,
            atse_start, atse_end, event_type
    FROM atse_data 
    WHERE (gene_id_clean = ? OR gene_name = ?)
    ORDER BY start, end
    """
    gene_atse = pd.read_sql_query(atse_query, conn, params=[gene_id_base, gene_name])
    conn.close()
    #memory_tracker.measure(f"after_atse_query_{gene_name}")
    
    if gene_atse.empty:
        return {'error': f"No ATSE data found for gene {gene_name}"}

    gene_info = gene_atse.iloc[0]
    strand = gene_info.get('event_strand', gene_info.get('strand', '+'))
    chromosome = gene_info.get('chromosome', gene_info.get('chrom', 'chr1'))
    
    junctions = []
    transcripts_set = set()
    
    #memory_tracker.measure(f"before_junction_processing_{gene_name}")
    valid_rows = gene_atse[
        (gene_atse['start'].notna()) & (gene_atse['end'].notna()) |
        (gene_atse['atse_start'].notna()) & (gene_atse['atse_end'].notna()) |
        (gene_atse['junction_id'].notna())
    ]
    
    for _, row in valid_rows.iterrows():
        junction_coords = []
        
        if pd.notna(row.get('start')) and pd.notna(row.get('end')):
            try:
                start = int(row['start'])
                end = int(row['end'])
                junction_coords.append((start, end))
            except (ValueError, TypeError):
                pass
        
        # Fallback to junction_id parsing
        if not junction_coords and pd.notna(row.get('junction_id')):
            junction_id = str(row['junction_id'])
            if '_' in junction_id:
                parts = junction_id.split('_')
                if len(parts) >= 3:
                    try:
                        start = int(parts[1])
                        end = int(parts[2])
                        junction_coords.append((start, end))
                    except (ValueError, IndexError):
                        pass
        
        transcript_columns = ['transcripts', 'perfect_match_3_prime', 'perfect_match_5_prime', 
                                'both_ends_transcripts', 'only_5_prime_transcripts', 'only_3_prime_transcripts']
        
        for col in transcript_columns:
            if col in row and pd.notna(row[col]):
                transcripts_str = str(row[col])
                if ',' in transcripts_str:
                    transcripts_list = transcripts_str.split(',')
                elif ';' in transcripts_str:
                    transcripts_list = transcripts_str.split(';')
                elif '|' in transcripts_str:
                    transcripts_list = transcripts_str.split('|')
                else:
                    transcripts_list = [transcripts_str]
                
                for transcript in transcripts_list:
                    transcript = transcript.strip()
                    if transcript and transcript not in ('nan', 'None', ''):
                        transcripts_set.add(transcript)
        
        for start, end in junction_coords:
            junctions.append({
                'start': start,
                'end': end,
                'event_id': row.get('event_id', ''),
                'event_type': row.get('event_type', 'unknown'),
                'transcripts': list(transcripts_set)
            })
    
    if not junctions:
        if len(gene_atse) > 0:
            first_row = gene_atse.iloc[0]
        
        # use atse_start/atse_end as fallback
        for _, row in gene_atse.iterrows():
            if 'atse_start' in row and 'atse_end' in row and pd.notna(row['atse_start']) and pd.notna(row['atse_end']):
                try:
                    start = int(row['atse_start'])
                    end = int(row['atse_end'])
                    junctions.append({
                        'start': start,
                        'end': end,
                        'event_id': row.get('event_id', ''),
                        'event_type': row.get('event_type', 'unknown'),
                        'transcripts': list(transcripts_set)
                    })
                except (ValueError, TypeError):
                    continue
    
    # Remove duplicates and sort junctions
    unique_junctions = []
    seen = set()
    for j in junctions:
        key = (j['start'], j['end'])
        if key not in seen:
            seen.add(key)
            unique_junctions.append(j)
    
    unique_junctions.sort(key=lambda x: x['start'])
    
    if filtered_junction_ids and len(filtered_junction_ids) > 0:
        valid_junction_ids = [str(jid).strip() for jid in filtered_junction_ids if jid and str(jid).strip()]
        
        if valid_junction_ids:
            filtered_unique_junctions = []
            
            for junction in unique_junctions:
                junction_id = f"chr{chromosome.replace('chr', '')}_{junction['start']}_{junction['end']}_{strand}"
                if junction_id in valid_junction_ids:
                    filtered_unique_junctions.append(junction)
            
            unique_junctions = filtered_unique_junctions
    
    return {
        'gene_name': gene_name,
        'gene_id': gene_id_with_version,
        'strand': strand,
        'chromosome': chromosome,
        'junctions': unique_junctions,
        'transcripts': sorted(list(transcripts_set))
    }


def process_transcript_structure(psl_df: pd.DataFrame, 
                                 gene_name: str, 
                                 db_path: str) -> pd.DataFrame:
    """Process PSL data to get transcript structure for a specific gene"""
    
    # Get gene_id for the gene_name from the database
    gene_id = get_gene_id_for_gene_name(db_path, gene_name)
    
    if gene_id is None:
        print(f"Cannot process transcript structure: no gene_id found for {gene_name}")
        return pd.DataFrame()
    
    # Filter PSL data for specific gene
    gene_psl = psl_df[psl_df['gene_id'].str.contains(gene_id, na=False)]
    
    if gene_psl.empty:
        print(f"No PSL data found for gene_id: {gene_id}")
        return pd.DataFrame()
    
    #print(f"Found {len(gene_psl)} PSL records for gene {gene_name} (gene_id: {gene_id})")
    
    # Process block information
    transcript_data = []
    
    for _, row in gene_psl.iterrows():
        # Parse block sizes and starts
        try:
            block_sizes = [int(x) for x in row['blockSizes'].strip(',').split(',') if x]
            block_starts = [int(x) for x in row['tStarts'].strip(',').split(',') if x]
            
            # Create exon coordinates
            for i, (size, start) in enumerate(zip(block_sizes, block_starts)):
                transcript_data.append({
                    'trans_id': row['trans_id'],
                    'gene_id': row['gene_id'],
                    'chr': row['tName'],
                    'strand': row['strand'],
                    'transcript_start': row['tStart'],
                    'transcript_end': row['tEnd'],
                    'transcript_length': row['transcript_length'],
                    'exon_number': i + 1,
                    'exon_start': start,
                    'exon_end': start + size,
                    'exon_size': size
                })
        except Exception as e:
            print(f"Error processing transcript {row['trans_id']}: {e}")
            continue
    
    return pd.DataFrame(transcript_data)


@cached(cache_timeout=600)
def create_junction_exon_visualization(gene_data: dict, 
                                       height: int = 600,
                                       show_y_labels: bool = False,
                                       exon_color: str = '#2E86C1',
                                       junction_color: str = '#85929E',
                                       filtered_transcript_ids: list = None) -> go.Figure:
    """Create junction and exon structure plot for junction master table data"""
    if 'error' in gene_data:
        return create_empty_atse_message(gene_data['error'])
    
    #with PlotPerformanceContext(gene_data.get('gene_name', 'unknown'), "junction_exon_visualization"):
    junctions = gene_data['junctions']
    gene_name = gene_data['gene_name']
    gene_id = gene_data['gene_id']
    strand = gene_data['strand']
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "data", "isoformgazer.db")

    transcript_data = cached_transcript_structure_processing(db_path, gene_name, filtered_transcript_ids or [])    
    if transcript_data.empty and not junctions:
        return create_empty_atse_message(f"No transcript or junction data found for {gene_name}")
    
    fig = go.Figure()
    intron_color = '#85929E' 
    transcript_labels = []
    transcript_y_positions = []
    
    if not transcript_data.empty:
        transcript_data_opt = plot_optimizer.preprocess_dataframe_for_plotting(transcript_data)
        transcript_summary = transcript_data_opt.groupby('id', as_index=False).agg({
            'trans_id': 'first',
            'transcript_start': 'min',
            'transcript_end': 'max'
        })
        
        transcript_summary['transcript_length'] = transcript_summary['transcript_end'] - transcript_summary['transcript_start']
        transcript_summary = transcript_summary.sort_values('transcript_length', ascending=False).reset_index(drop=True)
        transcript_summary['trans_order'] = range(1, len(transcript_summary) + 1)
        
        # Calculate dynamic height if not provided or if using default slider value
        if height is None or height == 600:
            num_transcripts = len(transcript_summary)
            height = calculate_dynamic_structure_plot_height(num_transcripts, base_height=600)
        
        plot_data = transcript_data_opt.merge(transcript_summary[['id', 'trans_order']], on='id')
        
        min_start = plot_data['transcript_start'].min()
        max_end = plot_data['transcript_end'].max()
        y_max = len(transcript_summary) + 1
        
        transcript_labels = [abbreviate_transcript_name(trans_id) for trans_id in transcript_summary['trans_id'].tolist()]
        transcript_y_positions = transcript_summary['trans_order'].tolist()
        
        intron_x = []
        intron_y = []
        intron_text = []
        
        for _, transcript in transcript_summary.iterrows():
            intron_x.extend([transcript['transcript_start'], transcript['transcript_end'], None])
            intron_y.extend([transcript['trans_order'], transcript['trans_order'], None])
            intron_text.extend([f"Isoform ID: {transcript['id']}<br>Transcript: {abbreviate_transcript_name(transcript['trans_id'])}<br>Length: {transcript['transcript_length']:,} bp", "", ""])
        
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
        
        exon_shapes = []
        exon_hover_x = []
        exon_hover_y = []
        exon_hover_text = []
        
        for _, transcript in transcript_summary.iterrows():
            isoform_id = transcript['id']
            trans_id = transcript['trans_id']
            trans_order = transcript['trans_order']
            
            trans_exons = plot_data[plot_data['id'] == isoform_id].sort_values('exon_start')
            
            for _, exon in trans_exons.iterrows():
                exon_shapes.append({
                    'type': "rect",
                    'x0': exon['exon_start'], 'y0': trans_order - 0.3,
                    'x1': exon['exon_end'], 'y1': trans_order + 0.3,
                    'fillcolor': exon_color,
                    'line': {'color': exon_color, 'width': 1},
                    'opacity': 0.8
                })
                
                exon_hover_x.append((exon['exon_start'] + exon['exon_end']) / 2)
                exon_hover_y.append(trans_order)
                exon_hover_text.append(
                    f"Isoform ID: {isoform_id}<br>Transcript ID: {abbreviate_transcript_name(trans_id)}<br>Exon: {exon['exon_number']}<br>"
                    f"Size: {exon['exon_size']} bp<br>Coordinates: {exon['exon_start']:,} - {exon['exon_end']:,}"
                )
        
        junction_y_start = y_max + 0.5
        
    else:
        # Calculate dynamic height if not provided or if using default slider value (no transcript data case)
        if height is None or height == 600:
            height = calculate_dynamic_structure_plot_height(0, base_height=600)
        
        if junctions:
            all_coords = []
            for j in junctions:
                all_coords.extend([j['start'], j['end']])
            min_start = min(all_coords)
            max_end = max(all_coords)
        else:
            min_start, max_end = 0, 100000
        
        y_max = 1
        junction_y_start = 1.5
        exon_shapes = []
        exon_hover_x = []
        exon_hover_y = []
        exon_hover_text = []
    
    junction_labels = []
    junction_y_positions = []
    junction_shapes = []
    junction_hover_x = []
    junction_hover_y = []
    junction_hover_text = []
    
    if junctions:
        for i, junction in enumerate(junctions):
            start = junction['start']
            end = junction['end']
            junction_y_pos = junction_y_start + (i * 1.0)
    
            junction_id = f"chr{gene_data.get('chromosome', '').replace('chr', '')}_{start}_{end}_{strand}"
            junction_labels.append(junction_id)
            junction_y_positions.append(junction_y_pos)
            junction_shapes.append({
                'type': "rect",
                'x0': start, 'y0': junction_y_pos - 0.15,
                'x1': end, 'y1': junction_y_pos + 0.15,
                'fillcolor': junction_color,
                'line': {'color': junction_color, 'width': 1},
                'opacity': 0.8
            })
            
            junction_hover_x.append((start + end) / 2)
            junction_hover_y.append(junction_y_pos)
            junction_hover_text.append(
                f'Junction ID: {junction_id}<br>Coordinates: {start:,} - {end:,}<br>'
                f'Event: {junction.get("event_id", "")}<br>Type: {junction.get("event_type", "")}'
            )
    
    all_shapes = exon_shapes + junction_shapes
    if all_shapes:
        fig.update_layout(shapes=all_shapes)
    
    if exon_hover_x:
        fig.add_trace(go.Scatter(
            x=exon_hover_x,
            y=exon_hover_y,
            mode='markers',
            marker=dict(size=8, opacity=0),
            showlegend=False,
            hovertemplate='%{text}<extra></extra>',
            text=exon_hover_text,
            name='Exons'
        ))
    
    if junction_hover_x:
        fig.add_trace(go.Scatter(
            x=junction_hover_x,
            y=junction_hover_y,
            mode='markers',
            marker=dict(size=10, opacity=0),
            showlegend=False,
            hovertemplate='%{text}<extra></extra>',
            text=junction_hover_text,
            name='Junctions'
        ))
    
    total_y_range = junction_y_start + len(junctions) * 1.0 + 0.5 if junctions else y_max
    
    fig.update_layout(
        title={
            'text': f"Splice Junctions and Exons for Gene {gene_name} ({gene_id})<br>(Coordinates: {min_start} - {max_end}, Strand: {strand})",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14}
        },
        xaxis=dict(
            title="Genomic Position",
            range=[min_start, max_end],
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
            range=[0, total_y_range],
            autorange=False,
            fixedrange=False
        ),
        height=height,
        margin=dict(
            l=MIN_MARGIN,
            r=160,
            t=MAX_MARGIN,
            b=MAX_MARGIN+7
        ),
        hovermode='closest',
        plot_bgcolor='white',
        autosize=True,
        uirevision='constant'
    )
    
    for idx, transcript in enumerate(transcript_labels):
        y_pos = transcript_y_positions[idx]
        fig.add_annotation(
            x=1.01,
            xref='paper',
            y=y_pos,
            text=transcript,
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            font=dict(size=10)
        )

    for idx, junction in enumerate(junction_labels):
        y_pos = junction_y_positions[idx]
        fig.add_annotation(
            x=1.01,
            xref='paper',
            y=y_pos,
            text=junction,
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            font=dict(size=10, color='black')
        )

    return fig


def create_exon_regions_from_junctions(junctions: list, region_start: int, region_end: int) -> list:
    """Create exon regions based on junction coordinates"""
    if not junctions:
        return [(region_start + 500, region_end - 500)]
    
    splice_sites = set()
    for junction in junctions:
        splice_sites.add(junction['start'])
        splice_sites.add(junction['end'])
    
    sorted_sites = sorted(splice_sites)
    exon_regions = []
    
    # First exon (from region start to first splice site)
    if sorted_sites:
        first_exon_end = sorted_sites[0] - 50
        if first_exon_end > region_start:
            exon_regions.append((region_start + 200, first_exon_end))
    
    # Internal exons (between splice sites)
    for i in range(len(sorted_sites) - 1):
        exon_start = sorted_sites[i] + 50
        exon_end = sorted_sites[i + 1] - 50
        if exon_end > exon_start:
            exon_regions.append((exon_start, exon_end))
    
    # Last exon (from last splice site to region end)
    if sorted_sites:
        last_exon_start = sorted_sites[-1] + 50
        if last_exon_start < region_end:
            exon_regions.append((last_exon_start, region_end - 200))
    
    if not exon_regions:
        exon_regions = [(region_start + 200, region_end - 200)]
    
    return exon_regions


def create_empty_atse_message(message: str) -> go.Figure:
    """Create empty figure with message for ATSE plots"""
    fig = go.Figure()
    
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False,
        font=dict(size=14, color="gray")
    )
    
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='white',
        height=250,
        margin=dict(l=MIN_MARGIN, 
                    r=MIN_MARGIN, 
                    t=MIN_MARGIN, 
                    b=MIN_MARGIN)
    )
    
    return fig