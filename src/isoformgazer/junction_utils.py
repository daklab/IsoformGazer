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
from data_utils import apply_distance_preprocessing, get_matplotlib_colormap, get_table_prefix
from isoform_utils import (load_psl_data, get_gene_id_for_gene_name, abbreviate_transcript_name,
                           calculate_dynamic_structure_plot_height, calculate_clustergram_min_height,
                           get_tissue_tpm_for_isoforms, get_organ_tpm_for_isoforms, get_organ_colors)
from isoformgazer.performance_utils import cached, cached_transcript_structure_processing, plot_optimizer

###################################################################
# MARGIN PRESETS FOR FIGURES
###################################################################
MIN_MARGIN = 8
MAX_MARGIN = 55
MAX_MARGIN_LABELS = 65

###################################################################
# TEXT WRAPPING UTILITIES
###################################################################
def wrap_colorbar_title(name_text, label_text, char_limit=15):
    """
    Formatting method for colorscale titles to wrap colorbar title 
    at char_limit (default: 15), with Tissue TPM / Organ TPM label 
    on its own separate line at the end."""
    words = name_text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        # Check if adding this word would exceed the limit
        potential_length = current_length + len(word) + (1 if current_line else 0)

        # Case 1: word fits in the current line
        if potential_length <= char_limit:
            current_line.append(word)
            current_length = potential_length

        # Case 2: word doesn't fit, so start a new line
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)

    # Add the last line of name
    if current_line:
        lines.append(' '.join(current_line))

    # Always add label on its own line
    lines.append(label_text)

    return '<br>'.join(lines)

###################################################################
# ORGAN EXTRACTION AND ANNOTATION UTILITIES FOR JUNCTIONS
###################################################################
def get_junction_organ_colors():
    """
    Get organ color mapping for junction data.
    Uses the same colors as isoform clustergram where organs overlap,
    and adds new distinct colors for junction-specific organs.
    """
    # Get base organ colors from isoform_utils
    base_colors = get_organ_colors()

    # Junction-specific organ color mapping
    # Organs that match isoform organs use the same color
    junction_organ_colors = {
        # Human & Mouse shared organs (using isoform clustergram mapped colors where available)
        'bladder': base_colors.get('bladder', '#9370DB'),  # Medium purple
        'blood': base_colors.get('blood', '#FF6B6B'),      # coral
        'brain': base_colors.get('brain', '#4ECDC4'),      # blue
        'heart': base_colors.get('heart', '#FF7675'),      # red
        'kidney': base_colors.get('kidney', '#00CED1'),    # cyan
        'liver': base_colors.get('liver', '#FFD700'),      # gold
        'lung': base_colors.get('lung', '#87CEEB'),        # lightblue
        'marrow': base_colors.get('bone', '#F5F5DC'),      # beige (marrow related to bone)
        'skin': '#FFB6C1',                                  # light pink
        'spleen': '#DC143C',                                # crimson
        'thymus': '#8A2BE2',                                # blue violet
        'tongue': '#FF69B4',                                # hot pink
        'trachea': '#20B2AA',                               # light sea green
        # Human-specific organs
        'eye': '#00BFFF',                                   # deep sky blue
        'fat': base_colors.get('adipose', '#FFA500'),      # orange
        'large intestine': '#D2691E',                       # chocolate
        'limb muscle': base_colors.get('muscle', '#00008B'), # dark blue
        'lymph node': '#BA55D3',                            # medium orchid
        'mammary': base_colors.get('breast', '#EE82EE'),   # violet
        'muscle': base_colors.get('muscle', '#00008B'),    # dark blue
        'prostate': base_colors.get('prostate', '#FFFF00'), # yellow
        'salivary gland': '#DDA0DD',                        # plum
        'small intestine': '#90EE90',                       # light green
        'uterus': '#FF1493',                                # deep pink
        'vasculature': base_colors.get('vessels', '#D2691E'), # chocolate
        # Mouse-specific organs
        'aorta': base_colors.get('vessels', '#D2691E'),     # chocolate (vessels)
        'brown adipose tissue': '#FF8C00',                  # dark orange (brown adipose tissue)
        'diaphragm': '#4682B4',                             # steel blue
        'gonadal adipose tissue': '#FFA500',                # orange (gonadal adipose tissue)
        'limb': base_colors.get('muscle', '#00008B'),       # dark blue (limb muscle)
        'mesenteric adipose tissue': '#FF7F50',             # coral (mesenteric adipose tissue)
        'mammary gland': base_colors.get('breast', '#EE82EE'), # violet
        'pancreas': base_colors.get('pancreas', '#90EE90'), # light green
        'SC adipose tissue': '#FFD700',                     # gold (subcutaneous adipose tissue)
        # Unknown/unmapped
        'unknown': '#CCCCCC'                                # light gray
    }

    return junction_organ_colors


def extract_organ_from_cell_type(cell_type):
    """
    Extract organ name from cell_type column.
    Cell_Type format: "Organ_CellType" (e.g., "Skin_T_Cell", "Large_Intestine_B_Cell")
    Handles multi-word organ names like "Large Intestine", "Limb Muscle", etc.
    Returns the organ name in lowercase with spaces (e.g., "large intestine", "limb muscle")
    """
    if pd.isna(cell_type) or not cell_type or cell_type == '':
        return 'unknown'

    # Convert to string and split by underscore
    parts = str(cell_type).split('_')

    if len(parts) < 2:
        return 'unknown'

    # Multi-word organ names to check for (in order of length, longest first)
    # This ensures "Large_Intestine" is matched before just "Large"
    multi_word_organs = [
        'large_intestine',
        'small_intestine',
        'limb_muscle',
        'lymph_node',
        'salivary_gland',
        'mammary_gland'
    ]
    cell_type_lower = cell_type.lower()
    for multi_organ in multi_word_organs:
        if cell_type_lower.startswith(multi_organ + '_'):
            return multi_organ.replace('_', ' ')

    # Special case: handle typo "lympha" -> map to "lymph node"
    if cell_type_lower.startswith('lympha_node_'):
        return 'lymph node'
    #elif cell_type_lower.startswith('BAT'):
    #    return 'brown adipose tissue'
    #elif cell_type_lower.startswith('GAT'):
    #    return 'gonadal adipose tissue'
    #elif cell_type_lower.startswith('SCAT'):
    #    return ''

    # Single word organ: just take the first part
    organ = parts[0].strip().lower()
    # handle typo "lympha" (single word) -> "lymph node"
    if organ == 'lympha':
        return 'lymph node'
    # abbreviation "si" -> "small intestine"
    elif organ == 'si':
        return 'small intestine'
    elif organ == 'bat': 
        return 'brown adipose tissue'
    elif organ == 'gat':
        return 'gonadal adipose tissue'
    elif organ == 'mat':
        return 'mesenteric adipose tissue'
    elif organ == 'scat':
        return 'SC adipose tissue'

    return organ


def create_junction_organ_annotation_bar(cell_type_cols, species='Human'):
    """
    Create organ color annotation bar for junction clustergram.
    Similar to create_organ_annotation_bar in isoform_utils but for junction cell types.

    Args:
        cell_type_cols: List of cell type column names (format: "Organ_CellType")
        species: 'Human' or 'Mouse'

    Returns:
        Tuple of (organ_list, color_list) where each list corresponds to the input cell_type_cols
    """
    organ_colors = get_junction_organ_colors()

    organ_list = []
    color_list = []

    for cell_type in cell_type_cols:
        organ = extract_organ_from_cell_type(cell_type)
        color = organ_colors.get(organ, '#CCCCCC')  # Default to light gray for unknown organs
        organ_list.append(organ)
        color_list.append(color)

    return organ_list, color_list


@cached(cache_timeout=300)
def get_unique_organs_from_junctions(db_path: str, species: str = 'Human') -> tuple:
    """
    Get unique organs and their colors from the junctions table.
    Cached to avoid repeated database queries.

    Args:
        db_path: Path to the database
        species: 'Human' or 'Mouse'

    Returns:
        Tuple of (unique_organs, unique_colors) - lists of unique organ names and their colors
    """
    conn = sqlite3.connect(db_path)
    table_prefix = get_table_prefix(species)

    # Query to get unique cell types from junction_psis table
    query = f"""
    SELECT DISTINCT cell_type
    FROM {table_prefix}junction_psis
    WHERE cell_type IS NOT NULL
    ORDER BY cell_type
    """

    result = pd.read_sql_query(query, conn)
    conn.close()

    if result.empty:
        return [], []

    cell_types = result['cell_type'].tolist()
    organ_list, color_list = create_junction_organ_annotation_bar(cell_types, species=species)

    # Get unique organs and their colors (preserve order of first appearance)
    unique_organs = []
    unique_colors = []
    seen_organs = set()

    for organ, color in zip(organ_list, color_list):
        if organ not in seen_organs:
            unique_organs.append(organ)
            unique_colors.append(color)
            seen_organs.add(organ)

    # Sort organs alphabetically
    organ_color_pairs = list(zip(unique_organs, unique_colors))
    organ_color_pairs.sort(key=lambda x: x[0].lower())
    unique_organs = [pair[0] for pair in organ_color_pairs]
    unique_colors = [pair[1] for pair in organ_color_pairs]

    return unique_organs, unique_colors

###################################################################
# VISUALIZATION METHODS
###################################################################
def create_summary_clustergram(db_path, height=600, colorscale='Viridis', show_tables='show', show_celltype_labels=False, distance_metric='euclidean', linkage_method='complete', show_gridlines=False, species="Human"):
    """Create summary-level clustergram across all cell types and top junctions"""
    conn = sqlite3.connect(db_path)
    table_prefix = get_table_prefix(species)
    # Query junction_psis table for PSI values per junction-cell_type
    query = f"""
    SELECT cell_type, junction_id, AVG(psi) as avg_psi, COUNT(*) as n_observations
    FROM {table_prefix}junction_psis
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

    # Get organ colors for column annotation
    cell_type_labels = list(psi_matrix_filtered.columns)
    organ_list, color_list = create_junction_organ_annotation_bar(cell_type_labels, species=species)

    # Hide cell type labels based only on toggle state
    hide_celltype_labels = not show_celltype_labels
    if hide_celltype_labels:
        actual_clustergram_height = height + 400
    else:
        actual_clustergram_height = height + 200

    clustergram = dash_bio.Clustergram(
        data=psi_matrix_filtered.values,
        column_labels=cell_type_labels,
        row_labels=list(psi_matrix_filtered.index),
        column_colors=color_list,
        height=actual_clustergram_height,
        color_threshold={
            'row': 0.5,
            'col': 0.5
        },
        hidden_labels='col' if hide_celltype_labels else None,
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

    if len(clustergram.data) > 0:
        heatmap_trace = clustergram.data[-1]
        heatmap_trace.z = psi_matrix_filtered_with_nan.values

        if show_gridlines:
            heatmap_trace.xgap = 1
            heatmap_trace.ygap = 1

    if hide_celltype_labels:
        bottom_margin = max(0, 120 - 400)
    else:
        bottom_margin = max(0, 120 - 150)

    clustergram.update_layout(
        title={
            'text': "Summary: Junction Usage Across All Cell Types",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        margin=dict(l=20,
                    r=350,
                    t=90,
                    b=bottom_margin),
        autosize=True,
        width=None,
        height=height,
        uirevision='constant',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        yaxis=dict(
            automargin=True,
            showgrid=True,
            gridcolor='white',
            gridwidth=1
        ),
        xaxis=dict(
            automargin=False,
            tickangle=90 if not hide_celltype_labels else 0,
            tickfont=dict(
                size=8 if not hide_celltype_labels else 1,
                color='rgba(0,0,0,0)' if hide_celltype_labels else None
            ),
            showticklabels=not hide_celltype_labels,
            showgrid=True,
            gridcolor='white',
            gridwidth=1
        )
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

    unique_organs, unique_colors = get_unique_organs_from_junctions(db_path, species=species)

    colorbar_x_paper = 1.0
    colorbar_pixel_offset = 150  
    colorbar_y_position = 1.005
    pixels_between_items = 25
    vertical_offset_pixels = 7
    base_spacing = 30
    legend_pixel_offset = colorbar_pixel_offset + 20 + base_spacing

    clustergram.add_annotation(
        x=colorbar_x_paper,
        y=colorbar_y_position,
        xref='paper',
        yref='paper',
        xshift=legend_pixel_offset,
        yshift=-vertical_offset_pixels,
        text="Organ Legend",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=16, family="Open Sans, verdana, arial, sans-serif")
    )

    for i, (organ, color) in enumerate(zip(unique_organs, unique_colors)):
        organ_display = organ.lower()
        clustergram.add_annotation(
            x=colorbar_x_paper,
            y=colorbar_y_position,
            xref='paper',
            yref='paper',
            xshift=legend_pixel_offset,
            yshift=-(vertical_offset_pixels + 30 + (i * pixels_between_items)),
            text=f'<span style="color:{color}; font-size:16px">&#9632;</span> {organ_display}',
            showarrow=False,
            xanchor='left',
            yanchor='top',
            font=dict(size=10)
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


def create_gene_clustergram(db_path, gene_name, height=600, colorscale='Viridis', show_tables='show', filtered_junction_ids=None, show_celltype_labels=False, distance_metric='euclidean', linkage_method='complete', show_gridlines=False, species="Human"):
    """Create ATSE-level clustergram with junctions ordered by average PSI (descending)"""
    conn = sqlite3.connect(db_path)
    table_prefix = get_table_prefix(species)
    # Get gene_id for gene_name
    gene_id = get_gene_id_from_atse(db_path, gene_name, species)
    if not gene_id:
        conn.close()
        return create_empty_clustergram_message(f"No gene_id found for gene: {gene_name}")

    # Join junctions master table with junction_psis table to get PSI values per cell type
    query = f"""
    SELECT jp.cell_type, jp.junction_id, jp.psi, jp.n_cells,
           jp.atse_count, jp.junction_count,
           j.event_id, j.gene_name, j.junction_average_psi
    FROM {table_prefix}junction_psis jp
    JOIN {table_prefix}junctions j ON jp.junction_id = j.junction_id
    WHERE j.gene_id LIKE ? AND jp.psi IS NOT NULL
    """
    if filtered_junction_ids:
        filtered_ids_str = [str(jid) for jid in filtered_junction_ids]
        placeholders = ','.join(['?'] * len(filtered_ids_str))
        query += f" AND jp.junction_id IN ({placeholders})"
        params = [f"{gene_id}%"] + filtered_ids_str
    else:
        params = [f"{gene_id}%"]

    query += " ORDER BY j.junction_average_psi DESC NULLS LAST, jp.junction_id"
    gene_vals = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if len(gene_vals) == 0:
        return create_empty_clustergram_message(f"No junction data found for gene: {gene_name}")

    ordered_junction_ids = gene_vals['junction_id'].drop_duplicates().tolist()
    # note: have to again reverse the order same as for transcripts because Dash Bio clustergrams display the first row at the bottom...
    ordered_junction_ids = ordered_junction_ids[::-1]

    psi_matrix = gene_vals.pivot(index='junction_id', columns='cell_type', values='psi')

    orig_matrix = gene_vals.pivot(index='junction_id', columns='cell_type', values='psi')
    valid_columns = orig_matrix.columns[~orig_matrix.isna().all()]
    psi_matrix = psi_matrix[valid_columns]
    psi_matrix_with_nan = psi_matrix.copy()

    # n_cells matrix to allow us to show cell count in tooltip on hover in dashboard
    n_cells_matrix = gene_vals.pivot(index='junction_id', columns='cell_type', values='n_cells')
    n_cells_matrix = n_cells_matrix.reindex(index=psi_matrix.index, columns=psi_matrix.columns).fillna(0)

    # Reorder matrices to match the sorted order from the database query (again, reversed for Dash Bio display...)
    psi_matrix = psi_matrix.loc[ordered_junction_ids]
    psi_matrix_with_nan = psi_matrix_with_nan.loc[ordered_junction_ids]
    n_cells_matrix = n_cells_matrix.loc[ordered_junction_ids]

    n_cells_values = n_cells_matrix.values.astype(int)
    junction_labels = list(psi_matrix.index)
    cell_type_labels = list(psi_matrix.columns)

    if psi_matrix.empty:
        return create_empty_clustergram_message(f"No PSI data available for gene: {gene_name}")

    if height <= 875:
        num_junctions = len(junction_labels)
        min_required_height = calculate_clustergram_min_height(num_junctions, base_height=750)
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

    organ_list, color_list = create_junction_organ_annotation_bar(cell_type_labels, species=species)

    num_junctions = len(junction_labels)
    num_cell_types = len(cell_type_labels)
    hide_junction_labels = (num_junctions > 30) or (show_tables == 'show')

    # Hide cell type labels based only on toggle state
    hide_celltype_labels = not show_celltype_labels

    if (num_junctions > 30):
        left_margin = 2  
    else:
        max_label_len = max((len(str(name)) for name in junction_labels), default=10)
        left_margin = min(70, max(20, 7 * max_label_len + 10))

    if hide_celltype_labels:
        actual_clustergram_height = height + 400
        base_bottom_margin = max(200, int(height * 0.35))
        bottom_margin = max(0, base_bottom_margin - 400)
    else:
        actual_clustergram_height = height + 200
        base_bottom_margin = max(200, int(height * 0.35))
        bottom_margin = max(0, base_bottom_margin - 150)

    try:
        clustergram, computed_traces = dash_bio.Clustergram(
            data=psi_matrix_processed.values,
            row_labels=junction_labels,
            column_labels=cell_type_labels,
            column_colors=color_list,
            height=actual_clustergram_height,
            color_threshold={'row': 0.7, 'col': 0.7},
            hidden_labels='col' if hide_celltype_labels else None,
            cluster='col',
            color_list={
                'row': ['#636EFA', '#EF553B', '#00CC96', '#AB63FA'],
                'col': ['#FFA15A', '#19D3F3', '#FF6692', '#B6E880'],
                'bg': '#506784'
            },
            line_width=2,
            display_ratio=[0.12, 0.08] if not hide_celltype_labels else [0.08, 0.05],
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

    try:
        if len(clustergram.data) > 0:
            heatmap_trace = clustergram.data[-1]

            if show_gridlines:
                heatmap_trace.xgap = 1
                heatmap_trace.ygap = 1

            if hasattr(heatmap_trace, 'colorbar'):
                heatmap_trace.colorbar.x = 1.0
                heatmap_trace.colorbar.xpad = 150
                heatmap_trace.colorbar.y = 1.005
                heatmap_trace.colorbar.yanchor = 'top'
                heatmap_trace.colorbar.len = 0.3
                heatmap_trace.colorbar.thickness = 20
                heatmap_trace.colorbar.title = dict(text='PSI', font=dict(size=16))
                heatmap_trace.colorbar.tickfont = dict(size=10)

    except Exception as e:
        print(f"Warning: Could not update colorbar position: {e}")

    unique_organs, unique_colors = get_unique_organs_from_junctions(db_path, species=species)

    colorbar_x_paper = 1.0
    colorbar_pixel_offset = 150  
    colorbar_y_position = 1.005
    pixels_between_items = 25
    vertical_offset_pixels = 7
    base_spacing = 30
    legend_pixel_offset = colorbar_pixel_offset + 20 + base_spacing

    clustergram.add_annotation(
        x=colorbar_x_paper,
        y=colorbar_y_position,
        xref='paper',
        yref='paper',
        xshift=legend_pixel_offset,
        yshift=-vertical_offset_pixels,
        text="Organ Legend",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=16, family="Open Sans, verdana, arial, sans-serif")
    )

    for i, (organ, color) in enumerate(zip(unique_organs, unique_colors)):
        organ_display = organ.lower()
        clustergram.add_annotation(
            x=colorbar_x_paper,
            y=colorbar_y_position,
            xref='paper',
            yref='paper',
            xshift=legend_pixel_offset,
            yshift=-(vertical_offset_pixels + 30 + (i * pixels_between_items)),  # Title height (30px) + item spacing
            text=f'<span style="color:{color}; font-size:16px">&#9632;</span> {organ_display}',
            showarrow=False,
            xanchor='left',
            yanchor='top',
            font=dict(size=11)
        )

    clustergram.update_layout(
        title={
            'text': f"Splicing PSI Clustermap for {gene_name} ({len(junction_labels)} junctions)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18 if hide_celltype_labels else 20}
        },
        margin=dict(
            l=min(20, left_margin + 50),
            r=350,
            t=90,
            b=bottom_margin
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
            automargin=False,
            tickangle=90 if not hide_celltype_labels else 0,
            tickfont=dict(
                size=8 if not hide_celltype_labels else 1,
                color='rgba(0,0,0,0)' if hide_celltype_labels else None
            ),
            showticklabels=not hide_celltype_labels,
            showgrid=True,
            gridcolor='white',
            gridwidth=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white'
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


def get_gene_id_from_atse(db_path: str, gene_name: str, species="Human") -> str:
    """Get gene_id for a given gene_name from the database"""
    conn = sqlite3.connect(db_path)
    table_prefix = get_table_prefix(species)
    query = f"SELECT DISTINCT gene_id FROM {table_prefix}junctions WHERE gene_name = ? LIMIT 1"
    result = pd.read_sql_query(query, conn, params=[gene_name])
    conn.close()

    if len(result) > 0:
        # Remove version number
        return result.iloc[0]['gene_id'].split('.')[0]
    return None


def filter_junctions_by_transcripts(db_path: str, gene_name: str, filtered_transcript_ids: list, species="Human") -> list:
    """
    Filter junction IDs based on transcript overlap using matched_transcript_ids column.
    Returns a list of junction IDs that have any of the filtered transcripts in their matched_transcript_ids.
    """
    if not filtered_transcript_ids:
        return []

    conn = sqlite3.connect(db_path)
    table_prefix = get_table_prefix(species)
    transcript_id_placeholders = ','.join(['?'] * len(filtered_transcript_ids))
    transcript_query = f"""
    SELECT DISTINCT transcript
    FROM {table_prefix}isoforms
    WHERE id IN ({transcript_id_placeholders})
    """
    transcript_result = pd.read_sql_query(transcript_query, conn, params=filtered_transcript_ids)

    if transcript_result.empty:
        conn.close()
        return []

    # Strip version numbers from transcript names for matching (for PSL mapping, matched_transcript_ids doesn't have versions)
    transcript_names = transcript_result['transcript'].tolist()
    transcript_names_no_version = set()
    for name in transcript_names:
        # Remove version number (e.g., ENST00000416721.6 -> ENST00000416721)
        name_no_version = name.split('.')[0] if '.' in name else name
        transcript_names_no_version.add(name_no_version)

    # Query junctions table for junctions that have these transcripts in matched_transcript_ids
    junction_query = f"""
    SELECT DISTINCT junction_id, matched_transcript_ids
    FROM {table_prefix}junctions
    WHERE gene_name = ?
    AND matched_transcript_ids IS NOT NULL
    """
    junction_result = pd.read_sql_query(junction_query, conn, params=[gene_name])
    conn.close()

    if junction_result.empty:
        return []

    matching_junction_ids = []

    for _, row in junction_result.iterrows():
        junction_id = row['junction_id']
        matched_transcripts_str = str(row['matched_transcript_ids'])

        if pd.notna(matched_transcripts_str) and matched_transcripts_str not in ('nan', 'None', ''):
            matched_transcripts = set(t.strip() for t in matched_transcripts_str.split(',') if t.strip())

            # Check if any matched transcript is in our filtered transcript set (both without versions)
            if matched_transcripts & transcript_names_no_version:
                matching_junction_ids.append(junction_id)

    return matching_junction_ids


def filter_transcripts_by_junctions(db_path: str, filtered_junction_ids: list, species="Human") -> list:
    """
    Filter transcript IDs based on junction overlap using matched_transcript_ids column.
    Returns a list of transcript IDs (from isoforms table) that align with the filtered junctions.
    """
    if not filtered_junction_ids:
        return []

    conn = sqlite3.connect(db_path)
    table_prefix = get_table_prefix(species)
    junction_id_placeholders = ','.join(['?'] * len(filtered_junction_ids))
    junction_query = f"""
    SELECT DISTINCT matched_transcript_ids
    FROM {table_prefix}junctions
    WHERE junction_id IN ({junction_id_placeholders})
    AND matched_transcript_ids IS NOT NULL
    """
    junction_result = pd.read_sql_query(junction_query, conn, params=filtered_junction_ids)

    if junction_result.empty:
        conn.close()
        return []

    # Get all transcript names (without version numbers from matched_transcript_ids)
    all_transcript_names_no_version = set()
    for _, row in junction_result.iterrows():
        transcript_ids_str = str(row['matched_transcript_ids'])

        if pd.notna(transcript_ids_str) and transcript_ids_str not in ('nan', 'None', ''):
            transcript_names = transcript_ids_str.split(',')

            for name in transcript_names:
                name = name.strip()

                if name:
                    all_transcript_names_no_version.add(name)

    if not all_transcript_names_no_version:
        conn.close()
        return []

    # Get all isoforms for this gene, then filter by matching transcript names (without version)
    # First, get all isoforms from the junctions we queried
    gene_query = f"""
    SELECT DISTINCT gene_name
    FROM {table_prefix}junctions
    WHERE junction_id IN ({junction_id_placeholders})
    LIMIT 1
    """
    gene_result = pd.read_sql_query(gene_query, conn, params=filtered_junction_ids)

    if gene_result.empty:
        conn.close()
        return []

    gene_name = gene_result.iloc[0]['gene_name']

    # Get all isoforms for this gene
    isoforms_query = f"""
    SELECT DISTINCT id, transcript
    FROM {table_prefix}isoforms
    WHERE gene_name = ?
    """
    isoforms_result = pd.read_sql_query(isoforms_query, conn, params=[gene_name])
    conn.close()

    if isoforms_result.empty:
        return []

    # Filter isoforms by matching transcript names (strip version numbers for comparison)
    matching_ids = []
    for _, row in isoforms_result.iterrows():
        transcript_with_version = row['transcript']
        # Strip version number (e.g., ENST00000416721.6 -> ENST00000416721)
        transcript_no_version = transcript_with_version.split('.')[0] if '.' in transcript_with_version else transcript_with_version

        if transcript_no_version in all_transcript_names_no_version:
            matching_ids.append(row['id'])

    return matching_ids


@cached(cache_timeout=300)
def process_gene_atse_data(gene_name: str, db_path: str, filtered_junction_ids=None, species="Human") -> dict:
    """Process ATSE data for a specific gene with caching and performance optimization"""
    #with ProfilerContext(f"process_atse_data_{gene_name}"):
    gene_id_with_version = None
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size = 10000")
    conn.execute("PRAGMA temp_store = MEMORY")

    table_prefix = get_table_prefix(species)
    # Get gene_id from db
    gene_id_query = f"SELECT DISTINCT gene_id FROM {table_prefix}junctions WHERE gene_name = ? LIMIT 1"
    gene_result = pd.read_sql_query(gene_id_query, conn, params=[gene_name])

    if gene_result.empty:
        conn.close()
        return {'error': f"No gene_id found for {gene_name}"}

    gene_id_with_version = gene_result.iloc[0]['gene_id']
    gene_id_base = gene_id_with_version.split('.')[0]
    #memory_tracker.measure(f"after_gene_lookup_{gene_name}")

    atse_query = f"""
    SELECT event_id, gene_id, gene_name, event_strand, chromosome,
            start, end, junction_id, transcripts, perfect_match_3_prime,
            perfect_match_5_prime, both_ends_transcripts,
            only_5_prime_transcripts, only_3_prime_transcripts,
            atse_start, atse_end, event_type
    FROM {table_prefix}atse_data
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

    conn = sqlite3.connect(db_path)
    junction_psi_query = f"""
    SELECT DISTINCT junction_id, junction_average_psi
    FROM {table_prefix}junctions
    WHERE gene_id LIKE ?
    """
    junction_psi_df = pd.read_sql_query(junction_psi_query, conn, params=[f"{gene_id_base}%"])
    conn.close()

    # Create a mapping of junction coordinates to average PSI
    junction_psi_map = {}
    for _, row in junction_psi_df.iterrows():
        junction_id = row['junction_id']
        avg_psi = row['junction_average_psi']

        parts = junction_id.split('_')
        if len(parts) >= 3:
            try:
                start = int(parts[1])
                end = int(parts[2])
                key = (start, end)

                if key not in junction_psi_map:
                    junction_psi_map[key] = avg_psi

                else:
                    junction_psi_map[key] = max(junction_psi_map[key], avg_psi)

            except (ValueError, IndexError):
                pass

    unique_junctions.sort(
        key=lambda x: (
            -junction_psi_map.get((x['start'], x['end']), 0),  # negative for descending
            x['start']
        )
    )

    # Reverse the order because junctions are displayed top to bottom
    unique_junctions = unique_junctions[::-1]

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


def create_junction_exon_visualization(gene_data: dict,
                                       height: int = 600,
                                       show_y_labels: bool = False,
                                       exon_color: str = '#2E86C1',
                                       junction_color: str = '#85929E',
                                       filtered_transcript_ids: list = None,
                                       color_by_abundance: bool = False,
                                       color_junctions_by_psi: bool = True,
                                       db_path: str = None,
                                       colorscale: str = 'Viridis',
                                       abundance_type: str = 'average',
                                       tissue_name: str = None,
                                       organ_name: str = None,
                                       individual_junction_colors: dict = None,
                                       individual_transcript_colors: dict = None,
                                       species: str = "Human") -> go.Figure:
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

    # Get table prefix early since needed for multiple queries
    table_prefix = get_table_prefix(species)

    transcript_data = cached_transcript_structure_processing(db_path, gene_name, filtered_transcript_ids or [], species)
    if transcript_data.empty and not junctions:
        return create_empty_atse_message(f"No transcript or junction data found for {gene_name}")

    fig = go.Figure()
    intron_color = '#85929E'
    transcript_labels = []
    transcript_y_positions = []

    if not transcript_data.empty:
        transcript_data_opt = plot_optimizer.preprocess_dataframe_for_plotting(transcript_data)
        conn = sqlite3.connect(db_path)
        # Get gene_id from gene_data
        gene_id_with_version = gene_data.get('gene_id', '')
        gene_id_base = gene_id_with_version.split('.')[0] if gene_id_with_version else ''

        if gene_id_base:
            transcript_ids_query = f"""
            SELECT DISTINCT id FROM {table_prefix}isoforms
            WHERE gene_id LIKE ?
            ORDER BY isoform_average_tpm DESC NULLS LAST
            """
            ordered_ids_df = pd.read_sql_query(transcript_ids_query, conn, params=[f"{gene_id_base}%"])
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

        transcript_summary['transcript_length'] = transcript_summary['transcript_end'] - transcript_summary['transcript_start']
        # Reindex transcript_summary to match the original sorted order from database and sort by isoform_average_tpm descending
        transcript_summary = transcript_summary.sort_values('id', key=lambda x: x.map({vid: idx for idx, vid in enumerate(ordered_transcript_ids)})).reset_index(drop=True)
        transcript_summary['trans_order'] = range(1, len(transcript_summary) + 1)

        if color_by_abundance and gene_id_base:
            conn = sqlite3.connect(db_path)
            try:
                tpm_query = f"""
                SELECT DISTINCT id, isoform_average_tpm
                FROM {table_prefix}isoforms
                WHERE gene_id LIKE ?
                """
                tpm_df = pd.read_sql_query(tpm_query, conn, params=[f"{gene_id_base}%"])
                transcript_summary = transcript_summary.merge(tpm_df, on='id', how='left')
            finally:
                conn.close()
        
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
        
        # Group introns by color (to minimize number of traces)
        intron_groups = {}  # color -> list of transcript data

        for _, transcript in transcript_summary.iterrows():
            isoform_id = transcript['id']
            color = intron_color

            if color not in intron_groups:
                intron_groups[color] = []

            intron_groups[color].append({
                'x': [transcript['transcript_start'], transcript['transcript_end'], None],
                'y': [transcript['trans_order'], transcript['trans_order'], None],
                'text': [f"Isoform ID: {transcript['id']}<br>Transcript: {abbreviate_transcript_name(transcript['trans_id'])}<br>Length: {transcript['transcript_length']:,} bp", "", ""]
            })

        # One trace per color group
        for color, transcripts in intron_groups.items():
            intron_x = []
            intron_y = []
            intron_text = []

            for t in transcripts:
                intron_x.extend(t['x'])
                intron_y.extend(t['y'])
                intron_text.extend(t['text'])

            fig.add_trace(go.Scatter(
                x=intron_x,
                y=intron_y,
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False,
                hovertemplate='%{text}<extra></extra>',
                text=intron_text,
                connectgaps=False
            ))
        
        exon_shapes = []
        exon_hover_x = []
        exon_hover_y = []
        exon_hover_text = []

        # Get color scale for transcripts if enabled
        transcript_tpm_dict = None
        tpm_min, tpm_max = 0, 0
        cmap = None

        if color_by_abundance:
            if abundance_type == 'tissue' and tissue_name:
                transcript_tpm_dict = get_tissue_tpm_for_isoforms(db_path, gene_data['gene_name'], tissue_name, species)

            elif abundance_type == 'organ' and organ_name:
                transcript_tpm_dict = get_organ_tpm_for_isoforms(db_path, gene_data['gene_name'], organ_name, species)

            else:
                if 'isoform_average_tpm' in transcript_summary.columns:
                    transcript_tpm_dict = {row['id']: row.get('isoform_average_tpm', 0)
                                          for _, row in transcript_summary.iterrows()}

            # Get min/max for normalization
            if transcript_tpm_dict:
                tpm_values = [v for v in transcript_tpm_dict.values() if v is not None]
                if tpm_values:
                    tpm_min, tpm_max = min(tpm_values), max(tpm_values)
                    cmap = get_matplotlib_colormap(colorscale)

        for _, transcript in transcript_summary.iterrows():
            isoform_id = transcript['id']
            trans_id = transcript['trans_id']
            trans_order = transcript['trans_order']

            trans_exons = plot_data[plot_data['id'] == isoform_id].sort_values('exon_start')

            # Check for TPM color first (highest priority when enabled)
            if color_by_abundance and transcript_tpm_dict and isoform_id in transcript_tpm_dict:
                tpm = transcript_tpm_dict[isoform_id]

                if tpm_max > tpm_min:
                    normalized = (tpm - tpm_min) / (tpm_max - tpm_min)

                # When all values are equal, use 0.0 to show the minimum color on the scale
                else:
                    normalized = 0.0

                rgba = cmap(normalized)
                exon_fill_color = f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{int(rgba[3]*255)})'

            # Check for individual transcript color (second priority, convert to string for comparison)
            elif individual_transcript_colors and str(isoform_id) in individual_transcript_colors:
                exon_fill_color = individual_transcript_colors[str(isoform_id)]

            else:
                exon_fill_color = exon_color

            for _, exon in trans_exons.iterrows():
                exon_shapes.append({
                    'type': "rect",
                    'x0': exon['exon_start'], 'y0': trans_order - 0.3,
                    'x1': exon['exon_end'], 'y1': trans_order + 0.3,
                    'fillcolor': exon_fill_color,
                    'line': {'color': exon_fill_color, 'width': 1},
                    'opacity': 0.8
                })

                # Build hover text with optional TPM info
                hover_text = f"Isoform ID: {isoform_id}<br>Transcript ID: {abbreviate_transcript_name(trans_id)}<br>Exon: {exon['exon_number']}<br>Size: {exon['exon_size']} bp<br>Coordinates: {exon['exon_start']:,} - {exon['exon_end']:,}"
                if color_by_abundance and transcript_tpm_dict and isoform_id in transcript_tpm_dict:
                    tpm = transcript_tpm_dict[isoform_id]
                    if tpm is not None:
                        if abundance_type == 'tissue' and tissue_name:
                            formatted_tissue = ' '.join(word.capitalize() for word in tissue_name.split())
                            hover_text += f"<br>{formatted_tissue} TPM: {tpm:.2f}"

                        elif abundance_type == 'organ' and organ_name:
                            formatted_organ = ' '.join(word.capitalize() for word in organ_name.split())
                            hover_text += f"<br>{formatted_organ} TPM: {tpm:.2f}"

                        else:
                            hover_text += f"<br>Average TPM: {tpm:.2f}"

                # Create multiple invisible points across the entire exon block so user can click anywhere on it
                exon_length = exon['exon_end'] - exon['exon_start']
                num_points = max(10, int(exon_length / 1000))  # At least 10 points, more for longer exons
                for i in range(num_points + 1):
                    x_point = exon['exon_start'] + (exon_length * i / num_points)
                    exon_hover_x.append(x_point)
                    exon_hover_y.append(trans_order)
                    exon_hover_text.append(hover_text)
                # Add separator
                exon_hover_x.append(None)
                exon_hover_y.append(None)
                exon_hover_text.append("")
        
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

    junction_psi_data = {}
    if junctions and color_junctions_by_psi:
        conn = sqlite3.connect(db_path)
        try:
            # Get gene_id from gene_data
            gene_id_with_version_for_psi = gene_data.get('gene_id', '')
            gene_id_base_for_psi = gene_id_with_version_for_psi.split('.')[0] if gene_id_with_version_for_psi else ''

            if gene_id_base_for_psi:
                psi_query = f"""
                SELECT DISTINCT junction_id, junction_average_psi
                FROM {table_prefix}junctions
                WHERE gene_id LIKE ?
                """
                psi_df = pd.read_sql_query(psi_query, conn, params=[f"{gene_id_base_for_psi}%"])
            else:
                psi_df = pd.DataFrame()
            # Create a mapping of (start, end) -> junction_average_psi
            for _, row in psi_df.iterrows():
                junction_id = row['junction_id']
                psi = row['junction_average_psi']
                # Parse junction_id format: chr19_123_456_strand
                parts = junction_id.rsplit('_', 1)
                if len(parts) == 2:
                    coords_part = parts[0]
                    coord_parts = coords_part.split('_')
                    if len(coord_parts) >= 3:
                        try:
                            start = int(coord_parts[-2])
                            end = int(coord_parts[-1])
                            junction_psi_data[(start, end)] = psi
                        except (ValueError, IndexError):
                            pass
        finally:
            conn.close()

        # Calculate PSI min/max for normalization
        psi_values = [v for v in junction_psi_data.values() if v is not None]
        if psi_values:
            psi_min = min(psi_values)
            psi_max = max(psi_values)
        else:
            psi_min = psi_max = 0

        cmap = get_matplotlib_colormap(colorscale)

    # Query conserved junctions data for cross-species mapping
    conserved_junctions_data = {}
    if junctions:
        conn = sqlite3.connect(db_path)
        try:
            conserved_query = """
            SELECT mouse_junction_id, human_junction_id, sequence_similarity
            FROM human_mouse_conserved_junctions
            """
            conserved_df = pd.read_sql_query(conserved_query, conn)

            # Create lookup based on species...
            # For human: map human_junction_id -> (mouse_junction_id, sequence_similarity)
            # For mouse: map mouse_junction_id -> (human_junction_id, sequence_similarity)
            if species == "Human":
                for _, row in conserved_df.iterrows():
                    conserved_junctions_data[row['human_junction_id']] = {
                        'conserved_junction_id': row['mouse_junction_id'],
                        'sequence_similarity': row['sequence_similarity']
                    }
            elif species == "Mouse":
                for _, row in conserved_df.iterrows():
                    conserved_junctions_data[row['mouse_junction_id']] = {
                        'conserved_junction_id': row['human_junction_id'],
                        'sequence_similarity': row['sequence_similarity']
                    }
        finally:
            conn.close()

    if junctions:
        for i, junction in enumerate(junctions):
            start = junction['start']
            end = junction['end']
            junction_y_pos = junction_y_start + (i * 1.0)

            junction_id = f"chr{gene_data.get('chromosome', '').replace('chr', '')}_{start}_{end}_{strand}"
            junction_labels.append(junction_id)
            junction_y_positions.append(junction_y_pos)

            # Color priority: Average PSI color > user selected individual junction color > global color
            if color_junctions_by_psi and (start, end) in junction_psi_data:
                psi = junction_psi_data[(start, end)]
                if psi is not None:
                    if psi_max > psi_min:
                        normalized = (psi - psi_min) / (psi_max - psi_min)
                    else:
                        normalized = 0.5
                    rgba = cmap(normalized)
                    junction_fill_color = f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{int(rgba[3]*255)})'
                else:
                    # PSI coloring enabled but no PSI data: so use individual color if available, otherwise global color
                    if individual_junction_colors and junction_id in individual_junction_colors:
                        junction_fill_color = individual_junction_colors[junction_id]
                    else:
                        junction_fill_color = junction_color

            # Check if junction has an individual color override
            elif individual_junction_colors and junction_id in individual_junction_colors:
                junction_fill_color = individual_junction_colors[junction_id]
            else:
                junction_fill_color = junction_color

            junction_shapes.append({
                'type': "rect",
                'x0': start, 'y0': junction_y_pos - 0.15,
                'x1': end, 'y1': junction_y_pos + 0.15,
                'fillcolor': junction_fill_color,
                'line': {'color': junction_fill_color, 'width': 1},
                'opacity': 0.8
            })

            # Build hover text with optional PSI info
            hover_text = f'Junction ID: {junction_id}<br>Coordinates: {start:,} - {end:,}<br>Event: {junction.get("event_id", "")}<br>Type: {junction.get("event_type", "")}'
            if (start, end) in junction_psi_data:
                psi = junction_psi_data[(start, end)]
                if psi is not None:
                    hover_text += f"<br>Average PSI: {psi:.2f}"

            # Add conserved junction information if available
            if junction_id in conserved_junctions_data:
                conserved_data = conserved_junctions_data[junction_id]
                if species == "Human":
                    hover_text += f"<br>Conserved Mouse Junction ID: {conserved_data['conserved_junction_id']}"
                    hover_text += f"<br>Conserved Mouse Junction Sequence Similarity: {conserved_data['sequence_similarity']:.2%}"
                elif species == "Mouse":
                    hover_text += f"<br>Conserved Human Junction ID: {conserved_data['conserved_junction_id']}"
                    hover_text += f"<br>Conserved Human Junction Sequence Similarity: {conserved_data['sequence_similarity']:.2%}"

            # Create invisible line segment with multiple points so user can click anywhere on junction to color it
            num_points = 20
            for i in range(num_points + 1):
                x_point = start + (end - start) * i / num_points
                junction_hover_x.append(x_point)
                junction_hover_y.append(junction_y_pos)
                junction_hover_text.append(hover_text)
            junction_hover_x.append(None)
            junction_hover_y.append(None)
            junction_hover_text.append("")
    
    all_shapes = exon_shapes + junction_shapes
    if all_shapes:
        fig.update_layout(shapes=all_shapes)
    
    if exon_hover_x:
        fig.add_trace(go.Scatter(
            x=exon_hover_x,
            y=exon_hover_y,
            mode='lines',
            line=dict(width=25, color='rgba(100,100,100,0)'),
            showlegend=False,
            hovertemplate='%{text}<extra></extra>',
            text=exon_hover_text,
            name='Exons',
            connectgaps=False 
        ))
    
    if junction_hover_x:
        fig.add_trace(go.Scatter(
            x=junction_hover_x,
            y=junction_hover_y,
            mode='lines',
            line=dict(width=20, color='rgba(100,100,255,0)'),
            showlegend=False,
            hovertemplate='%{text}<extra></extra>',
            text=junction_hover_text,
            name='Junctions',
            connectgaps=False
        ))

    # Setup colorscale (needed for both junction and transcript colorbars)
    if colorscale:
        selected_colorscale = colorscale
    else:
        selected_colorscale = 'Viridis'

    # Create PSI colorbar for junctions (independent of transcript coloring)
    if junctions and color_junctions_by_psi and 'psi_min' in locals() and 'psi_max' in locals():
        # Create invisible scatter trace with colorbar for PSI
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                colorscale=selected_colorscale,
                cmin=psi_min,
                cmax=psi_max,
                colorbar=dict(
                    title=dict(text="Junction<br>Average PSI", font=dict(size=16)),
                    thickness=20,
                    len=0.4,
                    x=1.22,
                    xpad=0,
                    y=0.78,
                    tickfont=dict(size=10)
                ),
                showscale=True,
                size=0,
                opacity=0
            ),
            showlegend=False,
            hoverinfo='none',
            name='PSI Scale'
        ))

    if color_by_abundance:
        if not transcript_summary.empty:
            if tpm_min is not None and tpm_max is not None:
                if abundance_type == 'tissue' and tissue_name:
                    formatted_tissue = ' '.join(word.capitalize() for word in tissue_name.split())
                    colorbar_title = wrap_colorbar_title(formatted_tissue, "Tissue TPM")
                elif abundance_type == 'organ' and organ_name:
                    formatted_organ = ' '.join(word.capitalize() for word in organ_name.split())
                    colorbar_title = wrap_colorbar_title(formatted_organ, "Organ TPM")
                else:
                    colorbar_title = "Transcript<br>Average TPM"

                if tpm_max > tpm_min:
                    cmin_val = tpm_min
                    cmax_val = tpm_max

                else:
                    # When all values are 0, just use a small range for display so colorbar is still visible
                    cmin_val = 0
                    cmax_val = 1

                # If junctions are not colored by PSI, move transcript colorbar up
                if not color_junctions_by_psi: 
                    transcript_colorbar_y = 0.78 
                else: 
                    transcript_colorbar_y = 0.33

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
                            len=0.4,
                            x=1.22,
                            xpad=0,
                            y=transcript_colorbar_y,
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

    total_y_range = junction_y_start + len(junctions) * 1.0 + 0.5 if junctions else y_max

    # keep labels and colorbars at fixed distance away from y-axis labels (accounting for length of isoform hashes)
    if color_by_abundance or color_junctions_by_psi:
        right_margin = 250
        label_xshift_pixels = 10
    else:
        right_margin = 200
        label_xshift_pixels = 10

    fig.update_layout(
        title={
            'text': f"Splice Junctions and Exons for Gene {gene_name} ({gene_id})<br>(Coordinates: {min_start} - {max_end}, Strand: {strand})",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
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
            r=right_margin,
            t=MAX_MARGIN,
            b=MAX_MARGIN+7
        ),
        hovermode='closest',
        dragmode='zoom',
        plot_bgcolor='white',
        autosize=True,
        uirevision='constant'
    )

    for idx, transcript in enumerate(transcript_labels):
        y_pos = transcript_y_positions[idx]
        fig.add_annotation(
            x=1.0,
            xref='paper',
            xshift=label_xshift_pixels,
            y=y_pos,
            text=transcript,
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            font=dict(size=12)
        )

    for idx, junction in enumerate(junction_labels):
        y_pos = junction_y_positions[idx]
        fig.add_annotation(
            x=1.0,
            xref='paper',
            xshift=label_xshift_pixels,
            y=y_pos,
            text=junction,
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            font=dict(size=12, color='black')
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