import os
import re
import sqlite3
import pandas as pd
import numpy as np
import dash_bio
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple

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
    

def get_gene_id_for_gene_name(db_path: str, gene_name: str) -> str:
    """Get gene_id for a given gene_name from the isoforms database"""
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT gene_id FROM isoforms WHERE gene_name = ? LIMIT 1"
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
    

def process_transcript_structure(db_path: str, gene_name: str, filtered_ids: list) -> pd.DataFrame:
    """Optimized transcript structure processing"""
    conn = sqlite3.connect(db_path)
    gene_query = "SELECT DISTINCT gene_id FROM isoforms WHERE gene_name = ? LIMIT 1"
    gene_result = pd.read_sql_query(gene_query, conn, params=[gene_name])
    
    if gene_result.empty:
        conn.close()
        return pd.DataFrame()
    
    gene_id = gene_result.iloc[0]['gene_id'].split('.')[0]
    
    if filtered_ids:
        placeholders = ','.join(['?'] * len(filtered_ids))
        isoform_query = f"""
        SELECT id FROM isoforms 
        WHERE gene_id LIKE ? AND id IN ({placeholders})
        """
        params = [f"{gene_id}%"] + filtered_ids
    else:
        isoform_query = "SELECT id FROM isoforms WHERE gene_id LIKE ?"
        params = [f"{gene_id}%"]
    
    isoform_ids = pd.read_sql_query(isoform_query, conn, params=params)['id'].tolist()
    
    if not isoform_ids:
        conn.close()
        return pd.DataFrame()
    
    placeholders = ','.join(['?'] * len(isoform_ids))
    psl_query = f"""
    SELECT 
        id, trans_id, gene_id, tName, strand, 
        tStart, tEnd, blockSizes, tStarts
    FROM psl_data 
    WHERE id IN ({placeholders})
    ORDER BY tStart
    """
    gene_psl = pd.read_sql_query(psl_query, conn, params=isoform_ids)
    conn.close()
    
    transcript_data = []
    for _, row in gene_psl.iterrows():
        try:
            block_sizes = [int(x) for x in row['blockSizes'].split(',') if x]
            block_starts = [int(x) for x in row['tStarts'].split(',') if x]
            
            for i, (size, start) in enumerate(zip(block_sizes, block_starts)):
                transcript_data.append({
                    'id': row['id'],
                    'trans_id': row['trans_id'],
                    'gene_id': row['gene_id'],
                    'chr': row['tName'],
                    'strand': row['strand'],
                    'transcript_start': row['tStart'],
                    'transcript_end': row['tEnd'],
                    'exon_number': i + 1,
                    'exon_start': start,
                    'exon_end': start + size,
                    'exon_size': size
                })
        except Exception:
            continue
    
    return pd.DataFrame(transcript_data)


def load_expression_data(db_path: str, gene_name: str, data_type: str = 'tpm') -> pd.DataFrame:
    """Load expression data from SQLite database"""
    conn = sqlite3.connect(db_path)
    gene_id = get_gene_id_for_gene_name(db_path, gene_name)
    if not gene_id:
        return pd.DataFrame()
    
    if data_type.lower() == 'tpm': 
        table_name = 'tpm_data'
    else: 
        table_name = 'ratio_data'
    
    query = f"""
    SELECT 
        exp.*,
        psl.trans_id,
        iso.gene_name
    FROM {table_name} exp
    JOIN psl_data psl ON exp.id = psl.id
    JOIN isoforms iso ON exp.id = iso.id
    WHERE iso.gene_name = ?
    """
    
    try:
        df = pd.read_sql_query(query, conn, params=[gene_name])
        df = df.drop(['index'], axis=1)

        numeric_cols = [col for col in df.columns if col not in ['transcript', 'trans_id', 'gene', 'gene_name']]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df
    
    except Exception as e:
        print(f"Error loading {data_type} data: {e}")
        return pd.DataFrame()
    
    finally:
        conn.close()
    

def create_transcript_structure_plot(db_path: str, 
                                     transcript_data: pd.DataFrame, 
                                     gene_name: str, 
                                     height: int = 400,
                                     show_y_labels: bool = False) -> go.Figure:
    """Create transcript structure plot similar to simplified Isoviz version"""
    orf_perplexity = "Unknown"
    conn = sqlite3.connect(db_path)

    gene_ensembl_column = 'gene_id'
    query = f"SELECT {gene_ensembl_column} FROM isoforms WHERE gene_name = ? LIMIT 1"
    gene_ensembl_result = pd.read_sql_query(query, conn, params=[gene_name])
    gene_ensembl_id = gene_ensembl_result.iloc[0][gene_ensembl_column]
    
    orf_column = 'ORF_perplexity'
    query = f"SELECT {orf_column} FROM isoforms WHERE gene_name = ? LIMIT 1"
    result = pd.read_sql_query(query, conn, params=[gene_name])
    
    if not result.empty:
        orf_value = result.iloc[0][orf_column]
        if pd.isna(orf_value) or orf_value is None:
            orf_perplexity = "None"
        else:
            orf_perplexity = f"{orf_value:.3f}"
    else:
        orf_perplexity = "No data available"
    conn.close()

    strand = "" 
    if not transcript_data.empty:
        unique_strands = transcript_data['strand'].unique()
        if len(unique_strands) > 0:
            strand = unique_strands[0]

    if transcript_data.empty:
        return create_empty_isoform_message(f"No transcript data for gene: {gene_name}")
    
    transcript_summary = transcript_data.groupby('trans_id').agg({
        'id': 'first',
        'transcript_start': 'min',
        'transcript_end': 'max'
    }).reset_index()

    transcript_summary['transcript_length'] = transcript_summary['transcript_end'] - transcript_summary['transcript_start']
    transcript_summary = transcript_summary.sort_values('transcript_length', ascending=False).reset_index(drop=True)
    
    transcript_summary['trans_order'] = range(1, len(transcript_summary) + 1)
    
    plot_data = transcript_data.merge(transcript_summary[['trans_id', 'trans_order']], on='trans_id')
    
    fig = go.Figure()
    
    exon_color = '#2E86C1'
    intron_color = '#85929E'
    min_start = plot_data['transcript_start'].min()
    max_end = plot_data['transcript_end'].max()
    y_max = len(transcript_summary) + 1
    
    for _, transcript in transcript_summary.iterrows():
        trans_id = transcript['trans_id']
        trans_order = transcript['trans_order']
        
        trans_exons = plot_data[plot_data['trans_id'] == trans_id].sort_values('exon_start')
        
        fig.add_trace(go.Scatter(
            x=[transcript['transcript_start'], transcript['transcript_end']],
            y=[trans_order, trans_order],
            mode='lines',
            line=dict(color=intron_color, width=2),
            showlegend=False,
            hovertemplate=f"Transcript: {trans_id}<br>Length: {transcript['transcript_length']:,} bp<extra></extra>"
        ))
        
        for _, exon in trans_exons.iterrows():
            fig.add_shape(
                type="rect",
                x0=exon['exon_start'], y0=trans_order - 0.3,
                x1=exon['exon_end'], y1=trans_order + 0.3,
                fillcolor=exon_color,
                line=dict(color=exon_color, width=1),
                opacity=0.8
            )
            
            fig.add_trace(go.Scatter(
                x=[(exon['exon_start'] + exon['exon_end']) / 2],
                y=[trans_order],
                mode='markers',
                marker=dict(size=1, opacity=0),
                showlegend=False,
                hovertemplate=f"Transcript ID: {trans_id}<br>Exon: {exon['exon_number']}<br>Size: {exon['exon_size']} bp<br>Coordinates: {exon['exon_start']:,} - {exon['exon_end']:,}<extra></extra>"
            ))
    
    title_text = f"Transcripts for Gene {gene_name} ({gene_ensembl_id})<br>(ORF Perplexity: {orf_perplexity}, Coordinates: {min_start} - {max_end}, Strand: {strand})"
    
    fig.update_layout(
        title={
            'text': title_text,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14}
        },
        xaxis=dict(
            title="Genomic Position",
            range=[min_start - 1000, max_end + 1000],
            showgrid=False,
            tickformat=',',
            rangeslider=dict(visible=True, range=[min_start, max_end]),
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
            r=160,  
            t=80, 
            b=50
        ),
        hovermode='closest',
        plot_bgcolor='white',
        autosize=True 
    )
    
    for _, transcript in transcript_summary.iterrows():
        fig.add_annotation(
            x=max_end + (max_end - min_start) * 0.02,
            y=transcript['trans_order'],
            text=transcript['trans_id'],
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            font=dict(size=10)
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
                                      collapse_tissues: bool = True) -> go.Figure:
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

    if collapse_tissues:
        heatmap_data, tissue_display_names, tissue_categories = collapse_tissues_by_average(tpm_data, tissue_cols)
    else:
        heatmap_data = tpm_data[tissue_cols].values.T
        tissue_display_names, tissue_categories = process_individual_tissues(tissue_cols)
    
    num_tissues = len(tissue_display_names)
    num_transcripts = len(transcript_names)
    
    calculated_height = calculate_optimal_height(transcript_names, num_tissues, show_tables, height)
    
    if show_labels and show_tables != 'show':
        max_tissue_name_length = max([len(name) for name in tissue_display_names]) if tissue_display_names else 10
        left_margin = max(200, min(350, max_tissue_name_length * 8))
    else:
        left_margin = 40
    
    max_transcript_length = max([len(str(name)) for name in transcript_names]) if transcript_names else 10
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
                    title=colorbar_title,   
                )
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                y=tissue_display_names,
                x=transcript_names,
                colorscale=colorscale,
                hovertemplate=f'Transcript: %{{x}}<br>Tissue: %{{y}}<br>{data_type}: %{{z:.2f}}<extra></extra>',
                colorbar=dict(title=data_type, x=1.02)
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
                x=transcript_names,
                colorscale=colorscale,
                hovertemplate=f'Transcript: %{{x}}<br>Tissue: %{{y}}<br>{data_type}: %{{z:.2f}}<extra></extra>',
                colorbar=dict(title=data_type)
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
            tickangle=45,
            tickfont=dict(size=8),
            automargin=True,
            tickmode='array',
            tickvals=list(range(len(transcript_names))),
            ticktext=transcript_names
        )
    
    if collapse_tissues: 
        heatmap_resolution_level = "averaged by tissue"
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


def calculate_bottom_margin(show_labels: bool, transcript_names: list) -> int:
    """Calculate bottom margin based on label visibility and name lengths"""
    if show_labels:
        max_name_length = max(len(str(name)) for name in transcript_names) if transcript_names else 10
        return max(100, min(200, max_name_length * 7))
    return 50


def create_isoform_expression_clustergram(tpm_data: pd.DataFrame, 
                                          gene_name: str, 
                                          height: int = 600,
                                          colorscale: str = 'Viridis',
                                          data_type: str = 'TPM',
                                          show_tables: str = 'show',
                                          show_labels: bool = False,
                                          collapse_tissues: bool = True) -> go.Figure:
    """Create responsive clustergram that behaves exactly like junction clustergram"""
    if tpm_data.empty:
        return create_empty_isoform_message(f"No data for gene: {gene_name}")
    
    if 'gene_name' not in tpm_data.columns:
        return create_empty_isoform_message("Data missing 'gene_name' column.")
    #tpm_data = tpm_data[tpm_data['gene_name'] == gene_name].copy()
    
    if tpm_data.empty:
        return create_empty_isoform_message(f"No isoform data found for gene {gene_name}.")
    
    metadata_cols = ['id', 'trans_id', 'transcript', 'gene', 'tpm_average', 'tpm_sum', 'gene_name', 'max_ratio', 'min_ratio', 'prob']
    tissue_cols = [col for col in tpm_data.columns if col not in metadata_cols]
    
    transcript_names = tpm_data['transcript'].tolist() if 'transcript' in tpm_data.columns else tpm_data.index.tolist()
    
    if collapse_tissues:
        heatmap_data, tissue_display_names, tissue_categories = collapse_tissues_by_average(tpm_data, tissue_cols)
        tissue_cols_for_organs = tissue_display_names
    else:
        heatmap_data = tpm_data[tissue_cols].values.T
        tissue_display_names, tissue_categories = process_individual_tissues(tissue_cols)
        tissue_cols_for_organs = tissue_cols
    
    organ_list, color_list = create_organ_annotation_bar(tissue_cols_for_organs)
    clean_tissue_names = [extract_tissue_name_from_column(col) for col in tissue_cols_for_organs]
    
    # Ensure proper data orientation: rows = transcripts, columns = tissues
    if heatmap_data.shape[0] == len(tissue_cols_for_organs):
        clustergram_data = heatmap_data.T
    else:
        clustergram_data = heatmap_data
    
    num_tissues = len(clean_tissue_names)
    hide_tissue_labels = (num_tissues > 30) or (show_tables == 'show')
    left_margin = max(120, int(height * 0.1))
    
    if hide_tissue_labels:
        bottom_margin = 50 
        actual_clustergram_height = height
    else:
        bottom_margin = max(200, int(height * 0.35)) 
        actual_clustergram_height = height - 80
    
    width = min(1500, max(800, len(clean_tissue_names) * 12, int(height * 1.2)))
    clustergram_data_processed = pd.DataFrame(clustergram_data).copy()
    clustergram_data_processed = clustergram_data_processed.replace([np.inf, -np.inf], 0)
    clustergram_data_processed = clustergram_data_processed.astype(float)
    clustergram_data_processed = clustergram_data_processed.fillna(0)
    
    num_transcripts = len(transcript_names)
    if num_transcripts == 1:
        return create_single_transcript_heatmap(
            heatmap_data=heatmap_data,
            transcript_names=transcript_names,
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
            row_labels=transcript_names,
            column_colors=color_list,
            height=actual_clustergram_height,
            width=width,
            color_threshold={'row': 0.7, 'col': 0.7},
            hidden_labels='col' if not show_labels else None,
            cluster='all',
            color_list={
                'row': ['#636EFA', '#EF553B', '#00CC96', '#AB63FA'],
                'col': ['#FFA15A', '#19D3F3', '#FF6692', '#B6E880'],
                'bg': '#506784'
            },
            line_width=2,
            display_ratio=[0.12, 0.08] if not hide_tissue_labels else [0.08, 0.05],
            standardize='none', 
            center_values=False,
            return_computed_traces=True
        )
        # Update heatmap trace with reordered organs
        column_ids = computed_traces['column_ids']
        reordered_organ_list = [organ_list[i] for i in column_ids]
        heatmap_trace = clustergram.data[-1]
        heatmap_trace.customdata = np.array([reordered_organ_list] * len(transcript_names))
        heatmap_trace.text = heatmap_trace.customdata
        heatmap_trace.hovertemplate = (
            "<b>Transcript:</b> %{y}<br>"
            "<b>Tissue:</b> %{x}<br>"
            "<b>Organ:</b> %{text}<br>"
            f"<b>{data_type}:</b> %{{z:.2f}}"
            "<extra></extra>"
        )
        # Title for continuous colorscale bar - currently breaks viz, TO DO: figure out why 
        #heatmap_trace.colorbar.title = data_type
        
    except Exception as e2:
        print(f"Error creating clustergram: {e2}")
        return create_empty_isoform_message(f"Error creating visualization for {gene_name}")
    
    clustergram = apply_colorscale_to_clustergram(clustergram, colorscale)
    colorbar_x = -0.35
    
    try:
        if len(clustergram.data) > 0:
            heatmap_trace = clustergram.data[-1]
            if hasattr(heatmap_trace, 'colorbar'):
                heatmap_trace.colorbar.x = colorbar_x 
                heatmap_trace.colorbar.y = 1.0     
                heatmap_trace.colorbar.yanchor = 'top'
                heatmap_trace.colorbar.len = 0.3    
                heatmap_trace.colorbar.thickness = 20

    except Exception as e:
        print(f"Warning: Could not update colorbar position: {e}")
    
    organ_list, color_list = create_organ_annotation_bar(tissue_cols_for_organs)
    unique_organs = []
    unique_colors = []
    for organ, color in zip(organ_list, color_list):
        if organ not in unique_organs:
            unique_organs.append(organ)
            unique_colors.append(color)

    # Adjust positioning to prevent cutoff on right and place legend correctly
    colorbar_x = -0.87
    legend_x = -0.34
    legend_y_start = 1.04
    legend_y_step = 0.04

    try:
        if len(clustergram.data) > 0:
            heatmap_trace = clustergram.data[-1]
            if hasattr(heatmap_trace, 'colorbar'):
                heatmap_trace.colorbar.x = colorbar_x
                heatmap_trace.colorbar.y = 1.005
                heatmap_trace.colorbar.yanchor = 'top'
                heatmap_trace.colorbar.len = 0.3    
                heatmap_trace.colorbar.thickness = 20
    except Exception as e:
        print(f"Warning: Could not update colorbar position: {e}")
    
    # Colorscale bar title
    clustergram.add_annotation(
        x=-0.52,
        y=1.04,
        xref="paper",
        yref="paper",
        text=f"<b>{data_type}</b>",
        showarrow=False,
        font={"size": 13},
        xanchor="left",
        yanchor="top"
    )

    # Add organ legend annotations (to the right of TPM / ratio abundance colorscale)
    clustergram.add_annotation(
        x=legend_x,                 
        y=legend_y_start,                    
        xref="paper",
        yref="paper",
        text="<b>Organ Legend</b>",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=12)
    )

    for i, (organ, color) in enumerate(zip(unique_organs, unique_colors)):
        clustergram.add_annotation(
            x=legend_x,  # note: positioned to the right of colorbar
            y=legend_y_start - 0.04 - (i * legend_y_step),
            xref='paper',
            yref='paper',
            text=f'<span style="color:{color}; font-size:14px">&#9632;</span> {organ}',
            showarrow=False,
            xanchor='left',
            yanchor='top',
            font=dict(size=11)
        )
    
    clustergram.update_layout(
        title={
            'text': f"Isoform Expression Clustergram for {gene_name} ({len(transcript_names)} isoforms, {data_type} data)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14 if hide_tissue_labels else 16}
        },
        margin=dict(
                l=max(200, left_margin + 50), 
                r=150, 
                t=90, 
                b=calculate_bottom_margin(show_labels, transcript_names)
            ),
        autosize=True,
        height=height,
        yaxis=dict(
            automargin=True,
            tickangle=0,
            tickfont=dict(size=min(11, max(8, int(height/60))))
        ),
        xaxis=dict(
            automargin=True,
            tickangle=45 if show_labels else 0,
            tickfont=dict(
                size=6 if show_labels else 1,
                color='rgba(0,0,0,0)' if not show_labels else None 
            ),
            showticklabels=show_labels
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return clustergram


def create_single_transcript_heatmap(heatmap_data, transcript_names, tissue_display_names,
                                     gene_name, height, colorscale, data_type):
    """Create simple heatmap when only one transcript remains"""
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=heatmap_data,
        x=transcript_names,
        y=tissue_display_names,
        colorscale=colorscale,
        colorbar=dict(title=data_type),
        hovertemplate=f'Transcript: %{{x}}<br>Tissue: %{{y}}<br>{data_type}: %{{z:.2f}}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"{gene_name} Expression (1 isoform)",
        height=height,
        xaxis=dict(title="Transcript"),
        yaxis=dict(title="Tissue"),
        margin=dict(l=100, r=50, t=80, b=100)
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
        pass

    return fig


def collapse_tissues_by_average(tpm_data: pd.DataFrame, 
                                tissue_cols: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
    """Collapse multiple experiments per tissue by averaging values"""
    tissue_to_organ_mapping = get_tissue_to_organ_mapping()
    tissue_mapping = {}
    
    for col in tissue_cols:
        tissue_mapping[col] = extract_tissue_name_from_column(col)
    
    # Group columns by tissue name
    tissue_groups = {}
    for col, tissue_name in tissue_mapping.items():
        if tissue_name not in tissue_groups:
            tissue_groups[tissue_name] = []
        tissue_groups[tissue_name].append(col)
    
    averaged_data = []
    tissue_display_names = []
    tissue_categories = []
    
    for tissue_name, columns in tissue_groups.items():
        tissue_data = tpm_data[columns].values
        
        averaged_values = np.mean(tissue_data, axis=1)
        averaged_data.append(averaged_values)
        
        tissue_display_names.append(tissue_name)
        tissue_category = tissue_to_organ_mapping[tissue_name]
        tissue_categories.append(tissue_category)
    
    heatmap_data = np.array(averaged_data)
    
    return heatmap_data, tissue_display_names, tissue_categories


def process_individual_tissues(tissue_cols: List[str]) -> Tuple[List[str], List[str]]:
    """Process individual tissue experiments (current behavior)"""
    tissue_display_names = []
    tissue_categories = []
    tissue_to_organ_mapping = get_tissue_to_organ_mapping()
    
    for col in tissue_cols:
        display_name = extract_tissue_name_from_column(col)
        tissue_display_names.append(display_name)
        tissue_category = tissue_to_organ_mapping[display_name]
        tissue_categories.append(tissue_category)
    
    return tissue_display_names, tissue_categories


def get_tissue_to_organ_mapping():
    """Create tissue to organ mapping with complete heart anatomy"""
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
        'vessels': '#D2691E'       # chocolate
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
                  .replace('.', '-')
                  .strip()
    )
    return tissue_name


def create_organ_annotation_bar(tissue_cols, height=20):
    """Create organ color annotation bar for clustergram"""
    tissue_to_organ = get_tissue_to_organ_mapping()
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
