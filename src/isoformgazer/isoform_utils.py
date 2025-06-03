import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
from typing import Dict, List, Tuple
import os

###################################################################
# VISUALIZATION METHODS
###################################################################
def load_psl_data(psl_file_path: str) -> pd.DataFrame:
    """Load and process PSL file for transcript structure"""
    try:
        # PSL files have specific column structure
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
        print(f"No gene_id found for gene_name '{gene_name}'")
        return None
    

def process_transcript_structure(psl_df: pd.DataFrame, gene_name: str, db_path: str) -> pd.DataFrame:
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


def load_tpm_data(tpm_file_path: str) -> pd.DataFrame:
    """Load TPM data for isoform expression heatmap"""
    try:
        # Load the TPM file
        tpm_df = pd.read_csv(tpm_file_path, sep='\t')
        
        # Check if required columns exist
        required_cols = ['transcript', 'gene', 'gene_name']
        missing_cols = [col for col in required_cols if col not in tpm_df.columns]
        if missing_cols:
            print(f"Warning: Missing required columns of {missing_cols}")
        
        # Show sample of gene_name column
        if 'gene_name' in tpm_df.columns:
            unique_genes = tpm_df['gene_name'].dropna().unique()
            #print(f"Number of unique genes in TPM data: {len(unique_genes)}")
            #print(f"First 10 genes: {list(unique_genes[:10])}")
        
        return tpm_df
    except Exception as e:
        print(f"Error loading TPM file: {e}")
        return pd.DataFrame()
    

def create_transcript_structure_plot(transcript_data: pd.DataFrame, gene_name: str, height: int = 400) -> go.Figure:
    """Create transcript structure plot similar to the R version"""
    
    if transcript_data.empty:
        return create_empty_isoform_message(f"No transcript data for gene: {gene_name}")
    
    transcript_summary = transcript_data.groupby('trans_id').agg({
        'transcript_length': 'first',
        'transcript_start': 'min',
        'transcript_end': 'max'
    }).sort_values('transcript_length', ascending=False).reset_index()
    
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
                hovertemplate=f"Exon {exon['exon_number']}<br>Size: {exon['exon_size']} bp<br>Position: {exon['exon_start']:,}-{exon['exon_end']:,}<extra></extra>"
            ))
    
    fig.update_layout(
        title={
            'text': f"{gene_name} Transcript Summary",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14}
        },
        xaxis=dict(
            title="Genomic Position",
            range=[min_start - 1000, max_end + 1000],
            showgrid=False,
            tickformat=',',
            rangeslider=dict(visible=True, range=[min_start, max_end])
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[0, y_max]
        ),
        height=height,
        margin=dict(l=100, r=150, t=50, b=50),
        hovermode='closest',
        plot_bgcolor='white'
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


def create_isoform_expression_heatmap(tpm_data: pd.DataFrame, 
                                      gene_name: str, 
                                      height: int = 600,
                                      colorscale: str = 'Viridis',
                                      data_type: str = 'TPM',
                                      show_tables: str = 'show',
                                      show_labels: bool = False,
                                      collapse_tissues: bool = True) -> go.Figure:
    """
    Creates fully responsive isoform expression heatmap 
    """
    if tpm_data.empty:
        return create_empty_isoform_message(f"No data for gene: {gene_name}")
    
    try:
        if 'gene_name' not in tpm_data.columns:
            return create_empty_isoform_message("Data missing 'gene_name' column.")
        
        gene_tpm = tpm_data[tpm_data['gene_name'] == gene_name].copy()
        
    except Exception as e:
        print(f"Error filtering data: {e}")
        return create_empty_isoform_message(f"Error filtering data for gene: {gene_name}")
    
    if gene_tpm.empty:
        return create_empty_isoform_message(f"No isoform data found for gene {gene_name}.")
    
    metadata_cols = ['transcript', 'gene', 'tpm_average', 'tpm_sum', 'gene_name', 'max_ratio', 'min_ratio', 'prob']
    tissue_cols = [col for col in gene_tpm.columns if col not in metadata_cols]
    #print(f"Found {len(tissue_cols)} tissue columns")
    
    heatmap_data = gene_tpm[tissue_cols].values.T
    transcript_names = gene_tpm['transcript'].tolist() if 'transcript' in gene_tpm.columns else gene_tpm.index.tolist()

    if collapse_tissues:
        heatmap_data, tissue_display_names, tissue_categories = collapse_tissues_by_average(gene_tpm, tissue_cols)
    else:
        heatmap_data = gene_tpm[tissue_cols].values.T
        tissue_display_names, tissue_categories = process_individual_tissues(tissue_cols)
    
    num_tissues = len(tissue_display_names)
    
    # Dynamic margin calculation...
    if show_tables == 'show':
        calculated_height = min(height, 350)
        if show_labels:
            left_margin = 100
        else:
            left_margin = 40
        right_margin = 60
        top_margin = 60
        # Calculate bottom margin for transcript names (always needed for x-axis)
        max_transcript_length = max([len(str(name)) for name in transcript_names]) if transcript_names else 10
        bottom_margin = 40
        
    else:
        calculated_height = max(height, 600)
        if show_labels:
            max_tissue_name_length = max([len(name) for name in tissue_display_names]) if tissue_display_names else 10
            left_margin = max(200, min(350, max_tissue_name_length * 8))
        else:
            left_margin = 40  
        right_margin = 100
        top_margin = 80
        # Calculate bottom margin for transcript names (always needed for x-axis)
        max_transcript_length = max([len(str(name)) for name in transcript_names]) if transcript_names else 10
        bottom_margin = max(120, min(200, max_transcript_length * 8))

    tissue_colors = get_tissue_colors()
    
    if show_tables == 'show':
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.06, 0.94],
            horizontal_spacing=0.005,
            shared_yaxes=True
        )
        
        # Add tissue color annotation
        tissue_color_values = []
        for category in tissue_categories:
            color = tissue_colors.get(category, '#CCCCCC')
            tissue_color_values.append([color])
        
        fig.add_trace(
            go.Heatmap(
                z=tissue_color_values,
                y=tissue_display_names,
                x=[''],
                colorscale=[[0, '#CCCCCC'], [1, '#CCCCCC']],
                showscale=False,
                hovertemplate='Tissue: %{y}<br>Category: %{customdata}<extra></extra>',
                customdata=tissue_categories
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
        
        # Show/hide labels based on show_labels toggle
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
            tickfont=dict(size=9),
            automargin=True
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
            'font': {'size': 16},
            'pad': {'b': 10}
        },
        font=dict(size=9),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        autosize=True
    )
    
    return fig


def collapse_tissues_by_average(gene_tpm: pd.DataFrame, 
                                tissue_cols: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
    """Collapse multiple experiments per tissue by averaging values"""
    tissue_mapping = {}
    for col in tissue_cols:
        if '.' in col:
            clean_tissue_name = '.'.join(col.split('.')[1:]).replace('_', ' ')
        else:
            clean_tissue_name = col.replace('_', ' ')
        tissue_mapping[col] = clean_tissue_name
    
    # Group columns by tissue name
    tissue_groups = {}
    for col, tissue_name in tissue_mapping.items():
        if tissue_name not in tissue_groups:
            tissue_groups[tissue_name] = []
        tissue_groups[tissue_name].append(col)
    
    print(f"Collapsing {len(tissue_cols)} experiments into {len(tissue_groups)} tissues")
    averaged_data = []
    tissue_display_names = []
    tissue_categories = []
    
    for tissue_name, columns in tissue_groups.items():
        tissue_data = gene_tpm[columns].values
        
        averaged_values = np.mean(tissue_data, axis=1)
        averaged_data.append(averaged_values)
        
        tissue_display_names.append(tissue_name)
        tissue_category = map_tissue_to_category(tissue_name)
        tissue_categories.append(tissue_category)
    
    heatmap_data = np.array(averaged_data)
    
    return heatmap_data, tissue_display_names, tissue_categories


def process_individual_tissues(tissue_cols: List[str]) -> Tuple[List[str], List[str]]:
    """Process individual tissue experiments (current behavior)"""
    
    tissue_display_names = []
    tissue_categories = []
    
    for col in tissue_cols:
        if '.' in col:
            tissue_ensembl_id = col.split('.')[0]
            tissue_name = '.'.join(col.split('.')[1:])
            display_name = f"{tissue_name} ({tissue_ensembl_id})"
        else:
            display_name = col
        
        display_name = display_name.replace('_', ' ')
        tissue_display_names.append(display_name)

        base_tissue_name = tissue_name if '.' in col else col
        tissue_category = map_tissue_to_category(base_tissue_name)
        tissue_categories.append(tissue_category)
    
    return tissue_display_names, tissue_categories


def map_tissue_to_category(tissue_name: str) -> str:
    """Map individual tissue name to broader category"""
    tissue_lower = tissue_name.lower().replace('-', '_').replace(' ', '_')
    
    tissue_keywords = {
        'blood': ['gm12878', 'k562', 'hl_60', 'hl.60'],
        'brain': ['brain', 'neuron', 'astrocyte', 'cortex', 'dorsolateral_prefrontal_cortex', 'glutamatergic_neuron'],
        'liver': ['hepg2', 'liver', 'right_lobe_of_liver'],
        'lung': ['lung', 'calu3', 'pc_9', 'left_lung', 'lower_lobe_of_left_lung'],
        'heart': ['heart', 'cardiac', 'cardiac_septum', 'heart_left_ventricle'],
        'kidney': ['kidney'],
        'muscle': ['muscle', 'psoas_muscle'],
        'colon': ['colon', 'caco_2', 'hct116', 'left_colon', 'mucosa_of_descending_colon'],
        'breast': ['mcf', 'mammary_epithelial_cell'],
        'pancreas': ['panc1', 'type_b_pancreatic_cell', 'progenitor_cell_of_endocrine_pancreas'],
        'stem_cell': ['wtc11', 'h1', 'h9', 'progenitor_cell'],
        'bone': ['chondrocyte'],
        'other': []
    }
    
    for category, keywords in tissue_keywords.items():
        if any(keyword in tissue_lower for keyword in keywords):
            return category
    
    return 'other'


def create_tissue_mapping(tissue_names: List[str]) -> Dict[str, str]:
    """Create tissue mapping for samples"""
    tissue_keywords = {
        'blood': ['GM12878', 'K562', 'HL-60', 'HL.60'],
        'brain': ['brain', 'neuron', 'astrocyte', 'cortex', 'dorsolateral_prefrontal_cortex', 'glutamatergic_neuron'],
        'liver': ['HepG2', 'liver', 'right_lobe_of_liver'],
        'lung': ['lung', 'Calu3', 'PC-9', 'left_lung', 'lower_lobe_of_left_lung', 'lower_lobe_of_right_lung', 'upper_lobe_of_right_lung'],
        'heart': ['heart', 'cardiac', 'cardiac_septum', 'heart_left_ventricle', 'heart_right_ventricle', 'left_cardiac_atrium', 'right_cardiac_atrium'],
        'kidney': ['kidney'],
        'muscle': ['muscle', 'psoas_muscle'],
        'colon': ['colon', 'Caco-2', 'HCT116', 'left_colon', 'mucosa_of_descending_colon'],
        'breast': ['MCF', 'mammary_epithelial_cell'],
        'pancreas': ['Panc1', 'type_B_pancreatic_cell', 'progenitor_cell_of_endocrine_pancreas'],
        'prostate': ['PC-3'],
        'ovary': ['ovary'],
        'stem_cell': ['WTC11', 'H1', 'H9'],
        'embryo': ['endodermal_cell', 'endothelial_cell', 'neural_crest_cell']
    }
    
    mapping = {}
    for tissue in tissue_names:
        tissue_lower = tissue.lower().replace('-', '_').replace(' ', '_')
        assigned = False
        for tissue_type, keywords in tissue_keywords.items():
            if any(keyword.lower() in tissue_lower for keyword in keywords):
                mapping[tissue] = tissue_type
                assigned = True
                break
        if not assigned:
            mapping[tissue] = 'other'
    
    return mapping


def get_tissue_colors() -> Dict[str, str]:
    """Get colors for different tissue types"""
    return {
        'blood': '#FF6B6B',
        'brain': '#4ECDC4', 
        'liver': '#45B7D1',
        'lung': '#96CEB4',
        'heart': '#FFEAA7',
        'kidney': '#DDA0DD',
        'muscle': '#98D8C8',
        'colon': '#F7DC6F',
        'breast': '#F8C471',
        'pancreas': '#82E0AA',
        'prostate': '#AED6F1',
        'ovary': '#E8DAEF',
        'stem_cell': '#D5DBDB',
        'embryo': '#FADBD8',
        'other': '#BDC3C7',
        'unknown': '#CCCCCC'
    }


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
