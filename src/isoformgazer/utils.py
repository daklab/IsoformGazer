import plotly.graph_objs as go
import numpy as np
import pandas as pd


def create_atse_map(gene_data, gene_name="Rpsa", gene_id="ENSMUSG00000032518.6", strand="+"):
    """
    Create an ATSE map visualization similar to the attached image.
    """
    fig = go.Figure()
    y_pos = 0
    labels = []
    
    # Add each transcript track
    for transcript_id, transcript in gene_data.items():
        labels.append(transcript_id)
        
        # Add intron lines  as dashed lines (TBD if this is really the best way to do it, since then these 
        # appear as traces you can toggle, but we really just want to able to toggle the entire junction visibility...)
        fig.add_trace(go.Scatter(
            x=[transcript['start'], transcript['end']],
            y=[y_pos, y_pos],
            mode='lines',
            line=dict(width=1, color='black', dash='dash'),
            name=transcript_id,
            hoverinfo='text',
            text=f"{transcript_id} ({transcript['annotation']})",
            customdata=[transcript_id],
            visible=True,
            showlegend=True
        ))
        
        for exon in transcript['exons']:
            fig.add_trace(go.Scatter(
                x=[exon['start'], exon['end'], exon['end'], exon['start'], exon['start']],
                y=[y_pos-0.3, y_pos-0.3, y_pos+0.3, y_pos+0.3, y_pos-0.3],
                fill="toself",
                fillcolor=exon['color'],
                line=dict(width=0),
                mode='lines',
                name=transcript_id,
                hoverinfo='text',
                text=f"Exon: {exon['id']} in {transcript_id}",
                showlegend=False,
                customdata=[transcript_id],
                visible=True
            ))
        
        y_pos -= 1  # have to move down for next track
    
    title = f"Splice Junctions and Exons for Gene {gene_name}<br>(ID: {gene_id}, Strand: {strand})"
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14}
        },
        showlegend=True,
        legend_title_text="Transcripts",
        xaxis=dict(
            title="Genomic Position",
            showgrid=False,
            zeroline=False,
            rangeslider=dict(visible=True),
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[y_pos-1, 1]
        ),
        margin=dict(l=100, r=20, t=50, b=50),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=300,
        hovermode='closest'
    )
    
    # Annotations for leftmost fig labels
    for i, label in enumerate(labels):
        annotation = dict(
            x=0,
            y=i * -1,
            xref="paper",
            yref="y",
            text=label,
            showarrow=False,
            font=dict(size=10, color="black"),
            align="right",
            xanchor="right",
            xshift=-10
        )
        fig.add_annotation(annotation)
    
    # coordinate markers!
    min_pos = min(t['start'] for t in gene_data.values())
    max_pos = max(t['end'] for t in gene_data.values())
    step = 500
    for pos in range(int(min_pos), int(max_pos) + step, step):
        fig.add_shape(
            type="line",
            x0=pos,
            y0=y_pos-1,
            x1=pos,
            y1=1,
            line=dict(color="Gray", width=1, dash="dot")
        )
        fig.add_annotation(
            x=pos,
            y=y_pos-1.2,
            text=str(pos),
            showarrow=False,
            font=dict(size=8),
            xanchor="center"
        )
    
    return fig


def generate_mock_data(random_seed=18):
    """
    Generates mock data for IsoformGazer prototype.
    """
    np.random.seed(random_seed)
    
    # 0) Random heatmap values
    data1 = np.random.rand(10, 10)
    data2 = np.random.rand(10, 10)
    
    # 1) Mock data for the left table (isoform-level)
    isoform_ids = [f"ENST0000{i:05d}" for i in range(1, 11)]
    num_effective_isoforms = np.random.randint(1, 10, size=10)
    nuclear_localization = np.random.choice(["High", "Medium", "Low"], size=10)
    coordinates = [f"chr{np.random.randint(1, 23)}:{np.random.randint(10000, 99999)}-{np.random.randint(100000, 999999)}" for _ in range(10)]
    annotations = np.random.choice(["Protein coding", "Nonsense mediated decay", "Retained intron", "Processed transcript", "Novel isoform"], size=10)

    df1 = pd.DataFrame({
        'Isoform_Id': isoform_ids,
        'Number of Effective Isoforms': num_effective_isoforms,
        'Nuclear Localization': nuclear_localization,
        'Coordinates': coordinates,
        'Annotation': annotations
    })

    # 2) Mock data for the right table (junction-level)
    genes = [f"GENE_{chr(65+i)}" for i in range(10)]
    splice_junction_ids = [f"SJ_{i:05d}" for i in range(1, 11)]
    atse_ids = [f"ATSE_{i:05d}" for i in range(1, 11)]
    strands = np.random.choice(["+", "-"], size=10)
    annotations_right = np.random.choice(["Exon Skipping", "Alt 5' SS", "Alt 3' SS", "Mutually Exclusive", "Intron Retention"], size=10)

    df2 = pd.DataFrame({
        'Gene': genes,
        'Splice_Junction_ID': splice_junction_ids,
        'ATSE_ID': atse_ids,
        'Strand': strands,
        'Annotation': annotations_right
    })
    
    # 3) Mock data for ATSE visualization (TO DO: adding logic for creating these on demand from junction master table)
    atse_data = {
        "Rpsa-201": {
            "start": 29500,
            "end": 31600,
            "annotation": "protein_coding",
            "exons": [
                {"id": "exon1", "start": 29600, "end": 29800, "color": "blue"},
                {"id": "exon2", "start": 30200, "end": 30400, "color": "blue"},
                {"id": "exon3", "start": 30600, "end": 30800, "color": "blue"},
                {"id": "exon4", "start": 31000, "end": 31200, "color": "blue"},
                {"id": "exon5", "start": 31400, "end": 31500, "color": "blue"}
            ]
        },
        "Rpsa-202": {
            "start": 29500,
            "end": 31600,
            "annotation": "retained_intron",
            "exons": [
                {"id": "exon1", "start": 29600, "end": 29800, "color": "green"},
                {"id": "exon2", "start": 30200, "end": 31500, "color": "green"}
            ]
        },
        "Rpsa-203": {
            "start": 29500,
            "end": 31600,
            "annotation": "nonsense_mediated_decay",
            "exons": [
                {"id": "exon1", "start": 29600, "end": 29800, "color": "green"},
                {"id": "exon2", "start": 30200, "end": 30400, "color": "green"},
                {"id": "exon3", "start": 31000, "end": 31200, "color": "green"},
                {"id": "exon4", "start": 31400, "end": 31500, "color": "green"}
            ]
        },
        "Rpsa-204": {
            "start": 29500,
            "end": 31600,
            "annotation": "nonsense_mediated_decay",
            "exons": [
                {"id": "exon1", "start": 29600, "end": 29800, "color": "green"},
                {"id": "exon2", "start": 30200, "end": 30400, "color": "green"},
                {"id": "exon3", "start": 30600, "end": 30800, "color": "green"},
                {"id": "exon4", "start": 31400, "end": 31500, "color": "green"}
            ]
        },
        "Rpsa-205": {
            "start": 29500,
            "end": 31600,
            "annotation": "protein_coding",
            "exons": [
                {"id": "exon1", "start": 29600, "end": 29800, "color": "blue"},
                {"id": "exon2", "start": 30600, "end": 30800, "color": "blue"},
                {"id": "exon3", "start": 31000, "end": 31200, "color": "blue"},
                {"id": "exon4", "start": 31400, "end": 31500, "color": "blue"}
            ]
        }
    }
    
    atse_fig = create_atse_map(atse_data)
    
    return {
        'data1': data1,
        'data2': data2,
        'df1': df1,
        'df2': df2,
        'atse_data': atse_data,
        'atse_fig': atse_fig
    }