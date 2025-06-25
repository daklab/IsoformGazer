import sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import html
import dash_bootstrap_components as dbc

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
    
    size = 100000
    
    # 1) Mock data for the left table (isoform-level)
    isoform_ids = [f"ENST0000{i:05d}" for i in range(1, size+1)]
    num_effective_isoforms = np.random.randint(1, 10, size=size)
    nuclear_localization = np.random.choice(["High", "Medium", "Low"], size=size)
    coordinates = [f"chr{np.random.randint(1, 23)}:{np.random.randint(10000, 99999)}-{np.random.randint(100000, 999999)}" for _ in range(size)]
    annotations = np.random.choice(["Protein coding", "Nonsense mediated decay", "Retained intron", "Processed transcript", "Novel isoform"], size=size)

    df1 = pd.DataFrame({
        'Isoform_Id': isoform_ids,
        'Number of Effective Isoforms': num_effective_isoforms,
        'Nuclear Localization': nuclear_localization,
        'Coordinates': coordinates,
        'Annotation': annotations
    })

    # 2) Mock data for the right table (junction-level)
    genes = [f"GENE_{chr(65+(i%26))}" for i in range(size)]  # Using modulo to avoid going beyond ASCII letters
    splice_junction_ids = [f"SJ_{i:05d}" for i in range(1, size+1)]
    atse_ids = [f"ATSE_{i:05d}" for i in range(1, size+1)]
    strands = np.random.choice(["+", "-"], size=size)
    annotations_right = np.random.choice(["Exon Skipping", "Alt 5' SS", "Alt 3' SS", "Mutually Exclusive", "Intron Retention"], size=size)

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


def query_master_table(db_path, table_name, page=0, page_size=10, sort_by=None, filters=None, gene_filter=None):
    """
    Query isoform data with pagination, sorting, and filtering.
    """
    conn = sqlite3.connect(db_path)

    if table_name == 'isoforms': 
        query = "SELECT * FROM isoforms"
    elif table_name == 'junctions': 
        query = "SELECT * FROM junctions"
    where_clauses = []
    params = []
    
    # Filter by gene if provided - should search both gene_name and gene_id
    if gene_filter:
        where_clauses.append("(gene_name = ? OR gene_id = ?)")
        params.extend([gene_filter, gene_filter])
    
    # Apply any/all filtering conditions from user vals 
    if filters:
        for column, operator, value in filters:
            if operator == 'contains':
                where_clauses.append(f"LOWER({column}) LIKE LOWER(?)")
                params.append(f"%{value}%")
            elif operator == 'eq':
                where_clauses.append(f"{column} = ?")
                params.append(value)
            elif operator == 'ne':
                where_clauses.append(f"{column} != ?")
                params.append(value)
            elif operator == 'lt':
                where_clauses.append(f"{column} < ?")
                params.append(value)
            elif operator == 'gt':
                where_clauses.append(f"{column} > ?")
                params.append(value)
            elif operator == 'le':
                where_clauses.append(f"{column} <= ?")
                params.append(value)
            elif operator == 'ge':
                where_clauses.append(f"{column} >= ?")
                params.append(value)
    
    # Add WHERE if needed
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    # Sorting
    if sort_by:
        order_clauses = []
        for col, direction in sort_by:
            order_clauses.append(f"{col} {'ASC' if direction == 'asc' else 'DESC'}")
        if order_clauses:
            query += " ORDER BY " + ", ".join(order_clauses)
    
    # Get total count for pagination info
    count_query = f"SELECT COUNT(*) FROM ({query})"
    total_count = pd.read_sql_query(count_query, conn, params=params).iloc[0, 0]
    query += f" LIMIT {page_size} OFFSET {page * page_size}"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    return df.to_dict('records'), total_count


def get_gene_options(db_path, search_term=None, limit=10):
    """Get gene options for dropdown from database"""
    conn = sqlite3.connect(db_path)
    
    if search_term:
        query = """
        SELECT DISTINCT 
            CASE 
                WHEN gene_name IS NULL OR gene_name = '' THEN 'Unknown'
                ELSE gene_name 
            END as gene_name, 
            gene_id 
        FROM isoforms 
        WHERE gene_id IS NOT NULL 
        AND (
            (gene_name IS NOT NULL AND gene_name LIKE ?) 
            OR gene_id LIKE ? 
            OR (gene_name IS NULL AND 'Unknown' LIKE ?)
        )
        ORDER BY gene_name
        LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=[f"%{search_term}%", f"%{search_term}%", f"%{search_term}%", limit])
    else:
        query = """
        SELECT DISTINCT 
            CASE 
                WHEN gene_name IS NULL OR gene_name = '' THEN 'Unknown'
                ELSE gene_name 
            END as gene_name, 
            gene_id 
        FROM isoforms 
        WHERE gene_id IS NOT NULL
        ORDER BY gene_name
        LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=[limit])
    
    conn.close()
    
    options = []
    for _, row in df.iterrows():
        gene_name = row['gene_name']
        gene_id = row['gene_id']
        
        # Skip any entries that are still somehow None/null?
        if gene_name and gene_id and not pd.isna(gene_name) and not pd.isna(gene_id):
            options.append({
                'label': f"{gene_name} ({gene_id})",
                'value': gene_name  # Use the processed gene_name (including 'Unknown' for no name genes)
            })
    
    return options


def get_master_table_columns(db_path: str, table_name: str) -> list:
    """Get columns for DataTables with index handling"""
    conn = sqlite3.connect(db_path)
    columns = [{
        'name': 'id',
        'id': 'id',
        'type': 'numeric'
    }]
    
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    cols = cursor.fetchall()
    
    for col in cols:
        if col[1] == 'id':
            continue
            
        columns.append({
            'name': col[1].replace('_', ' ').title(),
            'id': col[1],
            'type': 'numeric' if col[2] in ['REAL', 'INTEGER'] else 'text'
        })
    
    conn.close()
    return columns


def get_column_types(db_path, table_name):
    """Get data types for columns in a table"""
    conn = sqlite3.connect(db_path)
    
    query = f"PRAGMA table_info({table_name})"
    columns_info = pd.read_sql_query(query, conn)
    conn.close()
    
    # Dictionary of column name -> data type
    column_types = {}
    for _, row in columns_info.iterrows():
        column_types[row['name']] = row['type'].lower()
        
    return column_types


def parse_filter_query(db_path, filter_query, table_name=None):
    """Parse filter query with type validation"""
    if not filter_query:
        return []
    
    if table_name: 
        column_types = get_column_types(db_path, table_name)
    else: 
        column_types = {}
    
    filters = []
    expressions = filter_query.split(' && ')
    
    for expression in expressions:
        if not expression or expression.isspace():
            continue
        
        try:
            if ' scontains ' in expression:
                col, val = expression.split(' scontains ')
                operator = 'contains'
            elif ' s< ' in expression:
                col, val = expression.split(' s< ')
                operator = 'lt'
            elif ' s> ' in expression:
                col, val = expression.split(' s> ')
                operator = 'gt'
            elif ' s<= ' in expression:
                col, val = expression.split(' s<= ')
                operator = 'le'
            elif ' s>= ' in expression:
                col, val = expression.split(' s>= ')
                operator = 'ge'
            elif ' s= ' in expression:
                col, val = expression.split(' s= ')
                operator = 'eq'
            elif ' s!= ' in expression:
                col, val = expression.split(' s!= ')
                operator = 'ne'
            else:
                print(f"Unrecognized filter operation: {expression}")
                continue
            
            col = col.strip('{} ')
            val = val.strip('" ')
            
            col_type = column_types.get(col, 'string')
            if col_type in ('integer', 'float', 'numeric') and operator in ('eq', 'lt', 'gt', 'le', 'ge', 'ne'):
                try:
                    if col_type == 'integer':
                        val = int(val)
                    elif col_type in ('float', 'numeric'):
                        val = float(val)
                except ValueError:
                    print(f"Skipping filter: invalid numeric value '{val}' for column '{col}'")
                    continue
            
            if operator == 'contains' and col_type in ('integer', 'float', 'numeric'):
                try:
                    if col_type == 'integer':
                        val = int(val)
                    elif col_type in ('float', 'real', 'numeric'):
                        val = float(val)
                    operator = 'eq'
                except ValueError:
                    print(f"Skipping filter: invalid numeric value '{val}' for column '{col}'")
                    continue
            
            filters.append((col, operator, val))
            
        except Exception as e:
            print(f"Error parsing filter expression '{expression}': {e}")
    
    return filters


def create_custom_spinner(message):
    """Create custom spinner with both animation and text message"""
    return html.Div(
        style={
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'center',
            'justifyContent': 'center',
            'height': '100%',
            'width': '100%',
            'padding': '20px',
            'backgroundColor': 'rgba(255, 255, 255, 0.9)',
            'borderRadius': '8px'
        },
        children=[
            # Spinner container
            html.Div(
                className="dash-spinner dash-default-spinner",
                style={'marginBottom': '15px'},
                children=[
                    html.Div(className="dash-default-spinner-rect1"),
                    html.Div(className="dash-default-spinner-rect2"),
                    html.Div(className="dash-default-spinner-rect3"),
                    html.Div(className="dash-default-spinner-rect4"),
                    html.Div(className="dash-default-spinner-rect5"),
                ]
            ),
            # Message text
            html.P(
                message,
                style={
                    'fontSize': '14px',
                    'color': '#506784',
                    'textAlign': 'center',
                    'fontFamily': '"Open Sans", sans-serif',
                    'margin': '0',
                    'fontWeight': '500'
                }
            )
        ]
    )
