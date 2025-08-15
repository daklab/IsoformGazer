import sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import html
import dash_bootstrap_components as dbc


def query_master_table(db_path, table_name, page=0, page_size=10, sort_by=None, filters=None, gene_filter=None):
    """
    Query isoform data with pagination, sorting, and filtering.
    OPTIMIZED: Uses prepared statements and efficient indexing
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA optimize")
    conn.execute("PRAGMA cache_size = 10000") 

    # Select all columns for full data table functionality
    query = f"SELECT * FROM {table_name}"
    
    where_clauses = []
    params = []
    
    if gene_filter:
        where_clauses.append("(gene_name = ? OR gene_id LIKE ?)")
        params.extend([gene_filter, f"{gene_filter}%"])
    
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
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    if sort_by:
        order_clauses = []
        for col, direction in sort_by:
            order_clauses.append(f"{col} {'ASC' if direction == 'asc' else 'DESC'}")
        if order_clauses:
            query += " ORDER BY " + ", ".join(order_clauses)
    
    if where_clauses:
        count_where = " WHERE " + " AND ".join(where_clauses)
    else:
        count_where = ""
    
    count_query = f"SELECT COUNT(*) FROM {table_name}{count_where}"
    total_count = pd.read_sql_query(count_query, conn, params=params).iloc[0, 0]
    
    if page_size > 0:
        query += f" LIMIT {page_size} OFFSET {page * page_size}"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    return df.to_dict('records'), total_count


def get_gene_options(db_path, search_term=None, limit=10):
    """Get gene options for dropdown from database"""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size = 10000")
    
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
