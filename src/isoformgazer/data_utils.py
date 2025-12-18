import sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import html
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import os
import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path


def get_table_prefix(species="Human"):
    """Convert species name to table prefix for database queries"""
    return "mouse_" if species == "Mouse" else ""


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
    else:
        # Default sort order: isoforms by isoform_average_tpm DESC, junctions by junction_average_psi DESC
        if table_name.lower() == 'isoforms':
            query += " ORDER BY isoform_average_tpm DESC NULLS LAST"
        elif table_name.lower() == 'junctions':
            query += " ORDER BY junction_average_psi DESC NULLS LAST"
    
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


def get_all_gene_options(db_path, species="Human"):
    """Loads all gene options from database at startup for fast client-side filtering"""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size = 10000")

    table_prefix = get_table_prefix(species)
    query = f"""
    SELECT DISTINCT
        CASE
            WHEN gene_name IS NULL OR gene_name = '' THEN 'Unknown'
            ELSE gene_name
        END as gene_name,
        gene_id
    FROM {table_prefix}isoforms
    WHERE gene_id IS NOT NULL
    ORDER BY gene_name
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    options = []
    for _, row in df.iterrows():
        gene_name = row['gene_name']
        gene_id = row['gene_id']

        if gene_name and gene_id and not pd.isna(gene_name) and not pd.isna(gene_id):
            options.append({
                'label': f"{gene_name} ({gene_id})",
                'value': gene_name,
                'search': f"{gene_name.lower()} {gene_id.lower()}"
            })

    return options


def get_gene_options(db_path, search_term=None, limit=10, species="Human"):
    """Get gene options for dropdown from database"""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size = 10000")

    table_prefix = get_table_prefix(species)
    if search_term:
        query = f"""
        SELECT DISTINCT
            CASE
                WHEN gene_name IS NULL OR gene_name = '' THEN 'Unknown'
                ELSE gene_name
            END as gene_name,
            gene_id
        FROM {table_prefix}isoforms
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
        query = f"""
        SELECT DISTINCT
            CASE
                WHEN gene_name IS NULL OR gene_name = '' THEN 'Unknown'
                ELSE gene_name
            END as gene_name,
            gene_id
        FROM {table_prefix}isoforms
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


def validate_value(column, value, operator, rules):
    """Validate a single value against column rules"""
    if column not in rules:
        return None 
            
    rule = rules[column]
    val_type = rule['type']
    validation_rule = rule['rule']
    
    if val_type == 'integer':
        try:
            int_val = int(value)
        except ValueError:
            return f"Column '{column.replace('_', ' ').title()}' accepts integer values only."
        
    elif val_type == 'float':
        try:
            float_val = float(value)
        except ValueError:
            return f"Column '{column.replace('_', ' ').title()}' accepts numeric values only."
    
    # Rule-specific validation cases
    if validation_rule == 'any':
        return None
    
    elif validation_rule == 'numeric':
        return None
    
    elif validation_rule == 'no_numeric':
        if value.replace('.', '').replace('-', '').isdigit():
            return f"Column '{column.replace('_', ' ').title()}' does not accept numeric values."
        
    elif validation_rule == 'starts_with':
        if operator in ('eq', 'contains') and not value.startswith(rule['value']):
            return f"Column '{column.replace('_', ' ').title()}' values should begin with '{rule['value']}'."
        
    elif validation_rule == 'in_list':
        if operator == 'eq':
            if val_type == 'integer':
                if int(value) not in rule['values']:
                    return f"Column '{column.replace('_', ' ').title()}' only accepts values: {', '.join(map(str, rule['values']))}."
            elif val_type == 'string':
                if value not in rule['values']:
                    return f"Column '{column.replace('_', ' ').title()}' only accepts values: {', '.join(rule['values'])}."
                
    elif validation_rule == 'range':
        if val_type in ('float', 'integer'):
            num_val = float(value) if val_type == 'float' else int(value)
            min_val = rule.get('min', float('-inf'))
            max_val = rule.get('max', float('inf'))
            if not (min_val <= num_val <= max_val):
                return f"Column '{column.replace('_', ' ').title()}' values must be between {min_val} and {max_val}."
            
    elif validation_rule == 'min_value':
        if val_type in ('float', 'integer'):
            num_val = float(value) if val_type == 'float' else int(value)
            min_val = rule.get('min', float('-inf'))
            exclusive = rule.get('exclusive', False)
            if exclusive and num_val <= min_val:
                return f"Column '{column.replace('_', ' ').title()}' values must be greater than {min_val}."
            elif not exclusive and num_val < min_val:
                return f"Column '{column.replace('_', ' ').title()}' values must be greater than or equal to {min_val}."
    
    return None


def validate_filter_input(db_path, table_name, filter_query):
    """
    Validate user filter query input with specific column validation rules.
    Returns: (is_valid, errors_dict)
    """
    if not filter_query:
        return True, {}
    
    validation_rules = {
        'isoforms': {
            'gene': {'type': 'string', 'rule': 'starts_with', 'value': 'ENSG'},
            'transcript': {'type': 'string', 'rule': 'any'},
            'isoform_category': {'type': 'string', 'rule': 'in_list', 'values': ['NMD', 'RI', 'noORF', 'protein_coding']},
            'isoform_canonical': {'type': 'integer', 'rule': 'in_list', 'values': [0, 1]},
            'isoform_annotated': {'type': 'integer', 'rule': 'in_list', 'values': [0, 1]},
            'isoform_max_ratio': {'type': 'float', 'rule': 'range', 'min': 0.0, 'max': 1.0},
            'isoform_min_ratio': {'type': 'float', 'rule': 'range', 'min': 0.0, 'max': 1.0},
            'isoform_prob': {'type': 'float', 'rule': 'range', 'min': 0.0, 'max': 1.0},
            'gene_potential': {'type': 'integer', 'rule': 'numeric'},
            'gene_perplexity': {'type': 'float', 'rule': 'min_value', 'min': 0.0, 'exclusive': True},
            'gene_protein_category': {'type': 'string', 'rule': 'any'},
            'gene_gencode_v46_basic_transcript_counts': {'type': 'integer', 'rule': 'numeric'},
            'gene_average_tpm': {'type': 'float', 'rule': 'numeric'},
            'gene_total_tpm': {'type': 'float', 'rule': 'numeric'},
            'isoform_average_tpm': {'type': 'float', 'rule': 'numeric'},
            'isoform_total_tpm': {'type': 'float', 'rule': 'numeric'},
            'isoform_number_of_exons': {'type': 'integer', 'rule': 'numeric'},
            'isoform_ranking_within_gene': {'type': 'integer', 'rule': 'numeric'},
            'isoform_effective': {'type': 'integer', 'rule': 'in_list', 'values': [0, 1]},
            'ptc_potential': {'type': 'integer', 'rule': 'numeric'},
            'ptc_perplexity': {'type': 'float', 'rule': 'numeric'},
            'orf_id': {'type': 'string', 'rule': 'any'},
            'orf_prob': {'type': 'float', 'rule': 'range', 'min': 0.0, 'max': 1.0},
            'orf_potential': {'type': 'integer', 'rule': 'numeric'},
            'orf_perplexity': {'type': 'float', 'rule': 'numeric'},
            'orf_effective_tissue_count': {'type': 'integer', 'rule': 'numeric'},
            'orf_expressed_samples': {'type': 'integer', 'rule': 'numeric'},
            'orf_percent_effective_tissue_count': {'type': 'float', 'rule': 'numeric'},
            'orf_ratio_sd': {'type': 'float', 'rule': 'range', 'min': 0.0, 'max': 1.0},
            'gene_expressed_samples': {'type': 'integer', 'rule': 'numeric'},
            'isoform_ratio_sd': {'type': 'float', 'rule': 'range', 'min': 0.0, 'max': 1.0},
            'orf_quadrant': {'type': 'string', 'rule': 'in_list', 'values': ['I', 'II', 'III', 'IV']},
            'orf_ranking_within_gene': {'type': 'integer', 'rule': 'numeric'},
            'orf_effective': {'type': 'integer', 'rule': 'in_list', 'values': [0, 1]},
            'gene_name': {'type': 'string', 'rule': 'any'},
            'transcript_name': {'type': 'string', 'rule': 'any'},
            'novel_junction': {'type': 'integer', 'rule': 'in_list', 'values': [0, 1]},
            'gene_id': {'type': 'string', 'rule': 'starts_with', 'value': 'ENSG'}
        },
        'junctions': {
            'gene_name': {'type': 'string', 'rule': 'any'},
            'gene_id': {'type': 'string', 'rule': 'starts_with', 'value': 'ENSG'},
            'event_id': {'type': 'string', 'rule': 'starts_with', 'value': 'ENSG'},
            'junction_id': {'type': 'string', 'rule': 'any'},
            'junction_id_index': {'type': 'integer', 'rule': 'numeric'},
            'atse_count': {'type': 'integer', 'rule': 'numeric'},
            'junction_count': {'type': 'integer', 'rule': 'numeric'},
            'cell_type': {'type': 'string', 'rule': 'no_numeric'},
            'n_cells': {'type': 'integer', 'rule': 'numeric'},
            'psi': {'type': 'float', 'rule': 'numeric'}
        }
    }
    
    errors = {}
    expressions = filter_query.split(' && ')
    table_rules = validation_rules.get(table_name, {})
    
    for expression in expressions:
        if not expression or expression.isspace():
            continue
        
        try:
            col = None
            val = None
            operator = None
            
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
                continue
            
            col = col.strip('{} ')
            val = val.strip('" ')
            
            error_msg = validate_value(col, val, operator, table_rules)
            if error_msg:
                errors[col] = error_msg
        
        except Exception:
            continue
    
    return len(errors) == 0, errors


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


###################################################################
# DATA PREPROCESSING UTILITIES
###################################################################
def apply_distance_preprocessing(data_matrix, distance_metric):
    """Apply specific preprocessing for problematic distance metrics in clustergram - 
    used to check invalid values and correct prior to passing to Dash Bio Clustergram"""
    processed = data_matrix.copy()
    
    # Correlation requires non-constant rows/columns, so should check for rows where there is no variance and add tiny amount of noise
    if distance_metric == 'correlation':
        
        row_std = processed.std(axis=1)
        col_std = processed.std(axis=0)
        
        constant_rows = row_std == 0
        if constant_rows.any():
            noise = np.random.normal(0, 1e-7, processed.loc[constant_rows].shape)
            processed.loc[constant_rows] += noise
            
        constant_cols = col_std == 0
        if constant_cols.any():
            noise = np.random.normal(0, 1e-7, processed.loc[:, constant_cols].shape)
            processed.loc[:, constant_cols] += noise
    
    # Standardized Euclidean needs non-zero variance for all features, and cosine needs non-zero magnitude vectors: 
    # can use same fix as correlation, but only necessary for columns
    elif distance_metric == 'seuclidean' or distance_metric == 'cosine':
        col_std = processed.std(axis=0)
        constant_cols = col_std == 0
        if constant_cols.any():
            noise = np.random.normal(0, 1e-7, processed.loc[:, constant_cols].shape)
            processed.loc[:, constant_cols] += noise
    
    return processed


###################################################################
# DEFAULT GENE CACHING
###################################################################
def get_default_gene_cache_path(base_dir):
    """Returns path for default gene cache"""
    return os.path.join(base_dir, "data", "default_gene_cache.pkl")


def get_cache_metadata_path(base_dir):
    """Returns path for cache metadata"""
    return os.path.join(base_dir, "data", "cache_metadata.json")


def get_database_hash(db_path):
    """Generates hash of database file for cache invalidation"""
    try:
        with open(db_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print(f"Warning: Could not compute database hash: {e}")
        return None


def is_cache_valid(base_dir, db_path):
    """Check if default gene cache is still valid"""
    cache_path = get_default_gene_cache_path(base_dir)
    metadata_path = get_cache_metadata_path(base_dir)

    if not Path(cache_path).exists() or not Path(metadata_path).exists():
        return False

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        current_db_hash = get_database_hash(db_path)
        cached_db_hash = metadata.get('db_hash')

        if current_db_hash != cached_db_hash:
            return False

        return True
    except Exception as e:
        print(f"Warning: Cache validation failed: {e}")
        return False


def load_default_gene_cache(base_dir):
    """Load cached data and plots for default gene (A1BG-AS1)"""
    cache_path = get_default_gene_cache_path(base_dir)

    if not Path(cache_path).exists():
        return None

    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        return cache_data
    except Exception as e:
        print(f"Warning: Could not load cache: {e}")
        return None


def save_default_gene_cache(base_dir, db_path, cache_data):
    """Save cached data and plots for default gene (A1BG-AS1)"""
    cache_path = get_default_gene_cache_path(base_dir)
    metadata_path = get_cache_metadata_path(base_dir)

    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

        metadata = {
            'gene_name': 'A1BG-AS1',
            'timestamp': datetime.now().isoformat(),
            'db_hash': get_database_hash(db_path),
            'cache_version': 1
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

    except Exception as e:
        print(f"Warning: Could not save cache: {e}")


def clear_default_gene_cache(base_dir):
    """Clear the default gene cache"""
    cache_path = get_default_gene_cache_path(base_dir)
    metadata_path = get_cache_metadata_path(base_dir)

    try:
        if Path(cache_path).exists():
            os.remove(cache_path)
        if Path(metadata_path).exists():
            os.remove(metadata_path)
        print("✓ Default gene cache cleared")
    except Exception as e:
        print(f"Warning: Could not clear cache: {e}")


def generate_default_gene_cache(db_path, gene_name='A1BG-AS1', species="Human"):
    """
    Generate cache for default gene data and plots.
    This function queries all necessary data for the default gene to populate
    the initial stores with pre-computed results.

    Returns a dictionary with cached data structures.
    """
    cache_data = {}

    try:
        table_prefix = get_table_prefix(species)
        isoform_data, _ = query_master_table(
            db_path,
            table_name=f'{table_prefix}isoforms',
            page=0,
            page_size=10,
            sort_by=None,
            filters=None,
            gene_filter=gene_name
        )
        cache_data['isoform_full_data'] = isoform_data

        junction_data, _ = query_master_table(
            db_path,
            table_name=f'{table_prefix}junctions',
            page=0,
            page_size=100,
            sort_by=None,
            filters=None,
            gene_filter=gene_name
        )
        cache_data['junction_full_data'] = junction_data
        cache_data['filtered_isoform_ids'] = [row.get('id', '') for row in isoform_data]
        cache_data['filtered_junction_ids'] = [row.get('junction_id', '') for row in junction_data]

        return cache_data

    except Exception as e:
        print(f"Warning: Could not generate default gene cache: {e}")
        return None


def extract_gtf_attr_val(attr_str):
    """Extract value from GTF attribute"""
    parts = attr_str.split(None, 1)

    if len(parts) < 2:
        return None
    
    value = parts[1].strip()

    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]

    return value


def get_matplotlib_colormap(colorscale_name: str):
    """Convert a Plotly colorscale name to a matplotlib colormap"""
    colormap_mapping = {
        'Viridis': plt.cm.viridis,
        'Plasma': plt.cm.plasma,
        'Inferno': plt.cm.inferno,
        'Magma': plt.cm.magma,
        'Cividis': plt.cm.cividis,
        'Blues': plt.cm.Blues,
        'Reds': plt.cm.Reds,
        'RdBu_r': plt.cm.RdBu_r,
        'RdYlBu': plt.cm.RdYlBu,
        'Spectral': plt.cm.Spectral,
        'YlOrRd': plt.cm.YlOrRd,
        'Turbo': plt.cm.turbo,
    }

    return colormap_mapping.get(colorscale_name, plt.cm.viridis)