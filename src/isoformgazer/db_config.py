"""
Database configuration module for IsoformGazer.
Supports both SQLite (local development) and PostgreSQL (production) using SQLAlchemy.
"""
import os
import pandas as pd
from contextlib import contextmanager
from sqlalchemy import create_engine, event, text
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy.orm import sessionmaker, scoped_session
import logging

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration and connection management"""

    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._db_type = None
        self._connection_string = None

    def initialize(self, db_path=None, use_postgresql=None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database (for local development)
            use_postgresql: If True, use PostgreSQL; if False, use SQLite.
                          If None, auto-detect based on environment variables.
        """
        # Auto-detect database type from environment if not specified
        if use_postgresql is None:
            use_postgresql = os.getenv('DB_TYPE', 'sqlite').lower() == 'postgresql'

        if use_postgresql:
            self._initialize_postgresql()
        else:
            self._initialize_sqlite(db_path)

    def _initialize_postgresql(self):
        """Initialize PostgreSQL connection using environment variables"""
        pg_host = os.getenv('PG_HOST', 'isoformgazer-db01.nygenome.org')
        pg_port = os.getenv('PG_PORT', '5432')
        pg_database = os.getenv('PG_DATABASE', 'isoformgazer')
        pg_user = os.getenv('PG_USER', 'isoformgazer_app')
        pg_password = os.getenv('PG_PASSWORD', '')

        if not pg_password:
            raise ValueError(
                "PostgreSQL password not found. Please set PG_PASSWORD environment variable."
            )
        self._connection_string = (
            f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
        )

        self.engine = create_engine(
            self._connection_string,
            poolclass=QueuePool,
            pool_size=10,           # Maintain 10 connections in the pool
            max_overflow=20,        # Allow up to 20 additional connections under load
            pool_pre_ping=True,     # Verify connections before using them
            pool_recycle=3600,      # Recycle connections after 1 hour
            echo=False,             # Optional: set to True for SQL query debugging
            connect_args={
                'connect_timeout': 10,
                'application_name': 'isoformgazer_app'
            }
        )

        self._db_type = 'postgresql'
        logger.info(f"Initialized PostgreSQL connection to {pg_host}:{pg_port}/{pg_database}")

        # Configure session factory
        self.SessionLocal = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        )

    def _initialize_sqlite(self, db_path=None):
        """Initialize SQLite connection for local development"""
        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, "data", "isoformgazer.db")

        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"SQLite database not found at {db_path}. "
                f"Please run setup_local_database() first."
            )

        self._connection_string = f"sqlite:///{db_path}"
        self.engine = create_engine(
            self._connection_string,
            poolclass=NullPool,  
            echo=False,  
            connect_args={
                'check_same_thread': False, 
                'timeout': 30
            }
        )

        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode = WAL")
            cursor.execute("PRAGMA synchronous = NORMAL")
            cursor.execute("PRAGMA cache_size = 10000")
            cursor.execute("PRAGMA temp_store = MEMORY")
            cursor.execute("PRAGMA optimize")
            cursor.close()

        self._db_type = 'sqlite'
        logger.info(f"Initialized SQLite connection to {db_path}")

        # Configure session factory
        self.SessionLocal = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        )

    @contextmanager
    def get_connection(self):
        """
        Get a raw database connection (context manager).
        Automatically commits on success and rolls back on error.

        Usage:
            with db_config.get_connection() as conn:
                result = conn.execute(text("SELECT * FROM table"))
        """
        connection = self.engine.connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def get_session(self):
        """
        Get an ORM session (context manager).
        Automatically commits on success and rolls back on error.

        Usage:
            with db_config.get_session() as session:
                result = session.query(Model).all()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def execute_query(self, query, params=None):
        """
        Execute a query and return results as pandas DataFrame.

        Args:
            query: SQL query string using :param style parameters
            params: Dictionary of query parameters

        Returns:
            pandas DataFrame with query results
        """
        if params:
            if self.is_postgresql():
                # Convert :param to %(param)s for psycopg2
                # Sort param names by length (longest first) to avoid partial replacements
                # e.g., :jid_10 should be replaced before :jid_1
                converted_query = query
                for param_name in sorted(params.keys(), key=len, reverse=True):
                    converted_query = converted_query.replace(f':{param_name}', f'%({param_name})s')
                query_to_execute = converted_query
            else:
                query_to_execute = query

            result = pd.read_sql_query(query_to_execute, self.engine, params=params)
            
        else:
            result = pd.read_sql_query(query, self.engine)

        return result

    def execute_raw(self, query, params=None):
        """
        Execute a raw SQL query (for INSERT, UPDATE, DELETE, etc.)

        Args:
            query: SQL query string
            params: Dictionary of query parameters

        Returns:
            Result proxy object
        """
        with self.get_connection() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))

        return result

    def quote_identifier(self, identifier):
        """
        Quote table/column names appropriately for the database type.
        PostgreSQL uses double quotes, SQLite can use double quotes too.

        Args:
            identifier: Table or column name to quote

        Returns:
            Quoted identifier string
        """
        return f'"{identifier}"'

    def get_db_type(self):
        """Return the current database type ('sqlite' or 'postgresql')"""
        return self._db_type

    def is_postgresql(self):
        """Check if using PostgreSQL"""
        return self._db_type == 'postgresql'

    def is_sqlite(self):
        """Check if using SQLite"""
        return self._db_type == 'sqlite'

    def close(self):
        """Close all database connections"""
        if self.SessionLocal:
            self.SessionLocal.remove()
        if self.engine:
            self.engine.dispose()
            logger.info(f"Closed {self._db_type} database connections")


db_config = DatabaseConfig()


def get_db_config():
    """Get the global database configuration instance"""
    return db_config


def initialize_database(db_path=None, use_postgresql=None):
    """
    Initialize the global database configuration.

    Args:
        db_path: Path to SQLite database (for local development)
        use_postgresql: If True, use PostgreSQL; if False, use SQLite.
                       If None, auto-detect based on environment variables.
    """
    db_config.initialize(db_path=db_path, use_postgresql=use_postgresql)
    return db_config
