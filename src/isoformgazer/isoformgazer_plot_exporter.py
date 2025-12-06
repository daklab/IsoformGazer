#!/usr/bin/env python3

"""
This script allows users to export plots from IsoformGazer for multiple genes in bulk, with support for:
- Multiple export formats (PNG, PDF, SVG, interactive HTML)
- Customizable plot styling options
- Data from CSV files or gene name lists

Example usage:
    python3 isoformgazer_plot_exporter.py --genes "FUS,TARDBP,RBFOX2" --format png --output-dir ./exports
    python3 isoformgazer_plot_exporter.py --gene-file genes.txt --format pdf --output-dir ./exports
    python3 isoformgazer_plot_exporter.py --genes "FUS" --format svg --output-dir ./exports
    python3 isoformgazer_plot_exporter.py --genes "FUS" --format html --output-dir ./exports --data-dir /path/to/data
"""

import argparse
import os
import sys
import traceback
import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
import plotly.graph_objs as go
import plotly.io as pio
from PIL import Image

from isoform_utils import (
    create_transcript_structure_plot,
    create_isoform_expression_clustergram,
    process_transcript_structure,
    load_expression_data,
)
from junction_utils import (
    create_gene_clustergram,
    create_junction_exon_visualization,
    process_gene_atse_data,
)

from app import setup_local_database

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


###################################################################
# DATABASE SETUP (copied from app.py)
###################################################################
def check_database_status(db_path: str) -> bool:
    """
    Checks if database exists and is valid.
    """
    if Path(db_path).exists():
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("SELECT 1 FROM isoforms LIMIT 1")
            conn.close()
            return True
        except:
            return False
    return False


class BulkPlotExporter:
    """Main class for bulk exporting plots from IsoformGazer data using SQLite database."""
    def __init__(
        self,
        data_dir: str = None,
        output_dir: str = "./isoformgazer_exports",
        format: str = "png",
        plot_types: List[str] = None,
        skip_missing_genes: bool = True,
        verbose: bool = True,
        title_font_size: int = 16,
        axis_font_size: int = 12,
        clustergram_height: int = 700,
        colorscale: str = "Viridis",
        distance_metric: str = "euclidean",
        linkage_method: str = "complete",
        show_gridlines: bool = False,
        gridline_color: str = "#ffffff",
        collapse_mode: str = "tissue",
        color_by_abundance: bool = False,
        exon_color: str = '#2E86C1',
        junction_color: str = '#85929E',
        structure_type: str = "both",
        transparent_background: bool = False,
        show_celltype_labels: bool = True,
        export_width: int = 1200,
        export_height: int = 800,
    ):
        """
        Initializes bulk plot exporter.

        Args:
            data_dir: Path to data directory. Defaults to src/isoformgazer/data
            output_dir: Directory to save exported plots
            format: Export format - 'png', 'pdf', or 'html'
            plot_types: List of plot types to export ('structure', 'isoform', 'junction')
            skip_missing_genes: If True, skip genes with missing data instead of failing
            verbose: Print progress messages
            title_font_size: Font size for plot titles and legend titles
            axis_font_size: Font size for axis labels
            clustergram_height: Height of clustergram plots in pixels
            colorscale: Plotly colorscale name (e.g., 'Viridis', 'Plasma', 'Blues')
            distance_metric: Distance metric for clustering ('euclidean', 'correlation', 'cosine')
            linkage_method: Linkage method for clustering ('complete', 'single', 'average', 'weighted')
            show_gridlines: Whether to show gridlines in clustergrams
            gridline_color: Color of gridlines
            collapse_mode: How to collapse samples ('tissue' or 'replicate')
            color_by_abundance: Color structure plot transcripts by TPM
            exon_color: Color for exons in structure plot
            junction_color: Color for junctions in structure plot (default: #85929E)
            structure_type: Type of structure plot ('both' for transcripts+junctions, 'transcripts' for transcripts only)
            transparent_background: If True, use transparent background; if False (default), use white background
            show_celltype_labels: If True (default), show cell type labels on x-axis of junction clustergram
        """
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data")

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.format = format.lower()
        self.plot_types = plot_types or ["structure", "isoform", "junction"]
        self.skip_missing_genes = skip_missing_genes
        self.verbose = verbose
        self.title_font_size = title_font_size
        self.axis_font_size = axis_font_size
        self.clustergram_height = clustergram_height
        self.colorscale = colorscale
        self.distance_metric = distance_metric
        self.linkage_method = linkage_method
        self.show_gridlines = show_gridlines
        self.gridline_color = gridline_color
        self.collapse_mode = collapse_mode
        self.color_by_abundance = color_by_abundance
        self.exon_color = exon_color
        self.junction_color = junction_color
        self.structure_type = structure_type.lower()
        self.transparent_background = transparent_background
        self.show_celltype_labels = show_celltype_labels
        self.export_width = export_width
        self.export_height = export_height

        self._validate_inputs()

        db_file_path = Path(self.data_dir) / "isoformgazer.db"
        db_existed = db_file_path.exists()

        self._log("Setting up database from:", self.data_dir)
        self.db_path = setup_local_database(data_dir=str(self.data_dir))

        if db_existed:
            self._log("Using existing database at:", self.db_path)
        else:
            self._log("Created new database at:", self.db_path)

    def _validate_inputs(self):
        """
        Validates input params prior to plotting.
        """
        if self.format not in ["png", "pdf", "svg", "html"]:
            raise ValueError(f"Invalid format: {self.format}. Must be 'png', 'pdf', 'svg', or 'html'")

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # For PDF, require at least one plot type (will combine into single PDF)
        if self.format == "pdf" and not self.plot_types:
            raise ValueError("At least one plot type required for PDF export")

    def _log(self, *args, level: str = "info"):
        """Print log messages."""
        if self.verbose:
            prefix = {
                "info": "[INFO]",
                "warning": "[WARNING]",
                "error": "[ERROR]",
                "success": "[SUCCESS]",
            }.get(level, "[LOG]")
            print(f"{prefix} {' '.join(str(arg) for arg in args)}")

    def parse_genes(self, genes_str: str = None, gene_file: str = None) -> List[str]:
        """
        Parse gene names from string or file.

        Args:
            genes_str: Comma-separated string of gene names
            gene_file: Path to file with one gene name per line

        Returns:
            List of gene names
        """
        genes = []

        if genes_str:
            genes.extend([g.strip() for g in genes_str.split(",") if g.strip()])

        if gene_file:
            gene_file = Path(gene_file)
            if not gene_file.exists():
                raise FileNotFoundError(f"Gene file not found: {gene_file}")
            with open(gene_file, "r") as f:
                genes.extend([line.strip() for line in f if line.strip()])

        if not genes:
            raise ValueError("No genes provided")

        seen = set()
        unique_genes = []
        for gene in genes:
            if gene not in seen:
                seen.add(gene)
                unique_genes.append(gene)

        return unique_genes

    def _get_gene_data(self, gene_name: str) -> Tuple[bool, dict]:
        """
        Check if gene exists in database and return gene-specific data.

        Args:
            gene_name: Name of gene to check

        Returns:
            Tuple of (gene_exists, gene_data_dict)
        """
        try:
            conn = sqlite3.connect(self.db_path)

            # Check if gene exists in isoforms table
            gene_check = "SELECT 1 FROM isoforms WHERE gene_name = ? LIMIT 1"
            result = pd.read_sql_query(gene_check, conn, params=[gene_name])

            if result.empty:
                conn.close()
                return False, {}

            tpm_query = "SELECT * FROM tpm_data WHERE gene_name = ?"
            ratio_query = "SELECT * FROM ratio_data WHERE gene_name = ?"
            log_tpm_query = "SELECT * FROM log_tpm_data WHERE gene_name = ?"

            tpm_df = pd.read_sql_query(tpm_query, conn, params=[gene_name])
            ratio_df = pd.read_sql_query(ratio_query, conn, params=[gene_name])
            log_tpm_df = pd.read_sql_query(log_tpm_query, conn, params=[gene_name])

            conn.close()

            return True, {
                "tpm": tpm_df,
                "ratio": ratio_df,
                "log_tpm": log_tpm_df,
            }
        except Exception as e:
            self._log(f"Error getting data for gene {gene_name}: {e}", level="warning")
            return False, {}

    def _apply_font_sizes(self, fig: go.Figure) -> go.Figure:
        """
        Apply font size customizations to figure.
        """
        fig.update_layout(title={"font": {"size": self.title_font_size}})
        fig.update_xaxes(tickfont={"size": self.axis_font_size})
        fig.update_yaxes(tickfont={"size": self.axis_font_size})
        return fig

    def _apply_styling(self, fig: go.Figure) -> go.Figure:
        """
        Apply background and styling customizations to figure.
        """
        if self.transparent_background:
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
        else:
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
        return fig

    def export_plots(self, genes: List[str]) -> dict:
        """
        Export plots for a list of genes.

        Args:
            genes: List of gene names to export

        Returns:
            Dictionary with export statistics
        """
        stats = {
            "total_genes": len(genes),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "failed_genes": [],
        }

        if self.format == "pdf":
            return self._export_pdf_batch(genes, stats)
        else:
            return self._export_individual(genes, stats)

    def _export_individual(self, genes: List[str], stats: dict) -> dict:
        """
        Export individual files (PNG or HTML) for each gene.
        """
        self._log(f"Exporting {len(genes)} genes as {self.format.upper()} files...")

        for gene in genes:
            gene_exists, gene_data = self._get_gene_data(gene)

            if not gene_exists:
                self._log(f"Gene not found: {gene}", level="warning")
                if self.skip_missing_genes:
                    stats["skipped"] += 1
                    continue
                else:
                    stats["failed"] += 1
                    stats["failed_genes"].append(gene)
                    continue

            try:
                self._log(f"Exporting plots for gene: {gene}")
                gene_dir = self.output_dir / gene
                gene_dir.mkdir(parents=True, exist_ok=True)

                for plot_type in self.plot_types:
                    try:
                        fig = self._create_plot(gene, plot_type, gene_data)
                        if fig is not None:
                            self._save_figure(fig, gene, plot_type, gene_dir)
                    except Exception as e:
                        self._log(
                            f"Error creating {plot_type} plot for {gene}: {e}",
                            level="warning",
                        )
                        traceback.print_exc()

                stats["successful"] += 1
                self._log(f"Successfully exported all {gene} plots.", level="success")

            except Exception as e:
                self._log(f"Error exporting plots for {gene}: {e}", level="error")
                stats["failed"] += 1
                stats["failed_genes"].append(gene)
                traceback.print_exc()

        return stats

    def _export_pdf_batch(self, genes: List[str], stats: dict) -> dict:
        """
        Export plots to single or multiple PDF files (one per plot type).
        """
        self._log(f"Exporting {len(genes)} genes to PDF format...")

        pdf_images = {plot_type: [] for plot_type in self.plot_types}
        temp_files = []

        for gene in genes:
            gene_exists, gene_data = self._get_gene_data(gene)

            if not gene_exists:
                self._log(f"Gene not found: {gene}", level="warning")
                if self.skip_missing_genes:
                    stats["skipped"] += 1
                    continue
                else:
                    stats["failed"] += 1
                    stats["failed_genes"].append(gene)
                    continue

            try:
                self._log(f"Processing gene: {gene}")

                for plot_type in self.plot_types:
                    try:
                        fig = self._create_plot(gene, plot_type, gene_data)
                        if fig is not None:
                            # Export to temporary PNG
                            temp_png = f"/tmp/{gene}_{plot_type}_temp.png"
                            fig.write_image(temp_png, width=1200, height=800)

                            # Open and convert to RGB (required for PDF)
                            image = Image.open(temp_png)
                            if image.mode != 'RGB':
                                image = image.convert('RGB')

                            pdf_images[plot_type].append(image)
                            temp_files.append(temp_png)

                            self._log(f"  Prepared {plot_type} plot for {gene}")

                    except Exception as e:
                        self._log(
                            f"Error creating {plot_type} plot for {gene}: {e}",
                            level="warning",
                        )

                stats["successful"] += 1
                self._log(f"Successfully exported all {gene} plots.", level="success")

            except Exception as e:
                self._log(f"Error processing {gene}: {e}", level="error")
                stats["failed"] += 1
                stats["failed_genes"].append(gene)

        for plot_type, images in pdf_images.items():
            if images:
                try:
                    if plot_type == "structure":
                        pdf_filename = "isoformgazer_structure_plots.pdf"
                    elif plot_type == "isoform":
                        pdf_filename = "isoformgazer_isoform_clustergrams.pdf"
                    elif plot_type == "junction":
                        pdf_filename = "isoformgazer_junctions_clustergrams.pdf"
                    else:
                        pdf_filename = f"isoformgazer_{plot_type}_plots.pdf"

                    pdf_path = self.output_dir / pdf_filename
                    # Single PDF with one image for gene per page
                    images[0].save(
                        str(pdf_path),
                        save_all=True,
                        append_images=images[1:] if len(images) > 1 else [],
                        duration=0,
                        loop=0
                    )
                    self._log(f"Saved {plot_type} plots to: {pdf_path}", level="success")
                except Exception as e:
                    self._log(f"Error saving PDF for {plot_type}: {e}", level="error")

        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                self._log(f"Warning: Could not delete temporary file {temp_file}: {e}", level="warning")

        return stats

    def _create_plot(self, gene_name: str, plot_type: str, gene_data: dict) -> Optional[go.Figure]:
        """
        Create a plot for the given gene and plot type.

        Args:
            gene_name: Name of the gene
            plot_type: Type of plot ('structure', 'isoform', 'junction')
            gene_data: Dictionary with gene data from database

        Returns:
            Plotly figure object or None if plot creation failed
        """
        try:
            if plot_type == "structure":
                # Choose structure plot type based on user preference
                if self.structure_type == "transcripts":
                    transcript_data = process_transcript_structure(self.db_path, gene_name, filtered_ids=[])
                    fig = create_transcript_structure_plot(
                        db_path=self.db_path,
                        transcript_data=transcript_data,
                        gene_name=gene_name,
                        height=None,
                        show_y_labels=True,
                        exon_color=self.exon_color,
                        color_by_abundance=self.color_by_abundance,
                        colorscale=self.colorscale,
                    )

                else:
                    # Default: transcripts + junctions plot (junction_exon_visualization)
                    gene_data_dict = process_gene_atse_data(gene_name, self.db_path, filtered_junction_ids=None)

                    if 'error' in gene_data_dict:
                        self._log(f"No junction/transcript data for {gene_name}", level="warning")
                        return None

                    fig = create_junction_exon_visualization(
                        gene_data=gene_data_dict,
                        height=None,
                        show_y_labels=True,
                        exon_color=self.exon_color,
                        junction_color=self.junction_color,
                        color_by_abundance=self.color_by_abundance,
                        db_path=self.db_path,
                        colorscale=self.colorscale,
                    )

            elif plot_type == "isoform":

                # Load expression data using the proper function that handles joining with PSL data
                tpm_data = load_expression_data(self.db_path, gene_name, data_type='tpm')
                ratio_data = load_expression_data(self.db_path, gene_name, data_type='ratio')
                log_tpm_data = load_expression_data(self.db_path, gene_name, data_type='log_tpm')

                if tpm_data.empty or ratio_data.empty or log_tpm_data.empty:
                    return None

                fig = create_isoform_expression_clustergram(
                    tpm_data=tpm_data,
                    ratio_data=ratio_data,
                    log_tpm_data=log_tpm_data,
                    gene_name=gene_name,
                    height=self.clustergram_height,
                    colorscale=self.colorscale,
                    data_type="Ratio",
                    show_tables="hide",
                    show_labels=True,
                    collapse_mode=self.collapse_mode,
                    distance_metric=self.distance_metric,
                    linkage_method=self.linkage_method,
                    show_gridlines=self.show_gridlines,
                    gridline_color=self.gridline_color,
                    db_path=self.db_path,
                )

            elif plot_type == "junction":
                fig = create_gene_clustergram(
                    db_path=self.db_path,
                    gene_name=gene_name,
                    height=self.clustergram_height,
                    colorscale=self.colorscale,
                    show_tables="hide",
                    show_celltype_labels=self.show_celltype_labels,
                    distance_metric=self.distance_metric,
                    linkage_method=self.linkage_method,
                    show_gridlines=self.show_gridlines,
                )
            else:
                self._log(f"Unknown plot type: {plot_type}", level="warning")
                return None

            if fig is not None:
                fig = self._apply_font_sizes(fig)
                fig = self._apply_styling(fig)

            return fig

        except Exception as e:
            self._log(f"Error creating {plot_type} plot for {gene_name}: {e}", level="warning")
            traceback.print_exc()
            return None

    def _save_figure(self, fig: go.Figure, gene_name: str, plot_type: str, output_dir: Path):
        """
        Save figure in the specified format.

        Args:
            fig: Plotly figure object
            gene_name: Name of the gene
            plot_type: Type of plot ('structure', 'isoform', 'junction')
            output_dir: Directory to save the file
        """
        if plot_type == "structure":
            filename = f"{gene_name}_structure_plot.{self.format}"
        elif plot_type == "isoform":
            filename = f"{gene_name}_isoform_clustergram.{self.format}"
        elif plot_type == "junction":
            filename = f"{gene_name}_junction_clustergram.{self.format}"
        else:
            filename = f"{gene_name}_{plot_type}_plot.{self.format}"

        filepath = output_dir / filename

        try:
            if self.format == "png":
                fig.write_image(str(filepath), width=self.export_width, height=self.export_height)
            elif self.format == "svg":
                fig.write_image(str(filepath), format="svg", width=self.export_width, height=self.export_height)
            elif self.format == "html":
                fig.write_html(str(filepath))
            else:
                self._log(f"Unknown format: {self.format}", level="warning")
                return

            self._log(f"  Saved: {filepath}")

        except Exception as e:
            self._log(f"Error saving {plot_type} plot to {filepath}: {e}", level="error")
            raise


def main():
    """Main entry point for the bulk export script."""
    parser = argparse.ArgumentParser(
    description="""
        IsoformGazer Bulk Gene Plot Export Script

        Exports plots for multiple genes at once with customizable settings. 
        Data is loaded from the local isoformgazer.db that is automatically created from the prerelease data files.

        The following export formats are supported (choose one with --format):
        - PNG: Individual image files for each plot type and gene (1200x800px, raster)
        - SVG: Scalable vector graphics files for each plot type and gene (vector, editable)
        - HTML: Interactive Plotly HTML files for each plot type and gene (interactive, zoomable)
        - PDF: Combined PDF files (one per plot type) with genes on separate pages

        Available Plot Types (--plot-types, comma-separated):
        - structure: Transcript structure plot (either with transcripts and junctions, or just transcripts) 
        - isoform: Isoform expression clustergram showing ENCODE4 bulk long-read RNA-seq expression data
        - junction: Junction expression clustergram showing cell type PSI patterns for Tabula Sapiens 2.0 
                    pseudobulked smart-seq2 single cell and Allen Brain single nuclei for brain data

        Quick Examples:
            # Export PNG plots for multiple genes
            python3 isoformgazer_plot_exporter.py --genes "FUS,TARDBP,RBFOX2" --format png --output-dir ./exports

            # Export SVG plots for a single gene
            python3 isoformgazer_plot_exporter.py --genes "FUS" --format svg --output-dir ./svg_exports

            # Export all formats from a gene file
            python3 isoformgazer_plot_exporter.py --gene-file genes.txt --format html --output-dir ./html_plots

            # Export with custom clustering and styling
            python3 isoformgazer_plot_exporter.py --genes "FUS" --format png --distance-metric correlation \\
                --colorscale Plasma --title-font-size 18 --output-dir ./styled_exports

        Customization Options:

            Font Sizes:
                --title-font-size INT          Font size for plot titles (default: 16)
                --axis-font-size INT           Font size for axis labels (default: 12)

            Clustering & Distance Metrics:
                --distance-metric STR          euclidean, correlation, or cosine (default: euclidean)
                --linkage-method STR           complete, single, average, or weighted (default: complete)
                --collapse-mode STR            tissue or replicate (default: tissue)

            Clustergram Styling:
                --colorscale STR               Plotly colorscale name, e.g. Viridis, Plasma, Blues, Reds (default: Viridis)
                --clustergram-height INT       Height in pixels for clustergram plots (default: 700)
                --show-gridlines               Add gridlines to heatmaps (default: false)
                --gridline-color STR           Gridline color in hex format (default: #ffffff)

            Structure Plot Styling:
                --exon-color STR               Color for exons in hex format (default: #2E86C1)
                --junction-color STR           Color for junctions in hex format (default: #85929E)
                --color-by-abundance           Color transcripts/junctions by abundance (TPM/PSI) (default: false)
                --structure-type STR           both (transcripts+junctions) or transcripts only (default: both)

            Junction Clustergram Options:
                --show-celltype-labels         Show cell type labels on junction clustergram x-axis (default: true)
                --hide-celltype-labels         Hide cell type labels on junction clustergram x-axis (default: false)

            Export Dimensions (PNG/SVG only):
                --export-width INT             Width of exported plots in pixels (default: 1200)
                --export-height INT            Height of exported plots in pixels (default: 800)

            General Options:
                --transparent-background       Use transparent background instead of white (default: false)
                --skip-missing                 Skip genes not found in database instead of failing (default: true)
                --verbose                      Print detailed progress messages (default: true)

        Output Structure:

            - For PNG/SVG/HTML Formats, each gene gets its own directory with individual plot files:
                    exports/
                    ├── FUS/
                    │   ├── FUS_structure_plot.{ext}
                    │   ├── FUS_isoform_clustergram.{ext}
                    │   └── FUS_junction_clustergram.{ext}
                    └── TARDBP/
                        ├── TARDBP_structure_plot.{ext}
                        ├── TARDBP_isoform_clustergram.{ext}
                        └── TARDBP_junction_clustergram.{ext}

            For PDF Format, combined PDF files are created for each plot type (one file per plot type, with one page per gene):
                    exports/
                    ├── isoformgazer_structure_plots.pdf
                    ├── isoformgazer_isoform_clustergrams.pdf
                    └── isoformgazer_junctions_clustergrams.pdf
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ############################
    # Gene input args
    ############################
    gene_input = parser.add_mutually_exclusive_group(required=True)
    gene_input.add_argument(
        "--genes",
        help='Comma-separated list of gene names (e.g., "FUS,TARDBP,RBFOX2")',
    )
    gene_input.add_argument(
        "--gene-file",
        help="Path to file with one gene name per line",
    )

    ############################
    # Output arguments
    ############################
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg", "html"],
        default="png",
        help="Export format (default: png)",
    )
    parser.add_argument(
        "--output-dir",
        default="./exports",
        help="Output directory for exported plots (default: ./exports)",
    )
    parser.add_argument(
        "--data-dir",
        help="Path to data directory containing TSV files (default: src/isoformgazer/data)",
    )
    parser.add_argument(
        "--plot-types",
        default="structure,isoform,junction",
        help="Comma-separated list of plot types to export (default: structure,isoform,junction)",
    )

    ############################
    # Styling arguments
    ############################
    parser.add_argument(
        "--title-font-size",
        type=int,
        default=16,
        help="Font size for plot titles and legend titles (default: 16)",
    )
    parser.add_argument(
        "--axis-font-size",
        type=int,
        default=12,
        help="Font size for axis labels (default: 12)",
    )
    parser.add_argument(
        "--clustergram-height",
        type=int,
        default=700,
        help="Height of clustergram plots in pixels (default: 700)",
    )
    parser.add_argument(
        "--colorscale",
        default="Viridis",
        help="Plotly colorscale name (default: Viridis)",
    )
    parser.add_argument(
        "--exon-color",
        default="#2E86C1",
        help="Color for exons in structure plot (default: #EDAE49)",
    )
    parser.add_argument(
        "--junction-color",
        default="#85929E",
        help="Color for junctions in structure plot (default: #85929E)",
    )

    ############################
    # Clustering arguments
    ############################
    parser.add_argument(
        "--distance-metric",
        choices=["euclidean", "correlation", "cosine"],
        default="euclidean",
        help="Distance metric for clustering (default: euclidean)",
    )
    parser.add_argument(
        "--linkage-method",
        choices=["complete", "single", "average", "weighted"],
        default="complete",
        help="Linkage method for hierarchical clustering (default: complete)",
    )
    parser.add_argument(
        "--collapse-mode",
        choices=["tissue", "replicate"],
        default="tissue",
        help="How to collapse samples (default: tissue)",
    )

    ############################
    # Display arguments
    ############################
    parser.add_argument(
        "--show-gridlines",
        action="store_true",
        help="Add gridlines to clustergrams",
    )
    parser.add_argument(
        "--gridline-color",
        default="#ffffff",
        help="Color of gridlines (default: #ffffff)",
    )
    parser.add_argument(
        "--color-by-abundance",
        action="store_true",
        help="Color structure plot transcripts by average TPM",
    )
    parser.add_argument(
        "--structure-type",
        choices=["both", "transcripts"],
        default="both",
        help="Type of structure plot: 'both' for transcripts+junctions (default), 'transcripts' for transcripts only",
    )
    parser.add_argument(
        "--transparent-background",
        action="store_true",
        help="Use transparent background for plots (default: white background)",
    )
    parser.add_argument(
        "--show-celltype-labels",
        dest="show_celltype_labels",
        action="store_true",
        default=True,
        help="Show cell type labels on x-axis of junction clustergram (default: True)",
    )
    parser.add_argument(
        "--hide-celltype-labels",
        dest="show_celltype_labels",
        action="store_false",
        help="Hide cell type labels on x-axis of junction clustergram",
    )

    ############################
    # Export dimensions args
    ############################
    parser.add_argument(
        "--export-width",
        type=int,
        default=1200,
        help="Width of exported PNG/SVG plots in pixels (default: 1200)",
    )
    parser.add_argument(
        "--export-height",
        type=int,
        default=800,
        help="Height of exported PNG/SVG plots in pixels (default: 800)",
    )

    ############################
    # General options args
    ############################
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        default=True,
        help="Skip genes with missing data instead of failing (default: True)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress messages (default: True)",
    )

    args = parser.parse_args()

    try:
        exporter = BulkPlotExporter(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            format=args.format,
            plot_types=[p.strip() for p in args.plot_types.split(",")],
            skip_missing_genes=args.skip_missing,
            verbose=args.verbose,
            title_font_size=args.title_font_size,
            axis_font_size=args.axis_font_size,
            clustergram_height=args.clustergram_height,
            colorscale=args.colorscale,
            distance_metric=args.distance_metric,
            linkage_method=args.linkage_method,
            show_gridlines=args.show_gridlines,
            gridline_color=args.gridline_color,
            collapse_mode=args.collapse_mode,
            color_by_abundance=args.color_by_abundance,
            exon_color=args.exon_color,
            junction_color=args.junction_color,
            structure_type=args.structure_type,
            transparent_background=args.transparent_background,
            show_celltype_labels=args.show_celltype_labels,
            export_width=args.export_width,
            export_height=args.export_height,
        )

        genes = exporter.parse_genes(genes_str=args.genes, gene_file=args.gene_file)
        print(f"\nFound {len(genes)} genes to export: {', '.join(genes[:5])}" + ("..." if len(genes) > 5 else ""))

        print(f"\nStarting export as {args.format.upper()}...\n")
        stats = exporter.export_plots(genes)

        if stats["failed_genes"]:
            print(f"Failed genes:     {', '.join(stats['failed_genes'])}")

    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
