#!/usr/bin/env python3
"""
Client for IsoformGazer Image Export Service

This module provides the client interface for the main IsoformGazer application
to communicate with the image export microservice running on an internal host.

Example usage:
    from export_client import ImageExportClient
    client = ImageExportClient(service_url="http://internal-host:5001")
    svg_bytes = client.export_figure(fig, format='svg', width=1200, height=800)
"""

import os
import time
import kaleido
import traceback
from typing import Optional, Dict, Any, List
import requests
import plotly.graph_objs as go

class ImageExportClient:
    """Client for the IsoformGazer Image Export Service"""

    def __init__(
        self,
        service_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        fallback_to_local: bool = True
    ):
        """
        Initialize the export client.

        Args:
            service_url: URL of the image export service (e.g., "http://internal-host:5001")
                        If None, will read from EXPORT_SERVICE_URL environment variable
            timeout: Request timeout in seconds (default: 60)
            max_retries: Maximum number of retry attempts (default: 3)
            fallback_to_local: If True, fall back to local kaleido if service is unavailable
        """
        self.service_url = service_url or os.getenv('EXPORT_SERVICE_URL')
        self.timeout = timeout
        self.max_retries = max_retries
        self.fallback_to_local = fallback_to_local
        self._service_available = None  # Cache for service availability

        if not self.service_url:
            print("WARNING: EXPORT_SERVICE_URL not configured. Image export will try to use local kaleido if available.")


    def health_check(self) -> bool:
        """
        Check if the export service is available.

        Returns:
            True if service is healthy, False otherwise
        """
        if not self.service_url:
            return False

        try:
            response = requests.get(
                f"{self.service_url}/health",
                timeout=5
            )
            healthy = response.status_code == 200
            self._service_available = healthy
            return healthy
        
        except Exception as e:
            print(f"Export service health check failed: {e}")
            self._service_available = False
            return False

    def export_figure(
        self,
        figure: go.Figure,
        format: str = 'svg',
        width: int = 1200,
        height: int = 800,
        scale: int = 1
    ) -> Optional[bytes]:
        """
        Export a Plotly figure to the specified format.

        Args:
            figure: Plotly Figure object to export
            format: Output format ('svg', 'png', 'pdf', 'webp', 'jpeg')
            width: Width in pixels
            height: Height in pixels
            scale: Scale factor for raster formats

        Returns:
            Image bytes if successful, None otherwise
        """
        if not self.service_url:
            return self._local_export(figure, format, width, height, scale)

        for attempt in range(self.max_retries):
            try:
                figure_dict = figure.to_dict()
                payload = {
                    'figure': figure_dict,
                    'format': format,
                    'width': width,
                    'height': height,
                    'scale': scale
                }
                response = requests.post(
                    f"{self.service_url}/export",
                    json=payload,
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    self._service_available = True
                    return response.content
                else:
                    error_msg = response.json().get('error', 'Unknown error')
                    print(f"Export service error (attempt {attempt + 1}/{self.max_retries}): {error_msg}")

            except requests.exceptions.Timeout:
                print(f"Export service timeout (attempt {attempt + 1}/{self.max_retries})")

            except requests.exceptions.ConnectionError:
                print(f"Cannot connect to export service (attempt {attempt + 1}/{self.max_retries})")
                self._service_available = False
                
            except Exception as e:
                print(f"Unexpected error during export (attempt {attempt + 1}/{self.max_retries}): {e}")
                traceback.print_exc()

            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)

        print(f"Failed to export via service after {self.max_retries} attempts")
        if self.fallback_to_local:
            print("Attempting local export as fallback...")
            return self._local_export(figure, format, width, height, scale)

        return None

    def _local_export(
        self,
        figure: go.Figure,
        format: str,
        width: int,
        height: int,
        scale: int
    ) -> Optional[bytes]:
        """
        Fallback to local Kaleido export.

        Args:
            figure: Plotly Figure object to export
            format: Output format
            width: Width in pixels
            height: Height in pixels
            scale: Scale factor

        Returns:
            Image bytes if successful, None otherwise
        """
        try:
            image_bytes = figure.to_image(
                format=format,
                width=width,
                height=height,
                scale=scale
            )
            print(f"Successfully exported using local kaleido (version {kaleido.__version__})")
            return image_bytes
        
        except ImportError:
            print("ERROR: Kaleido not available locally and remote export service failed")
            return None
        
        except Exception as e:
            print(f"Local export failed: {e}")
            traceback.print_exc()
            return None

    def batch_export(
        self,
        export_specs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Export multiple figures at once.

        Args:
            export_specs: List of export specifications, each containing:
                - figure: Plotly Figure object
                - format: Output format
                - width: Width in pixels (optional)
                - height: Height in pixels (optional)
                - filename: Desired filename (optional)

        Returns:
            List of results, each containing success status and either data or error
        """
        if not self.service_url:
            print("Batch export requires export service URL")
            return []

        try:
            payload_exports = []
            for spec in export_specs:
                figure = spec.get('figure')
                if not isinstance(figure, go.Figure):
                    continue

                payload_exports.append({
                    'figure': figure.to_dict(),
                    'format': spec.get('format', 'svg'),
                    'width': spec.get('width', 1200),
                    'height': spec.get('height', 800),
                    'scale': spec.get('scale', 1),
                    'filename': spec.get('filename', 'plot')
                })

            response = requests.post(
                f"{self.service_url}/batch-export",
                json={'exports': payload_exports},
                timeout=self.timeout * len(payload_exports)
            )

            if response.status_code == 200:
                return response.json()['results']
            else:
                error_msg = response.json().get('error', 'Unknown error')
                print(f"Batch export error: {error_msg}")
                return []

        except Exception as e:
            print(f"Batch export failed: {e}")
            traceback.print_exc()
            return []

    def is_available(self) -> bool:
        """
        Check if export service is available (uses cached result if recent).

        Returns:
            True if service is available, False otherwise
        """
        if self._service_available is None:
            return self.health_check()
        return self._service_available


# Singleton instance for convenience
_default_client = None


def get_export_client() -> ImageExportClient:
    """
    Get the default export client instance.

    Returns:
        ImageExportClient instance
    """
    global _default_client
    if _default_client is None:
        _default_client = ImageExportClient()
    return _default_client


def export_figure(
    figure: go.Figure,
    format: str = 'svg',
    width: int = 1200,
    height: int = 800
) -> Optional[bytes]:
    """
    Convenience function to export a figure using the default client.

    Args:
        figure: Plotly Figure object
        format: Output format
        width: Width in pixels
        height: Height in pixels

    Returns:
        Image bytes if successful, None otherwise
    """
    client = get_export_client()
    return client.export_figure(figure, format, width, height)
