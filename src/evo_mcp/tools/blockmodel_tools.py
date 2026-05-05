# SPDX-FileCopyrightText: 2026 Bentley Systems, Incorporated
#
# SPDX-License-Identifier: Apache-2.0

"""
MCP tools for querying and downloading block models as CSV.
"""

import logging
import os
from pathlib import Path
from typing import Optional
from uuid import UUID

from evo_mcp.context import evo_context, ensure_initialized

logger = logging.getLogger(__name__)


def register_blockmodel_tools(mcp):
    """Register block model tools with the FastMCP server."""

    @mcp.tool()
    async def query_block_model(
        workspace_id: str,
        block_model_id: str,
        output_filename: str = "",
        columns: list[str] = ["*"],
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        use_column_names: bool = True,
        include_coordinates: bool = True,
        exclude_null_rows: bool = True,
    ) -> dict:
        """Query a block model and download the results as a CSV file for analysis.

        Queries block model data from Evo and saves it locally as a CSV file.
        Supports filtering by columns and spatial bounding box.

        Args:
            workspace_id: Workspace UUID containing the block model
            block_model_id: Block model UUID to query
            output_filename: Output CSV filename (default: auto-generated from block model name)
            columns: List of column names to include (default: ["*"] for all columns)
            x_min: Minimum X for bounding box filter (optional)
            x_max: Maximum X for bounding box filter (optional)
            y_min: Minimum Y for bounding box filter (optional)
            y_max: Maximum Y for bounding box filter (optional)
            z_min: Minimum Z for bounding box filter (optional)
            z_max: Maximum Z for bounding box filter (optional)
            use_column_names: Use column names instead of UUIDs as headers (default: True)
            include_coordinates: Include x, y, z coordinate columns (default: True)
            exclude_null_rows: Exclude rows where all attribute values are null (default: True)

        Returns:
            Dictionary with file path, row count, column count, and column names.
        """
        await ensure_initialized()

        from evo.blockmodels import BlockModelAPIClient
        from evo.blockmodels.endpoints.models import (
            BBoxXYZ,
            FloatRange,
            ColumnHeaderType,
            GeometryColumns,
        )
        from evo.common.utils import Cache

        # Get workspace environment and create block model client
        workspace = await evo_context.workspace_client.get_workspace(UUID(workspace_id))
        environment = workspace.get_environment()
        cache = Cache(root=evo_context.cache_path, mkdir=True)
        service_client = BlockModelAPIClient(environment, evo_context.connector, cache)

        bm_id = UUID(block_model_id)

        # Build bounding box filter if any spatial params provided
        bbox = None
        if all(v is not None for v in [x_min, x_max, y_min, y_max, z_min, z_max]):
            bbox = BBoxXYZ(
                x_minmax=FloatRange(min=x_min, max=x_max),
                y_minmax=FloatRange(min=y_min, max=y_max),
                z_minmax=FloatRange(min=z_min, max=z_max),
            )

        # Determine column header type
        column_headers = ColumnHeaderType.name if use_column_names else ColumnHeaderType.id

        # Determine geometry columns
        geometry_columns = GeometryColumns.coordinates if include_coordinates else GeometryColumns.indices

        # Query the block model
        table = await service_client.query_block_model_as_table(
            bm_id=bm_id,
            columns=columns,
            bbox=bbox,
            column_headers=column_headers,
            geometry_columns=geometry_columns,
            exclude_null_rows=exclude_null_rows,
        )

        # Convert to pandas DataFrame
        df = table.to_pandas()

        # Determine output filename
        if not output_filename:
            bm_info = await service_client.get_block_model(bm_id)
            safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in bm_info.name)
            output_filename = f"{safe_name}.csv"

        # Save to the configured local data directory
        local_data_dir = os.environ.get("LOCAL_DATA_DIR", str(Path.cwd() / "data"))
        output_dir = Path(local_data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename

        df.to_csv(output_path, index=False)

        logger.info(f"Block model exported to CSV: {output_path} ({len(df)} rows)")

        return {
            "file_path": str(output_path),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "message": f"Block model downloaded to {output_path}",
        }

    @mcp.tool()
    async def list_block_models(
        workspace_id: str,
    ) -> list[dict]:
        """List all block models in a workspace.

        Args:
            workspace_id: Workspace UUID to list block models from

        Returns:
            List of block model summaries with id, name, description, and CRS.
        """
        await ensure_initialized()

        from evo.blockmodels import BlockModelAPIClient
        from evo.common.utils import Cache

        workspace = await evo_context.workspace_client.get_workspace(UUID(workspace_id))
        environment = workspace.get_environment()
        cache = Cache(root=evo_context.cache_path, mkdir=True)
        service_client = BlockModelAPIClient(environment, evo_context.connector, cache)

        block_models = await service_client.list_block_models()

        return [
            {
                "id": str(bm.id),
                "name": bm.name,
                "description": bm.description,
                "coordinate_reference_system": bm.coordinate_reference_system,
                "created_at": bm.created_at.isoformat() if bm.created_at else None,
            }
            for bm in block_models
        ]

    @mcp.tool()
    async def block_model_statistics(
        file_path: str,
    ) -> dict:
        """Compute statistics on a downloaded block model CSV file.

        Provides spatial extents, block size distribution, numeric column
        statistics (mean, median, std, percentiles), categorical column
        value counts, null counts, and volume-weighted grade estimates.

        Args:
            file_path: Path to the block model CSV file (as returned by query_block_model)

        Returns:
            Dictionary with comprehensive statistics about the block model.
        """
        import pandas as pd
        import numpy as np

        df = pd.read_csv(file_path)

        stats = {
            "file_path": file_path,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
        }

        # Null counts
        stats["null_counts"] = {col: int(df[col].isna().sum()) for col in df.columns}

        # Spatial extents (if x, y, z present)
        coord_cols = [c for c in ["x", "y", "z"] if c in df.columns]
        if coord_cols:
            stats["spatial_extents"] = {}
            for col in coord_cols:
                stats["spatial_extents"][col] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "range": float(df[col].max() - df[col].min()),
                }

        # Block size distribution (if dx, dy, dz present)
        size_cols = [c for c in ["dx", "dy", "dz"] if c in df.columns]
        if size_cols:
            stats["block_sizes"] = {}
            for col in size_cols:
                stats["block_sizes"][col] = {
                    str(k): int(v) for k, v in df[col].value_counts().sort_index().items()
                }

        # Numeric column statistics
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in coord_cols + size_cols]
        if numeric_cols:
            stats["numeric_statistics"] = {}
            for col in numeric_cols:
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                stats["numeric_statistics"][col] = {
                    "count": int(len(series)),
                    "null_count": int(df[col].isna().sum()),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "p25": float(series.quantile(0.25)),
                    "p75": float(series.quantile(0.75)),
                    "p90": float(series.quantile(0.90)),
                    "p95": float(series.quantile(0.95)),
                    "p99": float(series.quantile(0.99)),
                }

        # Categorical column value counts
        cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns]
        if cat_cols:
            stats["categorical_statistics"] = {}
            for col in cat_cols:
                vc = df[col].value_counts()
                stats["categorical_statistics"][col] = {
                    "unique_values": int(vc.nunique()),
                    "top_values": {str(k): int(v) for k, v in vc.head(20).items()},
                }

        # Volume and volume-weighted statistics (if dx, dy, dz all present)
        if all(c in df.columns for c in ["dx", "dy", "dz"]):
            df["_volume"] = df["dx"] * df["dy"] * df["dz"]
            stats["volume"] = {
                "total_volume_m3": float(df["_volume"].sum()),
            }
            # Volume-weighted stats for numeric attribute columns
            if numeric_cols:
                stats["volume"]["volume_weighted_means"] = {}
                for col in numeric_cols:
                    populated = df[df[col].notna()]
                    if len(populated) > 0:
                        vol_weighted_mean = float(
                            (populated[col] * populated["_volume"]).sum() / populated["_volume"].sum()
                        )
                        stats["volume"]["volume_weighted_means"][col] = vol_weighted_mean
                        stats["volume"][f"{col}_mineralized_volume_m3"] = float(populated["_volume"].sum())

        return stats
