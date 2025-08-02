"""
FlowKit Agent Functions for Flow Cytometry Data Analysis

This module provides a comprehensive set of functions for flow cytometry data analysis
using the FlowKit library. These functions are designed to be used by an AI agent to
perform various flow cytometry analysis tasks.

Author: FlowKit Agent
"""

import os
import warnings
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd

# FlowKit imports
import flowkit as fk
from flowkit import Sample, Session, Workspace, GatingStrategy, Matrix
from flowkit import gates, transforms
from flowkit.exceptions import FlowKitException
from flowkit._models.gating_results import GatingResults

# Bokeh imports
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models import Title, Range1d
from bokeh.palettes import Category10, Category20, Viridis256


def load_fcs_file(
    file_path: str,
    sample_id: Optional[str] = None,
    compensation: Optional[Union[str, np.ndarray]] = None,
    subsample: int = 10000,
) -> Sample:
    """
    Load a single FCS file and create a Sample object.

    This function loads flow cytometry data from an FCS file and creates a FlowKit Sample
    object that can be used for further analysis including compensation, transformation,
    and gating.

    Args:
        file_path (str): Path to the FCS file to load
        sample_id (Optional[str]): Custom sample identifier. If None, uses filename
        compensation (Optional[Union[str, np.ndarray]]): Compensation matrix as file path,
            CSV string, or NumPy array. If None, uses $SPILL keyword from FCS file
        subsample (int): Number of events to use for subsampling in plots (default: 10000)

    Returns:
        Sample: FlowKit Sample object containing the loaded FCS data

    Raises:
        FileNotFoundError: If the FCS file doesn't exist
        FlowKitException: If there's an error loading the FCS file

    Example:
        >>> sample = load_fcs_file('data/sample001.fcs', sample_id='Sample_001')
        >>> print(f"Loaded {sample.event_count} events with {len(sample.pnn_labels)} channels")
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FCS file not found: {file_path}")

        sample = Sample(
            file_path,
            sample_id=sample_id,
            compensation=compensation,
            subsample=subsample,
        )

        return sample

    except Exception as e:
        raise FlowKitException(f"Error loading FCS file {file_path}: {str(e)}")


def load_multiple_fcs_files(
    directory_path: str, file_pattern: str = "*.fcs"
) -> List[Sample]:
    """
    Load multiple FCS files from a directory.

    This function loads all FCS files matching a pattern from a directory and returns
    a list of Sample objects. Useful for batch processing of flow cytometry data.

    Args:
        directory_path (str): Path to directory containing FCS files
        file_pattern (str): File pattern to match (default: "*.fcs")

    Returns:
        List[Sample]: List of FlowKit Sample objects

    Raises:
        FileNotFoundError: If the directory doesn't exist

    Example:
        >>> samples = load_multiple_fcs_files('data/', '*.fcs')
        >>> print(f"Loaded {len(samples)} samples")
    """
    try:
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        samples = fk.load_samples(directory_path)
        return samples

    except Exception as e:
        raise FlowKitException(
            f"Error loading FCS files from {directory_path}: {str(e)}"
        )


def get_sample_info(sample: Sample) -> Dict[str, Any]:
    """
    Get comprehensive information about a flow cytometry sample.

    This function extracts key information from a Sample object including event count,
    channel information, metadata, and data quality metrics.

    Args:
        sample (Sample): FlowKit Sample object

    Returns:
        Dict[str, Any]: Dictionary containing sample information including:
            - sample_id: Sample identifier
            - event_count: Number of events
            - channel_count: Number of channels
            - pnn_labels: Parameter names (PnN)
            - pns_labels: Parameter descriptions (PnS)
            - acquisition_date: Date of acquisition
            - cytometer: Cytometer information
            - has_compensation: Whether compensation matrix is available
            - has_transformation: Whether transformation is applied

    Example:
        >>> info = get_sample_info(sample)
        >>> print(f"Sample: {info['sample_id']}, Events: {info['event_count']}")
    """
    try:
        info = {
            "sample_id": sample.id,
            "event_count": sample.event_count,
            "channel_count": len(sample.pnn_labels),
            "pnn_labels": sample.pnn_labels,
            "pns_labels": sample.pns_labels,
            "fluoro_channels": [sample.pnn_labels[i] for i in sample.fluoro_indices],
            "scatter_channels": [sample.pnn_labels[i] for i in sample.scatter_indices],
            "time_channel": (
                sample.pnn_labels[sample.time_index]
                if sample.time_index is not None
                else None
            ),
            "has_compensation": hasattr(sample, "_comp_events")
            and sample._comp_events is not None,
            "has_transformation": hasattr(sample, "_transformed_events")
            and sample._transformed_events is not None,
            "version": sample.version,
            "metadata_keys": list(sample.metadata.keys()),
        }

        # Extract common metadata if available
        metadata = sample.metadata
        if "$DATE" in metadata:
            info["acquisition_date"] = metadata["$DATE"]
        if "$CYT" in metadata:
            info["cytometer"] = metadata["$CYT"]
        if "$VOL" in metadata:
            info["sample_volume"] = metadata["$VOL"]

        return info

    except Exception as e:
        raise FlowKitException(f"Error getting sample info: {str(e)}")


def apply_compensation(
    sample: Sample, compensation_matrix: Optional[Union[str, np.ndarray]] = None
) -> Sample:
    """
    Apply compensation to a flow cytometry sample.

    Compensation corrects for spectral overlap between fluorescent channels.
    This function applies compensation using either a provided matrix or the
    matrix stored in the FCS file's $SPILL keyword.

    Args:
        sample (Sample): FlowKit Sample object
        compensation_matrix (Optional[Union[str, np.ndarray]]): Compensation matrix as:
            - File path to CSV/TSV file
            - CSV string
            - NumPy array
            - If None, uses $SPILL keyword from FCS metadata

    Returns:
        Sample: Sample object with compensation applied

    Raises:
        FlowKitException: If compensation fails

    Example:
        >>> compensated_sample = apply_compensation(sample)
        >>> # Or with custom matrix
        >>> compensated_sample = apply_compensation(sample, 'compensation.csv')
    """
    try:
        sample.apply_compensation(compensation_matrix)
        return sample

    except Exception as e:
        raise FlowKitException(f"Error applying compensation: {str(e)}")


def apply_logicle_transform(
    sample: Sample,
    channels: Optional[List[str]] = None,
    t: float = 262144.0,
    w: float = 0.5,
    m: float = 4.5,
    a: float = 0.0,
) -> Sample:
    """
    Apply Logicle transformation to flow cytometry data.

    Logicle transformation is a biexponential transformation that provides a smooth
    transition between linear and logarithmic scales, making it ideal for flow cytometry
    data visualization. It handles both positive and negative values appropriately.

    Args:
        sample (Sample): FlowKit Sample object
        channels (Optional[List[str]]): List of channel names to transform.
            If None, transforms all fluorescent channels
        t (float): Top of scale value (default: 262144.0)
        w (float): Width of linear region in decades (default: 0.5)
        m (float): Number of decades in the logarithmic region (default: 4.5)
        a (float): Additional negative decades (default: 0.0)

    Returns:
        Sample: Sample object with Logicle transformation applied

    Raises:
        FlowKitException: If transformation fails

    Example:
        >>> # Apply logicle transformation to all fluorescent channels
        >>> transformed_sample = apply_logicle_transform(sample)
        >>> # Apply with custom parameters to specific channels
        >>> transformed_sample = apply_logicle_transform(sample,
        ...     channels=['FITC-A', 'PE-A'], t=100000, w=0.3, m=4.0)
    """
    try:
        transform = transforms.LogicleTransform(
            param_t=t, param_w=w, param_m=m, param_a=a
        )

        # Apply transformation
        if channels is not None:
            # Create dictionary for specific channels
            transform_dict = {}
            for channel in channels:
                if channel in sample.pnn_labels:
                    transform_dict[channel] = transform
                else:
                    warnings.warn(f"Channel {channel} not found in sample")

            if transform_dict:
                sample.apply_transform(transform_dict)
        else:
            # Apply to all fluorescent channels
            sample.apply_transform(transform)

        return sample

    except Exception as e:
        raise FlowKitException(f"Error applying Logicle transformation: {str(e)}")


def apply_asinh_transform(
    sample: Sample,
    channels: Optional[List[str]] = None,
    cofactor: float = 5.0,
) -> Sample:
    """
    Apply inverse hyperbolic sine (asinh) transformation to flow cytometry data.

    Asinh transformation is useful for flow cytometry data as it provides a smooth
    transition between linear and logarithmic scales, similar to logicle but with
    different mathematical properties. It's particularly good for data with a wide
    dynamic range.

    Args:
        sample (Sample): FlowKit Sample object
        channels (Optional[List[str]]): List of channel names to transform.
            If None, transforms all fluorescent channels
        cofactor (float): Cofactor for the transformation (default: 5.0)

    Returns:
        Sample: Sample object with asinh transformation applied

    Raises:
        FlowKitException: If transformation fails

    Example:
        >>> # Apply asinh transformation to all fluorescent channels
        >>> transformed_sample = apply_asinh_transform(sample)
        >>> # Apply with custom cofactor to specific channels
        >>> transformed_sample = apply_asinh_transform(sample,
        ...     channels=['FITC-A', 'PE-A'], cofactor=10.0)
    """
    try:
        transform = transforms.AsinhTransform(cofactor=cofactor)

        # Apply transformation
        if channels is not None:
            # Create dictionary for specific channels
            transform_dict = {}
            for channel in channels:
                if channel in sample.pnn_labels:
                    transform_dict[channel] = transform
                else:
                    warnings.warn(f"Channel {channel} not found in sample")

            if transform_dict:
                sample.apply_transform(transform_dict)
        else:
            # Apply to all fluorescent channels
            sample.apply_transform(transform)

        return sample

    except Exception as e:
        raise FlowKitException(f"Error applying asinh transformation: {str(e)}")


def apply_log_transform(
    sample: Sample,
    channels: Optional[List[str]] = None,
    base: float = 10.0,
    offset: float = 1.0,
) -> Sample:
    """
    Apply logarithmic transformation to flow cytometry data.

    Logarithmic transformation is a classic method for flow cytometry data visualization.
    It compresses high values and expands low values, making it easier to visualize
    data with a wide dynamic range. The offset parameter helps handle zero and negative values.

    Args:
        sample (Sample): FlowKit Sample object
        channels (Optional[List[str]]): List of channel names to transform.
            If None, transforms all fluorescent channels
        base (float): Base of the logarithm (default: 10.0)
        offset (float): Offset to add before taking log (default: 1.0)

    Returns:
        Sample: Sample object with log transformation applied

    Raises:
        FlowKitException: If transformation fails

    Example:
        >>> # Apply log transformation to all fluorescent channels
        >>> transformed_sample = apply_log_transform(sample)
        >>> # Apply with custom parameters to specific channels
        >>> transformed_sample = apply_log_transform(sample,
        ...     channels=['FITC-A', 'PE-A'], base=2.0, offset=0.1)
    """
    try:
        transform = transforms.LogTransform(base=base, offset=offset)

        # Apply transformation
        if channels is not None:
            # Create dictionary for specific channels
            transform_dict = {}
            for channel in channels:
                if channel in sample.pnn_labels:
                    transform_dict[channel] = transform
                else:
                    warnings.warn(f"Channel {channel} not found in sample")

            if transform_dict:
                sample.apply_transform(transform_dict)
        else:
            # Apply to all fluorescent channels
            sample.apply_transform(transform)

        return sample

    except Exception as e:
        raise FlowKitException(f"Error applying log transformation: {str(e)}")


def apply_linear_transform(
    sample: Sample,
    channels: Optional[List[str]] = None,
    scale: float = 1.0,
    offset: float = 0.0,
) -> Sample:
    """
    Apply linear transformation to flow cytometry data.

    Linear transformation applies a simple linear scaling to the data.
    This is useful when you want to scale the data without changing its distribution
    or when working with data that is already appropriately scaled.

    Args:
        sample (Sample): FlowKit Sample object
        channels (Optional[List[str]]): List of channel names to transform.
            If None, transforms all fluorescent channels
        scale (float): Scaling factor (default: 1.0)
        offset (float): Offset to add (default: 0.0)

    Returns:
        Sample: Sample object with linear transformation applied

    Raises:
        FlowKitException: If transformation fails

    Example:
        >>> # Apply linear transformation to all fluorescent channels
        >>> transformed_sample = apply_linear_transform(sample)
        >>> # Apply with custom scaling to specific channels
        >>> transformed_sample = apply_linear_transform(sample,
        ...     channels=['FSC-A', 'SSC-A'], scale=0.001, offset=0.0)
    """
    try:
        transform = transforms.LinearTransform(scale=scale, offset=offset)

        # Apply transformation
        if channels is not None:
            # Create dictionary for specific channels
            transform_dict = {}
            for channel in channels:
                if channel in sample.pnn_labels:
                    transform_dict[channel] = transform
                else:
                    warnings.warn(f"Channel {channel} not found in sample")

            if transform_dict:
                sample.apply_transform(transform_dict)
        else:
            # Apply to all fluorescent channels
            sample.apply_transform(transform)

        return sample

    except Exception as e:
        raise FlowKitException(f"Error applying linear transformation: {str(e)}")


def apply_hyperlog_transform(
    sample: Sample,
    channels: Optional[List[str]] = None,
    t: float = 262144.0,
    w: float = 0.5,
    m: float = 4.5,
    a: float = 0.0,
) -> Sample:
    """
    Apply Hyperlog transformation to flow cytometry data.

    Hyperlog transformation is an alternative to logicle transformation that provides
    similar benefits for flow cytometry data visualization. It offers a smooth
    transition between linear and logarithmic scales with different mathematical properties.

    Args:
        sample (Sample): FlowKit Sample object
        channels (Optional[List[str]]): List of channel names to transform.
            If None, transforms all fluorescent channels
        t (float): Top of scale value (default: 262144.0)
        w (float): Width of linear region in decades (default: 0.5)
        m (float): Number of decades in the logarithmic region (default: 4.5)
        a (float): Additional negative decades (default: 0.0)

    Returns:
        Sample: Sample object with Hyperlog transformation applied

    Raises:
        FlowKitException: If transformation fails

    Example:
        >>> # Apply hyperlog transformation to all fluorescent channels
        >>> transformed_sample = apply_hyperlog_transform(sample)
        >>> # Apply with custom parameters to specific channels
        >>> transformed_sample = apply_hyperlog_transform(sample,
        ...     channels=['FITC-A', 'PE-A'], t=100000, w=0.3, m=4.0)
    """
    try:
        transform = transforms.HyperlogTransform(t=t, w=w, m=m, a=a)

        # Apply transformation
        if channels is not None:
            # Create dictionary for specific channels
            transform_dict = {}
            for channel in channels:
                if channel in sample.pnn_labels:
                    transform_dict[channel] = transform
                else:
                    warnings.warn(f"Channel {channel} not found in sample")

            if transform_dict:
                sample.apply_transform(transform_dict)
        else:
            # Apply to all fluorescent channels
            sample.apply_transform(transform)

        return sample

    except Exception as e:
        raise FlowKitException(f"Error applying Hyperlog transformation: {str(e)}")


def create_rectangle_gate(
    gate_name: str, dimensions: List[Dict[str, Any]]
) -> gates.RectangleGate:
    """
    Create a rectangular gate for flow cytometry analysis.

    Rectangle gates are used to select events within specified ranges for one or more
    parameters. They can be 1D (range gate) or multi-dimensional.

    Args:
        gate_name (str): Name of the gate
        dimensions (List[Dict[str, Any]]): List of dimension specifications, each containing:
            - 'id': Channel name (e.g., 'FSC-A', 'FITC-A')
            - 'min': Minimum value (optional)
            - 'max': Maximum value (optional)
            - 'compensation_ref': Compensation reference (optional)

    Returns:
        gates.RectangleGate: FlowKit RectangleGate object

    Raises:
        ValueError: If dimension specifications are invalid

    Example:
        >>> # Create a 2D rectangle gate
        >>> dims = [
        ...     {'id': 'FSC-A', 'min': 10000, 'max': 250000},
        ...     {'id': 'SSC-A', 'min': 5000, 'max': 200000}
        ... ]
        >>> gate = create_rectangle_gate('Cells', dims)
    """
    try:
        gate_dimensions = []

        for dim_spec in dimensions:
            if "id" not in dim_spec:
                raise ValueError("Each dimension must have an 'id' field")

            dim = fk.Dimension(
                dim_spec["id"],
                compensation_ref=dim_spec.get("compensation_ref"),
                min_value=dim_spec.get("min"),
                max_value=dim_spec.get("max"),
            )
            gate_dimensions.append(dim)

        gate = gates.RectangleGate(gate_name, gate_dimensions)
        return gate

    except Exception as e:
        raise FlowKitException(f"Error creating rectangle gate: {str(e)}")


def create_polygon_gate(
    gate_name: str, channel_x: str, channel_y: str, vertices: List[Tuple[float, float]]
) -> gates.PolygonGate:
    """
    Create a polygon gate for flow cytometry analysis.

    Polygon gates are used to select events within a polygonal region defined by
    vertices in 2D space. They are commonly used for identifying cell populations
    with irregular boundaries.

    Args:
        gate_name (str): Name of the gate
        channel_x (str): X-axis channel name (e.g., 'FSC-A')
        channel_y (str): Y-axis channel name (e.g., 'SSC-A')
        vertices (List[Tuple[float, float]]): List of (x, y) coordinates defining the polygon

    Returns:
        gates.PolygonGate: FlowKit PolygonGate object

    Raises:
        ValueError: If less than 3 vertices are provided

    Example:
        >>> # Create a triangular gate around lymphocytes
        >>> vertices = [(20000, 10000), (80000, 15000), (50000, 60000)]
        >>> gate = create_polygon_gate('Lymphocytes', 'FSC-A', 'SSC-A', vertices)
    """
    try:
        if len(vertices) < 3:
            raise ValueError("Polygon gate requires at least 3 vertices")

        dimensions = [fk.Dimension(channel_x), fk.Dimension(channel_y)]

        gate = gates.PolygonGate(gate_name, dimensions, vertices)
        return gate

    except Exception as e:
        raise FlowKitException(f"Error creating polygon gate: {str(e)}")


def create_ellipsoid_gate(
    gate_name: str,
    channel_x: str,
    channel_y: str,
    center: Tuple[float, float],
    covariance_matrix: np.ndarray,
    distance_square: float,
) -> gates.EllipsoidGate:
    """
    Create an ellipsoid gate for flow cytometry analysis.

    Ellipsoid gates define elliptical regions based on statistical parameters
    and are useful for automated gating based on population statistics.

    Args:
        gate_name (str): Name of the gate
        channel_x (str): X-axis channel name
        channel_y (str): Y-axis channel name
        center (Tuple[float, float]): Center coordinates (mean_x, mean_y)
        covariance_matrix (np.ndarray): 2x2 covariance matrix
        distance_square (float): Square of the Mahalanobis distance

    Returns:
        gates.EllipsoidGate: FlowKit EllipsoidGate object

    Example:
        >>> center = (50000, 30000)
        >>> cov_matrix = np.array([[1000, 500], [500, 800]])
        >>> gate = create_ellipsoid_gate('Cells', 'FSC-A', 'SSC-A', center, cov_matrix, 2.0)
    """
    try:
        dimensions = [fk.Dimension(channel_x), fk.Dimension(channel_y)]

        gate = gates.EllipsoidGate(
            gate_name, dimensions, center, covariance_matrix, distance_square
        )
        return gate

    except Exception as e:
        raise FlowKitException(f"Error creating ellipsoid gate: {str(e)}")


def create_gating_strategy() -> GatingStrategy:
    """
    Create a new gating strategy for flow cytometry analysis.

    A gating strategy defines a hierarchical set of gates that can be applied
    to flow cytometry samples to identify different cell populations.

    Returns:
        GatingStrategy: Empty FlowKit GatingStrategy object

    Example:
        >>> strategy = create_gating_strategy()
        >>> # Add gates to the strategy
        >>> strategy.add_gate(cells_gate, ('root',))
    """
    try:
        return GatingStrategy()
    except Exception as e:
        raise FlowKitException(f"Error creating gating strategy: {str(e)}")


def add_gate_to_strategy(
    strategy: GatingStrategy,
    gate: Union[
        gates.RectangleGate,
        gates.PolygonGate,
        gates.EllipsoidGate,
        gates.QuadrantGate,
        gates.BooleanGate,
    ],
    gate_path: Tuple[str, ...],
    sample_id: Optional[str] = None,
):
    """
    Add a gate to an existing gating strategy.

    This function adds a gate to a gating strategy at the specified path in the
    gate hierarchy. Gates can be applied to all samples or specific samples.
    The gate hierarchy allows for sequential gating where child gates are applied
    only to events that pass through their parent gates.

    Args:
        strategy (GatingStrategy): FlowKit GatingStrategy object to add the gate to
        gate (Union[gates.RectangleGate, gates.PolygonGate, gates.EllipsoidGate,
            gates.QuadrantGate, gates.BooleanGate]): Gate object to add to the strategy
        gate_path (Tuple[str, ...]): Path in the gate hierarchy where the gate should be added.
            Use ('root',) for top-level gates, or ('root', 'parent_gate') for child gates
        sample_id (Optional[str]): If specified, the gate applies only to this specific sample.
            If None, the gate applies to all samples processed with this strategy

    Returns:
        None: This function modifies the strategy in-place

    Raises:
        FlowKitException: If adding the gate fails due to invalid gate path,
            incompatible gate type, or other FlowKit errors

    Example:
        >>> strategy = create_gating_strategy()
        >>> # Create and add a top-level gate for cell identification
        >>> cells_gate = create_rectangle_gate('Cells', [{'id': 'FSC-A', 'min': 10000}])
        >>> add_gate_to_strategy(strategy, cells_gate, ('root',))
        >>>
        >>> # Add a child gate for live cells
        >>> live_gate = create_rectangle_gate('Live', [{'id': 'PI-A', 'max': 1000}])
        >>> add_gate_to_strategy(strategy, live_gate, ('root', 'Cells'))
    """
    try:
        strategy.add_gate(gate, gate_path, sample_id=sample_id)
    except Exception as e:
        raise FlowKitException(f"Error adding gate to strategy: {str(e)}")


def apply_gating_strategy(
    sample: Sample, strategy: GatingStrategy, cache_events: bool = False
) -> GatingResults:
    """
    Apply a gating strategy to a flow cytometry sample.

    This function applies all gates in a gating strategy to a sample and returns
    the results, including event counts and percentages for each gate.

    Args:
        sample (Sample): FlowKit Sample object
        strategy (GatingStrategy): GatingStrategy to apply
        cache_events (bool): Whether to cache preprocessed events for performance

    Returns:
        GatingResults: Results of applying the gating strategy

    Raises:
        FlowKitException: If gating fails

    Example:
        >>> results = apply_gating_strategy(sample, strategy)
        >>> report = results.get_report()
        >>> print(report)
    """
    try:
        results = strategy.gate_sample(sample, cache_events=cache_events)
        return results
    except Exception as e:
        raise FlowKitException(f"Error applying gating strategy: {str(e)}")


def get_gating_report(gating_results: GatingResults) -> pd.DataFrame:
    """
    Get a summary report from gating results.

    This function extracts a pandas DataFrame containing gate statistics including
    event counts, percentages, and gate hierarchy information.

    Args:
        gating_results (GatingResults): GatingResults object from apply_gating_strategy

    Returns:
        pd.DataFrame: DataFrame with columns:
            - sample: Sample ID
            - gate: Gate name
            - path: Gate path in hierarchy
            - count: Number of events in gate
            - absolute_percent: Percentage of total events
            - relative_percent: Percentage of parent gate events

    Example:
        >>> report = get_gating_report(gating_results)
        >>> print(report[['gate', 'count', 'absolute_percent']])
    """
    try:
        return gating_results.get_report()
    except Exception as e:
        raise FlowKitException(f"Error getting gating report: {str(e)}")


def get_gate_events(
    sample: Sample,
    gating_results: GatingResults,
    gate_name: str,
    gate_path: Optional[Tuple[str, ...]] = None,
) -> pd.DataFrame:
    """
    Extract events that fall within a specific gate.

    This function returns the flow cytometry events that were classified as positive
    for a specific gate, allowing for downstream analysis of gated populations.

    Args:
        sample (Sample): Original Sample object
        gating_results (GatingResults): Results from gating analysis
        gate_name (str): Name of the gate
        gate_path (Optional[Tuple[str, ...]]): Path to the gate in hierarchy

    Returns:
        pd.DataFrame: DataFrame containing events within the gate

    Raises:
        FlowKitException: If gate is not found or extraction fails

    Example:
        >>> lymph_events = get_gate_events(sample, results, 'Lymphocytes')
        >>> print(f"Found {len(lymph_events)} lymphocyte events")
    """
    try:
        gate_membership = gating_results.get_gate_membership(gate_name, gate_path)
        events_df = sample.as_dataframe(source="xform")
        gated_events = events_df[gate_membership]
        return gated_events
    except Exception as e:
        raise FlowKitException(f"Error extracting gate events: {str(e)}")


def calculate_population_statistics(
    events: pd.DataFrame, channels: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistical measures for cell populations.

    This function computes common statistical measures (mean, median, std, etc.)
    for specified channels in a cell population.

    Args:
        events (pd.DataFrame): DataFrame containing flow cytometry events
        channels (List[str]): List of channel names to analyze

    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary with statistics:
            - First level: channel names
            - Second level: statistic names (mean, median, std, min, max, count)

    Example:
        >>> stats = calculate_population_statistics(lymph_events, ['FITC-A', 'PE-A'])
        >>> print(f"FITC-A mean: {stats['FITC-A']['mean']:.2f}")
    """
    try:
        statistics = {}

        for channel in channels:
            if channel in events.columns:
                channel_data = events[channel].dropna()
                statistics[channel] = {
                    "count": len(channel_data),
                    "mean": float(channel_data.mean()),
                    "median": float(channel_data.median()),
                    "std": float(channel_data.std()),
                    "min": float(channel_data.min()),
                    "max": float(channel_data.max()),
                    "q25": float(channel_data.quantile(0.25)),
                    "q75": float(channel_data.quantile(0.75)),
                }
            else:
                warnings.warn(f"Channel {channel} not found in events data")

        return statistics

    except Exception as e:
        raise FlowKitException(f"Error calculating population statistics: {str(e)}")


def create_session(samples: List[Sample]) -> Session:
    """
    Create a FlowKit Session for multi-sample analysis.

    A Session combines multiple samples with a single gating strategy, enabling
    batch analysis of multiple flow cytometry samples.

    Args:
        samples (List[Sample]): List of Sample objects to include in the session

    Returns:
        Session: FlowKit Session object

    Example:
        >>> samples = load_multiple_fcs_files('data/')
        >>> session = create_session(samples)
        >>> # Add gating strategy and analyze all samples
    """
    try:
        session = Session(samples)
        return session
    except Exception as e:
        raise FlowKitException(f"Error creating session: {str(e)}")


def analyze_session(
    session: Session,
    gating_strategy: GatingStrategy,
    use_multiprocessing: bool = True,
    cache_events: bool = False,
) -> pd.DataFrame:
    """
    Analyze all samples in a session with a gating strategy.

    This function applies a gating strategy to all samples in a session and returns
    a comprehensive report with results for all samples.

    Args:
        session (Session): FlowKit Session object
        gating_strategy (GatingStrategy): Gating strategy to apply
        use_multiprocessing (bool): Whether to use multiprocessing for speed
        cache_events (bool): Whether to cache preprocessed events

    Returns:
        pd.DataFrame: Combined analysis report for all samples

    Example:
        >>> report = analyze_session(session, strategy)
        >>> # Group by gate to see population statistics across samples
        >>> summary = report.groupby('gate')['count'].agg(['mean', 'std'])
    """
    try:
        session.gating_strategy = gating_strategy
        session.analyze_samples(use_mp=use_multiprocessing, cache_events=cache_events)
        report = session.get_analysis_report()
        return report
    except Exception as e:
        raise FlowKitException(f"Error analyzing session: {str(e)}")


def load_workspace(wsp_file_path: str, fcs_directory: str) -> Workspace:
    """
    Load a FlowJo workspace file (.wsp) for analysis.

    This function loads a FlowJo workspace file along with the associated FCS files,
    enabling analysis of pre-defined gating strategies and compensation matrices.

    Args:
        wsp_file_path (str): Path to the FlowJo workspace (.wsp) file
        fcs_directory (str): Directory containing the FCS files referenced in the workspace

    Returns:
        Workspace: FlowKit Workspace object

    Raises:
        FileNotFoundError: If workspace file or FCS directory doesn't exist

    Example:
        >>> workspace = load_workspace('analysis.wsp', 'data/')
        >>> sample_ids = workspace.get_sample_ids()
        >>> print(f"Loaded workspace with {len(sample_ids)} samples")
    """
    try:
        if not os.path.exists(wsp_file_path):
            raise FileNotFoundError(f"Workspace file not found: {wsp_file_path}")
        if not os.path.exists(fcs_directory):
            raise FileNotFoundError(f"FCS directory not found: {fcs_directory}")

        workspace = Workspace(wsp_file_path, fcs_directory)
        return workspace
    except Exception as e:
        raise FlowKitException(f"Error loading workspace: {str(e)}")


def export_gated_events(
    sample: Sample,
    gating_results: GatingResults,
    gate_name: str,
    output_path: str,
    file_format: str = "csv",
) -> str:
    """
    Export gated events to a file for further analysis.

    This function exports the events that fall within a specific gate to various
    file formats for use in other analysis tools.

    Args:
        sample (Sample): Original Sample object
        gating_results (GatingResults): Results from gating analysis
        gate_name (str): Name of the gate to export
        output_path (str): Path for the output file
        file_format (str): Format for export ('csv', 'fcs', 'tsv')

    Returns:
        str: Path to the exported file

    Raises:
        ValueError: If file format is not supported
        FlowKitException: If export fails

    Example:
        >>> output_file = export_gated_events(sample, results, 'Lymphocytes',
        ...                                  'lymphocytes.csv', 'csv')
        >>> print(f"Exported events to {output_file}")
    """
    try:
        # Get gated events
        gated_events = get_gate_events(sample, gating_results, gate_name)

        if file_format.lower() == "csv":
            gated_events.to_csv(output_path, index=False)
        elif file_format.lower() == "tsv":
            gated_events.to_csv(output_path, sep="\t", index=False)
        elif file_format.lower() == "fcs":
            # For FCS export, we need to create a new FCS file
            # This is more complex and would require additional FlowKit functionality
            raise NotImplementedError("FCS export not yet implemented")
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        return output_path

    except Exception as e:
        raise FlowKitException(f"Error exporting gated events: {str(e)}")


def compare_populations(
    events1: pd.DataFrame, events2: pd.DataFrame, channels: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compare statistical measures between two cell populations.

    This function compares mean fluorescence intensity and other statistics
    between two cell populations, useful for identifying differences between
    treatment groups or cell types.

    Args:
        events1 (pd.DataFrame): First population events
        events2 (pd.DataFrame): Second population events
        channels (List[str]): Channels to compare

    Returns:
        Dict[str, Dict[str, float]]: Comparison results with:
            - mean_diff: Difference in means
            - median_diff: Difference in medians
            - fold_change: Fold change (pop2/pop1)
            - effect_size: Cohen's d effect size

    Example:
        >>> comparison = compare_populations(treated_cells, control_cells, ['FITC-A'])
        >>> print(f"Fold change: {comparison['FITC-A']['fold_change']:.2f}")
    """
    try:
        comparison_results = {}

        for channel in channels:
            if channel in events1.columns and channel in events2.columns:
                data1 = events1[channel].dropna()
                data2 = events2[channel].dropna()

                mean1, mean2 = data1.mean(), data2.mean()
                median1, median2 = data1.median(), data2.median()
                std1, std2 = data1.std(), data2.std()

                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(
                    ((len(data1) - 1) * std1**2 + (len(data2) - 1) * std2**2)
                    / (len(data1) + len(data2) - 2)
                )
                effect_size = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0

                comparison_results[channel] = {
                    "mean_diff": mean2 - mean1,
                    "median_diff": median2 - median1,
                    "fold_change": mean2 / mean1 if mean1 > 0 else float("inf"),
                    "effect_size": effect_size,
                    "pop1_mean": mean1,
                    "pop2_mean": mean2,
                    "pop1_count": len(data1),
                    "pop2_count": len(data2),
                }
            else:
                warnings.warn(f"Channel {channel} not found in one or both populations")

        return comparison_results

    except Exception as e:
        raise FlowKitException(f"Error comparing populations: {str(e)}")


def generate_analysis_summary(
    samples: List[Sample], gating_results_list: List[GatingResults]
) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of flow cytometry analysis results.

    This function creates a summary report combining information from multiple
    samples and their gating results, useful for understanding overall experimental results.

    Args:
        samples (List[Sample]): List of analyzed samples
        gating_results_list (List[GatingResults]): List of gating results for each sample

    Returns:
        Dict[str, Any]: Summary containing:
            - total_samples: Number of samples analyzed
            - total_events: Total events across all samples
            - gate_summary: Statistics for each gate across samples
            - sample_info: Basic information for each sample

    Example:
        >>> summary = generate_analysis_summary(samples, results_list)
        >>> print(f"Analyzed {summary['total_samples']} samples with {summary['total_events']} total events")
    """
    try:
        total_events = sum(sample.event_count for sample in samples)
        total_samples = len(samples)

        # Combine all reports
        all_reports = []
        for results in gating_results_list:
            report = results.get_report()
            if len(report) > 0:
                all_reports.append(report)

        if all_reports:
            combined_report = pd.concat(all_reports, ignore_index=True)

            # Calculate gate summary statistics
            gate_summary = (
                combined_report.groupby(["gate", "path"])
                .agg(
                    {
                        "count": ["mean", "std", "min", "max"],
                        "absolute_percent": ["mean", "std", "min", "max"],
                        "relative_percent": ["mean", "std", "min", "max"],
                    }
                )
                .round(2)
            )
        else:
            gate_summary = pd.DataFrame()

        # Sample information
        sample_info = []
        for sample in samples:
            info = get_sample_info(sample)
            sample_info.append(
                {
                    "sample_id": info["sample_id"],
                    "event_count": info["event_count"],
                    "channel_count": info["channel_count"],
                    "has_compensation": info["has_compensation"],
                    "has_transformation": info["has_transformation"],
                }
            )

        summary = {
            "total_samples": total_samples,
            "total_events": total_events,
            "average_events_per_sample": (
                total_events / total_samples if total_samples > 0 else 0
            ),
            "gate_summary": gate_summary,
            "sample_info": sample_info,
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
        }

        return summary

    except Exception as e:
        raise FlowKitException(f"Error generating analysis summary: {str(e)}")


# Advanced Analysis Functions


def auto_gate_populations(
    sample: Sample,
    channels: List[str],
    method: str = "kmeans",
    n_clusters: int = 2,
    **kwargs,
) -> List[
    Union[
        gates.RectangleGate,
        gates.PolygonGate,
        gates.EllipsoidGate,
        gates.QuadrantGate,
        gates.BooleanGate,
    ]
]:
    """
    Automatically generate gates for cell populations using clustering algorithms.

    This function uses machine learning clustering algorithms to automatically identify
    cell populations and create corresponding gates. Useful for exploratory analysis
    and automated gating workflows.

    Args:
        sample (Sample): FlowKit Sample object
        channels (List[str]): List of channel names to use for clustering
        method (str): Clustering method ('kmeans', 'gaussian_mixture', 'dbscan')
        n_clusters (int): Number of clusters to find (for methods that require it)
        **kwargs: Additional parameters for the clustering algorithm

    Returns:
        List[Union[gates.RectangleGate, gates.PolygonGate, gates.EllipsoidGate,
            gates.QuadrantGate, gates.BooleanGate]]: List of automatically generated gates

    Raises:
        ImportError: If required ML libraries are not available
        FlowKitException: If clustering fails

    Example:
        >>> # Auto-gate lymphocytes and monocytes based on scatter
        >>> auto_gates = auto_gate_populations(sample, ['FSC-A', 'SSC-A'],
        ...                                   method='kmeans', n_clusters=3)
        >>> print(f"Generated {len(auto_gates)} automatic gates")
    """
    try:
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        from scipy.spatial import ConvexHull

        # Get events data
        events_df = sample.as_dataframe(source="xform")

        # Extract data for specified channels
        cluster_data = events_df[channels].dropna()

        if len(cluster_data) == 0:
            raise FlowKitException("No valid data found for clustering")

        # Standardize data for clustering
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)

        # Apply clustering algorithm
        if method.lower() == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, **kwargs)
        elif method.lower() == "gaussian_mixture":
            clusterer = GaussianMixture(n_components=n_clusters, **kwargs)
        elif method.lower() == "dbscan":
            clusterer = DBSCAN(**kwargs)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        cluster_labels = clusterer.fit_predict(scaled_data)

        # Generate gates for each cluster
        gates_list = []
        unique_labels = np.unique(cluster_labels)

        for i, label in enumerate(unique_labels):
            if label == -1:  # Skip noise points in DBSCAN
                continue

            cluster_points = cluster_data[cluster_labels == label]

            if len(cluster_points) < 3:  # Need at least 3 points for a polygon
                continue

            if len(channels) == 2:
                # Create polygon gate using convex hull
                try:
                    hull = ConvexHull(cluster_points.values)
                    vertices = [
                        (cluster_points.iloc[vertex, 0], cluster_points.iloc[vertex, 1])
                        for vertex in hull.vertices
                    ]

                    gate = create_polygon_gate(
                        f"AutoGate_Cluster_{i+1}", channels[0], channels[1], vertices
                    )
                    gates_list.append(gate)
                except:
                    # Fallback to rectangle gate if convex hull fails
                    dims = []
                    for ch in channels:
                        dims.append(
                            {
                                "id": ch,
                                "min": float(cluster_points[ch].min()),
                                "max": float(cluster_points[ch].max()),
                            }
                        )
                    gate = create_rectangle_gate(f"AutoGate_Cluster_{i+1}", dims)
                    gates_list.append(gate)
            else:
                # Multi-dimensional rectangle gate
                dims = []
                for ch in channels:
                    dims.append(
                        {
                            "id": ch,
                            "min": float(cluster_points[ch].min()),
                            "max": float(cluster_points[ch].max()),
                        }
                    )
                gate = create_rectangle_gate(f"AutoGate_Cluster_{i+1}", dims)
                gates_list.append(gate)

        return gates_list

    except ImportError:
        raise ImportError(
            "scikit-learn is required for automatic gating. Install with: pip install scikit-learn"
        )
    except Exception as e:
        raise FlowKitException(f"Error in automatic gating: {str(e)}")


def detect_outliers(
    events: pd.DataFrame,
    channels: List[str],
    method: str = "isolation_forest",
    contamination: float = 0.1,
) -> np.ndarray:
    """
    Detect outlier events in flow cytometry data.

    This function identifies outlier events that may represent debris, doublets,
    or other artifacts in flow cytometry data using various outlier detection methods.

    Args:
        events (pd.DataFrame): DataFrame containing flow cytometry events
        channels (List[str]): List of channel names to use for outlier detection
        method (str): Outlier detection method ('isolation_forest', 'one_class_svm', 'local_outlier_factor')
        contamination (float): Expected proportion of outliers (0.0 to 0.5)

    Returns:
        np.ndarray: Boolean array indicating outliers (True = outlier, False = normal)

    Raises:
        ImportError: If required ML libraries are not available
        FlowKitException: If outlier detection fails

    Example:
        >>> outliers = detect_outliers(events_df, ['FSC-A', 'SSC-A'],
        ...                           method='isolation_forest', contamination=0.05)
        >>> clean_events = events_df[~outliers]
        >>> print(f"Removed {outliers.sum()} outlier events")
    """
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.preprocessing import StandardScaler

        # Extract data for specified channels
        outlier_data = events[channels].dropna()

        if len(outlier_data) == 0:
            raise FlowKitException("No valid data found for outlier detection")

        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(outlier_data)

        # Apply outlier detection algorithm
        if method.lower() == "isolation_forest":
            detector = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = detector.fit_predict(scaled_data)
        elif method.lower() == "one_class_svm":
            detector = OneClassSVM(nu=contamination)
            outlier_labels = detector.fit_predict(scaled_data)
        elif method.lower() == "local_outlier_factor":
            detector = LocalOutlierFactor(contamination=contamination)
            outlier_labels = detector.fit_predict(scaled_data)
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")

        # Convert labels to boolean (outliers = True)
        outliers = outlier_labels == -1

        # Create full-length array for all events
        full_outliers = np.zeros(len(events), dtype=bool)
        valid_indices = events[channels].dropna().index
        full_outliers[valid_indices] = outliers

        return full_outliers

    except ImportError:
        raise ImportError(
            "scikit-learn is required for outlier detection. Install with: pip install scikit-learn"
        )
    except Exception as e:
        raise FlowKitException(f"Error in outlier detection: {str(e)}")


def calculate_mfi_comparison(
    populations: Dict[str, pd.DataFrame], channels: List[str]
) -> pd.DataFrame:
    """
    Calculate and compare Mean Fluorescence Intensity (MFI) across multiple populations.

    This function computes MFI values for specified channels across different cell
    populations and provides statistical comparisons between them.

    Args:
        populations (Dict[str, pd.DataFrame]): Dictionary where keys are population names
            and values are DataFrames containing events for each population
        channels (List[str]): List of channel names to analyze

    Returns:
        pd.DataFrame: DataFrame with MFI values and statistics for each population and channel

    Example:
        >>> populations = {
        ...     'CD4+ T cells': cd4_events,
        ...     'CD8+ T cells': cd8_events,
        ...     'B cells': b_cell_events
        ... }
        >>> mfi_results = calculate_mfi_comparison(populations, ['FITC-A', 'PE-A'])
        >>> print(mfi_results)
    """
    try:
        results = []

        for pop_name, events in populations.items():
            for channel in channels:
                if channel in events.columns:
                    channel_data = events[channel].dropna()

                    if len(channel_data) > 0:
                        results.append(
                            {
                                "population": pop_name,
                                "channel": channel,
                                "mfi": float(channel_data.mean()),
                                "median_fi": float(channel_data.median()),
                                "std_fi": float(channel_data.std()),
                                "cv": (
                                    float(channel_data.std() / channel_data.mean())
                                    if channel_data.mean() > 0
                                    else 0
                                ),
                                "count": len(channel_data),
                                "q25": float(channel_data.quantile(0.25)),
                                "q75": float(channel_data.quantile(0.75)),
                            }
                        )
                else:
                    warnings.warn(
                        f"Channel {channel} not found in population {pop_name}"
                    )

        return pd.DataFrame(results)

    except Exception as e:
        raise FlowKitException(f"Error calculating MFI comparison: {str(e)}")


def perform_dimensionality_reduction(
    events: pd.DataFrame,
    channels: List[str],
    method: str = "umap",
    n_components: int = 2,
    **kwargs,
) -> np.ndarray:
    """
    Perform dimensionality reduction on flow cytometry data for visualization.

    This function applies dimensionality reduction techniques to high-dimensional
    flow cytometry data to enable 2D/3D visualization and analysis.

    Args:
        events (pd.DataFrame): DataFrame containing flow cytometry events
        channels (List[str]): List of channel names to include in the analysis
        method (str): Dimensionality reduction method ('umap', 'tsne', 'pca')
        n_components (int): Number of dimensions in the reduced space (default: 2)
        **kwargs: Additional parameters for the dimensionality reduction algorithm

    Returns:
        np.ndarray: Array of reduced-dimension coordinates

    Raises:
        ImportError: If required libraries are not available
        FlowKitException: If dimensionality reduction fails

    Example:
        >>> # Reduce 10D data to 2D for visualization
        >>> reduced_coords = perform_dimensionality_reduction(
        ...     events_df, fluoro_channels, method='umap', n_components=2
        ... )
        >>> # Plot the results
        >>> plt.scatter(reduced_coords[:, 0], reduced_coords[:, 1])
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Extract data for specified channels
        reduction_data = events[channels].dropna()

        if len(reduction_data) == 0:
            raise FlowKitException("No valid data found for dimensionality reduction")

        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(reduction_data)

        # Apply dimensionality reduction
        if method.lower() == "pca":
            reducer = PCA(n_components=n_components, **kwargs)
            reduced_data = reducer.fit_transform(scaled_data)
        elif method.lower() == "umap":
            try:
                import umap

                reducer = umap.UMAP(n_components=n_components, **kwargs)
                reduced_data = reducer.fit_transform(scaled_data)
            except ImportError:
                raise ImportError(
                    "umap-learn is required for UMAP. Install with: pip install umap-learn"
                )
        elif method.lower() == "tsne":
            try:
                from sklearn.manifold import TSNE

                reducer = TSNE(n_components=n_components, **kwargs)
                reduced_data = reducer.fit_transform(scaled_data)
            except ImportError:
                raise ImportError("scikit-learn is required for t-SNE")
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")

        return reduced_data

    except Exception as e:
        raise FlowKitException(f"Error in dimensionality reduction: {str(e)}")


def quality_control_metrics(sample: Sample) -> Dict[str, Any]:
    """
    Calculate quality control metrics for flow cytometry data.

    This function computes various QC metrics to assess data quality including
    event rates, signal stability, and potential technical issues.

    Args:
        sample (Sample): FlowKit Sample object

    Returns:
        Dict[str, Any]: Dictionary containing QC metrics:
            - event_rate: Events per second (if time channel available)
            - signal_stability: CV of signal over time
            - negative_events: Percentage of events with negative values
            - debris_events: Estimated percentage of debris events
            - doublet_events: Estimated percentage of doublet events

    Example:
        >>> qc_metrics = quality_control_metrics(sample)
        >>> print(f"Event rate: {qc_metrics['event_rate']:.1f} events/sec")
        >>> print(f"Signal stability: {qc_metrics['signal_stability']:.2f}")
    """
    try:
        events_df = sample.as_dataframe(source="raw")
        metrics = {}

        # Event rate calculation
        if sample.time_index is not None:
            time_channel = sample.pnn_labels[sample.time_index]
            if time_channel in events_df.columns:
                time_data = events_df[time_channel]
                total_time = (
                    time_data.max() - time_data.min()
                ) / 100  # Convert to seconds (assuming centiseconds)
                if total_time > 0:
                    metrics["event_rate"] = sample.event_count / total_time
                else:
                    metrics["event_rate"] = 0
            else:
                metrics["event_rate"] = None
        else:
            metrics["event_rate"] = None

        # Signal stability (CV over time bins)
        if sample.time_index is not None and len(sample.fluoro_indices) > 0:
            time_channel = sample.pnn_labels[sample.time_index]
            fluoro_channel = sample.pnn_labels[
                sample.fluoro_indices[0]
            ]  # Use first fluoro channel

            if (
                time_channel in events_df.columns
                and fluoro_channel in events_df.columns
            ):
                # Divide data into time bins
                n_bins = min(
                    10, sample.event_count // 1000
                )  # At least 1000 events per bin
                if n_bins > 1:
                    time_bins = pd.cut(events_df[time_channel], bins=n_bins)
                    bin_means = events_df.groupby(time_bins)[fluoro_channel].mean()
                    cv = (
                        bin_means.std() / bin_means.mean()
                        if bin_means.mean() > 0
                        else 0
                    )
                    metrics["signal_stability"] = float(cv)
                else:
                    metrics["signal_stability"] = None
            else:
                metrics["signal_stability"] = None
        else:
            metrics["signal_stability"] = None

        # Negative events percentage
        negative_counts = []
        for idx in sample.fluoro_indices:
            channel = sample.pnn_labels[idx]
            if channel in events_df.columns:
                negative_count = (events_df[channel] < 0).sum()
                negative_counts.append(negative_count)

        if negative_counts:
            total_negative = max(negative_counts)
            metrics["negative_events_percent"] = (
                total_negative / sample.event_count
            ) * 100
        else:
            metrics["negative_events_percent"] = 0

        # Debris estimation (low FSC-A, low SSC-A)
        if len(sample.scatter_indices) >= 2:
            fsc_channel = sample.pnn_labels[sample.scatter_indices[0]]
            ssc_channel = sample.pnn_labels[sample.scatter_indices[1]]

            if fsc_channel in events_df.columns and ssc_channel in events_df.columns:
                fsc_data = events_df[fsc_channel]
                ssc_data = events_df[ssc_channel]

                # Define debris as events in bottom 10% of both scatter parameters
                fsc_threshold = fsc_data.quantile(0.1)
                ssc_threshold = ssc_data.quantile(0.1)

                debris_events = (
                    (fsc_data < fsc_threshold) & (ssc_data < ssc_threshold)
                ).sum()
                metrics["debris_events_percent"] = (
                    debris_events / sample.event_count
                ) * 100
            else:
                metrics["debris_events_percent"] = None
        else:
            metrics["debris_events_percent"] = None

        # Doublet estimation (high FSC-A/FSC-W ratio)
        fsc_a_idx = None
        fsc_w_idx = None

        for i, label in enumerate(sample.pnn_labels):
            if "FSC-A" in label.upper():
                fsc_a_idx = i
            elif "FSC-W" in label.upper():
                fsc_w_idx = i

        if fsc_a_idx is not None and fsc_w_idx is not None:
            fsc_a_channel = sample.pnn_labels[fsc_a_idx]
            fsc_w_channel = sample.pnn_labels[fsc_w_idx]

            if (
                fsc_a_channel in events_df.columns
                and fsc_w_channel in events_df.columns
            ):
                fsc_a_data = events_df[fsc_a_channel]
                fsc_w_data = events_df[fsc_w_channel]

                # Calculate area/width ratio
                ratio = fsc_a_data / fsc_w_data
                ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()

                if len(ratio) > 0:
                    # Define doublets as events with ratio > 95th percentile
                    doublet_threshold = ratio.quantile(0.95)
                    doublet_events = (ratio > doublet_threshold).sum()
                    metrics["doublet_events_percent"] = (
                        doublet_events / len(ratio)
                    ) * 100
                else:
                    metrics["doublet_events_percent"] = None
            else:
                metrics["doublet_events_percent"] = None
        else:
            metrics["doublet_events_percent"] = None

        return metrics

    except Exception as e:
        raise FlowKitException(f"Error calculating QC metrics: {str(e)}")


def create_compensation_matrix_from_controls(
    control_samples: Dict[str, Sample], channels: List[str]
) -> Matrix:
    """
    Create a compensation matrix from single-stain control samples.

    This function calculates a compensation matrix using single-stain control samples,
    which is essential for proper spectral overlap correction in multi-color flow cytometry.

    Args:
        control_samples (Dict[str, Sample]): Dictionary mapping fluorochrome names to
            Sample objects containing single-stain controls
        channels (List[str]): List of detector channel names in the desired order

    Returns:
        Matrix: FlowKit Matrix object containing the compensation matrix

    Raises:
        FlowKitException: If matrix calculation fails
        ValueError: If control samples are insufficient

    Example:
        >>> controls = {
        ...     'FITC': fitc_control_sample,
        ...     'PE': pe_control_sample,
        ...     'APC': apc_control_sample
        ... }
        >>> comp_matrix = create_compensation_matrix_from_controls(controls,
        ...                                                       ['FITC-A', 'PE-A', 'APC-A'])
    """
    try:
        if len(control_samples) < 2:
            raise ValueError(
                "At least 2 control samples are required for compensation matrix calculation"
            )

        # Initialize compensation matrix
        n_channels = len(channels)
        comp_matrix = np.eye(n_channels)

        # Calculate spillover for each control
        for i, (fluorochrome, control_sample) in enumerate(control_samples.items()):
            if i >= n_channels:
                warnings.warn(
                    f"More control samples than channels. Skipping {fluorochrome}"
                )
                continue

            # Get positive population (top 50% of primary channel)
            events_df = control_sample.as_dataframe(source="raw")
            primary_channel = channels[i]

            if primary_channel not in events_df.columns:
                warnings.warn(
                    f"Primary channel {primary_channel} not found in {fluorochrome} control"
                )
                continue

            # Define positive population as top 50% of primary channel
            threshold = events_df[primary_channel].quantile(0.5)
            positive_events = events_df[events_df[primary_channel] > threshold]

            if len(positive_events) == 0:
                warnings.warn(f"No positive events found in {fluorochrome} control")
                continue

            # Calculate median fluorescence for each channel in positive population
            for j, channel in enumerate(channels):
                if channel in positive_events.columns:
                    median_fi = positive_events[channel].median()
                    primary_median = positive_events[primary_channel].median()

                    if primary_median > 0:
                        spillover = median_fi / primary_median
                        comp_matrix[j, i] = spillover

        # Create FlowKit Matrix object
        fluorochrome_names = list(control_samples.keys())[:n_channels]
        matrix = Matrix(comp_matrix, channels, fluorochrome_names)

        return matrix

    except Exception as e:
        raise FlowKitException(f"Error creating compensation matrix: {str(e)}")


def plot_channel(
    sample: Sample,
    channel_label_or_number: Union[str, int],
    source: str = "xform",
    subsample: bool = True,
    color_density: bool = True,
    bin_width: int = 4,
    event_mask: Optional[np.ndarray] = None,
    highlight_mask: Optional[np.ndarray] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    plot_width: int = 1000,
    plot_height: int = 400,
) -> figure:
    """
    Plot a 2-D histogram of the specified channel data with the x-axis as the event index.
    This is similar to plotting a channel vs Time, except the events are equally
    distributed along the x-axis.

    Args:
        sample (Sample): FlowKit Sample object
        channel_label_or_number (Union[str, int]): A channel's PnN label or number
        source (str): 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events are used for plotting
        subsample (bool): Whether to use all events for plotting or just the
            subsampled events. Default is True (subsampled events). Plotting
            subsampled events is much faster.
        color_density (bool): Whether to color the events by density, similar
            to a heat map. Default is True.
        bin_width (int): Bin size to use for the color density, in units of
            event point size. Larger values produce smoother gradients.
            Default is 4 for a 4x4 grid size.
        event_mask (Optional[np.ndarray]): Boolean array of events to plot. Takes precedence
            over highlight_mask (i.e. events marked False in event_mask will
            never be plotted).
        highlight_mask (Optional[np.ndarray]): Boolean array of event indices to highlight
            in color. Non-highlighted events will be light grey.
        x_min (Optional[float]): Lower bound of x-axis. If None, channel's min value will
            be used with some padding to keep events off the edge of the plot.
        x_max (Optional[float]): Upper bound of x-axis. If None, channel's max value will
            be used with some padding to keep events off the edge of the plot.
        y_min (Optional[float]): Lower bound of y-axis. If None, channel's min value will
            be used with some padding to keep events off the edge of the plot.
        y_max (Optional[float]): Upper bound of y-axis. If None, channel's max value will
            be used with some padding to keep events off the edge of the plot.
        plot_width (int): Width of the plot in pixels
        plot_height (int): Height of the plot in pixels

    Returns:
        figure: Bokeh Figure object containing the interactive channel plot

    Raises:
        FlowKitException: If plotting fails

    Example:
        >>> fig = plot_channel(sample, 'FSC-A', source='xform')
        >>> show(fig)
    """
    try:
        # Use the sample's built-in plot_channel method
        fig = sample.plot_channel(
            channel_label_or_number,
            source=source,
            subsample=subsample,
            color_density=color_density,
            bin_width=bin_width,
            event_mask=event_mask,
            highlight_mask=highlight_mask,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

        # Set custom dimensions
        fig.width = plot_width
        fig.height = plot_height

        return fig

    except Exception as e:
        raise FlowKitException(f"Error plotting channel: {str(e)}")


def plot_contour(
    sample: Sample,
    x_label_or_number: Union[str, int],
    y_label_or_number: Union[str, int],
    source: str = "xform",
    subsample: bool = True,
    plot_events: bool = False,
    fill: bool = False,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    plot_width: int = 600,
    plot_height: int = 600,
) -> figure:
    """
    Create a contour plot of the specified channel events.

    Args:
        sample (Sample): FlowKit Sample object
        x_label_or_number (Union[str, int]): A channel's PnN label or number for x-axis data
        y_label_or_number (Union[str, int]): A channel's PnN label or number for y-axis data
        source (str): 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events are used for plotting
        subsample (bool): Whether to use all events for plotting or just the
            subsampled events. Default is True (subsampled events). Running
            with all events is not recommended, as the Kernel Density
            Estimation is computationally demanding.
        plot_events (bool): Whether to display the event data points in
            addition to the contours. Default is False.
        fill (bool): Whether to fill in color between contour lines. Default is False.
        x_min (Optional[float]): Lower bound of x-axis. If None, channel's min value will
            be used with some padding to keep events off the edge of the plot.
        x_max (Optional[float]): Upper bound of x-axis. If None, channel's max value will
            be used with some padding to keep events off the edge of the plot.
        y_min (Optional[float]): Lower bound of y-axis. If None, channel's min value will
            be used with some padding to keep events off the edge of the plot.
        y_max (Optional[float]): Upper bound of y-axis. If None, channel's max value will
            be used with some padding to keep events off the edge of the plot.
        plot_width (int): Width of the plot in pixels
        plot_height (int): Height of the plot in pixels

    Returns:
        figure: Bokeh Figure object containing the interactive contour plot

    Raises:
        FlowKitException: If plotting fails

    Example:
        >>> fig = plot_contour(sample, 'FSC-A', 'SSC-A', source='xform')
        >>> show(fig)
    """
    try:
        # Use the sample's built-in plot_contour method
        fig = sample.plot_contour(
            x_label_or_number,
            y_label_or_number,
            source=source,
            subsample=subsample,
            plot_events=plot_events,
            fill=fill,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

        # Set custom dimensions
        fig.width = plot_width
        fig.height = plot_height

        return fig

    except Exception as e:
        raise FlowKitException(f"Error plotting contour: {str(e)}")


def plot_scatter(
    sample: Sample,
    x_label_or_number: Union[str, int],
    y_label_or_number: Union[str, int],
    source: str = "xform",
    subsample: bool = True,
    color_density: bool = True,
    bin_width: int = 4,
    event_mask: Optional[np.ndarray] = None,
    highlight_mask: Optional[np.ndarray] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    plot_width: int = 600,
    plot_height: int = 600,
) -> figure:
    """
    Create a scatter plot of the specified channel events.

    Args:
        sample (Sample): FlowKit Sample object
        x_label_or_number (Union[str, int]): A channel's PnN label or number for x-axis data
        y_label_or_number (Union[str, int]): A channel's PnN label or number for y-axis data
        source (str): 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events are used for plotting
        subsample (bool): Whether to use all events for plotting or just the
            subsampled events. Default is True (subsampled events). Plotting
            subsampled events is much faster.
        color_density (bool): Whether to color the events by density, similar
            to a heat map. Default is True.
        bin_width (int): Bin size to use for the color density, in units of
            event point size. Larger values produce smoother gradients.
            Default is 4 for a 4x4 grid size.
        event_mask (Optional[np.ndarray]): Boolean array of events to plot. Takes precedence
            over highlight_mask (i.e. events marked False in event_mask will
            never be plotted).
        highlight_mask (Optional[np.ndarray]): Boolean array of event indices to highlight
            in color. Non-highlighted events will be light grey.
        x_min (Optional[float]): Lower bound of x-axis. If None, channel's min value will
            be used with some padding to keep events off the edge of the plot.
        x_max (Optional[float]): Upper bound of x-axis. If None, channel's max value will
            be used with some padding to keep events off the edge of the plot.
        y_min (Optional[float]): Lower bound of y-axis. If None, channel's min value will
            be used with some padding to keep events off the edge of the plot.
        y_max (Optional[float]): Upper bound of y-axis. If None, channel's max value will
            be used with some padding to keep events off the edge of the plot.
        plot_width (int): Width of the plot in pixels
        plot_height (int): Height of the plot in pixels

    Returns:
        figure: Bokeh Figure object containing the interactive scatter plot

    Raises:
        FlowKitException: If plotting fails

    Example:
        >>> fig = plot_scatter(sample, 'FSC-A', 'SSC-A', source='xform')
        >>> show(fig)
    """
    try:
        # Use the sample's built-in plot_scatter method
        fig = sample.plot_scatter(
            x_label_or_number,
            y_label_or_number,
            source=source,
            subsample=subsample,
            color_density=color_density,
            bin_width=bin_width,
            event_mask=event_mask,
            highlight_mask=highlight_mask,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

        # Set custom dimensions
        fig.width = plot_width
        fig.height = plot_height

        return fig

    except Exception as e:
        raise FlowKitException(f"Error plotting scatter: {str(e)}")


def plot_scatter_matrix(
    sample: Sample,
    channel_labels_or_numbers: Optional[List[Union[str, int]]] = None,
    source: str = "xform",
    subsample: bool = True,
    event_mask: Optional[np.ndarray] = None,
    highlight_mask: Optional[np.ndarray] = None,
    color_density: bool = False,
    plot_height: int = 256,
    plot_width: int = 256,
) -> gridplot:
    """
    Create an interactive scatter plot matrix for all channel combinations
    except for the Time channel.

    Args:
        sample (Sample): FlowKit Sample object
        channel_labels_or_numbers (Optional[List[Union[str, int]]]): List of channel PnN labels or channel
            numbers to use for the scatter plot matrix. If None, then all
            channels will be plotted (except Time).
        source (str): 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events are used for plotting
        subsample (bool): Whether to use all events for plotting or just the
            subsampled events. Default is True (subsampled events). Plotting
            subsampled events is much faster.
        event_mask (Optional[np.ndarray]): Boolean array of events to plot. Takes precedence
            over highlight_mask (i.e. events marked False in event_mask will
            never be plotted).
        highlight_mask (Optional[np.ndarray]): Boolean array of event indices to highlight
            in color. Non-highlighted events will be light grey.
        color_density (bool): Whether to color the events by density, similar
            to a heat map. Default is False.
        plot_height (int): Height of each plot in pixels (screen units)
        plot_width (int): Width of each plot in pixels (screen units)

    Returns:
        gridplot: Bokeh GridPlot object containing the interactive scatter plot matrix

    Raises:
        FlowKitException: If plotting fails

    Example:
        >>> grid = plot_scatter_matrix(sample, ['FSC-A', 'SSC-A', 'FITC-A'], source='xform')
        >>> show(grid)
    """
    try:
        # Use the sample's built-in plot_scatter_matrix method
        grid = sample.plot_scatter_matrix(
            channel_labels_or_numbers,
            source=source,
            subsample=subsample,
            event_mask=event_mask,
            highlight_mask=highlight_mask,
            color_density=color_density,
            plot_height=plot_height,
            plot_width=plot_width,
        )

        return grid

    except Exception as e:
        raise FlowKitException(f"Error plotting scatter matrix: {str(e)}")


def plot_histogram(
    sample: Sample,
    channel_label_or_number: Union[str, int],
    source: str = "xform",
    subsample: bool = False,
    bins: Optional[Union[int, str]] = None,
    data_min: Optional[float] = None,
    data_max: Optional[float] = None,
    x_range: Optional[Tuple[float, float]] = None,
    plot_width: int = 600,
    plot_height: int = 400,
) -> figure:
    """
    Create a histogram plot of the specified channel events.

    Args:
        sample (Sample): FlowKit Sample object
        channel_label_or_number (Union[str, int]): A channel's PnN label or number to use
            for plotting the histogram
        source (str): 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events are used for plotting
        subsample (bool): Whether to use all events for plotting or just the
            subsampled events. Default is False (all events).
        bins (Optional[Union[int, str]]): Number of bins to use for the histogram or a string compatible
            with the NumPy histogram function. If None, the number of bins is
            determined by the square root rule.
        data_min (Optional[float]): Filter event data, removing events below specified value
        data_max (Optional[float]): Filter event data, removing events above specified value
        x_range (Optional[Tuple[float, float]]): Tuple of lower & upper bounds of x-axis. Used for modifying
            plot view, doesn't filter event data.
        plot_width (int): Width of the plot in pixels
        plot_height (int): Height of the plot in pixels

    Returns:
        figure: Bokeh Figure object containing the histogram plot

    Raises:
        FlowKitException: If plotting fails

    Example:
        >>> fig = plot_histogram(sample, 'FSC-A', source='xform')
        >>> show(fig)
    """
    try:
        # Use the sample's built-in plot_histogram method
        fig = sample.plot_histogram(
            channel_label_or_number,
            source=source,
            subsample=subsample,
            bins=bins,
            data_min=data_min,
            data_max=data_max,
            x_range=x_range,
        )

        # Set custom dimensions
        fig.width = plot_width
        fig.height = plot_height

        return fig

    except Exception as e:
        raise FlowKitException(f"Error plotting histogram: {str(e)}")


def plot_gate_overlay(
    sample: Sample,
    gate_id: Tuple[str, Tuple[str, ...]],
    gating_strategy: fk.GatingStrategy,
    source: str = "xform",
    subsample_count: int = 10000,
    random_seed: int = 1,
    event_mask: Optional[np.ndarray] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    color_density: bool = True,
    bin_width: int = 4,
    plot_width: int = 600,
    plot_height: int = 600,
) -> figure:
    """
    Create a plot showing the specified gate overlaid on the data.

    Args:
        sample (Sample): FlowKit Sample object
        gate_id (Tuple[str, Tuple[str, ...]]): Tuple of gate name and gate path (also a tuple)
        gating_strategy (fk.GatingStrategy): GatingStrategy containing gate_id
        source (str): 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events are used for plotting
        subsample_count (int): Number of events to use as a subsample. If the number of
            events in the Sample is less than the requested subsample count, then the
            maximum number of available events is used for the subsample.
        random_seed (int): Random seed used for subsampling events
        event_mask (Optional[np.ndarray]): Boolean array of events to plot (i.e. parent gate event membership)
        x_min (Optional[float]): Lower bound of x-axis. If None, channel's min value will
            be used with some padding to keep events off the edge of the plot.
        x_max (Optional[float]): Upper bound of x-axis. If None, channel's max value will
            be used with some padding to keep events off the edge of the plot.
        y_min (Optional[float]): Lower bound of y-axis. If None, channel's min value will
            be used with some padding to keep events off the edge of the plot.
        y_max (Optional[float]): Upper bound of y-axis. If None, channel's max value will
            be used with some padding to keep events off the edge of the plot.
        color_density (bool): Whether to color the events by density, similar
            to a heat map. Default is True.
        bin_width (int): Bin size to use for the color density, in units of
            event point size. Larger values produce smoother gradients.
            Default is 4 for a 4x4 grid size.
        plot_width (int): Width of the plot in pixels
        plot_height (int): Height of the plot in pixels

    Returns:
        figure: Bokeh Figure object containing the gate overlay plot

    Raises:
        FlowKitException: If plotting fails

    Example:
        >>> fig = plot_gate_overlay(sample, ('Cells', ('root',)), strategy)
        >>> show(fig)
    """
    try:
        # Import plot_utils from FlowKit
        from flowkit._utils import plot_utils

        # Use the plot_utils.plot_gate function
        fig = plot_utils.plot_gate(
            gate_id,
            gating_strategy,
            sample,
            subsample_count=subsample_count,
            random_seed=random_seed,
            event_mask=event_mask,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            color_density=color_density,
            bin_width=bin_width,
        )

        # Set custom dimensions
        fig.width = plot_width
        fig.height = plot_height

        return fig

    except Exception as e:
        raise FlowKitException(f"Error plotting gate overlay: {str(e)}")


def plot_population_comparison(
    samples: List[Sample],
    channel_x: Union[str, int],
    channel_y: Union[str, int],
    source: str = "xform",
    subsample: bool = True,
    color_density: bool = True,
    plot_width: int = 800,
    plot_height: int = 600,
) -> figure:
    """
    Create a scatter plot comparing multiple samples on the same channels.

    Args:
        samples (List[Sample]): List of FlowKit Sample objects
        channel_x (Union[str, int]): Channel for x-axis
        channel_y (Union[str, int]): Channel for y-axis
        source (str): 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events are used for plotting
        subsample (bool): Whether to use all events for plotting or just the
            subsampled events. Default is True (subsampled events).
        color_density (bool): Whether to color the events by density, similar
            to a heat map. Default is True.
        plot_width (int): Width of the plot in pixels
        plot_height (int): Height of the plot in pixels

    Returns:
        figure: Bokeh Figure object containing the population comparison plot

    Raises:
        FlowKitException: If plotting fails

    Example:
        >>> fig = plot_population_comparison([sample1, sample2], 'FSC-A', 'SSC-A')
        >>> show(fig)
    """
    try:
        # Create a single scatter plot with all samples
        fig = figure(
            tools="crosshair,hover,pan,zoom_in,zoom_out,box_zoom,undo,redo,reset,save",
            width=plot_width,
            height=plot_height,
        )

        # Define colors for different samples
        colors = Category10[10] if len(samples) <= 10 else Category20[20]

        for i, sample in enumerate(samples):
            # Get channel data
            x_index = sample.get_channel_index(channel_x)
            y_index = sample.get_channel_index(channel_y)

            x_data = sample.get_channel_events(
                x_index, source=source, subsample=subsample
            )
            y_data = sample.get_channel_events(
                y_index, source=source, subsample=subsample
            )

            # Create scatter plot for this sample
            fig.circle(
                x_data,
                y_data,
                size=3,
                alpha=0.6,
                color=colors[i % len(colors)],
                legend_label=sample.id,
            )

        # Set axis labels
        x_label = sample.pnn_labels[x_index]
        y_label = sample.pnn_labels[y_index]
        fig.xaxis.axis_label = x_label
        fig.yaxis.axis_label = y_label

        fig.title = Title(
            text=f"Population Comparison: {x_label} vs {y_label}", align="center"
        )

        return fig

    except Exception as e:
        raise FlowKitException(f"Error plotting population comparison: {str(e)}")


def plot_channel_statistics(
    sample: Sample,
    channels: List[Union[str, int]],
    source: str = "xform",
    subsample: bool = True,
    plot_width: int = 800,
    plot_height: int = 400,
) -> figure:
    """
    Create a bar plot showing statistical measures for multiple channels.

    Args:
        sample (Sample): FlowKit Sample object
        channels (List[Union[str, int]]): List of channels to analyze
        source (str): 'raw', 'comp', 'xform' for whether the raw, compensated
            or transformed events are used for plotting
        subsample (bool): Whether to use all events for plotting or just the
            subsampled events. Default is True (subsampled events).
        plot_width (int): Width of the plot in pixels
        plot_height (int): Height of the plot in pixels

    Returns:
        figure: Bokeh Figure object containing the channel statistics plot

    Raises:
        FlowKitException: If plotting fails

    Example:
        >>> fig = plot_channel_statistics(sample, ['FSC-A', 'SSC-A', 'FITC-A'])
        >>> show(fig)
    """
    try:
        # Calculate statistics for each channel
        channel_names = []
        means = []
        medians = []
        stds = []

        for channel in channels:
            channel_index = sample.get_channel_index(channel)
            channel_data = sample.get_channel_events(
                channel_index, source=source, subsample=subsample
            )

            channel_names.append(sample.pnn_labels[channel_index])
            means.append(float(np.mean(channel_data)))
            medians.append(float(np.median(channel_data)))
            stds.append(float(np.std(channel_data)))

        # Create bar plot
        fig = figure(
            tools="crosshair,hover,pan,zoom_in,zoom_out,box_zoom,undo,redo,reset,save",
            width=plot_width,
            height=plot_height,
            x_range=channel_names,
        )

        # Plot mean values
        fig.vbar(
            x=channel_names,
            top=means,
            width=0.8,
            color="blue",
            alpha=0.7,
            legend_label="Mean",
        )

        # Plot median values
        fig.vbar(
            x=channel_names,
            top=medians,
            width=0.6,
            color="red",
            alpha=0.7,
            legend_label="Median",
        )

        fig.title = Title(text="Channel Statistics", align="center")
        fig.xaxis.axis_label = "Channels"
        fig.yaxis.axis_label = "Value"

        return fig

    except Exception as e:
        raise FlowKitException(f"Error plotting channel statistics: {str(e)}")


if __name__ == "__main__":
    # Example usage and testing
    print("FlowKit Agent Functions loaded successfully!")
    print("Available functions:")

    # Get all functions defined in this module
    import inspect

    functions = [
        name
        for name, obj in globals().items()
        if inspect.isfunction(obj) and not name.startswith("_")
    ]

    for func_name in sorted(functions):
        func = globals()[func_name]
        if hasattr(func, "__doc__") and func.__doc__:
            first_line = func.__doc__.strip().split("\n")[0]
            print(f"  - {func_name}: {first_line}")
        else:
            print(f"  - {func_name}")
