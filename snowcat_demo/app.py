from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from snowcat_demo.model import (
    PRECISIONS,
    GemmMapping,
    GemmWorkload,
    attainable_metrics,
    best_at_capacity,
    compare_next_area_increment,
    divisors,
    doubling_buffer_gain_percent,
    evaluate_capacity,
    enumerate_mappings,
    estimate_mapping_traffic,
    pareto_frontier,
)
from snowcat_demo.model.pareto import is_pareto_point
from snowcat_demo.model.performance import operational_intensity, throughput_tflops
from snowcat_demo.model.traffic import LOOP_ORDERS


st.set_page_config(
    page_title="Snowcat GEMM Explorer",
    page_icon=".",
    layout="wide",
)


PRESETS = {
    "64 x 64 x 64": (64, 64, 64),
    "32 x 32 x 32": (32, 32, 32),
    "64 x 128 x 64": (64, 128, 64),
    "128 x 32 x 128": (128, 32, 128),
    "Custom": None,
}


LECTURE_PRESETS = {
    "Manual": {
        "workload": "64 x 64 x 64",
        "m0": 16,
        "k0": 16,
        "n0": 16,
        "order": "M-N-K",
        "capacity_kib": 64.0,
        "bandwidth": 1000.0,
        "peak": 120.0,
        "sram_increment_kib": 64.0,
        "compute_increment": 25.0,
    },
    "Preset A: inefficient 64^3 mapping": {
        "workload": "64 x 64 x 64",
        "m0": 1,
        "k0": 1,
        "n0": 1,
        "order": "M-K-N",
        "capacity_kib": 4.0,
        "bandwidth": 100.0,
        "peak": 500.0,
        "sram_increment_kib": 64.0,
        "compute_increment": 50.0,
    },
    "Preset B: generate ski slope": {
        "workload": "64 x 64 x 64",
        "m0": 8,
        "k0": 8,
        "n0": 16,
        "order": "M-N-K",
        "capacity_kib": 16.0,
        "bandwidth": 100.0,
        "peak": 500.0,
        "sram_increment_kib": 64.0,
        "compute_increment": 50.0,
    },
    "Preset C: small-buffer architecture": {
        "workload": "64 x 128 x 64",
        "m0": 8,
        "k0": 8,
        "n0": 8,
        "order": "M-N-K",
        "capacity_kib": 8.0,
        "bandwidth": 100.0,
        "peak": 500.0,
        "sram_increment_kib": 64.0,
        "compute_increment": 50.0,
    },
    "Preset D: buffer-versus-compute trade-off": {
        "workload": "128 x 32 x 128",
        "m0": 16,
        "k0": 8,
        "n0": 16,
        "order": "M-N-K",
        "capacity_kib": 16.0,
        "bandwidth": 100.0,
        "peak": 100.0,
        "sram_increment_kib": 64.0,
        "compute_increment": 25.0,
    },
}


def format_bytes(value: int | float) -> str:
    value = float(value)
    units = ["B", "KiB", "MiB", "GiB"]
    unit = units[0]
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            break
        value /= 1024
    return f"{value:.2f} {unit}" if value < 100 else f"{value:.0f} {unit}"


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def default_tile(options: list[int], target: int) -> int:
    candidates = [value for value in options if value <= target]
    return candidates[-1] if candidates else options[0]


def point_key(point) -> tuple:
    return (
        point.mapping.m0,
        point.mapping.k0,
        point.mapping.n0,
        point.mapping.loop_order,
        point.buffer_bytes,
        point.backing_store_bytes,
    )


@st.cache_data(show_spinner=False)
def cached_points(m: int, k: int, n: int, bytes_per_element: int):
    return enumerate_mappings(GemmWorkload(m, k, n, bytes_per_element))


@st.cache_data(show_spinner=False)
def cached_mapspace(m: int, k: int, n: int, bytes_per_element: int):
    points = enumerate_mappings(GemmWorkload(m, k, n, bytes_per_element))
    return points, pareto_frontier(points)


def points_dataframe(points, workload: GemmWorkload, frontier) -> pd.DataFrame:
    frontier_keys = {point_key(point) for point in frontier}
    rows = []
    for point in points:
        is_frontier = point_key(point) in frontier_keys
        rows.append(
            {
                "buffer_kib": point.buffer_bytes / 1024,
                "traffic_kib": point.backing_store_bytes / 1024,
                "buffer_bytes": point.buffer_bytes,
                "traffic_bytes": point.backing_store_bytes,
                "pareto": is_frontier,
                "class": "Pareto" if is_frontier else "Dominated",
                "tile": f"{point.mapping.m0} x {point.mapping.k0} x {point.mapping.n0}",
                "loop_order": point.mapping.order_label,
                "oi": workload.operations / point.backing_store_bytes,
            }
        )
    return pd.DataFrame(rows)


def frontier_dataframe(frontier) -> pd.DataFrame:
    best_by_capacity: dict[int, object] = {}
    for point in frontier:
        existing = best_by_capacity.get(point.buffer_bytes)
        if existing is None or point.backing_store_bytes < existing.backing_store_bytes:
            best_by_capacity[point.buffer_bytes] = point
    rows = [
        {
            "buffer_kib": point.buffer_bytes / 1024,
            "traffic_kib": point.backing_store_bytes / 1024,
            "hover": (
                f"{point.mapping.label}"
                f"<br>Buffer: {format_bytes(point.buffer_bytes)}"
                f"<br>Traffic: {format_bytes(point.backing_store_bytes)}"
            ),
        }
        for point in sorted(best_by_capacity.values(), key=lambda item: item.buffer_bytes)
    ]
    return pd.DataFrame(rows)


def make_ski_slope_figure(
    points,
    frontier,
    workload: GemmWorkload,
    selected_point,
    capacity_kib: float,
    show_all: bool,
    show_dominated: bool,
    show_algorithmic_minimum: bool,
    show_selected: bool,
) -> go.Figure:
    df = points_dataframe(points, workload, frontier)
    fig = go.Figure()

    if show_all:
        fig.add_trace(
            go.Scattergl(
                x=df["buffer_kib"],
                y=df["traffic_kib"],
                mode="markers",
                name="All mappings",
                marker={"color": "rgba(98, 113, 130, 0.30)", "size": 7},
                hoverinfo="skip",
            )
        )

    if show_dominated:
        dominated = df[~df["pareto"]]
        fig.add_trace(
            go.Scattergl(
                x=dominated["buffer_kib"],
                y=dominated["traffic_kib"],
                mode="markers",
                name="Dominated",
                marker={"color": "rgba(205, 74, 74, 0.55)", "size": 7},
                hoverinfo="skip",
            )
        )

    fdf = frontier_dataframe(frontier)
    fig.add_trace(
        go.Scatter(
            x=fdf["buffer_kib"],
            y=fdf["traffic_kib"],
            mode="lines+markers",
            name="Pareto frontier",
            line={"color": "#168f5a", "width": 4},
            marker={"color": "#168f5a", "size": 9},
            text=fdf["hover"],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    if show_selected:
        fig.add_trace(
            go.Scatter(
                x=[selected_point.buffer_bytes / 1024],
                y=[selected_point.backing_store_bytes / 1024],
                mode="markers",
                name="Selected mapping",
                marker={
                    "color": "#2454d6",
                    "size": 15,
                    "line": {"color": "white", "width": 2},
                },
                text=[selected_point.mapping.label],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    if show_algorithmic_minimum:
        fig.add_hline(
            y=workload.algorithmic_minimum_bytes / 1024,
            line_dash="dash",
            line_color="#4d4d4d",
            annotation_text="Algorithmic minimum",
            annotation_position="bottom right",
        )

    fig.add_vline(
        x=capacity_kib,
        line_dash="dot",
        line_color="#111827",
        annotation_text="Capacity",
        annotation_position="top",
    )
    fig.update_layout(
        height=560,
        margin={"l": 16, "r": 16, "t": 20, "b": 16},
        xaxis_title="Required Snowcat buffer (KiB)",
        yaxis_title="Backing-store traffic (KiB)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        template="plotly_white",
    )
    return fig


def tile_heatmap(
    rows: int,
    cols: int,
    row_tile: int,
    col_tile: int,
    row_tile_size: int,
    col_tile_size: int,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=cols,
        y1=rows,
        fillcolor="#eef2f7",
        line={"color": "#cbd5e1", "width": 1},
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=col_tile * col_tile_size,
        y0=row_tile * row_tile_size,
        x1=(col_tile + 1) * col_tile_size,
        y1=(row_tile + 1) * row_tile_size,
        fillcolor="#2454d6",
        line={"color": "#1d4ed8", "width": 1},
    )
    fig.add_trace(
        go.Scatter(
            x=[cols / 2],
            y=[rows / 2],
            mode="text",
            text=[f"{rows} x {cols}"],
            textfont={"color": "#334155", "size": 14},
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=title,
        height=210,
        margin={"l": 8, "r": 8, "t": 36, "b": 8},
        xaxis={
            "showticklabels": False,
            "showgrid": False,
            "range": [0, cols],
            "zeroline": False,
        },
        yaxis={
            "showticklabels": False,
            "showgrid": False,
            "range": [rows, 0],
            "zeroline": False,
            "scaleanchor": "x",
        },
        template="plotly_white",
    )
    return fig


def architecture_figure(metrics, capacity_kib: float) -> go.Figure:
    df = pd.DataFrame([asdict(metric) for metric in metrics])
    df["capacity_kib"] = df["capacity_bytes"] / 1024
    df["traffic_kib"] = df["traffic_bytes"] / 1024

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Traffic bound", "Attainable OI", "Throughput ceiling"),
    )
    fig.add_trace(
        go.Scatter(
            x=df["capacity_kib"],
            y=df["traffic_kib"],
            mode="lines+markers",
            line={"color": "#168f5a", "width": 3},
            name="Traffic",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["capacity_kib"],
            y=df["operational_intensity"],
            mode="lines+markers",
            line={"color": "#2454d6", "width": 3},
            name="OI",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["capacity_kib"],
            y=df["performance_tflops"],
            mode="lines+markers",
            line={"color": "#9a5b13", "width": 3},
            name="Performance",
        ),
        row=3,
        col=1,
    )
    fig.add_vline(x=capacity_kib, line_dash="dot", line_color="#111827")
    fig.update_yaxes(title_text="KiB", row=1, col=1)
    fig.update_yaxes(title_text="OP/B", row=2, col=1)
    fig.update_yaxes(title_text="TFLOP/s", row=3, col=1)
    fig.update_xaxes(title_text="Required Snowcat buffer (KiB)", row=3, col=1)
    fig.update_layout(
        height=720,
        margin={"l": 16, "r": 16, "t": 56, "b": 16},
        showlegend=False,
        template="plotly_white",
    )
    return fig


def decision_figure(decision) -> go.Figure:
    labels = ["Current", "Add SRAM", "Add compute"]
    values = [
        decision.baseline.performance_tflops if decision.baseline else 0.0,
        decision.sram_option.performance_tflops if decision.sram_option else 0.0,
        decision.compute_option.performance_tflops if decision.compute_option else 0.0,
    ]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=["#64748b", "#168f5a", "#2454d6"],
            text=[f"{value:.2f}" for value in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        height=320,
        yaxis_title="TFLOP/s",
        margin={"l": 8, "r": 8, "t": 24, "b": 8},
        template="plotly_white",
    )
    return fig


def recommendation_label(recommendation) -> str:
    if recommendation == "sram":
        return "Add SRAM"
    if recommendation == "compute":
        return "Add compute"
    if recommendation == "tie":
        return "Either option"
    return "No eligible mapping"


with st.sidebar:
    st.header("Lecture preset")
    lecture_preset_name = st.selectbox("Preset", list(LECTURE_PRESETS), index=0)
    lecture_preset = LECTURE_PRESETS[lecture_preset_name]

    st.header("Workload")
    default_workload_index = list(PRESETS).index(lecture_preset["workload"])
    preset_name = st.selectbox(
        "Workload preset",
        list(PRESETS),
        index=default_workload_index,
        key=f"workload_{lecture_preset_name}",
    )
    preset = PRESETS[preset_name]
    if preset is None:
        m = st.number_input("M", min_value=1, max_value=8192, value=64, step=1)
        k = st.number_input("K", min_value=1, max_value=8192, value=64, step=1)
        n = st.number_input("N", min_value=1, max_value=8192, value=64, step=1)
    else:
        m, k, n = preset
        st.caption(f"M={m}, K={k}, N={n}")

    precision = st.selectbox("Precision", list(PRECISIONS), index=1)
    bytes_per_element = PRECISIONS[precision]
    workload = GemmWorkload(int(m), int(k), int(n), bytes_per_element)

    st.header("Mapping")
    m_divs = divisors(workload.m)
    k_divs = divisors(workload.k)
    n_divs = divisors(workload.n)
    m0_default = default_tile(m_divs, int(lecture_preset["m0"]))
    k0_default = default_tile(k_divs, int(lecture_preset["k0"]))
    n0_default = default_tile(n_divs, int(lecture_preset["n0"]))
    m0 = st.selectbox(
        "M0",
        m_divs,
        index=m_divs.index(m0_default),
        key=f"m0_{lecture_preset_name}_{preset_name}",
    )
    k0 = st.selectbox(
        "K0",
        k_divs,
        index=k_divs.index(k0_default),
        key=f"k0_{lecture_preset_name}_{preset_name}",
    )
    n0 = st.selectbox(
        "N0",
        n_divs,
        index=n_divs.index(n0_default),
        key=f"n0_{lecture_preset_name}_{preset_name}",
    )
    order_labels = ["-".join(order) for order in LOOP_ORDERS]
    order_default = lecture_preset["order"]
    order_label = st.selectbox(
        "Loop order",
        order_labels,
        index=order_labels.index(order_default),
        key=f"order_{lecture_preset_name}_{preset_name}",
    )
    loop_order = tuple(order_label.split("-"))
    selected_mapping = GemmMapping(int(m0), int(k0), int(n0), loop_order)

    st.header("Architecture")
    capacity_kib = st.number_input(
        "Available buffer (KiB)",
        min_value=0.0,
        value=float(lecture_preset["capacity_kib"]),
        step=16.0,
        key=f"capacity_{lecture_preset_name}",
    )
    bandwidth_gb_s = st.number_input(
        "Memory bandwidth (GB/s)",
        min_value=1.0,
        value=float(lecture_preset["bandwidth"]),
        step=50.0,
        key=f"bandwidth_{lecture_preset_name}",
    )
    peak_tflops = st.number_input(
        "Peak compute (TFLOP/s)",
        min_value=1.0,
        value=float(lecture_preset["peak"]),
        step=10.0,
        key=f"peak_{lecture_preset_name}",
    )
    sram_increment_kib = st.number_input(
        "SRAM increment (KiB)",
        min_value=0.0,
        value=float(lecture_preset["sram_increment_kib"]),
        step=16.0,
        key=f"sram_increment_{lecture_preset_name}",
    )
    compute_increment_tflops = st.number_input(
        "Compute increment (TFLOP/s)",
        min_value=0.0,
        value=float(lecture_preset["compute_increment"]),
        step=5.0,
        key=f"compute_increment_{lecture_preset_name}",
    )

    st.header("Display")
    st.caption("Traffic enumeration uses exact closed-form CPU accounting for this Snowcat model.")
    show_all = st.checkbox("Show all mappings", value=True)
    show_dominated = st.checkbox("Show dominated mappings", value=False)
    show_algorithmic_minimum = st.checkbox("Show algorithmic minimum", value=True)
    show_selected = st.checkbox("Show selected mapping", value=True)
    show_oi_mesa = st.checkbox("Show OI mesa", value=True)


points, frontier = cached_mapspace(
    workload.m, workload.k, workload.n, workload.bytes_per_element
)
selected_traffic = estimate_mapping_traffic(
    workload,
    selected_mapping.m0,
    selected_mapping.k0,
    selected_mapping.n0,
    selected_mapping.loop_order,
)
selected_point = next(
    point
    for point in points
    if point.mapping == selected_mapping
    and point.buffer_bytes == selected_traffic.buffer_bytes
    and point.backing_store_bytes == selected_traffic.total_bytes
)
capacity_bytes = int(capacity_kib * 1024)
best_point = best_at_capacity(points, capacity_bytes)

st.title("Snowcat GEMM Explorer")
st.caption(
    "A lecture demo for mapping buffer footprint to backing-store traffic, "
    "then reading the Pareto frontier as an Orojenesis-style ski-slope bound."
)

tab_mapping, tab_slope, tab_architecture, tab_challenge = st.tabs(
    ["Mapping anatomy", "Ski slope", "Decision lab", "Challenge mode"]
)

with tab_mapping:
    st.subheader("Selected mapping")
    cols = st.columns(4)
    cols[0].metric("Buffer footprint", format_bytes(selected_point.buffer_bytes))
    cols[1].metric("Backing-store traffic", format_bytes(selected_point.backing_store_bytes))
    cols[2].metric(
        "Operational intensity",
        f"{operational_intensity(workload, selected_point.backing_store_bytes):.2f} OP/B",
    )
    cols[3].metric("Loop order", selected_mapping.order_label)

    st.latex(
        r"S_{\mathrm{buffer}} = b(M_0K_0 + K_0N_0 + M_0N_0)"
    )
    st.latex(
        r"T_{\mathrm{backing}} = b(MN_1K + M_1NK + MN)"
    )

    a_tile, w_tile, b_tile = workload.tile_bytes(m0, k0, n0)
    footprint_df = pd.DataFrame(
        {
            "Tensor": ["A tile", "W tile", "B tile"],
            "Bytes": [a_tile, w_tile, b_tile],
        }
    )
    access_df = pd.DataFrame(
        {
            "Traffic": ["A reads", "W reads", "B reads", "B writes"],
            "Bytes": [
                selected_traffic.a_read_bytes,
                selected_traffic.w_read_bytes,
                selected_traffic.b_read_bytes,
                selected_traffic.b_write_bytes,
            ],
        }
    )

    left, right = st.columns(2)
    with left:
        fig = go.Figure(
            go.Bar(
                x=footprint_df["Tensor"],
                y=footprint_df["Bytes"],
                marker_color=["#2454d6", "#168f5a", "#9a5b13"],
                text=[format_bytes(value) for value in footprint_df["Bytes"]],
                textposition="auto",
            )
        )
        fig.update_layout(
            title="Buffer occupancy",
            yaxis_title="Bytes",
            height=320,
            template="plotly_white",
            margin={"l": 8, "r": 8, "t": 48, "b": 8},
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = go.Figure(
            go.Bar(
                x=access_df["Traffic"],
                y=access_df["Bytes"],
                marker_color=["#2454d6", "#168f5a", "#b65a5a", "#9a5b13"],
                text=[format_bytes(value) for value in access_df["Bytes"]],
                textposition="auto",
            )
        )
        fig.update_layout(
            title="Backing-store access breakdown",
            yaxis_title="Bytes",
            height=320,
            template="plotly_white",
            margin={"l": 8, "r": 8, "t": 48, "b": 8},
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Tile view")
    heat_cols = st.columns(3)
    heat_cols[0].plotly_chart(
        tile_heatmap(workload.m, workload.k, 0, 0, m0, k0, "A[M,K] tile"),
        width="stretch",
    )
    heat_cols[1].plotly_chart(
        tile_heatmap(workload.k, workload.n, 0, 0, k0, n0, "W[K,N] tile"),
        width="stretch",
    )
    heat_cols[2].plotly_chart(
        tile_heatmap(workload.m, workload.n, 0, 0, m0, n0, "B[M,N] tile"),
        width="stretch",
    )

with tab_slope:
    st.subheader("Mapping cloud and Pareto ski slope")
    st.plotly_chart(
        make_ski_slope_figure(
            points,
            frontier,
            workload,
            selected_point,
            capacity_kib,
            show_all,
            show_dominated,
            show_algorithmic_minimum,
            show_selected,
        ),
        width="stretch",
    )
    st.info(
        "Snowcat is an optimistic two-level capacity model. It does not model "
        "spatial duplication, distributed storage, interconnect traffic, bank "
        "conflicts, latency hiding, or implementation constraints."
    )
    st.caption(
        f"Enumerated {len(points):,} mappings; {len(frontier):,} mapping points lie on the Pareto frontier."
    )

with tab_architecture:
    st.subheader("Decision lab")
    st.caption("Fixed area thought experiment: should the next increment go to SRAM or compute?")
    metrics = attainable_metrics(workload, points, bandwidth_gb_s, peak_tflops)
    decision = compare_next_area_increment(
        workload,
        points,
        capacity_bytes,
        bandwidth_gb_s,
        peak_tflops,
        int(sram_increment_kib * 1024),
        compute_increment_tflops,
    )
    if best_point is None:
        st.warning("The selected capacity is smaller than every enumerated mapping footprint.")
    else:
        best_oi = operational_intensity(workload, best_point.backing_store_bytes)
        best_perf = throughput_tflops(best_oi, bandwidth_gb_s, peak_tflops)
        cols = st.columns(4)
        cols[0].metric("Best traffic at capacity", format_bytes(best_point.backing_store_bytes))
        cols[1].metric("Attainable OI", f"{best_oi:.2f} OP/B")
        cols[2].metric("Throughput ceiling", f"{best_perf:.2f} TFLOP/s")
        cols[3].metric("Best mapping", best_point.mapping.order_label)

    if decision.baseline and decision.sram_option and decision.compute_option:
        st.subheader("Next area increment")
        decision_cols = st.columns(4)
        decision_cols[0].metric(
            "Current bottleneck", decision.baseline.bottleneck.value.title()
        )
        decision_cols[1].metric(
            "Add SRAM gain", format_percent(decision.sram_gain_percent)
        )
        decision_cols[2].metric(
            "Add compute gain", format_percent(decision.compute_gain_percent)
        )
        decision_cols[3].metric(
            "Recommendation", recommendation_label(decision.recommendation)
        )
        st.plotly_chart(decision_figure(decision), width="stretch")
        decision_table = pd.DataFrame(
            [
                {
                    "Scenario": "Current",
                    "Capacity": format_bytes(decision.baseline.capacity_bytes),
                    "Traffic": format_bytes(decision.baseline.traffic_bytes),
                    "OI": f"{decision.baseline.operational_intensity:.2f}",
                    "Performance": f"{decision.baseline.performance_tflops:.2f}",
                    "Bottleneck": decision.baseline.bottleneck.value,
                },
                {
                    "Scenario": "Add SRAM",
                    "Capacity": format_bytes(decision.sram_option.capacity_bytes),
                    "Traffic": format_bytes(decision.sram_option.traffic_bytes),
                    "OI": f"{decision.sram_option.operational_intensity:.2f}",
                    "Performance": f"{decision.sram_option.performance_tflops:.2f}",
                    "Bottleneck": decision.sram_option.bottleneck.value,
                },
                {
                    "Scenario": "Add compute",
                    "Capacity": format_bytes(decision.compute_option.capacity_bytes),
                    "Traffic": format_bytes(decision.compute_option.traffic_bytes),
                    "OI": f"{decision.compute_option.operational_intensity:.2f}",
                    "Performance": f"{decision.compute_option.performance_tflops:.2f}",
                    "Bottleneck": decision.compute_option.bottleneck.value,
                },
            ]
        )
        st.dataframe(decision_table, width="stretch", hide_index=True)

    if show_oi_mesa and metrics:
        st.plotly_chart(architecture_figure(metrics, capacity_kib), width="stretch")

with tab_challenge:
    st.subheader("Challenge mode")
    st.caption("Use this tab during Q&A or after the live demo.")
    selected_is_pareto = is_pareto_point(selected_point, frontier)

    challenge_type = st.selectbox(
        "Challenge",
        [
            "Find the best mapping under capacity",
            "Guess whether the selected mapping is Pareto-optimal",
            "Predict whether doubling buffer capacity helps",
            "Choose SRAM or compute for the next area increment",
        ],
    )

    if challenge_type == "Find the best mapping under capacity":
        st.write("Task: find a mapping that fits under the capacity marker and minimizes traffic.")

        if selected_point.buffer_bytes > capacity_bytes:
            st.warning("The selected mapping does not fit under the current capacity marker.")

        if best_point is None:
            st.warning("No enumerated mapping fits under the current capacity marker.")
        else:
            gap = selected_point.backing_store_bytes / best_point.backing_store_bytes
            comparison = pd.DataFrame(
                [
                    {
                        "Mapping": "Selected",
                        "Tile": f"{m0} x {k0} x {n0}",
                        "Loop order": selected_mapping.order_label,
                        "Buffer": format_bytes(selected_point.buffer_bytes),
                        "Traffic": format_bytes(selected_point.backing_store_bytes),
                        "OI": f"{operational_intensity(workload, selected_point.backing_store_bytes):.2f}",
                    },
                    {
                        "Mapping": "Best under capacity",
                        "Tile": (
                            f"{best_point.mapping.m0} x {best_point.mapping.k0} x {best_point.mapping.n0}"
                        ),
                        "Loop order": best_point.mapping.order_label,
                        "Buffer": format_bytes(best_point.buffer_bytes),
                        "Traffic": format_bytes(best_point.backing_store_bytes),
                        "OI": f"{operational_intensity(workload, best_point.backing_store_bytes):.2f}",
                    },
                ]
            )
            st.dataframe(comparison, width="stretch", hide_index=True)
            st.metric("Selected traffic / attainable bound", f"{gap:.2f}x")

    elif challenge_type == "Guess whether the selected mapping is Pareto-optimal":
        guess = st.radio("Audience guess", ["Pareto-optimal", "Dominated"], horizontal=True)
        reveal = st.checkbox("Reveal answer", key="reveal_pareto")
        if reveal:
            answer = "Pareto-optimal" if selected_is_pareto else "Dominated"
            st.metric("Answer", answer)
            st.write("Correct." if guess == answer else "Not for this selected mapping.")

    elif challenge_type == "Predict whether doubling buffer capacity helps":
        guess = st.radio("Audience guess", ["Helps", "Does not help"], horizontal=True)
        reveal = st.checkbox("Reveal answer", key="reveal_double_buffer")
        gain = doubling_buffer_gain_percent(
            workload, points, capacity_bytes, bandwidth_gb_s, peak_tflops
        )
        if reveal:
            helps = gain is not None and gain > 1.0
            answer = "Helps" if helps else "Does not help"
            st.metric("Performance gain from 2x buffer", format_percent(gain))
            st.write("Correct." if guess == answer else "Not under this capacity and roofline setting.")

    elif challenge_type == "Choose SRAM or compute for the next area increment":
        guess = st.radio(
            "Audience guess",
            ["Add SRAM", "Add compute", "Either option"],
            horizontal=True,
        )
        reveal = st.checkbox("Reveal answer", key="reveal_area_choice")
        decision = compare_next_area_increment(
            workload,
            points,
            capacity_bytes,
            bandwidth_gb_s,
            peak_tflops,
            int(sram_increment_kib * 1024),
            compute_increment_tflops,
        )
        if reveal:
            answer = recommendation_label(decision.recommendation)
            st.metric("Recommendation", answer)
            st.write("Correct." if guess == answer else "The bound favors a different next increment.")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Option": "Add SRAM", "Gain": format_percent(decision.sram_gain_percent)},
                        {
                            "Option": "Add compute",
                            "Gain": format_percent(decision.compute_gain_percent),
                        },
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
