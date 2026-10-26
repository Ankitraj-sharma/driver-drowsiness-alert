"""
Analytics, Telemetry Charts & Trip Safety Reporting for DrowsyGuard AI.
"""

from collections import deque
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def vigilance_gauge(score: int) -> go.Figure:
    """Circular gauge chart displaying driver vigilance index (0-100%)."""
    color = "#00D2BA" if score >= 80 else ("#FBC531" if score >= 50 else "#FF4757")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Vigilance Index", "font": {"size": 18, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#888"},
                "bar": {"color": color},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.1)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(255, 71, 87, 0.25)"},
                    {"range": [50, 80], "color": "rgba(251, 197, 49, 0.25)"},
                    {"range": [80, 100], "color": "rgba(0, 210, 186, 0.25)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EDF3"),
    )
    return fig


def ear_mar_time_series(
    ear_history: list[float],
    mar_history: list[float],
    ear_threshold: float,
    mar_threshold: float,
) -> go.Figure:
    """Plotly live dual-line chart showing EAR and MAR trends with danger limits."""
    frames = list(range(len(ear_history)))
    fig = go.Figure()

    # EAR curve
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=ear_history,
            mode="lines",
            name="EAR (Eye Aspect Ratio)",
            line=dict(color="#00D2BA", width=2.5),
        )
    )
    # EAR Threshold line
    fig.add_hline(
        y=ear_threshold,
        line_dash="dot",
        line_color="#FF4757",
        annotation_text=f"Micro-sleep Threshold ({ear_threshold})",
        annotation_position="bottom right",
    )

    # MAR curve
    fig.add_trace(
        go.Scatter(
            x=frames,
            y=mar_history,
            mode="lines",
            name="MAR (Mouth Aspect Ratio)",
            line=dict(color="#FFA502", width=2.5),
        )
    )
    # MAR Threshold line
    fig.add_hline(
        y=mar_threshold,
        line_dash="dot",
        line_color="#FFA502",
        annotation_text=f"Yawn Threshold ({mar_threshold})",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Live Kinematic Trend (EAR & MAR)",
        xaxis_title="Recent Frames",
        yaxis_title="Ratio Value",
        height=280,
        margin=dict(l=0, r=0, t=35, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EDF3"),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def incidents_dataframe(incidents: list[dict]) -> pd.DataFrame:
    """Format incident records into a neat presentation table."""
    if not incidents:
        return pd.DataFrame(columns=["Time", "Event Type", "Telemetry Details"])
    return pd.DataFrame(
        [
            {
                "Time": inc["time"],
                "Event Type": inc["type"],
                "Telemetry Details": inc["details"],
            }
            for inc in reversed(incidents)
        ]
    )
