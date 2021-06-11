import numpy as np
import sys 
from readData import *
import plotly.graph_objs as go


def dynVisual(pointClouds, names, zaugment=1, s=2):
    data = []
    xlim = [0.0, 0.0]
    ylim = [0.0, 0.0]
    for points, name in zip(pointClouds, names):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        xlim[0] = min(x.min(), xlim[0])
        xlim[1] = max(x.max(), xlim[1])
        ylim[0] = min(y.min(), ylim[0])
        ylim[1] = max(y.max(), ylim[1])
        if points.shape[1] == 4:
            colors = points[:, 3]
        else:
            colors = z
    
        trace = go.Scatter3d(
            name=name,
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=s,
                color=colors,                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=1
            )
        )
        data.append(trace)

    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    camera = dict(
        up=dict(x=1., y=0., z=1.),
        eye=dict(x=0., y=0., z=2.5)
    )

    
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(scene_camera=camera)
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    fig.update_scenes(aspectmode='manual', aspectratio=dict(x=1, y=(ylim[1]-ylim[0])//(xlim[1]-xlim[0]), z=zaugment))
    return fig


def addCloud(fig, pointClouds, names, prange=(200, 150, 100), skip=20):
    for points, name in zip(pointClouds, names):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
    
        trace = go.Scatter3d(
            name=name,
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=skip/10,
                color=z,                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=1
            )
        )
        fig.add_trace(trace)
    return fig
