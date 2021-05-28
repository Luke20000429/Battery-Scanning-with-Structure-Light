import numpy as np
import sys 
from readData import *
import plotly.graph_objs as go


def dynVisual(pointClouds, names, zaugment=1, s=2):
    data = []
    for points, name in zip(pointClouds, names):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
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
    fig.update_scenes(aspectmode='manual', aspectratio=dict(x=1, y=2, z=zaugment))
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
