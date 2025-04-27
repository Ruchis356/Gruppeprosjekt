import plotly.graph_objects as go
from plotly.subplots import make_subplots

class InteractiveGraphs:
    def __init__(self, weather_vars=None):
        self.template = "plotly_white"
        self.weather_vars = weather_vars or []
        # Initialize default colors (Plotly's color sequence)
        self.colors = [
            '#1f77b4',  # muted blue
            '#ff7f0e',  # safety orange
            '#2ca02c',  # cooked asparagus green
            '#d62728',  # brick red
            '#9467bd',  # muted purple
            '#8c564b',  # chestnut brown
            '#e377c2',  # raspberry yogurt pink
            '#7f7f7f',  # middle gray
            '#bcbd22',  # curry yellow-green
            '#17becf'   # blue-teal
        ]
    
    def dot_graph(self, df, columns, title, x_axis, y_axis):
        """Create interactive scatter plot with toggle buttons"""
        fig = go.Figure()
        
        # Add all traces (initially all visible)
        for i, col in enumerate(columns):
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df[col],
                    name=col,
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=self.colors[i % len(self.colors)]  # Now using initialized colors
                    ),
                    visible=True
                )
            )
        
        # Create simple toggle buttons (no JavaScript for now)
        buttons = [
            dict(
                label='✓ All',
                method='update',
                args=[{'visible': [True]*len(columns)}],
                args2=[{'title': 'Showing all pollutants'}]
            ),
            dict(
                label='✗ None',
                method='update',
                args=[{'visible': [False]*len(columns)}],
                args2=[{'title': 'Hiding all pollutants'}]
            )
        ]
        
        # Add individual toggle buttons
        buttons += [
            dict(
                label=f'● {col}',  # Using bullet instead of checkmark
                method='restyle',
                args=[{'visible': [col == trace.name for trace in fig.data]}],
                args2=[{'title': f'Showing {col}'}]
            ) for col in columns
        ]
        
        fig.update_layout(
            title=title,
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            template=self.template,
            hovermode="x unified",
            updatemenus=[
                dict(
                    type='buttons',
                    direction='right',
                    buttons=buttons,
                    pad={'r': 10, 't': 10},
                    x=0.1,
                    xanchor='left',
                    y=1.15,
                    yanchor='top',
                    bgcolor='#F5F5F5'
                )
            ],
            annotations=[
                dict(
                    text="Toggle:",
                    x=0,
                    xref="paper",
                    y=1.1,
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12)
                )
            ]
        )
        
        fig.show()







    def comparative_graph(self, df, columns, df_predictor, predictor, title, x_axis, y_axis, y_lims=None, zero_align=False):
        """Create interactive comparison plot with weather variable buttons"""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add pollution traces (primary y-axis)
        for i, col in enumerate(columns):
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df[col],
                    name=col,
                    mode='lines+markers',
                    marker=dict(size=4, color=self.colors[i % len(self.colors)]),
                    line=dict(width=1)
                ),
                secondary_y=False  # Correct placement of secondary_y parameter
            )
        
        # Add weather trace (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=df_predictor['Date'],
                y=df_predictor[predictor],
                name=predictor,
                mode='lines',
                line=dict(dash='dot', color='#666666', width=2)
            ),
            secondary_y=True  # Correct placement of secondary_y parameter
        )
        
        # Create weather variable buttons
        weather_buttons = [
            dict(
                method='update',
                label=var,
                args=[
                    {'y': [df_predictor[var]]},  # Update weather data
                    {'yaxis2.title': var}        # Update axis label
                ]
            ) for var in self.weather_vars
        ]
        
        # Create pollutant toggle buttons
        toggle_buttons = [
            dict(
                label='✓ All',
                method='update',
                args=[{'visible': [True]*(len(columns)+1)}]  # +1 for weather trace
            ),
            dict(
                label='✗ None',
                method='update',
                args=[{'visible': [False]*(len(columns)+1)}]
            )
        ]
        toggle_buttons += [
            dict(
                label=f'● {col}',
                method='restyle',
                args=[{'visible': [col == trace.name for trace in fig.data]}]
            ) for col in columns
        ]
        
        fig.update_layout(
            title=title,
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            template=self.template,
            hovermode="x unified",
            updatemenus=[
                # Weather variable selector
                dict(
                    type='buttons',
                    buttons=weather_buttons,
                    direction='right',
                    pad={'r': 10, 't': 10},
                    x=0.05,
                    xanchor='left',
                    y=1.15,
                    yanchor='top',
                    bgcolor='#F5F5F5'
                ),
                # Pollutant toggle
                dict(
                    type='buttons',
                    buttons=toggle_buttons,
                    direction='right',
                    pad={'r': 10, 't': 10},
                    x=0.05,
                    xanchor='left',
                    y=1.25,
                    yanchor='top',
                    bgcolor='#F5F5F5'
                )
            ],
            annotations=[
                dict(
                    text="Weather:",
                    x=0,
                    xref="paper",
                    y=1.12,
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12)
                ),
                dict(
                    text="Pollutants:",
                    x=0,
                    xref="paper",
                    y=1.22,
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12)
                )
            ]
        )
        
        if zero_align:
            fig.update_layout(yaxis2=dict(zeroline=True, zerolinewidth=2))
        
        fig.show()