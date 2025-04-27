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
        
        # Create simple toggle button
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
        """Create interactive comparison plot with proper toggle behavior"""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add pollution traces (all visible initially)
        self.pol_visible = [True] * len(columns)
        for i, col in enumerate(columns):
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df[col],
                    name=col,
                    mode='lines+markers',
                    marker=dict(size=4, color=self.colors[i % len(self.colors)]),
                    line=dict(width=1),
                    visible=True,
                    legendgroup='pollutants'
                ),
                secondary_y=False
            )
        
        # Add weather trace (visible initially)
        self.weather_visible = True
        fig.add_trace(
            go.Scatter(
                x=df_predictor['Date'],
                y=df_predictor[predictor],
                name=predictor,
                mode='lines',
                line=dict(dash='dot', color='#666666', width=2),
                visible=True,
                legendgroup='weather'
            ),
            secondary_y=True
        )
        
        # Create weather variable buttons
        weather_buttons = [
            dict(
                method='update',
                label=var,
                args=[
                    {'y': [df_predictor[var]], 'name': [var]},  # Update weather data
                    {'yaxis2.title': var},                      # Update axis label
                    {'visible': self.pol_visible + [True]}      # Keep current pollutant visibility
                ]
            ) for var in self.weather_vars
        ]
        
        # Create pollutant toggle buttons
        toggle_buttons = [
            dict(
                label='✓ All Pollutants',
                method='update',
                args=[{'visible': [True]*len(columns) + [self.weather_visible]}],
                args2=[{'title': 'Showing all pollutants'}]
            ),
            dict(
                label='✗ No Pollutants',
                method='update',
                args=[{'visible': [False]*len(columns) + [self.weather_visible]}],
                args2=[{'title': 'Hiding all pollutants'}]
            )
        ]
        
        # Add individual pollutant toggles
        for i, col in enumerate(columns):
            toggle_buttons.append(
                dict(
                    label=f'● {col}',
                    method='restyle',
                    args=[{
                        'visible': [
                            True if j == i else vis 
                            for j, vis in enumerate(self.pol_visible)
                        ] + [self.weather_visible]
                    }],
                    args2=[{
                        'visible': [
                            not self.pol_visible[j] if j == i else vis 
                            for j, vis in enumerate(self.pol_visible)
                        ] + [self.weather_visible]
                    }]
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            yaxis2_title=predictor,
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