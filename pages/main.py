import io
import base64
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, callback, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots

# to indicate this is a page of the app
dash.register_page(__name__, title="JEJARD Analytics")

############################################### Layout ###############################################

# define styling for different sections of the application
HEADER_STYLE = {
    "position": "fixed",
    "width": "100%",
    "height": "5%",
    "backgroundColor": "#246700"
}

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": "5%",
    "width": "21%",
    "height": "100vh", 
    "backgroundColor": "#9bc6b1"
}

CONTENT_STYLE = {
    "position": "absolute",
    "top": "7%",
    "left": "22%",
    "width": "77%",
    "height": "93%",
    "overflowY": "scroll",
    "backgroundColor": "white"
}

# Add components and styling to the different sections of the application
header = html.Div(
    [
        # Display application name on top left-hand corner
        html.H3('JEJARD Analytics', style={'position': 'fixed', 'left': '1%', 'color': 'white'}),
        
        # Create horizontal checklist for user to select which visualizations to display
        dcc.Checklist(
            options=[
                {'label': ' Box Plot', 'value': 'Box Plot'},
                {'label': ' Bar Graph', 'value': 'Bar Graph'},
                {'label': ' Scatter Plot', 'value': 'Scatter Plot'},
                {'label': ' Pie Graph', 'value': 'Pie Graph'},
                {'label': ' Human Silhouette', 'value': 'Human Silhouette'},
            ], value=['Human Silhouette', 'Pie Graph'], id='checklist',
            inline=True, labelStyle={'color': 'white', 'float': 'right', 'marginRight': '1rem'}
        ),
    ],
    style=HEADER_STYLE
)

sidebar = html.Div(
    [
        # Display patient information
        # TODO: Automate this
        html.H4('Patient Information', style={'position': 'fixed', 'left': '1%', 'top': '7%'}),
        html.P('Age: XX', style={'position': 'fixed', 'left': '2%', 'top': '12%'}),
        html.P('Weight: XX', style={'position': 'fixed', 'left': '2%', 'top': '15%'}),
        html.P('Admitted: DD/MM/YYYY', style={'position': 'fixed', 'left': '2%', 'top': '18%'}),
        html.P('Hand Dominance: XX', style={'position': 'fixed', 'left': '2%', 'top': '21%'}),
        html.P('Medications:', style={'fontWeight': 'bold', 'position': 'fixed', 'left': '2%', 'top': '26%'}),
        html.Li('Blood Thinner ABC', style={'position': 'fixed', 'left': '3%', 'top': '29%'}),
        html.Li('Anti Epileptic DEF', style={'position': 'fixed', 'left': '3%', 'top': '32%'}),
        html.P('Other Medical Diagnoses:', style={'fontWeight': 'bold', 'position': 'fixed', 'left': '2%', 'top': '36%'}),
        html.Li('XYZ', style={'position': 'fixed', 'left': '3%', 'top': '39%'}),
        
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(
    [
        dbc.Card(id='card1', children=dbc.CardBody(html.Div(id='graph1'))),
        html.Pre(),
        dbc.Card(id='card2', children=dbc.CardBody(html.Div(id='graph2'))),
        html.Pre(),
        dbc.Card(id='card3', children=dbc.CardBody(html.Div(id='graph3'))),
        html.Pre(),
        dbc.Card(id='card4', children=dbc.CardBody(html.Div(id='graph4'))),
        html.Pre(),
        dbc.Card(id='card5', children=dbc.CardBody(html.Div(id='graph5'))),
    ],
    style=CONTENT_STYLE
)

layout = html.Div([header, sidebar, content])

############################################### Callbacks ###############################################

# Output visualizations based on checklist options 
@callback(Output('card1', 'children'),
              Output('card2', 'children'),
              Output('card3', 'children'),
              Output('card4', 'children'),
              Output('card5', 'children'),
			  Input('checklist', 'value'),
)
def display_page(checklist_options):
    i = 0
    graphs = [[]] * 5

    # Populating visualizations based on checklist options
    # TODO: return graphs with data
    if 'Human Silhouette' in checklist_options:
        graphs[i] = dcc.Graph(figure=make_subplots(rows=1, cols=1))
        i += 1
    if 'Pie Graph' in checklist_options:
        graphs[i] = dcc.Graph(figure=make_subplots(rows=1, cols=2))
        i += 1
    if 'Scatter Plot' in checklist_options:
        graphs[i] = dcc.Graph(figure=make_subplots(rows=1, cols=2))
        i += 1
    if 'Bar Graph' in checklist_options:
        graphs[i] = dcc.Graph(figure=make_subplots(rows=1, cols=2))
        i += 1
    if 'Box Plot' in checklist_options:
        graphs[i] = dcc.Graph(figure=make_subplots(rows=1, cols=1))
    return graphs[0], graphs[1], graphs[2], graphs[3], graphs[4]