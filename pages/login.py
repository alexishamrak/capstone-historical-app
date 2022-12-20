import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

############################################### Layout ###############################################

# to indicate this is a page of the app
dash.register_page(__name__, path='/', title='JEJARD Analytics')

# set layout for login page
login = html.Div(children=[
	html.Div([
        dbc.Card(
            [
                dbc.CardHeader('JEJARD LOGIN', style={'fontSize': '200%', 'fontWeight': 'bold', 'textAlign': 'center', 'backgroundColor': '#d9d9d9'}),
                dbc.CardBody([
                    dbc.Input(id='input-username', placeholder='Username', type='text'),
                    html.Pre(),
                    dbc.Input(id='input-password', placeholder='Password', type='password'),
                    html.Pre(),
                    dbc.Button('Login', color='success', id='login-btn', n_clicks=0, style={'width': '100%'}),
                ]),
                html.Div(id='redirect-from-login'),
            ]
        )
    ], style={'width': '80%', 'margin': 'auto'}),
])

layout = html.Div(children=[
    dbc.Container(dbc.Row([dbc.Col(login)], align='center', style={'height': '100vh'})),
], style={'background': '#246700'})

############################################### Callbacks ###############################################

# page navigation
@callback(Output('redirect-from-login', 'children'),
          Input('login-btn', 'n_clicks'),
          State('input-username', 'value'),
          State('input-password', 'value'))
def update_page(n_clicks, username, password):
    valid_users = {'john.doe':'password123'}
    if username == '' or username == None or password == '' or password == None:
        raise PreventUpdate()
    if username not in valid_users:
        return html.P('Invalid Username.', style={'textAlign': 'center', 'fontWeight': 'bold'})
    if valid_users[username] == password:
        return dcc.Location(pathname='/main', id='')
    else:
        return html.P('Incorrect Password.', style={'textAlign': 'center', 'fontWeight': 'bold'})