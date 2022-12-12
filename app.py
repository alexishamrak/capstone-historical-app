import dash
from dash import Dash, html
import dash_bootstrap_components as dbc

# initialize application
app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'JEJARD Analytics'

# display layout
app.layout = html.Div([
	dash.page_container
])

# run application       
if __name__ == '__main__':
	app.run_server(debug=True)
