import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# initialize application
app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'JEJARD Analytics'

# app layout / settings for page navigation
app.layout = html.Div([
	dcc.Store(id='user-authenticated', storage_type='session'),
	dcc.Location(id='url', refresh=False),
	html.Div(id='page-content'),
	dash.page_container,
])

# prevent unauthorized users from accessing main page
@app.callback(Output('page-content', 'children'),
			  Input('user-authenticated', 'data'),
			  State('url', 'pathname'),
)
def display_page(user_authenticated, pathname):
	# if access granted to user, redirect to main page
	# otherwise, stay on login page
	if pathname == '/main' and not user_authenticated:
		return dcc.Location(pathname='/', id='')
	if user_authenticated:
		if pathname == '/':
			return dcc.Location(pathname='/main', id='')

# run application       
if __name__ == '__main__':
	app.run_server(debug=True)
