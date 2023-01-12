import io
import base64
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, callback, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import scipy.signal
from agcounts.extract import get_counts
import plotly.express as px

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
    "position": "absolute", # to enable scroll through the different visualizations
    "top": "7%",
    "left": "22%",
    "width": "77%",
    "height": "93%",
    "overflowY": "scroll", # only want vertical scroll bar
    "backgroundColor": "white"
}

# add components and styling to the different sections of the application
header = html.Div(
    [
        # display application name on top left-hand corner
        html.H3('JEJARD Analytics', style={'position': 'fixed', 'left': '1%', 'color': 'white'}),
        # create horizontal checklist for user to select which visualizations to display
        dcc.Checklist(
            options=[
                {'label': ' Box Plots', 'value': 'Box Plots'},
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
        # display patient information
        # TODO: automate this
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
        # place each visualization inside a card (for neatness)
        dbc.Card(id='card1', children=dbc.CardBody(html.Div(id='graph1'))),
        html.Pre(),
        dbc.Card(id='card2', children=dbc.CardBody(html.Div(id='graph2'))),
        html.Pre(),
        dbc.Card(id='card3', children=dbc.CardBody(html.Div(id='graph3'))),
        html.Pre(),
        dbc.Card(id='card4', children=dbc.CardBody(html.Div(id='graph4'))),
        html.Pre(),
        dbc.Card(id='card5', children=dbc.CardBody(html.Div(id='graph5'))),
        html.Pre(),
        dbc.Card(id='card6', children=dbc.CardBody(html.Div(id='graph6'))),
        dcc.Store(id='filter-data', storage_type='session'),
        dcc.Store(id='bilateral-mag', storage_type='session')
    ],
    style=CONTENT_STYLE
)

layout = html.Div([header, sidebar, content])


#################################### functions ###############################################################


# function for sorting data into separate arrays for time, X, Y, and Z data
def sorting_data(dataset):
    time = dataset["time"]
    time = np.array(time)
    x = dataset["x-acceleration"]
    x = np.array(x)
    y = dataset["y-acceleration"]
    y = np.array(y)
    z = dataset["z-acceleration"]
    z = np.array(z)
    raw = dataset[["x-acceleration", "y-acceleration", "z-acceleration"]]
    raw = np.array(raw)
    return time, x, y, z, raw


# function for calculating the number of activity counts
def collecting_counts(raw_data, freq, epoch):
    # frequency is the sampling rate (50 Hz), epochs was arbitrarily set to 10
    # get_counts() is calculating the activity count from the accelerometer data
    # epoch is "grouping" the data into 10 second intervals
    # maybe we should try epoch=1 later
    raw_counts = get_counts(raw_data, freq=freq, epoch=epoch)
    raw_counts = pd.DataFrame(raw_counts, columns=["Axis1", "Axis2", "Axis3"])
    raw_count_mag = np.sqrt(raw_counts["Axis1"] ** 2 + raw_counts["Axis2"] ** 2 + raw_counts["Axis3"] ** 2)
    return raw_counts, raw_count_mag


# function for filtering raw data
def filter_data(x_data, y_data, z_data):
    # maybe we'll need to play around with the "51", "3" values
    x_hat = scipy.signal.savgol_filter(x_data, 51, 3)
    y_hat = scipy.signal.savgol_filter(y_data, 51, 3)
    z_hat = scipy.signal.savgol_filter(z_data, 51, 3)
    return x_hat, y_hat, z_hat


def use_ratio(paretic_count_mag, non_paretic_count_mag, final_time):
    paretic_count = sum(i >= 2 for i in paretic_count_mag)
    non_paretic_count = sum(i >= 2 for i in non_paretic_count_mag)
    use_ratio_calc = paretic_count / non_paretic_count
    paretic_limb_use = (paretic_count / len(paretic_count_mag)) * final_time
    non_paretic_limb_use = (non_paretic_count / len(non_paretic_count_mag)) * final_time
    return use_ratio_calc, paretic_limb_use, non_paretic_limb_use


def calc_mag(x_filt, y_filt, z_filt):
    mag = np.sqrt(x_filt ** 2 + y_filt ** 2 + z_filt ** 2)
    return mag


def bilateral_mag(leftside_mag, rightside_mag, left_time, right_time):
    # merge time and magnitude datasets for left and right limb
    left = np.transpose(np.vstack((left_time, leftside_mag)))
    right = np.transpose(np.vstack((right_time, rightside_mag)))
    # convert dataset into a Pandas Dataframe and add column names
    leftside_mag_df = pd.DataFrame(left, columns=['time', 'left_mag'])
    rightside_mag_df = pd.DataFrame(right, columns=['time', 'right_mag'])
    # merge datasets based on time column
    if len(left_time) > len(right_time):
        merged_dataset = leftside_mag_df.merge(rightside_mag_df, on='time', how='left')
    else:
        merged_dataset = rightside_mag_df.merge(leftside_mag_df, on='time', how='left')

    bilat_mag = np.nansum([merged_dataset['left_mag'], merged_dataset['right_mag']], axis=0)

    return bilat_mag


############################################### Callbacks ###############################################

# preprocess data
@callback(Output('filter-data', 'data'),
          Output('bilateral-mag', 'data'),
          Input('url', 'pathname'))

def preprocessing(url_pathname):
    left_hand = pd.read_csv('assets/left_hand_lm.csv')
    right_hand = pd.read_csv('assets/right_hand_hm.csv')
    # left_leg = pd.read_csv('Assets/left_leg_lm.csv')
    # right_leg = pd.read_csv('Assets/right_leg_hm.csv')

    freq = 50
    epoch = 1

    lh_time, lh_x, lh_y, lh_z, lh_raw = sorting_data(left_hand)
    rh_time, rh_x, rh_y, rh_z, rh_raw = sorting_data(right_hand)
    # ll_time, ll_X, ll_Y, ll_Z, ll_raw = sorting_data(left_leg)
    # rl_time, rl_X, rl_Y, rl_Z, rl_raw = sorting_data(right_leg)

    # creating arrays to store the start and end indices of time segments
    time_interval = 60  # should be changed to 3600s
    last_index_array = []
    first_index_array = [0]
    final_time = np.max(lh_time) # ASSUMPTION: lh_time and rh_time are the same (raspberry pi can ensure this)
    iteration = int(np.floor(final_time/time_interval))
    data_spacing = np.max(np.where(lh_time[lh_time < time_interval]))

    for i in range(iteration):
        val = data_spacing * (i + 1)
        last_index_array.append(val)
        first_index_array.append(val + 1)

    last_index_array.append(len(lh_time) - 1)

    lh_x_hat, lh_y_hat, lh_z_hat = filter_data(lh_x, lh_y, lh_z)
    rh_x_hat, rh_y_hat, rh_z_hat = filter_data(rh_x, rh_y, rh_z)
    # ll_X_hat, ll_Y_hat, ll_Z_hat = filter_data(ll_X, ll_Y, ll_Z)
    # rl_X_hat, rl_Y_hat, rl_Z_hat = filter_data(rl_X, rl_Y, rl_Z)

    h_non_paretic_limb_use_final = []
    hand_use_ratio_final = []
    h_paretic_limb_use_final = []

    for i in range(len(last_index_array)):
        lh_counts, lh_count_mag = collecting_counts(lh_raw[first_index_array[i]:last_index_array[i]], freq, epoch)
        rh_counts, rh_count_mag = collecting_counts(rh_raw[first_index_array[i]:last_index_array[i]], freq, epoch)
        # ll_counts, ll_count_mag = collecting_counts(ll_raw)
        # rl_counts, rl_count_mag = collecting_counts(rl_raw)

        # ASSUMPTION: left and right time is the same
        hand_use_ratio, h_paretic_limb_use, h_non_paretic_limb_use = use_ratio(lh_count_mag, rh_count_mag, final_time)
        # leg_use_ratio, l_paretic_limb_use, l_non_paretic_limb_use = use_ratio(ll_count_mag, rl_count_mag, final_time)

        h_non_paretic_limb_use_final.append(h_non_paretic_limb_use)
        hand_use_ratio_final.append(hand_use_ratio)
        h_paretic_limb_use_final.append(h_paretic_limb_use)

    lh_mag = calc_mag(lh_x_hat, lh_y_hat, lh_z_hat)
    rh_mag = calc_mag(rh_x_hat, rh_y_hat, rh_z_hat)
    # ll_mag = calc_mag(ll_X_hat, ll_Y_hat, ll_Z_hat)
    # rl_mag = calc_mag(rl_X_hat, rl_Y_hat, rl_Z_hat)

    bilateral_hand_mag = bilateral_mag(lh_mag, rh_mag, lh_time, rh_time)
    # print(f"Bilateral magnitude between hands is: {bilateral_hand_mag}")
    # bilateral_leg_mag = bilateral_mag(ll_mag, rl_mag)
    # print(f"Bilateral magnitude between legs is: {bilateral_leg_mag}")

    # create dataframe for bilateral magnitude (for boxplots)
    bilateral_mag_merge = pd.Series(bilateral_hand_mag, name='bilateral_magnitude') # TODO: add 'bilateral_leg_mag' 
    region = pd.Series(["upper extremities"] * len(bilateral_hand_mag), name='region_of_body') # TODO: add '+ ["lower extremities"] * len(bilateral_leg_mag)'
    bilat_temp = pd.DataFrame(bilateral_mag_merge)
    bilateral_mag_df = bilat_temp.join(region)

    # TODO: pass in data for legs
    data = np.transpose([h_non_paretic_limb_use_final, h_paretic_limb_use_final, hand_use_ratio_final]) 
    df = pd.DataFrame(data, columns=['Limb Use RH', 'Limb Use LH', 'Use Ratio U']).to_dict('records')
    return df, bilateral_mag_df.to_dict('records') 


# output visualizations based on checklist options
@callback(Output('card1', 'children'),
          Output('card2', 'children'),
          Output('card3', 'children'),
          Output('card4', 'children'),
          Output('card5', 'children'),
          Output('card6', 'children'),
          Input('checklist', 'value'),
          State('filter-data', 'data'),
          State('bilateral-mag', 'data')
)
def display_page(checklist_options, data, bilat_mag):
    # initialize variables needed
    i = 0
    graphs = [[]] * 6
    data = pd.DataFrame(data) 
    bilat_mag = pd.DataFrame(bilat_mag)

    # TODO: Find way to differentiate paretic vs nonparetic limb (for now, assume left side of body is paretic) 

    # populating visualizations based on checklist options
    # in the following order: Human Silhouette > Pie Graph > Scatter Plot > Bar Graph > Box Plot
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
    if 'Box Plots' in checklist_options:
        # TODO: find columns containing data about the paretic limb(s) - assume LH ONLY for right now

        # creating paretic limb acceleration dataset for boxplots
        num_datapoints = data.shape[0] # ASSUMPTION: hands/legs have the same number of datapoints
        paretic_limbs_merge = pd.Series(data['Limb Use LH'], name='paretic_acceleration') # TODO: add 'Limb Use LL'
        region = pd.Series(["Paretic Arm"] * num_datapoints, name='region_of_body') # TODO: add '+ ["Paretic Leg"] * num_datapoints'
        paretic_acc_temp = pd.DataFrame(paretic_limbs_merge)
        paretic_acc_boxplot_df = paretic_acc_temp.join(region)

        # plot boxplots for paretic limb acceleration
        paretic_acc_boxplot = px.box(paretic_acc_boxplot_df, x="region_of_body", y="paretic_acceleration") 
        graphs[i] = dcc.Graph(figure=paretic_acc_boxplot)

        # plot boxplots for bilateral magnitude
        bilat_mag_boxplot = px.box(bilat_mag, x="region_of_body", y="bilateral_magnitude") 
        graphs[i+1] = dcc.Graph(figure=bilat_mag_boxplot)

    return graphs[0], graphs[1], graphs[2], graphs[3], graphs[4], graphs[5]
