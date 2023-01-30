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
from PIL import Image
import plotly.graph_objects as go

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


def use_ratio(paretic_count_mag, non_paretic_count_mag, tot_time):
    paretic_count = sum(i >= 2 for i in paretic_count_mag)
    non_paretic_count = sum(i >= 2 for i in non_paretic_count_mag)
    use_ratio_calc = paretic_count / non_paretic_count
    paretic_limb_use = (paretic_count / len(paretic_count_mag)) * tot_time
    non_paretic_limb_use = (non_paretic_count / len(non_paretic_count_mag)) * tot_time
    return use_ratio_calc, paretic_limb_use, non_paretic_limb_use


############################################### Callbacks ###############################################

# preprocess data
@callback(Output('filter-data', 'data'),
          Input('url', 'pathname')
)
def preprocessing(url_pathname):
    left_arm = pd.read_csv('assets/left_hand_lm.csv')
    right_arm = pd.read_csv('assets/right_hand_hm.csv')

    freq = 50
    epoch = 1

    la_time, la_x, la_y, la_z, la_raw = sorting_data(left_arm)
    ra_time, ra_x, ra_y, ra_z, ra_raw = sorting_data(right_arm)

    # creating arrays to store the start and end indices of time segments
    time_interval = 60  # should be changed to 3600s
    last_index_array = []
    first_index_array = [0]
    final_time = np.max(la_time) # ASSUMPTION: lh_time and rh_time are the same (raspberry pi can ensure this)
    iteration = int(np.floor(final_time/time_interval))
    data_spacing = np.max(np.where(la_time[la_time < time_interval]))

    for i in range(iteration):
        val = data_spacing * (i + 1)
        last_index_array.append(val)
        first_index_array.append(val + 1)

    last_index_array.append(len(la_time) - 1)

    la_x_hat, la_y_hat, la_z_hat = filter_data(la_x, la_y, la_z)
    ra_x_hat, ra_y_hat, ra_z_hat = filter_data(ra_x, ra_y, ra_z)

    a_non_paretic_limb_use_final = []
    arm_use_ratio_final = []
    a_paretic_limb_use_final = []

    for i in range(len(last_index_array)):
        la_counts, la_count_mag = collecting_counts(la_raw[first_index_array[i]:last_index_array[i]], freq, epoch)
        ra_counts, ra_count_mag = collecting_counts(ra_raw[first_index_array[i]:last_index_array[i]], freq, epoch)

        tot_time_arm = np.ceil(la_time[last_index_array[i]] - la_time[first_index_array[i]])

        # ASSUMPTION: left and right time is the same
        arm_use_ratio, a_paretic_limb_use, a_non_paretic_limb_use = use_ratio(la_count_mag, ra_count_mag, tot_time_arm)

        a_non_paretic_limb_use_final.append(a_non_paretic_limb_use)
        arm_use_ratio_final.append(arm_use_ratio)
        a_paretic_limb_use_final.append(a_paretic_limb_use)

    data = np.transpose([a_non_paretic_limb_use_final, a_paretic_limb_use_final, arm_use_ratio_final]) 
    df = pd.DataFrame(data, columns=['Limb Use RA', 'Limb Use LA', 'Use Ratio U']).to_dict('records')
    return df


# output visualizations based on checklist options
@callback(Output('graph1', 'children'),
          Output('graph2', 'children'),
          Output('graph3', 'children'),
          Output('graph4', 'children'),
          Input('checklist', 'value'),
          Input('filter-data', 'data')
)
def display_page(checklist_options, data):
    if data is not None:
        # initialize variables needed
        i = 0
        graphs = [[]] * 4
        data = pd.DataFrame(data) 

        # generally, the paretic limb will have lower activity counts than its non-paretic counterpart. find an 
        # incident of low use ratio and classify limbs as paretic or non-paretic (limb use will be lower for paretic limb)
        # for human silhouetter below, if use ratio is 0.5+ both limbs are classified as moderate. if any value of 
        # use ratio falls between 0-0.5, the paretic limb falls under severe while the non-paretic limb is moderate
        thres = 0.5 # use ratio between 0-0.5 correlates to ARAT score of 0 or 1 (severe)
        trouble_idx_U = data['Use Ratio U'][data['Use Ratio U'] < thres].index
        non_paretic_arm_idx, paretic_arm_idx = 0, 1 # paretic limb = left
        if len(trouble_idx_U):
            if data['Limb Use LH'][trouble_idx_U[0]] > data['Limb Use RH'][trouble_idx_U[0]]: # paretic limb = right
                non_paretic_arm_idx, paretic_arm_idx = 1, 0
            
        # populating visualizations based on checklist options
        # in the following order: Human Silhouette > Pie Graph > Scatter Plot > Bar Graph > Box Plots
        if 'Human Silhouette' in checklist_options:
            im = Image.open("assets/silhouette_bw.png") # open image
            width, height = im.size # get the size of the image

            # TODO: is there a better algorithm for classification?
            # set color coding scheme
            severe = (255, 0, 0) # red (ARAT score 0-19 > Use Ratio 0-0.5)
            # warning = (255, 127, 0) # orange
            moderate = (107,142,35) # olive green (ARAT score 19+ > Use Ratio 0.5+)
            
            # assign color to limbs based on severity of movement
            color_LA, color_RA = moderate, moderate
            if len(trouble_idx_U): # if any use ratio value falls between 0-0.5, paretic limb is categorized as severe
                if 'RA' in data.columns[paretic_arm_idx]:
                    color_RA = severe
                else:
                    color_LA = severe 
            
            # change color of the image pixels
            for x in range(width):    
                for y in range(height):  
                    current_color = im.getpixel( (x,y) )
                    if (x < 288) and (current_color != (255, 255, 255) ): # left arm
                        im.putpixel( (x,y), color_LA) 
                    if (x > 482) and (current_color != (255, 255, 255) ): # right Arm
                        im.putpixel( (x,y), color_RA) 
            
            fig = px.imshow(im)
            fig.update_layout(margin=dict(l=10, r=10, b=10, t=10), hovermode=False)
            fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
            graphs[i] = html.Div(
                [
                    html.H4([html.Span("Severity of Limb Impairment", id="tooltip1", style={'paddingLeft': '3%'}),]),
                    dbc.Tooltip(
                        "Taking advantage of the positive correlation between the use ratio and ARAT scores, " 
                        "a use ratio of 0.5 is used as the threshold to indicate severity of the limb impairment. "
                        "“Severe” is shown with red, with an ARAT score of 19 or less (score of 0-1). "
                        "“Moderate” is shown with green, with an ARAT score of 19 or above (score of 2-3).",
                        target="tooltip1"
                    ),
                    dcc.Graph(figure=fig)
                ]
            )
            i += 1
        if 'Pie Graph' in checklist_options:

            paretic_arm = data.iloc[:, paretic_arm_idx]
            paretic_arm = np.floor(np.array(paretic_arm))

            # TODO: Adjust time vector based on final dataset length
            time = [50, 50, 50, 50, 50, 50]
            remaining_arm_time = np.ceil(time - paretic_arm)
            labels_arm = ['Paretic Arm Use Time (minutes)', 'Remaining Time to Meet Goal (minutes)']

            # TODO: Will automate N, M, specs, and subplot_titles
            # TODO: Will have to change depending on subplot size for full data (cannot do now)
            N = 2
            M = 3

            # TODO: Will have to populate specs with size of full data (cannot do now)
            specs = [[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]]
            pie_graph_arm = make_subplots(rows=N, cols=M, specs=specs, subplot_titles=['Hour 1', 'Hour 2', 'Hour 3', 'Hour 4', 'Hour 5', 'Hour 6'])

            row = 1
            column = 1

            for idx in range(len(paretic_arm)):
                if remaining_arm_time[idx] < 0:
                    remaining_arm_time[idx] = 0
                pie_graph_arm.add_trace(go.Pie(labels=labels_arm, values=[paretic_arm[idx], remaining_arm_time[idx]], 
                name=idx), row, column)
                pie_graph_arm.update_traces(hoverinfo='label+percent', textinfo='value', hole=0.3)
                if idx >= N and row < N:
                    row = row + 1
                if column < M:
                    column = column + 1
                if idx == N:
                    column = 1

            graphs[i] = html.Div(
                [
                    html.H4([html.Span("Hourly Paretic Arm Use (Target = 50 minutes)", id="tooltip2", style={'paddingLeft': '3%'}),]),
                    dbc.Tooltip(
                        "These pie charts help visualize the amount of movement seen by the paretic limb compared " 
                        "to the goal set by the doctor, which in our example, is set to 50 minutes per hour. The blue " 
                        "ring represents the amount of minutes of movement seen by the paretic limb, whereas the " 
                        "red ring represents the remaining time that the paretic limb should move to meet the doctor’s goal. " 
                        , target="tooltip2"
                    ),
                    dcc.Graph(figure=pie_graph_arm)
                ]
            )

            i += 1

        if 'Scatter Plot' in checklist_options:
            
            use_ratio_arm = data['Use Ratio U']

            num_dots = len(use_ratio_arm) + 1
            ind = np.arange(1, num_dots) 

            scatter_plot_arm = px.scatter(x=ind, y=use_ratio_arm)
            scatter_plot_arm.update_traces(marker_size=20)
            scatter_plot_arm.add_hline(y=0.79, line_dash="dash", line_color="red", annotation_text="Lower threshold = 0.79")
            scatter_plot_arm.add_hline(y=1.1, line_dash="dash", line_color="red", annotation_text="Upper threshold = 1.1")
            scatter_plot_arm.update_layout(xaxis_title="Hours",  yaxis_title="Arm Use Ratio", yaxis_range=[0,2])

            graphs[i] = html.Div(
                [
                    html.H4([html.Span("Use Ratio of Arms Relative to Typical Range", id="tooltip3", style={'paddingLeft': '3%'}),]),
                    dbc.Tooltip(
                        "These scatter plots help visualize the movement use ratio between paretic and " 
                        "non-paretic limbs (split into upper and lower extremities). The two dashed red " 
                        "lines represent an expected threshold for the use ratio between equally performing " 
                        "limbs. The blue dots are the actual use ratios collected from the sensor data. "
                        , target="tooltip3"
                    ),
                    dcc.Graph(figure=scatter_plot_arm)
                ]
            )

            i += 1

        if 'Bar Graph' in checklist_options:

            paretic_arm = data.iloc[:, paretic_arm_idx]
            non_paretic_arm = data.iloc[:, non_paretic_arm_idx]

            # ASSUMPTION: vector length of all four limbs is the same
            num_bars = len(paretic_arm) + 1
            ind = np.arange(1, num_bars)

            bar_graph_arm = go.Figure(data=[go.Bar(name='Non-Paretic', x=ind, y=non_paretic_arm), 
            go.Bar(name='Paretic', x=ind, y=paretic_arm)])

            bar_graph_arm.update_layout(barmode='stack', xaxis_title="Hours",  yaxis_title="Activity Count")

            graphs[i] = html.Div(
                [
                    html.H4([html.Span("Activity Count of Paretic and Non-Paretic Arms", id="tooltip4", style={'paddingLeft': '3%'}),]),
                    dbc.Tooltip(
                        "These bar graphs help visualize the number of activity counts collected from "
                        "the paretic and non-paretic limbs (split into upper and lower extremities). Ideally, " 
                        "the two colors stacked on top of eachother will be equal meaning the paretic and "
                        "non-paretic limbs were used a similar amount. This graph should be able to help doctors "
                        "quickly evaluate any differences seen between the movement in paretic/non-paretic limbs. "
                        , target="tooltip4"
                    ),
                    dcc.Graph(figure=bar_graph_arm)
                ]
            )

            i += 1

        return graphs[0], graphs[1], graphs[2], graphs[3]
    else:
        raise PreventUpdate