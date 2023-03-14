import numpy as np
import pandas as pd
from dash import Dash, dcc, html, callback, Input, Output
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from agcounts.extract import get_counts
import plotly.express as px
from PIL import Image
from PIL import Image, ImageDraw, ImageFont    
import plotly.graph_objects as go

# initialize application
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'JEJARD Analytics'

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

# setting the style of tabs for choosing between day 1, day 2, and day 3 data
tabs_styles = {'zIndex': 99, 'display': 'inlineBlock', 'height': '4vh', 'width': '12vw',
               'position': 'fixed', "background": "#246700", 'top': '1%', 'left': '21%',
               'border': 'white', 'border-radius': '4px'}

tab_style = {
    "background": "#246700",
    'text-transform': 'uppercase',
    'color': 'white',
    'border': 'white',
    'font-size': '14px',
    'font-weight': 600,
    'align-items': 'center',
    'justify-content': 'center',
    'border-radius': '4px',
    'padding':'6px'
}

tab_selected_style = {
    "background": "white",
    'text-transform': 'uppercase',
    'color': '#246700',
    'font-size': '14px',
    'font-weight': 600,
    'align-items': 'center',
    'justify-content': 'center',
    'border-radius': '4px',
    'padding':'6px'
}

# add components and styling to the header of the application
header = html.Div(
    [
        # display application name on top left-hand corner
        html.H3('JEJARD Analytics', style={'position': 'fixed', 'left': '1%', 'color': 'white'}),
        # create horizontal checklist for user to select which visualizations to display
        dcc.Checklist(
            options=[
                {'label': ' Bar Graph', 'value': 'Bar Graph'},
                {'label': ' Scatter Plot', 'value': 'Scatter Plot'},
                {'label': ' Line Graph', 'value': 'Line Graph'},
                {'label': ' Human Silhouette', 'value': 'Human Silhouette'},
            ], value=['Human Silhouette', 'Line Graph'], id='checklist',
            inline=True, labelStyle={'color': 'white', 'float': 'right', 'marginRight': '1rem'}
        ),
    ],
    style=HEADER_STYLE
)

# add components and styling to the application sidebar
sidebar = html.Div(
    [
        # display patient information
        html.H4('Ramriez Santos', style={'position': 'fixed', 'left': '1%', 'top': '7%'}),
        html.P('Age: 67', style={'position': 'fixed', 'left': '2%', 'top': '12%'}),
        html.P('Weight: 77kg', style={'position': 'fixed', 'left': '2%', 'top': '15%'}),
        html.P('Admitted: 28/01/2023', style={'position': 'fixed', 'left': '2%', 'top': '18%'}),
        html.P('Hand Dominance: Right', style={'position': 'fixed', 'left': '2%', 'top': '21%'}),
        html.P('Medications:', style={'fontWeight': 'bold', 'position': 'fixed', 'left': '2%', 'top': '26%'}),
        html.Li('Thrombolytic (tPA)', style={'position': 'fixed', 'left': '3%', 'top': '29%'}),
        html.Li('Acetylsalicyclic Acid, Aspirin', style={'position': 'fixed', 'left': '3%', 'top': '32%'}),
        html.Li('Benazepril (Lotensin)', style={'position': 'fixed', 'left': '3%', 'top': '35%'}),
        html.P('Stroke Information:', style={'fontWeight': 'bold', 'position': 'fixed', 'left': '2%', 'top': '39%'}),
        html.Li('ARAT Score: 53 (Left Arm)', style={'position': 'fixed', 'left': '3%', 'top': '42%'}),
        html.Li('NIHSS Score: 12', style={'position': 'fixed', 'left': '3%', 'top': '45%'}),
        html.Li('Previous stroke in 2019', style={'position': 'fixed', 'left': '3%', 'top': '48%'}),
        
        # input field to set target for line graph
        html.P('Target Arm Movement (Minutes): ', style={'fontWeight': 'bold', 'position': 'fixed', 'left': '1%', 'top': '89%'}),
        dbc.Input(id='limb-movement-target', placeholder='Target Time', type='text', value=60, style={'position': 'fixed', 'left': '1%', 'top': '93%', 'width': '18%'}),
    ],
    style=SIDEBAR_STYLE,
)

# creating a content section with tabs to display three days worth of data and the four different visualizations
# the tabs are broken down into day 1, day 2, and day 3
# the visualizations are each given their own cards for displaying the graphs
content = html.Div([
    dcc.Tabs(id='tabs', value='day-1', children=[
        dcc.Tab(label='Day 1', value='day-1', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Day 2', value='day-2', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Day 3', value='day-3', style=tab_style, selected_style=tab_selected_style),
    ], style=tabs_styles),
    html.Div(
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
    style=CONTENT_STYLE)
])


app.layout = html.Div([header, sidebar, content])


#################################### functions ###############################################################


# function for sorting data into separate arrays for time, X-, Y-, and Z-data
def sorting_data(dataset):
    time = dataset["time"]
    time = np.array(time)
    raw = dataset[["Accelerometer X", "Accelerometer Y", "Accelerometer Z"]]
    raw = np.array(raw)
    return time, raw


# function for calculating the number of activity counts
def collecting_counts(raw_data):
    # frequency is the sampling rate (30 Hz), epochs is set to 10
    # get_counts() is calculating the activity count from the accelerometer data
    # epoch is "grouping" the data into 10 second intervals
    raw_counts = get_counts(raw_data, freq=30, epoch=10)
    raw_counts = pd.DataFrame(raw_counts, columns=["Axis1", "Axis2", "Axis3"])
    # compute vector magnitude of activity counts
    raw_count_mag = np.sqrt(raw_counts["Axis1"] ** 2 + raw_counts["Axis2"] ** 2 + raw_counts["Axis3"] ** 2)
    return raw_count_mag

# function for calculating the use of limbs (in time), the use ratio between limbs, and the activity counts from limbs
def use_ratio(paretic_count_mag, non_paretic_count_mag, tot_time):
    paretic_count = sum(i >= 2 for i in paretic_count_mag)
    non_paretic_count = sum(i >= 2 for i in non_paretic_count_mag)
    use_ratio_calc = paretic_count / non_paretic_count
    paretic_limb_use = (paretic_count / len(paretic_count_mag)) * tot_time
    non_paretic_limb_use = (non_paretic_count / len(non_paretic_count_mag)) * tot_time
    return use_ratio_calc, paretic_limb_use, non_paretic_limb_use, paretic_count, non_paretic_count


############################################### Callbacks ###############################################

# preprocess data
@callback(Output('filter-data', 'data'),
          Input('checklist', 'value'), 
          Input('tabs', 'value'), 
)
def preprocessing(checklist, tabs):

    # load in dataset - 30 Hz raw accelerometer data (m/s2) - 24 hours worth of data total
    # based on the tab selected a different dataset will be selected
    # we are using this data because it has been collected from real stroke patients so it 
    # will help demonstrate expected trends
    if tabs == 'day-1':
        left_arm = pd.read_csv('assets/570_week2_LUE.csv', header=10)
        right_arm = pd.read_csv('assets/570_week2_RUE.csv', header=10)
    elif tabs == 'day-2':
        left_arm = pd.read_csv('assets/570_week4_LUE.csv', header=10)
        right_arm = pd.read_csv('assets/570_week4_RUE.csv', header=10)
    elif tabs == 'day-3':
        left_arm = pd.read_csv('assets/569_week4_LUE.csv', header=10)
        right_arm = pd.read_csv('assets/569_week4_RUE.csv', header=10)

    # add time stamp (time in seconds)
    left_arm['time'] = np.arange(0, 86400, 1/30) # 86400 seconds = 24 hours
    right_arm['time'] = np.arange(0, 86400, 1/30)

    # saving dataset that is input to use for data processing
    la_time, la_raw = sorting_data(left_arm)
    ra_time, ra_raw = sorting_data(right_arm)

    # creating arrays to store the start and end indices of each 4-hour time segment
    time_interval = 14400 # (14400s = 4 hours) - each data point should show the aggregated data for 4 hours
    last_index_array = []
    first_index_array = [0]
    final_time = np.max(la_time) # ASSUMPTION: la_time and ra_time are the same (raspberry pi can ensure this)
    iteration = int(np.floor(final_time/time_interval))
    data_spacing = np.max(np.where(la_time[la_time < time_interval]))

    # splitting the data into sections - we have chosen four hours
    # four hours worth of data will be processed and displayed as one point on a graph
    # this will be done six times because we are working with 24 hours of data (24 hours/4 hours = 6)
    for i in range(iteration):
        val = data_spacing * (i + 1)
        last_index_array.append(val)
        first_index_array.append(val + 1)

    # appending last timestamp from sectioned data for processing
    last_index_array.append(len(la_time) - 1)

    # creating empty lists to populate with calculated values
    a_non_paretic_limb_use_final = []
    arm_use_ratio_final = []
    a_paretic_limb_use_final = []
    paretic_count_final = []
    non_paretic_count_final = []

    # calculating the activity counts, limb use times, and use ratio
    for i in range(len(last_index_array)):
        # calling collecting_counts to calculate activity counts based on preset threshold
        la_count_mag = collecting_counts(la_raw[first_index_array[i]:last_index_array[i]])
        ra_count_mag = collecting_counts(ra_raw[first_index_array[i]:last_index_array[i]])

        # calculating the total time each limb was in use
        tot_time_arm = np.ceil(la_time[last_index_array[i]] - la_time[first_index_array[i]])

        # for calculations left arm is considered paretic
        # ASSUMPTION: left and right time is the same
        arm_use_ratio, a_paretic_limb_use, a_non_paretic_limb_use, paretic_count, non_paretic_count = use_ratio(la_count_mag, ra_count_mag, tot_time_arm)

        # appending calculations from segmented data
        a_non_paretic_limb_use_final.append(a_non_paretic_limb_use)
        arm_use_ratio_final.append(arm_use_ratio)
        a_paretic_limb_use_final.append(a_paretic_limb_use)
        paretic_count_final.append(paretic_count)
        non_paretic_count_final.append(non_paretic_count)

    # packaging data into a dataframe for displaying data in the visualizations
    data = np.transpose([a_non_paretic_limb_use_final, a_paretic_limb_use_final, arm_use_ratio_final, non_paretic_count_final, paretic_count_final]) 
    df = pd.DataFrame(data, columns=['Limb Use RA', 'Limb Use LA', 'Use Ratio U', 'RA Activity Count', 'LA Activity Count']).to_dict('records')
    
    return df


# output visualizations based on checklist options
@callback(Output('graph1', 'children'),
          Output('graph2', 'children'),
          Output('graph3', 'children'),
          Output('graph4', 'children'),
          Input('checklist', 'value'),
          Input('filter-data', 'data'),
          Input('limb-movement-target', 'value'),
)
def display_page(checklist_options, data, hourly_target):
    # run this code only if there is data contained in the component with id 'filter-data'
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
        non_paretic_arm_idx, paretic_arm_idx = [0,3], [1,4] # paretic limb = left
        if len(trouble_idx_U):
            if data['Limb Use LA'][trouble_idx_U[0]] > data['Limb Use RA'][trouble_idx_U[0]]: # paretic limb = right
                non_paretic_arm_idx, paretic_arm_idx = [1,4], [0,3]
            
        # populating visualizations based on checklist options
        # in the following order: Human Silhouette > Line Graph > Scatter Plot > Bar Graph
        if 'Human Silhouette' in checklist_options:
            im = Image.open("assets/silhouette_bw.png") # open image
            width, height = im.size # get the size of the image

            # set color coding scheme
            severe = (255, 0, 0) # red (ARAT score 0-19 > Use Ratio 0-0.5)
            moderate = (107,142,35) # olive green (ARAT score 19+ > Use Ratio 0.5+)
            
            # assign color to limbs based on severity of movement
            color_LA, color_RA = moderate, moderate
            if len(trouble_idx_U): # if any use ratio value falls between 0-0.5, paretic limb is categorized as severe
                if 'RA' in data.columns[paretic_arm_idx[0]]:
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
            
            # Add annotations to the human silhouette to differentiate the left from the right side
            draw = ImageDraw.Draw(im)
            font = ImageFont.truetype("arial.ttf", 40, encoding="unic")
            draw.text((10, 550), u"Left", fill='#000000', font=font)
            draw.text((671, 550), u"Right", fill='#000000', font=font)
            
            # show image and adjust image display on application
            fig = px.imshow(im)
            fig.update_layout(margin=dict(l=10, r=10, b=10, t=10), hovermode=False)
            fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

            # adding description of graph to graph title when mouse is hovered over top
            graphs[i] = html.Div(
                [
                    html.H4([html.Span("Severity of Arm Impairment", id="tooltip1", style={'paddingLeft': '3%'}),]),
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

            # update counter to display other visualizations on application
            i += 1

        if 'Line Graph' in checklist_options:
            
            # the line graph displays the hourly arm movement for each limb which has been calculated based on activity counts over time
            paretic_arm = data.iloc[:, paretic_arm_idx[0]]              # locating which limb is paretic
            paretic_arm = np.floor(np.array(paretic_arm/60))            # rounding down to the nearest minute of paretic limb time 
            non_paretic_arm = data.iloc[:, non_paretic_arm_idx[0]]      # locating which limb is non-paretic
            non_paretic_arm = np.floor(np.array(non_paretic_arm/60))    # rounding down to the nearest minute of non-paretic limb time
            diff = non_paretic_arm - paretic_arm                        # calculating difference in time between paretic and non-paretic limb

            # data.shape[0] retrieves the number of rows in the dataset
            hours = list(range(1, data.shape[0]+1))
            
            # creating the line graph which will have 4 different lines
            # there is a line for paretic movement, non-paretic movement, difference between limbs, and a threshold (expected movement time)
            line_graph_arm = go.Figure()
            line_graph_arm.add_trace(go.Scatter(x=hours, y=paretic_arm, mode='lines+markers', name='Paretic Arm Movement', line=dict(width=4)))
            line_graph_arm.add_trace(go.Scatter(x=hours, y=non_paretic_arm, mode='lines+markers', name='Non-Paretic Arm Movement', line=dict(color='rgb(231,107,243)', width=4)))
            line_graph_arm.add_trace(go.Scatter(x=hours, y=diff, mode='lines+markers', name='Difference between Arms', line=dict(width=4)))
            line_graph_arm.add_hline(y=int(hourly_target), line_dash="dash", line_color="red", annotation_text="Target")
            
            # adding x-axis tick values, axis labels, and increasing marker size
            line_graph_arm.update_xaxes(tickvals=hours, ticktext=['4', '8', '12', '16', '20', '24'])
            line_graph_arm.update_layout(xaxis_title='Hours', yaxis_title='Minutes of Movement for Every 4 Hours')
            line_graph_arm.update_traces(marker_size=14)

            # adding description of graph to graph title when mouse is hovered over top
            graphs[i] = html.Div(
                [
                    html.H4([html.Span("Hourly Arm Movement", id="tooltip2", style={'paddingLeft': '3%'}),]),
                    dbc.Tooltip(
                        "This line graph helps visualize the amount of movement seen by the paretic and "
                        "non-paretic limbs, which are the blue and magenta lines, respectively. The green "
                        "line is the difference between the non-paretic and paretic limb movement and "
                        "the red line is the movement goal set by a doctor, which in our example, is "
                        "40 minutes. The x-axis represents the hour when data was collected, whereas the "
                        "y-axis represents the minutes of movement seen per hour from each limb. "
                        , target="tooltip2"
                    ),
                    dcc.Graph(figure=line_graph_arm)
                ]
            )

            # update counter to display other visualizations on application
            i += 1

        if 'Scatter Plot' in checklist_options:

            # this scatter plot displays the use ratio between limbs over time in comparison to a lower/upper threshold
            # blue dots represent the use ratio calculated over four hours, two red dashed lines represent an upper 
            # and lower threshold of where the use ratio is expected to be
            scatter_plot_arm = px.scatter(x=np.arange(1, 7), y=data['Use Ratio U'])
            scatter_plot_arm.update_traces(marker_size=20)
            scatter_plot_arm.add_hline(y=0.79, line_dash="dash", line_color="red", annotation_text="Lower threshold = 0.79")
            scatter_plot_arm.add_hline(y=1.1, line_dash="dash", line_color="red", annotation_text="Upper threshold = 1.1")
            scatter_plot_arm.update_layout(xaxis_title="Hours",  yaxis_title="Arm Use Ratio", yaxis_range=[0,2])
            scatter_plot_arm.update_xaxes(tickvals=np.arange(1, 7), ticktext=['4', '8', '12', '16', '20', '24'])

            # adding description of graph to graph title when mouse is hovered over top
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

            # update counter to display other visualizations on application
            i += 1

        if 'Bar Graph' in checklist_options:

            # this graph will display the activity count between limbs as a bar graph
            paretic_arm = data.iloc[:, paretic_arm_idx[1]]          # locating which limb is paretic
            non_paretic_arm = data.iloc[:, non_paretic_arm_idx[1]]  # locating which limb is non-paretic
            ind = np.arange(1, len(paretic_arm) + 1)                # calculating how many bars will be needed        

            # adding both paretic and non-paretic data to bar graph
            bar_graph_arm = go.Figure(data=[go.Bar(name='Non-Paretic', x=ind, y=non_paretic_arm), 
            go.Bar(name='Paretic', x=ind, y=paretic_arm)])

            # updating graph to have x-axis ticks and axis labels
            bar_graph_arm.update_xaxes(tickvals=ind, ticktext=['4', '8', '12', '16', '20', '24'])
            bar_graph_arm.update_layout(barmode='stack', xaxis_title="Hours",  yaxis_title="Activity Count")
            
            # adding description of graph to graph title when mouse is hovered over top
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

            # update counter to display other visualizations on application
            i += 1

        # update cards based on checkbox selection(s) with visualizations stored in the variable 'graphs'
        return graphs[0], graphs[1], graphs[2], graphs[3]
    else:
        # prevent any updates from occuring to the application if there is not 
        # data contained in the component with id 'filter-data'
        raise PreventUpdate


# run application       
if __name__ == '__main__':
	app.run_server(debug=True)