# capstone-historical-app

This is the historical application for a project that involves the development of a wireless, wearable system for monitoring movement in recovering stroke patients for 72 hours. This system provides doctors with data and visualizations to help understand if the patient’s stroke has resulted in a loss of movement on either side of the body. The system involves two small accelerometers that can be worn on both wrists of the patient to continuously stream x-, y-, and z-data. This data is then transmitted to this application where calculations are made to provide further insights on the analytics of the data collected. This analytics application will provide caregivers with insight into the patient's level of movement on both sides of the body and help indicate any imbalances that may have resulted from the stroke. The visualizations included in this application include the following:

- A human silhouette that displays the severity of limb impairment,
- a line graph that shows the amount of limb movement in minutes over a 24 hour period,
- a scatter plot that displays the use ratio calculated over 4 hour segments with respect to what is deemed as normal, and 
- a bar graph that displays the activity counts seen from each limb.

## How to run this application

1. Ensure all dependencies listed are installed:
    - agcounts 
    - dash
    - dash-bootstrap-components
    - dash-core-components
    - matplotlib
    - numpy
    - pandas
    - plotly

2. Clone this repository:

    git clone https://github.com/alexishamrak/capstone-historical-app.git

3. In the terminal, run the following piece of code:

    python historical_app.py / python3 historical_app.py

## Accreditations:

The data used in this application was collected by Washington University, retrieved from SimTK: https://simtk.org/plugins/datashare/view.php. This accelerometer data was collected from stroke patients two to 24 weeks post-stroke. Patients wore accelerometers on both wrists for 24-hour segments. It should be noted that this application only makes use of the 30 Hz raw accelerometer data that is provided. 

> Lang CE, Waddell KJ, Barth J, Holleran CL, Strube MJ, Bland MD (2021) Upper limb performance in daily 
life approaches plateau around three to six weeks post stroke. Neurorehabilitation and Neural Repair, 
35:903:914.