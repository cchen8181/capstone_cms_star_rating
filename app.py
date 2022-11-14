import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import base64
from scipy.stats import spearmanr
from scipy.stats import pearsonr

#main section text
st.title("CMS Star Ratings")

#select box to show different pages of the app
page = st.sidebar.selectbox("Select Page",
    ['Star Rating Explorer', 'Star Measure Details (2022)','Correlations Dashboard', 'Contract Star Details'], index=0)

#use session state to switch between pages
st.session_state.page = page

#initialize measures dictionary for simulations later
if 'measures' not in st.session_state:
    st.session_state.measures = {}

#@st.cache(persist=True)

# define a function that loads data
def load_data(DATA_URL):
    # read data
    data = pd.read_csv(DATA_URL)
    
    # any modification to the read-in dataset can be put here
    return data

# function to show the treemap
def show_treemap(df, size):
    extra_cols = ['contract_id', 'contract_name', 'marketing_name', 'parent_org_name',
                  'overall_star', 'part_c_star', 'part_d_star',
                  'top_states', size, 'org_type_name','SNP']

    #create the plotly treemap object
    fig = px.treemap(
        data_frame = df,
        path=['parent_org_name', 'contract_id'],     # how the blocks are organized   
        values = size,      # value determining block size
        custom_data = extra_cols,
        color='Rating',
        color_continuous_scale ='rdylgn',
        range_color = [0,5]
    )

    #basic display formatting
    fig.update_traces(root_color="lightgrey")
    fig.update_layout(margin = dict(t=10, l=10, r=10, b=10))

    #set up hover display
    fig.update_traces(
        hovertemplate ='<b>%{customdata[0]} - %{customdata[1]}</b><br><br>' +
            'Marketing Name: %{customdata[2]}<br>'
            'Parent: %{customdata[3]}<br>' +
            'Overall: %{customdata[4]}<br>' +
            'Part C: %{customdata[5]}<br>' +
            'Part D: %{customdata[6]}<br>' +
            'States: %{customdata[7]}<br>' +
            'Enrollment: %{customdata[8]}<br>' +
            'Organization type: %{customdata[9]}<br>' +
            'SNP: %{customdata[10]}<br>'
    )
    #customize hoverlabel appearance
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Arial"
        )  
    )
    st.plotly_chart(fig)

# function to display PDF
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    
    # return file to display
    return pdf_display

#display scatter plot of correlations between a measure and a predictor
def show_scatter(df, x, y, hover):
    title= y + " VS " + x
    fig=px.scatter(df, x= x, y=y, title=title, hover_data=hover)
    st.plotly_chart(fig)
    
    tmp_df = df[[x,y]].dropna() 
    
    pearson = pearsonr(tmp_df[x], tmp_df[y])
    spearman = spearmanr(df[x], df[y], nan_policy='omit')
    st.markdown(f"__Pearson Correlation:__ {pearson[0]:.4f} (p-value: {pearson[1]:.4f}) __Spearman Correlation:__ {spearman[0]:.4f} (p-value: {spearman[1]:.4f})")
    st.markdown("""---""")

def style_table_details(styler):
    styler.format({"score": "{:.1f}", "star": "{:.0f}"}, na_rep= "N/A")
    styler.background_gradient(axis=None, vmin=1, vmax=5, cmap="RdYlGn", subset='star')
    return styler

def show_measures_table(df):
    st.table(df.style.pipe(style_table_details))
    
def style_table_rec(styler):
    styler.format({"score": "{:.1f}", "star": "{:.0f}", "weight":"{:.0f}", "lower":"{:.1f}", "upper": "{:.1f}", "penetration": "{:.1f}%"})
    styler.background_gradient(axis=None, vmin=1, vmax=5, cmap="RdYlGn", subset='star')
    return styler
    
def show_recommendations(df):
    st.table(df.style.pipe(style_table_rec))
    
def style_table_results(styler):
    styler.format({"Rounded": "{:.1f}", "Raw": "{:.2f}", "Actual":"{:.1f}"})
    styler.background_gradient(axis=None, vmin=1, vmax=5, cmap="RdYlGn", subset=['Rounded','Actual'])
    return styler

def show_star_results(df):
    st.table(df.style.pipe(style_table_results))

#update the session state value to allow star simulations
def update_star(measure, star):
    st.session_state.measures[measure] = star

#remove all session state values to reset star simulation
def clear_simulation():
    #for key in st.session_state.measures.keys():
    #    del st.session_state[key]
    del st.session_state.measures
    
def create_simulated_measures_df(df):
    """
    Given a df of a single contract and year's measure stars, return a df updated with the measure stars to simulate.
    """
    
    for key in st.session_state.measures.keys():
        df.loc[df['measure'] == key, 'star'] = st.session_state.measures[key]
        
    return df
    
#calculate the overall or summary star rating
def overall_summary_star(df, star_type='overall', star_col='star', weight_col='weight'):
    """
    Given a dataframe df for a single contract and year, calculate the chosen star_type (overall, part_c, part_d) using
    weighted average of the individual measure stars.
    
    Returns both the raw weighted average star and the star rounded to the nearest 0.5
    """
    
    #for overall star rating, don't double count these measures
    if star_type == 'overall':
        df = df[df['measure'] != 'D-Members Choosing to Leave the Plan']
        df = df[df['measure'] != 'D-Complaints about the Drug Plan']
    #for part c summary star, keep only part c measures
    elif star_type == 'part_c':
        df = df[df['is_part_c'] == 1]
    #for part d summary star, keep only part d measures
    elif star_type == 'part_d':
        df = df[df['is_part_d'] == 1]
    
    #remove measures where star was not assigned
    df = df.dropna(subset = star_col)
    
    #calculate weighted average
    raw_star = np.average(df[star_col], weights= df[weight_col])
    
    #perform rounding to nearest 0.5
    rounded_star = round(raw_star * 2) / 2
    
    return raw_star, rounded_star

### start of page for the Star Rating Explorer (treemap)
if st.session_state.page == 'Star Rating Explorer':
    st.markdown("""Every year, CMS rates Part C and Part D health plan contracts on a 5 star quality rating system. Higher rated plans are more attractive to patients and can lead to increased enrollment and plans that receive at least 4 stars receive additional quality bonus payments from Medicare, so there is strong financial incensive for a health plant to improve their star rating.
    """)
    
    st.markdown("Use the filters in the sidebar to control which contracts show up in the visualization.")
    st.markdown("Use the following visualization to explore how each contract compares against others")

    #sidebar text
    st.sidebar.title("Filters")

    df = load_data("data/visualization_data.csv")
    
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    
    #pick the year
    year = st.sidebar.slider("Year", min_value = min_year, max_value = max_year, value = max_year,
        help="Select the year you want to view the Star Ratings for.")
        
    #pick the plan type
    plan_type = st.sidebar.selectbox('Plan Type', ['MA-PD', 'MA only', 'PDP', 'All'], index=3, key='1',
        help="Select the type of plans you want to view and compare.")
        
    states_list = ['All']
    states_list.extend(df.columns[13:68])
    
    #pick the state
    state = st.sidebar.selectbox('State', states_list, index=0, key='3',
        help="Select the state to view only contracts with enrollment in that state.")    
        
    #pick the quartile of enrollment
    quartile = st.sidebar.selectbox('Enrollment size quartile', ['Top 25%', '50-75%', '25-50%', 'Bottom 25%', 'All'], index=0, key='2',
        help="Select the quartile (25% range) of health plan contracts to view and compare based on enrollment size.")

        
    #apply filters
    df_filtered = df[df['year'] == year]
    
    if plan_type !='All':
        if plan_type =='MA-PD':
            df_filtered = df_filtered[(df_filtered['has_part_c'] == 1) & (df_filtered['has_part_d'] == 1)]
        elif plan_type =='MA only':
            df_filtered = df_filtered[(df_filtered['has_part_c'] == 1) & (df_filtered['has_part_d'] == 0)]
        elif plan_type =='PDP':
            df_filtered = df_filtered[(df_filtered['has_part_c'] == 0) & (df_filtered['has_part_d'] == 1)]
    
    #if state is selected, filter list and also change the enrollment size to size within the selected state
    if state != 'All':
        df_filtered = df_filtered[df_filtered[state] > 0]
        size = state
    else:
        size = 'total_enrollment'
    
    which_q = {'Top 25%': [3],
                '50-75%': [2],
                '25-50%': [1],
                'Bottom 25%': [0],
                'All': [0,1,2,3]}

    df_filtered = df_filtered.loc[pd.qcut(df_filtered[size], q=4, labels=[0,1,2,3]).isin(which_q[quartile])]

    #make sure there are no 0s for the Total enrollment which is needed for the treemap to draw boxes
    df_filtered = df_filtered[df_filtered[size] != 0]

    #show treemap visualization
    show_treemap(df_filtered, size)
    
    #info to understand the visualization
    st.markdown(
    """
    **Legend**
    - Each box represents an individual contract
    - The size of the box corresponds to the number of enrollments for the contract
    - The color of the box represents the star rating assigned to the contract, using whichever is first available out of overall, Part C, and Part D star rating.
    - Each contract is organized under their parent organization
      - The color and size of the parent organization's box  will reflect the average color and the sum of enrollment size from contracts under the organization.
      - Hovering over a parent organization will be missing all information besides enrollment number, since those details are not available at the organization level. As a result, the other fields will appear as "?"
    - Hovering over the contract will reveal additional details
      - Full name of the contract
      - Marketing name
      - Overall, Part C, and Part D Star Rating
      - Percent enrollment in top 6 states in which the contract has members.
      - Total contract enrollment
      - Organization type
      - SNP = whether the contract is a Special Needs Plan that is specifically designed to provide targeted care to special needs individuals
	""")
    
### start of page for Star measure details (2022)
elif st.session_state.page == 'Star Measure Details (2022)':
    st.markdown("""
    This page allows you to view the detailed documentation of the each measure for the year 2022.
    
    Note: PDFs may not always display correctly depending on the browser you are using.
    """)
    measure_to_pdf = {
        "C01: Breast Cancer Screening": "HED-BCS.pdf",
        "C02: Colorectal Cancer Screening": "HED-COL.pdf",
        "C03: Annual Flu Vaccine": "HOS-FLU.pdf",
        "C04: Monitoring Physical Activity": "HOS-MPA.pdf",
        "C05: Special Needs Plan (SNP) Care Management": "CMS-SNPCM.pdf",
        "C06: Care for Older Adults – Medication Review": "HED-COAMR.pdf",
        "C07: Care for Older Adults – Pain Assessment": "HED-COAPS.pdf",
        "C08: Osteoporosis Management in Women who had a Fracture": "HED-OMW.pdf",
        "C09: Diabetes Care – Eye Exam": "HED-CDCEYE.pdf",
        "C10: Diabetes Care – Kidney Disease Monitoring": "HED-CDCNPH.pdf",
        "C11: Diabetes Care – Blood Sugar Controlled": "HED-CDCA1C.pdf",
        "C12: Rheumatoid Arthritis Management": "HED-ART.pdf",
        "C13: Reducing the Risk of Falling": "HOS-FRM.pdf",
        #"C14: Improving Bladder Control": ,
        "C15: Medication Reconciliation Post-Discharge": "HED-MPR.pdf",
        "C16: Statin Therapy for Patients with Cardiovascular Disease": "HED-SUPC.pdf",
        "C17: Getting Needed Care": "CHA-GNC.pdf",
        "C18: Getting Appointments and Care Quickly": "CHA-APTQ.pdf",
        "C19: Customer Service": "CHA-SVC.pdf",
        "C20: Rating of Health Care Quality": "CHA-HCQ.pdf",
        "C21: Rating of Health Plan": "CHA-RPLA.pdf",
        "C22: Care Coordination": "CHA-COORD.pdf",
        "C23: Complaints about the Health Plan": "CMS-C-CMPL.pdf",
        "C24: Members Choosing to Leave the Plan": "CMS-C-DSNR.pdf",
        "C25: Health Plan Quality Improvement": "CMS-C-IMPR.pdf",
        "C26: Plan Makes Timely Decisions about Appeals": "CMS-C-TIME.pdf",
        "C27: Reviewing Appeals Decisions": "CMS-C-FAIR.pdf",
        "C28: Call Center – Foreign Language Interpreter and TTY Availability": "CMS-C-LANG.pdf",
        "D01: Call Center – Foreign Language Interpreter and TTY Availability": "CMS-D-LANG.pdf",
        "D02: Complaints about the Drug Plan": "CMS-D-CMPL.pdf",
        "D03: Members Choosing to Leave the Plan": "CMS-D-DSNR.pdf",
        "D04: Drug Plan Quality Improvement": "CMS-D-IMPR.pdf",
        "D05: Rating of Drug Plan": "CMS-D-RDRG.pdf",
        "D06: Getting Needed Prescription Drugs": "CHA-GETRX.pdf",
        "D07: MPF Price Accuracy": "CMS-MPF.pdf",
        "D08: Medication Adherence for Diabetes Medications": "PQA-DIAB.pdf",
        "D09: Medication Adherence for Hypertension (RAS antagonists)": "PQA-HTN.pdf",
        "D10: Medication Adherence for Cholesterol (Statins)": "PQA-CHOL.pdf",
        "D11: MTM Program Completion Rate for CMR": "CMS-MTM.pdf",
        "D12: Statin Use in Persons with Diabetes (SUPD)": "PQA-SUPD.pdf"
    }
    
    measures = measure_to_pdf.keys()

    #user selects which measure to look at
    measure = st.sidebar.radio("Select the measure you want to explore", options=measures, index=0)

    #construct file path for PDF
    filepath = "measure_pdfs_2022/" + measure_to_pdf[measure]

    #display the PDF
    st.markdown(displayPDF(filepath), unsafe_allow_html=True)
    
### start of page for Correlations dashboard 
elif st.session_state.page == 'Correlations Dashboard':
    st.markdown("""
    This page of shows how a specific measure correlates with additional data sources for the purpose of identifying significant correlations that could be used as predictors in machine learning models. Select the desired measure in the sidebar and then toggle the groups of data sources to show correlation plots.
    
    The Y axis are the selected measure's scores, while the X axis are the additional predictor's values. For predictors only available at the state level, the Y axis is the weighted average of all contracts in each state based on the contract's enrollment size in each state.
    """)
    df = load_data("data/visualization_data_correlations.csv")

    measure_list = ['C-Breast Cancer Screening', 'C-Colorectal Cancer Screening',
       'C-Care for Older Adults - Pain Assessment',
       'C-Osteoporosis Management in Women who had a Fracture',
       'C-Diabetes Care - Eye Exam',
       'C-Diabetes Care - Kidney Disease Monitoring',
       'C-Diabetes Care - Blood Sugar Controlled',
       'C-Controlling Blood Pressure',
       'C-Rheumatoid Arthritis Management',
       'C-Reducing the Risk of Falling', 'C-Plan All-Cause Readmissions',
       'C-Getting Needed Care', 'C-Annual Flu Vaccine',
       'C-Getting Appointments and Care Quickly', 'C-Customer Service',
       'C-Rating of Health Care Quality', 'C-Rating of Health Plan',
       'C-Care Coordination', 'C-Complaints about the Health Plan',
       'C-Members Choosing to Leave the Plan',
       'C-Beneficiary Access and Performance Problems',
       'C-Health Plan Quality Improvement',
       'C-Plan Makes Timely Decisions about Appeals',
       'C-Improving or Maintaining Physical Health',
       'C-Reviewing Appeals Decisions',
       'C-Call Center - Foreign Language Interpreter and TTY Availability',
       'D-Call Center - Foreign Language Interpreter and TTY Availability',
       'D-Appeals Auto-Forward', 'D-Appeals Upheld',
       'D-Complaints about the Drug Plan',
       'D-Members Choosing to Leave the Plan',
       'D-Beneficiary Access and Performance Problems',
       'D-Drug Plan Quality Improvement', 'D-Rating of Drug Plan',
       'C-Improving or Maintaining Mental Health',
       'D-Getting Needed Prescription Drugs', 'D-MPF Price Accuracy',
       'D-Medication Adherence for Diabetes Medications',
       'D-Medication Adherence for Hypertension (RAS antagonists)',
       'D-Medication Adherence for Cholesterol (Statins)',
       'D-MTM Program Completion Rate for CMR',
       'C-Improving Bladder Control',
       'C-Medication Reconciliation Post-Discharge',
       'C-Monitoring Physical Activity', 'C-Adult BMI Assessment',
       'C-Special Needs Plan (SNP) Care Management',
       'C-Care for Older Adults - Medication Review',
       'C-Care for Older Adults - Functional Status Assessment',
       'C-Statin Therapy for Patients with Cardiovascular Disease',
       'D-Statin Use in Persons with Diabetes (SUPD)']
       
    measure = st.sidebar.selectbox('Measure', measure_list, index=0, key='1',
        help="Select the measure to see correlations for.")
    
    hover_data = ['year', 'state_id']
    
    #filters for census data
    demo_cols = ['american_indian_alaska_native', 'asian', 'black', 'hawaiian_pacific_islander', 'white', 'population']
    if st.sidebar.checkbox('Show Demographics correlations', False, key = '2'):
        st.subheader("Demographics correlations")
        for c in demo_cols:
            show_scatter(df, c, measure, hover_data)
    
    income_cols = ['personal_income', 'income_per_capita']
    if st.sidebar.checkbox('Show Income correlations', False, key = '3'):
        st.subheader("Income correlations")
        for c in income_cols:
            show_scatter(df, c, measure, hover_data)
    
    #filters for various disease indicators
    #arthritis
    arthritis_cols = ['Fair or poor health among adults aged >= 18 years with arthritis',
       'Severe joint pain due to arthritis among adults aged >= 18 years who have doctor-diagnosed arthritis',
       'Arthritis among adults aged >= 18 years who have heart disease',
       'Adults aged >= 18 years with arthritis who have taken a class to learn how to manage arthritis symptoms',
       'Arthritis among adults aged >= 18 years who are obese',
       'Physical inactivity among adults aged >= 18 years with arthritis',
       'Arthritis among adults aged >= 18 years who have diabetes',
       'Arthritis among adults aged >= 18 years',
       'Work limitation due to arthritis among adults aged 18-64 years who have doctor-diagnosed arthritis']
    if st.sidebar.checkbox('Show Arthritis indicators correlations', False, key = '4'):
        st.subheader("Arthrtis correlations")
        for c in arthritis_cols:
            show_scatter(df, c, measure, hover_data)
            
    #diabetes
    diabetes_cols = ['Hospitalization with diabetes as a listed diagnosis',
       'Mortality due to diabetes reported as any listed cause of death',
       'Mortality with diabetic ketoacidosis reported as any listed cause of death',
       'Amputation of a lower extremity attributable to diabetes',
       'Adults with diagnosed diabetes aged >= 18 years who have taken a diabetes self-management course',
       'Influenza vaccination among noninstitutionalized adults aged >= 65 years with diagnosed diabetes',
       'Influenza vaccination among noninstitutionalized adults aged 18-64 years with diagnosed diabetes',
       'Prevalence of depressive disorders among adults aged >= 18 years with diagnosed diabetes',
       'Prevalence of high blood pressure among adults aged >= 18 years with diagnosed diabetes',
       'Prevalence of high cholesterol among adults aged >= 18 years with diagnosed diabetes',
       'Visits to dentist or dental clinic among adults aged >= 18 years with diagnosed diabetes',
       'Foot examination among adults aged >= 18 years with diagnosed diabetes',
       'Pneumococcal vaccination among noninstitutionalized adults aged 18-64 years with diagnosed diabetes',
       'Glycosylated hemoglobin measurement among adults aged >= 18 years with diagnosed diabetes',
       'Prevalence of diagnosed diabetes among adults aged >= 18 years',
       'Dilated eye examination among adults aged >= 18 years with diagnosed diabetes',
       'Pneumococcal vaccination among noninstitutionalized adults aged >= 65 years with diagnosed diabetes']
    if st.sidebar.checkbox('Show Diabetes indicators correlations', False, key = '5'):
        st.subheader("Diabetes correlations")
        for c in diabetes_cols:
            show_scatter(df, c, measure, hover_data)
    
    #cancer
    cancer_cols=['Invasive cancer of the female breast, incidence',
       'Cancer of the female breast, mortality',
       'Cancer of the colon and rectum (colorectal), mortality',
       'Cancer of the colon and rectum (colorectal), incidence',
       'Mammography use among women aged 50-74 years',
       'Fecal occult blood test, sigmoidoscopy, or colonoscopy among adults aged 50-75 years']
    if st.sidebar.checkbox('Show Cancer indicators correlations', False, key = '6'):
        st.subheader("Cancer correlations")
        for c in cancer_cols:
            show_scatter(df, c, measure, hover_data)
    
    #kidney disease   
    kidney_cols = ['Mortality with end-stage renal disease',
       'Incidence of treated end-stage renal disease attributed to diabetes',
       'Incidence of treated end-stage renal disease',
       'Prevalence of chronic kidney disease among adults aged >= 18 years']
    if st.sidebar.checkbox('Show Kidney Disease indicators correlations', False, key = '7'):
        st.subheader("Kidney disease correlations")
        for c in kidney_cols:
            show_scatter(df, c, measure, hover_data)
    
    #cardiovascular disease
    cardio_cols = ['Hospitalization for heart failure among Medicare-eligible persons aged >= 65 years',
       'Mortality from total cardiovascular diseases',
       'Mortality from cerebrovascular disease (stroke)',
       'Hospitalization for acute myocardial infarction',
       'Mortality from diseases of the heart',
       'Mortality from heart failure',
       'Mortality from coronary heart disease',
       'Hospitalization for stroke',
       'Taking medicine for high blood pressure control among adults aged >= 18 years with high blood pressure',
       'Influenza vaccination among noninstitutionalized adults aged >= 65 years with a history of coronary heart disease or stroke',
       'High cholesterol prevalence among adults aged >= 18 years',
       'Influenza vaccination among noninstitutionalized adults aged 18-64 years with a history of coronary heart disease or stroke',
       'Cholesterol screening among adults aged >= 18 years',
       'Pneumococcal vaccination among noninstitutionalized adults aged 18-64 years with a history of coronary heart disease',
       'Pneumococcal vaccination among noninstitutionalized adults aged >= 65 years with a history of coronary heart disease',
       'Awareness of high blood pressure among adults aged >= 18 years']
    if st.sidebar.checkbox('Show Cardiovascular Disease indicators correlations', False, key = '8'):
        st.subheader("Cardiovascular disease correlations")
        for c in cardio_cols:
            show_scatter(df, c, measure, hover_data)
    
    #osteoporosis
    osteo_cols = ['Osteoporosis-All Females', 'Osteoporosis-Females 65 and older']
    if st.sidebar.checkbox('Show Osteoporosis correlations', False, key = '9'):
        st.subheader("Osteoporosis correlations")
        for c in osteo_cols:
            show_scatter(df, c, measure, hover_data)
    
    #annual flu vaccination
    flu_cols = ['seasonal influenca vaccine coverage for 65 and older']
    if st.sidebar.checkbox('Show Flu vaccine coverage correlations', False, key = '10'):
        st.subheader("Flu vaccine coverage correlations")
        for c in flu_cols:
            show_scatter(df, c, measure, hover_data)

    #disenrollment reasons
    reason_cols = ['Problems Getting Needed Care, Coverage, and Cost Information',
       'Problems Getting Information and Help from the Plan',
       'Problems with Coverage of Doctors and Hospitals',
       'Financial Reasons for Disenrollment',
       'Problems with Prescription Drug Benefits and Coverage']
    if st.sidebar.checkbox('Show Disenrollment Reasons correlations', False, key = '11'):
        st.subheader("Disenrollment reasons correlations")
        df_reasons = load_data("data/visualization_data_correlations_disenrollment.csv")
        for c in reason_cols:
            show_scatter(df_reasons, c, measure, ['year', 'contract_id'])

### start of page for Contract Star Details
elif st.session_state.page == 'Contract Star Details':
    df = load_data("data/visualization_data_contract_details.csv")
    df_cutpoints = load_data("data/visualization_data_cutpoints.csv")
    
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    #pick the year
    year = st.sidebar.slider("Year", min_value = min_year, max_value = max_year, value=max_year,
        help="Select the year you want to view the Star Ratings for.")
    
    df_filtered = df[df['year'] == year]
    
    #filter contracts by parent org
    parent_list = ['All']
    parent_list.extend(df_filtered['parent_org_name'].unique())
    
    parent = st.sidebar.selectbox('Parent Organization', parent_list, index=0, key='1',
        help="Filter contract selection by parent organization.")
    
    if parent != "All":
        df_filtered = df_filtered[df_filtered['parent_org_name'] == parent]
    

    #pick the plan type
    plan_type = st.sidebar.selectbox('Plan Type', ['MA-PD', 'MA only', 'PDP', 'All'], index=3, key='100',
        help="Select the type of plans you want to view and compare.")
    
    if plan_type !='All':
        if plan_type =='MA-PD':
            df_filtered = df_filtered[(df_filtered['has_part_c'] == 1) & (df_filtered['has_part_d'] == 1)]
        elif plan_type =='MA only':
            df_filtered = df_filtered[(df_filtered['has_part_c'] == 1) & (df_filtered['has_part_d'] == 0)]
        elif plan_type =='PDP':
            df_filtered = df_filtered[(df_filtered['has_part_c'] == 0) & (df_filtered['has_part_d'] == 1)]
    
    #select contract
    contracts = df_filtered[['contract_id','contract_name']].drop_duplicates()
    contracts_list = contracts['contract_id'] + ' - ' + contracts['contract_name']
    
    select_contract = st.sidebar.selectbox('Select Contract', contracts_list, key='3', index=0,
        help="Select the contract.")
    
    contract = select_contract.split(' - ')[0]
    df_filtered = df_filtered[df_filtered['contract_id'] == contract]
    
    
    #select measure
    #only allow Part C/D measures if contract has that part
    if (df_filtered['has_part_c'].iloc[0] == 1) and (df_filtered['has_part_d'].iloc[0] == 0):
        measure_list = df_filtered[df_filtered['is_part_c'] == 1]['measure'].unique()
    elif (df_filtered['has_part_c'].iloc[0] == 0) and (df_filtered['has_part_d'].iloc[0] == 1):
        measure_list = df_filtered[df_filtered['is_part_d'] == 1]['measure'].unique()
    else:
        measure_list = df_filtered['measure'].unique()
    
    measure = st.sidebar.selectbox('Select Measure to View Trends and for Simulations', measure_list, key='4', index=0,
        help="Select the measure to view historical trends on. The selected measure can also have its measure star altered to simulate changes to star rating")
    
    #info about the contract
    st.subheader("Display CMS measure details for " + select_contract)
    single_contract = df_filtered.iloc[0]
    yes_no = {1: "Yes", 0: "No"}

    st.markdown(f"Marketing name: {single_contract['marketing_name']}")
    st.markdown(f"Parent Organization: {single_contract['parent_org_name']}")
    st.markdown(f"Organization Type: {single_contract['org_type_name']}")
    
    #display part C
    if np.isnan(single_contract['part_c_star']):
        st.markdown(f"Has Part C: {yes_no[single_contract['has_part_c']]}")
    else:
        st.markdown(f"Has Part C: {yes_no[single_contract['has_part_c']]} -- Part C Star: {single_contract['part_c_star']}")
    
    #display part D
    if np.isnan(single_contract['part_d_star']):
        st.markdown(f"Has Part D: {yes_no[single_contract['has_part_d']]}")
    else:
        st.markdown(f"Has Part D: {yes_no[single_contract['has_part_d']]} -- Part D Star: {single_contract['part_d_star']}")
    
    #display overall
    if np.isnan(single_contract['overall_star']):
        st.markdown("Overall Star: Not available")
    else:
        st.markdown(f"Overall Star: {single_contract['overall_star']}")

    ### use CSS to hide the index column in the streamlit table
    # CSS to inject contained in a string
    hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    
    cols = ['domain_id', 'domain_name', 'measure', 'score', 'star']
    
    ### display Part C/D measures if the plan type has the corresponding part
    st.subheader("Measure Performance for Selected Contract and Year")
    st.markdown("""
    **Legend**
    - measure = name of the quality measure that CMS uses to calculate Part C and D contract performance
    - score = measure score for selected contract and year
    - star = measure star for selected contract and year
    """)
    st.markdown("**Note:** It is possible for the selected contract to not receive a score for one or more measures, in which case N/A will be displayed")
    
    if df_filtered['has_part_c'].iloc[0] == 1:
        st.subheader("Part C measures")
        df_c = df_filtered[df_filtered['is_part_c'] == 1]
        df_c = df_c[cols].sort_values(by='domain_id')
        
        #show each domain's measures
        for d in df_c['domain_id'].unique():
            df_domain = df_c[df_c['domain_id'] == d]
            st.markdown("Domain " + d + " - " + df_domain['domain_name'].iloc[0])
            show_measures_table(df_domain[['measure', 'score', 'star']])
    if df_filtered['has_part_d'].iloc[0] == 1:
        st.subheader("Part D measures")
        df_d = df_filtered[df_filtered['is_part_d'] == 1]
        df_d = df_d[cols].sort_values(by='domain_id')

        #show each domain's measures
        for d in df_d['domain_id'].unique():
            df_domain = df_d[df_d['domain_id'] == d]
            st.markdown("Domain " + d + " - " + df_domain['domain_name'].iloc[0])
            show_measures_table(df_domain[['measure', 'score', 'star']])
    
    ### display recommended measures to focus on for improving star rating
    st.subheader("Recommendations: Top measures to focus on")
    st.markdown("This is the list of recommended measures to focus on with the highest recommended at the top.")
    st.markdown("""The list is created by first removing measures not scored and measures already at 5 stars. Next, the remaining measures are sorted with highest weighted measures at the top, and then within each weight, again sorted with highest penetration to the top.
    """)
    st.markdown("""Through this methodology, we are left with the measures that still have room to improve ranked by their impact to the overall Star Rating, and how close they are to achieving the next star.
    """)
    
    #Legend for recommendations table
    st.markdown("""
    **Legend**
    - score = measure score for selected contract and year
    - star = measure star for selected contract and year
    - weight = measure weight given by CMS for how much they contribute to the overall and summary star rating calculation (1 to 5)
    - lower = lower bound of the measure star cut points given by CMS
    - upper = upper bound of the measure star cut points given by CMS
    - penetration = % calculation of how much of the star cut point's range the measure has covered
      - For higher is better measures: $$penetration = \dfrac{score - lower}{upper - lower} * 100$$
      - For lower is better measures: $$penetration = \dfrac{upper - score}{upper - lower} * 100$$
      - Measures related to patient experience (comes from CAHPS surveys) rely on significance testing in addition to cut points when assigning stars, so it is possible for the score to be outside the cut points of the assigned star, resulting in over 100% or negative penetration
    """)
    
    #filter out measures not scored (if no score, we cannot know if it is worth recommending)
    recommended = df_filtered.dropna(subset='score')
    #filter to measures with less than 5 stars
    recommended = recommended[recommended['star'] < 5]
    #sort highest weighted measures to the top then sort highest penetration to the top
    recommended = recommended.sort_values(by=['weight', 'penetration'], ascending=False)
    
    #display recommendations
    show_recommendations(recommended[['measure', 'score', 'star', 'weight', 'lower', 'upper','penetration']])
    
    
    ### allow simulation for how changes in specific measure stars could impact overall star rating
    
    st.subheader("Simulations: See how specific measure changes impact overall and summary star")
    st.markdown("""
    - Choose the measure to add to the simulation from the sidebar
    - Use the number input to select the desired star
    - Click Add Measure to Simulation to store the modified measure star and impact the simulated calculations
    """)
    
    #select the star value to simulate
    simulate_value = st.number_input("Set Selected Measure's Star to: ", min_value=1, max_value=5, value=5, key='Simulated value')
    
    #confirm and store the star value
    st.button('Add Measure to Simulation', on_click=update_star, kwargs={'measure':measure, 'star':simulate_value}, key='Store simulated value')
    
    
    # display all currently simulated measures
    st.markdown("**Simulating the following measure changes**")
    
    if len(st.session_state.measures.keys()) == 0:
        st.markdown("No measure changes currently selected")
    else:
        for key in st.session_state.measures.keys():
            message = key + " = " + str(st.session_state.measures[key])
            st.markdown(message)
    
    #calculate simulated  star rating
    st.markdown("**Calculations**")
    
    sim_df = create_simulated_measures_df(df_filtered)
    
    #create pandas df of results
    list_df_results = []
    
    #add a row for each star type
    if np.isnan(single_contract['part_c_star']) == False:
        raw, rounded = overall_summary_star(sim_df, 'part_c')
        tmp_result_df = pd.DataFrame({'Star Type': ['Part C Summary Star Rating'], 'Rounded': [rounded], 'Raw': [raw], 'Actual':[single_contract['part_c_star']]})
        list_df_results.append(tmp_result_df)
        
    if np.isnan(single_contract['part_d_star']) == False:
        raw, rounded = overall_summary_star(sim_df, 'part_d')
        tmp_result_df = pd.DataFrame({'Star Type': ['Part D Summary Star Rating'], 'Rounded': [rounded], 'Raw': [raw], 'Actual':[single_contract['part_d_star']]})
        list_df_results.append(tmp_result_df)
        
    if np.isnan(single_contract['overall_star']) == False:
        raw, rounded = overall_summary_star(sim_df, 'overall')
        tmp_result_df = pd.DataFrame({'Star Type': ['Overall Star Rating'], 'Rounded': [rounded], 'Raw': [raw], 'Actual':[single_contract['overall_star']]})
        list_df_results.append(tmp_result_df)
    
    #put the df together, and reset index to get around error with the styler
    if len(list_df_results) > 0:
        calc_result_df = pd.concat(list_df_results)
        calc_result_df = calc_result_df.reset_index(drop=True)
        show_star_results(calc_result_df)
        
        st.markdown("""
        **Legend**
        - Rounded = raw star rounded to the nearest 0.5 star
        - Raw = unrounded weighted average of individual measure stars using all available measures and then overridden by the simulated measure stars as specified
        - Actual = actual star assigned by CMS for the selected contract and year
        """)
        
        st.markdown("""
        The actual star ratings are adjusted with a reward factor that and a Categorical Adjustment Index (CAI) after the weighted average of measure stars. The reward factor can increase the star as a reward for having both high and stable relative performance. The CAI add or subtract from the star rating to adjust for the within-contract disparity in performance for Low Income Subsidy/Dual Eligible and disabled beneficiaries.
        """)
        
        st.markdown("The above calculations do not take into account these adjustments and therefore can be different from the actual Star Rating.")
    else:
        st.markdown("Plan does not have enough data to receive overall and summary stars")
    
    
    #clear and reset all simulated values
    st.button('Clear and Reset', on_click=clear_simulation, key='Reset simulated value')
    
    
    ##### measure specific info
    df_contract_measure = df[(df['contract_id'] == contract) & (df['measure'] == measure)]
    
    ### display historical trend and cut points boundary for selected measure
    st.subheader("Measure Historical Trend")
    st.markdown("Display how the current contract performed on the selected measure over the past few years.")
    st.markdown("""
    Also includes the predicted 2023 measure score created using machine learning models. The predicted score appears in a lighter shade of blue and can be compared against the actual 2023 measure score in dark blue.
    """)
    
    #use PDP cut points if contract has no part C but has part D
    if (df_filtered['has_part_c'].iloc[0] == 0) and (df_filtered['has_part_d'].iloc[0] == 1):
        use_PDP = 1
    else:
        use_PDP = 0
    
    measure_cutpoints = df_cutpoints[(df_cutpoints['measure'] == measure) & (df_cutpoints['is_PDP'] == use_PDP)]
    
    # if higher is better
    if measure_cutpoints['higher_is_better'].iloc[0] == 1:
        star_order = range(1,5)
        star_color = {1:'orange',
            2:'yellow',
            3:'yellowgreen',
            4:'green'}
    else:
        star_order = range(5,1,-1)
        star_color = {2:'orange',
            3:'yellow',
            4:'yellowgreen',
            5:'green'}
        
    #show graph
    fig = go.Figure()

    for star in star_order:
        #filter to specific star
        star_cutpoints = measure_cutpoints[measure_cutpoints['star'] == star]
        
        #set up cutpoints to create 0.5 difference between the year of the cutpoint
        newest = star_cutpoints[star_cutpoints['year'] == star_cutpoints['year'].max()]
        newest['year'] = newest['year'] + 1
        star_cutpoints = pd.concat([star_cutpoints, newest])
        star_cutpoints['year'] = star_cutpoints['year'] - 0.5
        
        #add line for star cut points and fill the boundary
        fig.add_trace(go.Scatter(x=star_cutpoints['year'], y=star_cutpoints['upper'],
                        mode='lines', line=dict(shape='vh', dash='dash', color=star_color[star]),
                        fill='tonexty', name=str(star) + ' star',
                        hovertemplate='Upper bound: %{y}')
                        )

    #line graph of the contract's measure scores
    fig.add_trace(go.Scatter(x=df_contract_measure['year'], y=df_contract_measure['score'],
                        mode='lines+markers', line=dict(color='darkblue', width=4),
                        marker=dict(size=12), name='Measure score',
                        hovertemplate='Year: %{x}<br>' +
                        'Score: %{y}')
                        )
    
    df_pred = pd.read_csv("data/Complete_2023_pred.csv", index_col=0)
    #st.table(df_pred)\
    
    #use try catch since there are cases where the contract does not have a cell for the measure
    try:
        #also check if NaN since the contract/measure cell could be null
        if np.isnan(df_pred.loc[contract, measure]) == False:
            pred_x= [2023]
            pred_y= [df_pred.loc[contract, measure]]
            
            #line graph of the contract's predicted 2023 measure score
            fig.add_trace(go.Scatter(x=pred_x, y=pred_y,
                                mode='lines+markers', line=dict(color='cornflowerblue', width=4),
                                marker=dict(size=10), name='Predicated score',
                                hovertemplate='Year: %{x}<br>' +
                                'Predicted Score: %{y}')
                                )
        else:
            st.markdown("No predicted score available for selected contract and measure.")
    except KeyError:
        st.markdown("No predicted score available for selected contract and measure.")
        
    #add title                  
    fig.update_layout(
        title="Historical Trend for " + measure,
        xaxis_title="Year",
        yaxis_title="Measure Score"
    )

    st.plotly_chart(fig)