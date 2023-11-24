import pickle
import numpy as np
import pandas as pd
import streamlit as st
import os
from PIL import Image

# Load the model
model_path = '../model/LR_clas.pkl'
if os.path.exists(model_path):
    model = pickle.load(open(model_path, 'rb'))
else:
    print(f"File not found: {model_path}")

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def create_age_group(age):
    # Define the age bins and labels
    bins = [0, 1, 10, 19, 44, 59, age.max() + 1]  # Ensure the bins increase monotonically
    labels = ["Babies", "Children", "Teenager", "Adult", "Pra-Senior Age", "Senior"]
    
    # Sort the bins in increasing order
    bins.sort()
    
    # Bin the age data and return the age group
    return pd.cut(age, bins=bins, labels=labels, right=False)

def create_duration_category(duration):
    # Define the duration bins and labels
    bins = [1, 120, 241, 361, duration.max() + 1]  # Ensure the bins increase monotonically
    labels = ["1-4 months", "5-8 months", "9-12 months", ">12 months"]
    
    # Sort the bins in increasing order
    bins.sort()
    
    # Bin the duration data and return the duration category
    return pd.cut(duration, bins=bins, labels=labels, right=True)

def main():
    # Load picture
    image_california = Image.open('../img/hospital.jpg')

    # Add explanatory text and picture in the sidebar
    st.sidebar.info('This app is created to predict claim of travel insurance.')    
    st.sidebar.image(image_california)

    # Add title
    st.title("Travel Insurance Predict Claim App")

    # Set up the form to fill in the required data
    agency = st.text_input('Agency')
    agency_type = st.selectbox('Agency Type', ['Airlines', 'Travel Agency']) # Replace with actual types
    distribution_channel = st.selectbox('Distribution Channel', ['Online', 'Offline']) # Replace with actual channels
    product_name = st.selectbox('Product Name', ['Annual Silver Plan','Cancellation Plan','Basic Plan','2 way Comprehensive Plan','Bronze Plan','1 way Comprehensive Plan','Rental Vehicle Excess Insurance','Single Trip Travel Protect Gold','Silver Plan','Value Plan','24 Protect','Annual Travel Protect Gold','Comprehensive Plan','Ticket Protector','Travel Cruise Protect','Single Trip Travel Protect Silver','Individual Comprehensive Plan','Gold Plan','Annual Gold Plan','Child Comprehensive Plan','Premier Plan','Annual Travel Protect Silver','Single Trip Travel Protect Platinum','Annual Travel Protect Platinum','Spouse or Parents Comprehensive Plan','Travel Cruise Protect Family'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    duration = st.number_input('Duration', min_value=1)
    destination = st.text_input('Destination')
    net_sales = st.number_input('Net Sales', min_value=1.0)
    commission = st.number_input('Commision (in value)', min_value=1.0)
    age = st.number_input('Age', min_value=1)
    
    # Create 'Age Group' and 'Duration Category' based on the provided logic
    age_group = create_age_group(pd.Series([age]))
    duration_category = create_duration_category(pd.Series([duration]))

    # Convert form to data frame
    input_df = pd.DataFrame([{
        'Agency': agency,
        'Agency Type': agency_type,
        'Distribution Channel': distribution_channel,
        'Product Name': product_name,
        'Destination': destination,
        'Net Sales': net_sales,
        'Commision (in value)': commission,
        'Duration Category': duration_category[0],  # Add this column if required
        'Age Group': age_group[0]  # Add this column if required
    }])
        
    # Set a variabel to store the output
    output = ""
    result = ""

    # Make a prediction 
    if st.button("Predict"):
        output = model.predict(input_df)
        if output[0] == 1:
            result = "The user will be use claim"
        else:
            result = "The user won't be use claim"

    # Show prediction result
    st.success(result)          

if __name__ == '__main__':
    main()