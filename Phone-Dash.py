import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
data = "C:\\Users\\USER\\OneDrive\\Desktop\\2nd Semester\\Ai Lab\\Phone-Dash\\Mobile phone price.csv"
import pandas as pd

def clean_data(df):

#    This function removes all null values from the dataframe
#   and resets the index.

    # Drop rows with any null values
    df = df.dropna()
    
    # Reset the index after dropping
    df = df.reset_index(drop=True)
    
    # Print success message
    print("✅ Data cleaned successfully! No missing values remaining.")
    
    return df

df=pd.read_csv(data)
print(df)
print(df['Brand'])
df=clean_data(df)
st.title("Phone Dash")
st.write("### Data Preview")
st.dataframe(df.head())             # Show first rows
st.write(df.info())                 # Print info to console or use st.write
st.write(df.describe())             # Summary statistics
st.sidebar.header("Filter Options") 
brand_list = sorted(df['Brand'].unique())
selected_brand = st.sidebar.selectbox("Choose Brand:", ["All"] + brand_list)



