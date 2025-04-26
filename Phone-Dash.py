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


