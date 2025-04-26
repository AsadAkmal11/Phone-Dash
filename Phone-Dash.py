import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
data = "C:\\Users\\USER\\OneDrive\\Desktop\\2nd Semester\\Ai Lab\\Phone-Dash\\Mobile phone price.csv"

df=pd.read_csv(data)
print(df)
print(df['Brand'])
