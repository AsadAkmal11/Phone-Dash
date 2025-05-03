import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(layout="wide", page_title="PhoneDash Simple", page_icon="📱")
st.title("📱 PhoneDash: Simple Mobile Dashboard")

# --- 1. File Upload & Data Cleaning ---
st.sidebar.header("Upload Your Mobile Price CSV")
file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

def clean_data(df):
    df.columns = df.columns.str.strip()  # Remove spaces in col names
    df = df.drop_duplicates().dropna(subset=["Brand", "Price ($)"])
    # Convert RAM, Storage, Battery to numeric (extract numbers)
    for col in ["RAM", "Storage", "Battery Capacity (mAh)"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.extract(r"(\d+)").astype(float)
    df["Price ($)"] = pd.to_numeric(df["Price ($)"], errors="coerce")
    df = df.dropna(subset=["Price ($)"])
    return df

if file:
    df = pd.read_csv(file)
    df = clean_data(df)
    st.success("Data loaded and cleaned!")
else:
    st.info("Please upload a CSV file to get started.")
    st.stop()

# --- 2. Visualization ---
st.header("🔍 Data Overview")
st.dataframe(df.head(20))

col1, col2 = st.columns(2)
with col1:
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    df["Price ($)"].plot.hist(ax=ax, bins=20, color='skyblue')
    ax.set_xlabel("Price ($)")
    st.pyplot(fig)
with col2:
    st.subheader("Phones per Brand")
    brand_counts = df["Brand"].value_counts()
    fig2, ax2 = plt.subplots()
    brand_counts.plot.bar(ax=ax2, color='orange')
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

# --- 3. Phone Comparison ---
st.header("⚖ Compare Two Phones")
st.write("Select two phones to compare their specs side by side.")

# Create a unique label for each phone
if "Storage" in df.columns:
    df["CompareLabel"] = df["Brand"].astype(str) + " | " + df["Model"].astype(str) + " | " + df["Storage"].astype(str) + "GB"
else:
    df["CompareLabel"] = df["Brand"].astype(str) + " | " + df["Model"].astype(str)

selected = st.multiselect(
    "Select Phones:", df["CompareLabel"], max_selections=2
)
if len(selected) == 2:
    comp = df[df["CompareLabel"].isin(selected)].set_index("CompareLabel").T
    st.dataframe(comp)
elif len(selected) > 2:
    st.warning("Please select only two phones.")

# --- 4. Recommendation ---
st.header("🏆 Recommend Phones")
st.write("Set your budget and minimum RAM to get suggestions.")
min_price, max_price = int(df["Price ($)"].min()), int(df["Price ($)"].max())
budget = st.slider("Max Price ($):", min_price, max_price, max_price)
min_ram = st.slider("Min RAM (GB):", int(df["RAM"].min()), int(df["RAM"].max()), int(df["RAM"].min()))

recs = df[(df["Price ($)"] <= budget) & (df["RAM"] >= min_ram)]
if not recs.empty:
    st.dataframe(recs[["Brand", "Model", "Price ($)", "RAM", "Storage", "Battery Capacity (mAh)"]].sort_values("Price ($)"))
else:
    st.info("No phones match your criteria.")

# --- 5. Machine Learning: Predict Phone Price ---
st.header("🤖 Predict Phone Price (ML)")
st.write("Enter phone specs to predict the price:")

# Prepare data for ML
ml_df = df.copy()
features = ["RAM", "Storage", "Battery Capacity (mAh)", "Brand"]
target = "Price ($)"

# Encode Brand
le = LabelEncoder()
ml_df["Brand_encoded"] = le.fit_transform(ml_df["Brand"].astype(str))

# Drop rows with missing values in features
ml_df = ml_df.dropna(subset=["RAM", "Storage", "Battery Capacity (mAh)", "Price ($)"])
X = ml_df[["RAM", "Storage", "Battery Capacity (mAh)", "Brand_encoded"]]
y = ml_df["Price ($)"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# User input for prediction
brand_input = st.selectbox("Brand", df["Brand"].unique())
ram_input = st.number_input("RAM (GB)", int(df["RAM"].min()), int(df["RAM"].max()), int(df["RAM"].median()))
storage_input = st.number_input("Storage (GB)", int(df["Storage"].min()), int(df["Storage"].max()), int(df["Storage"].median()))
battery_input = st.number_input("Battery Capacity (mAh)", int(df["Battery Capacity (mAh)"].min()), int(df["Battery Capacity (mAh)"].max()), int(df["Battery Capacity (mAh)"].median()))

if st.button("Predict Price"):
    input_array = np.array([[ram_input, storage_input, battery_input, le.transform([brand_input])[0]]])
    pred_price = model.predict(input_array)[0]
    st.success(f"Predicted Price: ${pred_price:,.0f}")

st.markdown("---")
st.markdown("Sir Asim Shah We Love You!")