# Phase 1: Setup & Dataset Prep

import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # Using RandomForest as requested
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib # For saving/loading model (optional but good practice)
import os # To check if model file exists

# --- Configuration ---
st.set_page_config(layout="wide", page_title="PhoneDash", page_icon="📱")
DATA_FILE = "Mobile phone price.csv" # Make sure this file is in the same directory
MODEL_FILE = "phone_price_predictor.joblib"
df=pd.read_csv(DATA_FILE)
print(df.head())
# --- Data Loading & Caching ---
# Use st.cache_data for functions that return data (like loading from CSV)
@st.cache_data
def load_data(file_path):
    """Loads data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        print("✅ Data loaded successfully.")
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please make sure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- Data Cleaning ---
def clean_data(df):
    """Cleans the dataframe: handles missing values, removes duplicates, resets index."""
    if df is None:
        return None

    print(f"Initial data shape: {df.shape}")
    # Basic check for essential columns (adjust based on your actual CSV)
    required_cols = ['Brand', 'Price ($)', 'RAM', 'Storage', 'Battery Capacity (mAh)'] # Add spaces if they exist in your CSV header!
    missing_essential = [col for col in required_cols if col not in df.columns]
    if missing_essential:
        st.warning(f"Warning: Essential columns missing from dataset: {', '.join(missing_essential)}. Functionality might be limited.")
        # Attempt to find similar columns (case-insensitive, trim whitespace)
        df.columns = df.columns.str.strip() # Clean column names
        required_cols = [col.strip() for col in required_cols]
        missing_essential = [col for col in required_cols if col not in df.columns]
        if missing_essential:
             st.error(f"Still missing essential columns after cleaning names: {', '.join(missing_essential)}. Please check your CSV.")
             return None # Stop processing if essential columns aren't there

    # Handle missing values (strategy: drop rows with missing price or brand, impute others if needed)
    initial_rows = len(df)
    df.dropna(subset=['Price ($)', 'Brand'], inplace=True) # Crucial columns
    rows_after_dropna = len(df)
    print(f"Dropped {initial_rows - rows_after_dropna} rows with missing Price or Brand.")

    # Example: Impute numerical columns like RAM, Storage, Battery with median (if they exist)
    for col in ['RAM ', 'Storage ', 'Battery Capacity (mAh)']: # Add spaces if needed
        col_clean = col.strip()
        if col_clean in df.columns:
             if df[col_clean].isnull().sum() > 0:
                  median_val = df[col_clean].median()
                  df[col_clean].fillna(median_val, inplace=True)
                  print(f"Imputed missing values in '{col_clean}' with median ({median_val}).")

    # Remove duplicate rows
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    rows_after_dedup = len(df)
    print(f"Dropped {initial_rows - rows_after_dedup} duplicate rows.")

    # Reset index
    df = df.reset_index(drop=True)

    # Add 'Price Range' Column (Example bins, adjust as needed)
    # Ensure 'Price' is numeric
    df['Price ($)'] = pd.to_numeric(df['Price ($)'], errors='coerce')
    df.dropna(subset=['Price ($)'], inplace=True) # Drop rows where price couldn't be converted
    
    try:
        # Make sure bins cover the range of your data
        min_price, max_price = df['Price ($)'].min(), df['Price ($)'].max()
        bins = [0, 10000, 30000, 60000, max_price + 1] # Adjust these PKR thresholds
        labels = ['Budget (<10k)', 'Mid-Range (10k-30k)', 'Upper Mid-Range (30k-60k)', 'Premium (>60k)']
        if len(bins) -1 != len(labels):
             raise ValueError("Number of bins must be one more than the number of labels.")
             
        df['Price_Range'] = pd.cut(df['Price ($)'], bins=bins, labels=labels, right=False)
        print("✅ 'Price_Range' column added.")
    except Exception as e:
         st.error(f"Error creating price range: {e}. Min Price: {min_price}, Max Price: {max_price}")
         # Fallback or skip adding the column if error
         df['Price_Range'] = 'N/A'


    print(f"✅ Data cleaned successfully! Final shape: {df.shape}")
    return df

# --- Load and Clean Data ---
df_raw = load_data(DATA_FILE)
df = clean_data(df_raw.copy() if df_raw is not None else None) # Work on a copy

# ==============================================================================
# PHASE 2: Dashboard Layout (Streamlit UI)
# ==============================================================================

st.title("📱 PhoneDash: Find Your Perfect Mobile!")
st.markdown("Explore mobile phone specifications and prices. Use the filters in the sidebar to narrow down your search.")

if df is None:
    st.warning("Dataset could not be loaded or processed. Please check the data file and console output.")
    st.stop() # Stop execution if data isn't loaded

# --- Sidebar Filters ---
st.sidebar.header("⚙ Filter Options")

# 1. Brand Filter (Multiselect)
brand_list = sorted(df['Brand'].unique())
selected_brands = st.sidebar.multiselect("Choose Brand(s):", brand_list, default=brand_list) # Default to all selected

# Handle case where user deselects a555555555555555555555ll brands
if not selected_brands:
    selected_brands = brand_list # If nothing selected, assume all are selected

# 2. Price Range Filter (Slider)
min_price, max_price = int(df['Price ($)'].min()), int(df['Price ($)'].max())
price_range = st.sidebar.slider(
    "Price Range (PKR):",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price) # Default to full range
)

# 3. RAM Filter (Slider/Selectbox - using slider for flexibility)
# Clean RAM column if necessary (e.g., remove 'GB', convert to numeric)
# Assuming 'RAM ' column exists and might have 'GB' or needs cleaning
ram_col = 'RAM ' # Adjust if column name is different
if ram_col in df.columns:
    # Attempt to extract numeric RAM value
    df['RAM_GB'] = df[ram_col].astype(str).str.extract('(\d+)').astype(float) # Extract numbers
    min_ram, max_ram = int(df['RAM_GB'].min()), int(df['RAM_GB'].max())
    
    # Check if min and max RAM are the same (e.g., all phones have 8GB)
    if min_ram == max_ram:
        st.sidebar.markdown(f"*RAM:* All listed phones have {min_ram} GB RAM.")
        selected_ram = (min_ram, max_ram) # Set filter to the only available value
    else:
         selected_ram = st.sidebar.slider(
             "RAM (GB):",
             min_value=min_ram,
             max_value=max_ram,
             value=(min_ram, max_ram) # Default to full range
         )
else:
    st.sidebar.warning("RAM column not found or processed correctly.")
    selected_ram = (0, 128) # Default placeholder if RAM not available


# 4. Battery Filter (Slider - optional, based on your data)
battery_col = 'Battery ' # Adjust if column name is different
if battery_col in df.columns:
    # Clean Battery column if necessary (e.g., remove 'mAh', convert to numeric)
    df['Battery_mAh'] = df[battery_col].astype(str).str.extract('(\d+)').astype(float) # Extract numbers
    df['Battery_mAh'].fillna(df['Battery_mAh'].median(), inplace=True) # Handle potential NaNs after extraction
    
    min_battery, max_battery = int(df['Battery_mAh'].min()), int(df['Battery_mAh'].max())
    
    if min_battery == max_battery:
         st.sidebar.markdown(f"*Battery:* All listed phones have {min_battery} mAh.")
         selected_battery = (min_battery, max_battery)
    elif min_battery < max_battery:
         selected_battery = st.sidebar.slider(
             "Battery Capacity (mAh):",
             min_value=min_battery,
             max_value=max_battery,
             value=(min_battery, max_battery) # Default to full range
        )
    else: # Handle potential issue where min > max after cleaning
        st.sidebar.warning("Could not determine valid battery range.")
        selected_battery = (0, 10000) # Default placeholder
else:
    st.sidebar.warning("Battery column not found or processed correctly.")
    selected_battery = (0, 10000) # Default placeholder if Battery not available

# --- Apply Filters ---
filtered_df = df[
    (df['Brand'].isin(selected_brands)) &
    (df['Price ($)'] >= price_range[0]) &
    (df['Price ($)'] <= price_range[1])
]

# Apply RAM filter only if 'RAM_GB' column exists and is valid
if 'RAM_GB' in df.columns and df['RAM_GB'].isnull().sum() == 0:
     # Check if selected_ram is iterable (a tuple)
    if isinstance(selected_ram, (list, tuple)) and len(selected_ram) == 2:
         filtered_df = filtered_df[
             (filtered_df['RAM_GB'] >= selected_ram[0]) &
             (filtered_df['RAM_GB'] <= selected_ram[1])
         ]
    # Add handling if it's not a tuple (e.g., if min_ram==max_ram)
    elif isinstance(selected_ram, (int, float)):
         filtered_df = filtered_df[filtered_df['RAM_GB'] == selected_ram]


# Apply Battery filter only if 'Battery_mAh' column exists and is valid
if 'Battery_mAh' in df.columns and df['Battery_mAh'].isnull().sum() == 0:
    # Check if selected_battery is iterable (a tuple)
    if isinstance(selected_battery, (list, tuple)) and len(selected_battery) == 2:
        filtered_df = filtered_df[
            (filtered_df['Battery_mAh'] >= selected_battery[0]) &
            (filtered_df['Battery_mAh'] <= selected_battery[1])
        ]
     # Add handling if it's not a tuple (e.g., if min_battery==max_battery)
    elif isinstance(selected_battery, (int, float)):
         filtered_df = filtered_df[filtered_df['Battery_mAh'] == selected_battery]


# --- Display Filtered Phones ---
st.header(f"📱 Matching Phones ({len(filtered_df)})")

if filtered_df.empty:
    st.warning("No phones match your current filter criteria. Try adjusting the filters in the sidebar!")
else:
    # Sorting Options
    sort_options = ['Price (Low to High)', 'Price (High to Low)', 'Battery (High to Low)', 'RAM (High to Low)']
    # Only add battery/RAM sort if columns exist
    if 'Battery_mAh' not in filtered_df.columns:
        sort_options.remove('Battery (High to Low)')
    if 'RAM_GB' not in filtered_df.columns:
        sort_options.remove('RAM (High to Low)')
        
    sort_by = st.selectbox("Sort results by:", sort_options)

    # Apply Sorting
    if sort_by == 'Price (Low to High)':
        filtered_df = filtered_df.sort_values('Price ($)', ascending=True)
    elif sort_by == 'Price (High to Low)':
        filtered_df = filtered_df.sort_values('Price ($)', ascending=False)
    elif sort_by == 'Battery (High to Low)' and 'Battery_mAh' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('Battery_mAh', ascending=False)
    elif sort_by == 'RAM (High to Low)' and 'RAM_GB' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('RAM_GB', ascending=False)

    # Select columns to display (adjust based on your CSV)
    display_columns = ['Brand', 'Model', 'Price ($)', 'Price_Range', 'RAM ', 'Storage ', 'Battery ', 'Camera ', 'Display '] # Add/remove as needed
    # Filter display_columns to only those that actually exist in filtered_df
    display_columns = [col for col in display_columns if col in filtered_df.columns]
    
    st.dataframe(filtered_df[display_columns].reset_index(drop=True))

    # --- Export Option (Phase 5) ---
    csv = filtered_df[display_columns].to_csv(index=False).encode('utf-8')
    st.download_button(
       label="📥 Download Results as CSV",
       data=csv,
       file_name='filtered_phones.csv',
       mime='text/csv',
    )


# ==============================================================================
# PHASE 3: Machine Learning Integration (Simple Price Prediction)
# ==============================================================================

st.sidebar.header("🤖 Price Predictor")
st.sidebar.markdown("Estimate price based on specs (Experimental)")

# --- Model Training (Cached) ---
# Use st.cache_resource for things that shouldn't be serialized, like ML models
@st.cache_resource 
def train_model(df):
    """Trains a RandomForestRegressor model to predict price."""
    try:
        # Feature Selection (Choose relevant numeric/categorical features)
        # Ensure these columns exist and are cleaned
        features = ['RAM_GB', 'Storage_Num', 'Battery_mAh', 'Brand_Encoded'] # Example features
        target = 'Price'

        # Preprocessing for ML:
        df_ml = df.copy()
        
        # Convert Storage (e.g., '128 GB') to numeric
        if 'Storage ' in df_ml.columns:
            df_ml['Storage_Num'] = df_ml['Storage '].astype(str).str.extract('(\d+)').astype(float)
            df_ml['Storage_Num'].fillna(df_ml['Storage_Num'].median(), inplace=True) # Impute missing
        else:
             st.sidebar.warning("Storage column not found for ML.")
             return None, None # Cannot train without features
        
        # Convert Brand to numeric using Label Encoding (simple approach)
        # Note: OneHotEncoding is often preferred to avoid implied order, but LabelEncoding is simpler here.
        if 'Brand' in df_ml.columns:
            label_encoder = LabelEncoder()
            df_ml['Brand_Encoded'] = label_encoder.fit_transform(df_ml['Brand'])
            # Store encoder mapping for prediction later if needed (optional here)
        else:
             st.sidebar.warning("Brand column not found for ML.")
             return None, None
             
        # Ensure all selected features exist and are numeric
        final_features = []
        for f in features:
             if f in df_ml.columns and pd.api.types.is_numeric_dtype(df_ml[f]):
                 final_features.append(f)
             else:
                 print(f"Warning: Feature '{f}' is missing or not numeric. Skipping for ML.")
        
        if not final_features or target not in df_ml.columns:
             st.sidebar.error("Not enough valid features or target ('Price') column missing for ML.")
             return None, None

        # Drop rows with NaN in features or target after preprocessing
        df_ml.dropna(subset=final_features + [target], inplace=True)

        if df_ml.empty:
             st.sidebar.error("No valid data remaining after cleaning for ML.")
             return None, None

        X = df_ml[final_features]
        y = df_ml[target]

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Training (Random Forest Regressor)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # Use more estimators for potentially better results
        model.fit(X_train, y_train)

        # Evaluation
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Model Trained. RMSE: {rmse:.2f}, R2 Score: {r2:.2f}")
        
        # Store feature names with the model
        model.feature_names_ = final_features 
        
        # Optional: Save model
        # joblib.dump(model, MODEL_FILE)
        # print(f"Model saved to {MODEL_FILE}")

        return model, label_encoder # Return model and encoder

    except Exception as e:
        st.sidebar.error(f"Error during model training: {e}")
        print(f"Error during model training: {e}") # Print to console for debugging
        return None, None

# --- Check for existing model or train ---
# This part is commented out to avoid needing a pre-saved file.
# It trains every time the script runs (or uses Streamlit's cache).
# if os.path.exists(MODEL_FILE):
#     print(f"Loading existing model from {MODEL_FILE}")
#     model = joblib.load(MODEL_FILE)
#     # Need to load label_encoder separately if saved, or retrain it
#     label_encoder_ml = LabelEncoder().fit(df['Brand']) # Re-fit encoder on full data
# else:
#     print("Training new model...")
#     model, label_encoder_ml = train_model(df)

# Simple approach: Train model when script runs (using cache)
model, label_encoder_ml = train_model(df)

# --- User Input for Prediction ---
if model and label_encoder_ml and hasattr(model, 'feature_names_'):
    st.sidebar.subheader("Enter Specs to Predict Price:")

    # Create input fields dynamically based on model features
    input_data = {}
    feature_display_names = { # Nicer names for UI
         'RAM_GB': 'RAM (GB)',
         'Storage_Num': 'Storage (GB)',
         'Battery_mAh': 'Battery (mAh)',
         'Brand_Encoded': 'Brand'
     }

    for feature in model.feature_names_:
        display_name = feature_display_names.get(feature, feature) # Use nicer name if available
        
        if feature == 'Brand_Encoded':
            # Use the trained label encoder's classes for the dropdown
            brand_options = list(label_encoder_ml.classes_)
            selected_brand_pred = st.sidebar.selectbox(f"{display_name}:", brand_options, key='pred_brand')
            # Encode the selected brand
            try:
                input_data[feature] = label_encoder_ml.transform([selected_brand_pred])[0]
            except ValueError:
                st.sidebar.warning(f"Brand '{selected_brand_pred}' not seen during training. Prediction might be less accurate.")
                input_data[feature] = -1 # Handle unknown category if needed by model
        elif feature == 'RAM_GB':
             # Get min/max from the original df column if possible for reasonable slider limits
             min_val, max_val = (int(df['RAM_GB'].min()), int(df['RAM_GB'].max())) if 'RAM_GB' in df else (1, 16)
             input_data[feature] = st.sidebar.slider(f"{display_name}:", min_val, max_val, int(df['RAM_GB'].median()) if 'RAM_GB' in df else 4, key=f'pred_{feature}')
        elif feature == 'Storage_Num':
             min_val, max_val = (int(df['Storage_Num'].min()), int(df['Storage_Num'].max())) if 'Storage_Num' in df else (32, 512)
             # Use selectbox for common storage sizes
             storage_options = [16, 32, 64, 128, 256, 512, 1024]
             # Filter options based on data range
             valid_storage_options = [s for s in storage_options if s >= min_val and s <= max_val]
             if not valid_storage_options: valid_storage_options = [int(df['Storage_Num'].median())] # Fallback
             input_data[feature] = st.sidebar.select_slider(f"{display_name}:", options=valid_storage_options, value=valid_storage_options[len(valid_storage_options)//2], key=f'pred_{feature}')
             # Alternatively, use number input:
             # input_data[feature] = st.sidebar.number_input(f"{display_name}:", min_value=min_val, max_value=max_val, value=int(df['Storage_Num'].median()) if 'Storage_Num' in df else 128, key=f'pred_{feature}')
        elif feature == 'Battery_mAh':
             min_val, max_val = (int(df['Battery_mAh'].min()), int(df['Battery_mAh'].max())) if 'Battery_mAh' in df else (2000, 7000)
             input_data[feature] = st.sidebar.slider(f"{display_name}:", min_val, max_val, int(df['Battery_mAh'].median()) if 'Battery_mAh' in df else 4000, step=100, key=f'pred_{feature}')
        else: # Generic number input for other features
             min_val, max_val = (int(df[feature].min()), int(df[feature].max())) if feature in df else (0, 1000)
             input_data[feature] = st.sidebar.number_input(f"{display_name}:", min_value=min_val, max_value=max_val, value=int(df[feature].median()) if feature in df else 0, key=f'pred_{feature}')


    if st.sidebar.button("Predict Price"):
        # Create DataFrame in the correct order
        predict_df = pd.DataFrame([input_data])
        predict_df = predict_df[model.feature_names_] # Ensure column order matches training

        try:
            predicted_price = model.predict(predict_df)[0]
            st.sidebar.success(f"Predicted Price: PKR {predicted_price:,.0f}")
            st.sidebar.markdown("(Note: This is an estimate based on the available data.)")

            # Bonus: Find phones with similar specs in the filtered data
            st.sidebar.markdown("---")
            st.sidebar.markdown("*Find similar phones in current results:*")
             # Simple similarity: within +/- 10% of predicted price and same RAM/Brand
            similar_phones = filtered_df[
                (filtered_df['Price ($)'] >= predicted_price * 0.9) &
                (filtered_df['Price ($)'] <= predicted_price * 1.1) &
                (filtered_df['Brand'] == selected_brand_pred) & # Match selected brand for prediction
                (filtered_df['RAM_GB'] == input_data['RAM_GB']) # Match selected RAM for prediction
            ]
            if not similar_phones.empty:
                st.sidebar.dataframe(similar_phones[['Brand', 'Model', 'Price', 'RAM ', 'Storage ']].head(3))
            else:
                st.sidebar.info("No phones found in the current results with very similar price/specs to the prediction.")


        except Exception as e:
            st.sidebar.error(f"Error during prediction: {e}")
            print(f"Error during prediction with input {input_data}: {e}")

elif not model:
    st.sidebar.warning("Price prediction model could not be trained. Check data and feature availability.")

# ==============================================================================
# PHASE 4: Visualization & UX
# ==============================================================================

st.header("📊 Visual Insights")
st.markdown("Explore trends in the filtered data.")

# Use columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Distribution")
    if not filtered_df.empty:
        fig_price = px.histogram(filtered_df, x="Price ($)", nbins=20, title="Distribution of Phone Prices",
                                 labels={'Price':'Price (PKR)'}, color='Price_Range',
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_price.update_layout(bargap=0.1)
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.info("No data to display for Price Distribution based on filters.")

with col2:
    st.subheader("Phones per Brand")
    if not filtered_df.empty:
        brand_count = filtered_df['Brand'].value_counts().reset_index()
        brand_count.columns = ['Brand', 'Count']
        fig_brand = px.bar(brand_count.head(15), x='Brand', y='Count', title="Number of Phones per Brand (Top 15)",
                            color='Brand', color_discrete_sequence=px.colors.qualitative.Vivid)
        fig_brand.update_layout(xaxis_title=None, yaxis_title="Number of Phones")
        st.plotly_chart(fig_brand, use_container_width=True)
    else:
        st.info("No data to display for Brand Count based on filters.")

st.subheader("Feature Relationships")
# Correlation Heatmap (on numerical features of the original cleaned df)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# Select a subset of potentially interesting numeric columns for heatmap
heatmap_cols = ['Price', 'RAM_GB', 'Storage_Num', 'Battery_mAh'] # Add others like screen size if available
heatmap_cols = [col for col in heatmap_cols if col in numeric_cols] # Keep only existing cols

if len(heatmap_cols) > 1:
    try:
        # Calculate correlation on the main cleaned dataframe df before filtering
        # because filtering might remove too much data for meaningful correlation
        corr = df[heatmap_cols].corr()
        
        fig_corr, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
        ax.set_title("Correlation Matrix of Key Specs (Overall Data)")
        st.pyplot(fig_corr)
    except Exception as e:
        st.warning(f"Could not generate correlation heatmap: {e}")
else:
    st.info("Not enough numerical data found to generate a correlation heatmap.")
    
# Feature Importance (if model was trained successfully)
if model and hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_'):
     st.subheader("Feature Importance for Price Prediction")
     try:
         importance_df = pd.DataFrame({
             'Feature': model.feature_names_,
             'Importance': model.feature_importances_
         }).sort_values(by='Importance', ascending=False)
         
         # Map encoded brand back for display if possible
         if 'Brand_Encoded' in importance_df['Feature'].values:
              importance_df['Feature'] = importance_df['Feature'].replace({'Brand_Encoded': 'Brand (Encoded)'})


         fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                           title="Which Specs Most Influence Predicted Price?")
         fig_imp.update_layout(yaxis_title=None)
         st.plotly_chart(fig_imp, use_container_width=True)
     except Exception as e:
         st.warning(f"Could not display feature importance: {e}")


# --- Data Preview Expander ---
with st.expander("Show Raw Data Overview"):
    st.subheader("Data Preview (First 10 Rows)")
    st.dataframe(df.head(10))
    st.subheader("Data Info")
    # Capture df.info() output to display in Streamlit
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.subheader("Summary Statistics (Numerical Columns)")
    st.dataframe(df[numeric_cols].describe()) # Show describe only for numeric

# ==============================================================================
# PHASE 5: Bonus & Creativity (Placeholder Ideas)
# ==============================================================================
# --- Ideas for further development ---
# - Comparison Tool: Allow selecting 2 rows from the table and show specs side-by-side.
# - Recommendation Engine: Based on budget/priority (e.g., "Best camera under 30k"), suggest top picks.
# - More Visualizations: Scatter plots (e.g., Price vs RAM colored by Brand), Box plots for distributions.
# - User Upload: Allow users to upload their own CSV file (st.file_uploader).
# - More Sophisticated ML: Use OneHotEncoder, try different models (XGBoost, LightGBM), hyperparameter tuning.
# - UI Enhancements: Use st.metric for key stats, add icons, potentially use custom components or HTML/CSS.
# [PREVIOUS CODE REMAINS THE SAME UNTIL THE END OF PHASE 4 VISUALIZATIONS]
# ... (Keep all the code from the previous response up to the end of the visualization section) ...

# ==============================================================================
# PHASE 4: Visualization & UX (Continued)
# ==============================================================================

# --- Data Preview Expander ---
with st.expander("Show Raw Data Overview"):
    st.subheader("Data Preview (First 10 Rows)")
    st.dataframe(df.head(10))
    st.subheader("Data Info")
    # Capture df.info() output to display in Streamlit
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.subheader("Summary Statistics (Numerical Columns)")
    # Ensure numeric_cols is defined if not already
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.dataframe(df[numeric_cols].describe()) # Show describe only for numeric


# ==============================================================================
# PHASE 5: Bonus & Creativity
# ==============================================================================

st.header("✨ Bonus Features")

# --- 1. Phone Comparison Tool ---
st.subheader("⚖ Compare Phones")
st.markdown("Select exactly two phones from the 'Matching Phones' table above using the 'Select' checkbox, then click the button below.")

# Modify the filtered data display to use st.data_editor for selection
if not filtered_df.empty:
    # Add a 'Select' column for the editor
    filtered_df_copy = filtered_df.copy() # Work on a copy
    if 'Select' not in filtered_df_copy.columns:
         # Insert 'Select' column at the beginning
         filtered_df_copy.insert(0, "Select", False)

    # Define column configuration for the editor (optional: customize widths, etc.)
    column_config = {
        "Select": st.column_config.CheckboxColumn("Select", default=False),
        "Price": st.column_config.NumberColumn("Price (PKR)", format="₨ %d"),
        # Add more config if needed for other columns
    }
    
    # Filter display_columns to only those that actually exist in filtered_df_copy
    # Ensure 'Select' is the first column if added
    display_columns_compare = ['Select'] + [col for col in display_columns if col in filtered_df_copy.columns and col != 'Select']
    
    st.markdown("---") # Separator before the table
    st.write("*Select Phones to Compare from this Table:*") # Add instruction context
    
    edited_df = st.data_editor(
        filtered_df_copy[display_columns_compare], # Only show relevant columns + Select
        key="phone_selector",
        column_config=column_config,
        use_container_width=True,
        hide_index=True, # Hide the default index
        num_rows="dynamic" # Allow dynamic rows (though we are not editing data here)
    )

    selected_rows = edited_df[edited_df.Select]
    num_selected = len(selected_rows)

    if st.button(f"Compare {num_selected} Selected Phone(s)"):
        if num_selected == 2:
            st.success("Comparing the two selected phones:")
            # Prepare data for comparison view (transpose)
            # Get the original data for selected rows using index if possible, or merge back if needed
            # Using the selected rows directly from the potentially modified edited_df:
            compare_data = selected_rows.drop(columns=['Select']).T # Transpose
            compare_data.columns = [f"Phone {i+1}" for i in range(num_selected)] # Rename columns

            # Improve display - get Model name for columns if 'Model' column exists
            if 'Model' in compare_data.index:
                 model_names = compare_data.loc['Model'].tolist()
                 compare_data.columns = model_names # Use model names as columns
                 # Optionally drop the Model row from the comparison table itself
                 # compare_data = compare_data.drop('Model')

            st.dataframe(compare_data, use_container_width=True)
        elif num_selected == 0:
            st.warning("Please select two phones from the table above to compare.")
        elif num_selected == 1:
             st.warning("Please select one more phone to compare (exactly two needed).")
        else:
            st.warning(f"Please select exactly two phones to compare (you selected {num_selected}).")
else:
     st.info("No phones match the current filters to select for comparison.")


# --- 2. Simple Recommendation Engine ("Top Picks") ---
st.subheader("🏆 Top Picks Recommendation")
st.markdown("Get recommendations based on your budget and priorities within the currently filtered results.")

if not filtered_df.empty:
    rec_col1, rec_col2 = st.columns([1, 2]) # Columns for input and results

    with rec_col1:
        st.markdown("*Your Priorities:*")
        max_budget = st.number_input("Maximum Budget (PKR):", min_value=price_range[0], max_value=price_range[1], value=price_range[1])
        
        # Define priorities - Add 'Camera' if camera data is usable
        priority_options = ['Balanced', 'Best Battery', 'Best RAM']
        # Attempt to process Camera data for recommendation
        camera_col = 'Camera ' # Adjust if needed
        if camera_col in df.columns:
            # Simple extraction of the first number (MP) as a proxy
            df['Camera_MP'] = df[camera_col].astype(str).str.extract('(\d+)').astype(float)
            # Check if extraction worked for a reasonable number of rows
            if df['Camera_MP'].isnull().sum() < len(df) * 0.5: # If less than 50% are NaN
                priority_options.append('Best Camera (MP)')
                
                # Impute missing MP values with median for scoring (optional)
                if 'Camera_MP' in df.columns: 
                     median_cam_mp = df['Camera_MP'].median()
                     df['Camera_MP'].fillna(median_cam_mp, inplace=True)
                     # Also apply to filtered_df if not already there
                     if 'Camera_MP' not in filtered_df.columns:
                         filtered_df['Camera_MP'] = filtered_df[camera_col].astype(str).str.extract('(\d+)').astype(float)
                         filtered_df['Camera_MP'].fillna(median_cam_mp, inplace=True)

        selected_priority = st.selectbox("Prioritize:", priority_options)

    # Filter by budget first
    recommendation_pool = filtered_df[filtered_df['Price ($)'] <= max_budget].copy()

    if recommendation_pool.empty:
        with rec_col2:
            st.warning("No phones found within your specified budget in the current filtered results.")
    else:
        # --- Scoring Logic ---
        # Normalize relevant columns (0 to 1 scale, higher is better)
        def normalize(series):
             # Check if max == min to avoid division by zero
             if series.max() == series.min():
                 return pd.Series([0.5] * len(series), index=series.index) # Return mid-value if all are same
             return (series - series.min()) / (series.max() - series.min())

        if 'RAM_GB' in recommendation_pool.columns:
            recommendation_pool['RAM_Score'] = normalize(recommendation_pool['RAM_GB'])
        else: recommendation_pool['RAM_Score'] = 0

        if 'Battery_mAh' in recommendation_pool.columns:
            recommendation_pool['Battery_Score'] = normalize(recommendation_pool['Battery_mAh'])
        else: recommendation_pool['Battery_Score'] = 0
        
        # Normalize Price (lower is better, so invert score: 1 - normalized)
        if 'Price' in recommendation_pool.columns:
             recommendation_pool['Price_Score'] = 1 - normalize(recommendation_pool['Price ($)'])
        else: recommendation_pool['Price_Score'] = 0

        if 'Camera_MP' in recommendation_pool.columns and recommendation_pool['Camera_MP'].isnull().sum() == 0:
             recommendation_pool['Camera_Score'] = normalize(recommendation_pool['Camera_MP'])
        else: recommendation_pool['Camera_Score'] = 0

        # Calculate final score based on priority
        if selected_priority == 'Balanced':
            # Example: Equal weight to Price, RAM, Battery (adjust weights as needed)
            recommendation_pool['Final_Score'] = (recommendation_pool['Price_Score'] * 0.4 +
                                                  recommendation_pool['RAM_Score'] * 0.3 +
                                                  recommendation_pool['Battery_Score'] * 0.3)
        elif selected_priority == 'Best Battery':
            # Example: Higher weight to Battery, moderate to Price
            recommendation_pool['Final_Score'] = (recommendation_pool['Battery_Score'] * 0.6 +
                                                  recommendation_pool['Price_Score'] * 0.3 +
                                                  recommendation_pool['RAM_Score'] * 0.1) # Small RAM weight
        elif selected_priority == 'Best RAM':
            # Example: Higher weight to RAM, moderate to Price
             recommendation_pool['Final_Score'] = (recommendation_pool['RAM_Score'] * 0.6 +
                                                   recommendation_pool['Price_Score'] * 0.3 +
                                                   recommendation_pool['Battery_Score'] * 0.1) # Small Battery weight
        elif selected_priority == 'Best Camera (MP)' and 'Camera_Score' in recommendation_pool:
            # Example: Higher weight to Camera, moderate to Price
             recommendation_pool['Final_Score'] = (recommendation_pool['Camera_Score'] * 0.5 +
                                                   recommendation_pool['Price_Score'] * 0.3 +
                                                   recommendation_pool['RAM_Score'] * 0.1 +
                                                   recommendation_pool['Battery_Score'] * 0.1)

        # Sort by score and get top N
        top_picks = recommendation_pool.sort_values('Final_Score', ascending=False).head(5) # Get Top 5

        with rec_col2:
            st.markdown(f"*Top {len(top_picks)} Picks for '{selected_priority}' (Budget ≤ PKR {max_budget:,.0f}):*")
            
            # Select columns to display for recommendations
            rec_display_cols = ['Brand', 'Model', 'Price', 'RAM ', 'Battery ', 'Storage ']
            if 'Camera ' in df.columns: rec_display_cols.append('Camera ') # Add camera if exists
            rec_display_cols = [col for col in rec_display_cols if col in top_picks.columns] # Ensure columns exist

            st.dataframe(
                top_picks[rec_display_cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True
                )

else:
     st.info("Apply filters in the sidebar to see phones and get recommendations.")


# --- Final Touches & Footer ---
st.markdown("---")
st.markdown("Created with ❤ using Streamlit and Pandas.")
# End of script
