import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

# Try to import optional packages
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    st.error("❌ joblib not installed. Please run: pip install joblib")

try:
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Amazon Delivery Time Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF9900;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF9900;
        margin: 20px 0;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Simple distance calculation function (backup for geopy)
def simple_distance(lat1, lon1, lat2, lon2):
    """Calculate approximate distance using haversine formula"""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

# Load models and encoders with error handling
@st.cache_resource
def load_models():
    if not JOBLIB_AVAILABLE:
        return None, None, None
    
    try:
        best_model = joblib.load('best_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        return best_model, scaler, label_encoders
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Load data with error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('amazon_delivery_cleaned.csv')
        return df
    except FileNotFoundError:
        st.error("❌ amazon_delivery_cleaned.csv not found! Please run the data cleaning notebook first.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Demo prediction function (when models are not available)
def demo_prediction(agent_age, agent_rating, distance, order_hour, weather, traffic, vehicle, area, category):
    """Simple rule-based prediction for demo purposes"""
    
    base_time = 2.0  # Base delivery time in hours
    
    # Distance factor
    distance_factor = distance * 0.1
    
    # Traffic factor
    traffic_factors = {'Low': -0.5, 'Medium': 0, 'High': 1.0, 'Normal': 0}
    traffic_factor = traffic_factors.get(traffic, 0)
    
    # Weather factor
    weather_factors = {'Clear': -0.2, 'Sunny': -0.2, 'Cloudy': 0, 'Rainy': 0.8, 'Stormy': 1.5}
    weather_factor = weather_factors.get(weather, 0)
    
    # Vehicle factor
    vehicle_factors = {'Bike': -0.3, 'Car': 0, 'Van': 0.2, 'Truck': 0.5}
    vehicle_factor = vehicle_factors.get(vehicle, 0)
    
    # Agent rating factor (higher rating = faster delivery)
    rating_factor = (5 - agent_rating) * 0.2
    
    # Time of day factor
    if order_hour in [12, 13, 14]:  # Lunch rush
        time_factor = 0.5
    elif order_hour in [18, 19, 20]:  # Dinner rush
        time_factor = 0.7
    elif order_hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # Night time
        time_factor = 0.3
    else:
        time_factor = 0
    
    # Calculate final prediction
    prediction = base_time + distance_factor + traffic_factor + weather_factor + vehicle_factor + rating_factor + time_factor
    
    return max(0.5, prediction)  # Minimum 30 minutes

# Main app function
def main():
    st.markdown('<h1 class="main-header">📦 Amazon Delivery Time Predictor</h1>', unsafe_allow_html=True)
    
    # Load models and data
    model, scaler, label_encoders = load_models()
    df = load_data()
    
    # Show warning if models are not available
    if model is None:
        st.warning("⚠️ ML models not found. Using demo prediction mode.")
        st.info("To use ML predictions, please run the model training notebook first to generate: best_model.pkl, feature_scaler.pkl, label_encoders.pkl")
    
    # Sidebar for inputs
    st.sidebar.header("📋 Order Details")
    
    # Input fields
    agent_age = st.sidebar.slider("Agent Age", min_value=18, max_value=65, value=30)
    agent_rating = st.sidebar.slider("Agent Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
    
    # Distance input
    distance = st.sidebar.number_input("Distance (km)", min_value=0.1, max_value=100.0, value=5.0, step=0.1)
    
    # Time input
    order_hour = st.sidebar.selectbox("Order Hour", options=list(range(24)), index=12)
    
    # Categorical inputs with default values
    weather_options = ['Clear', 'Cloudy', 'Rainy', 'Stormy', 'Sunny']
    traffic_options = ['High', 'Low', 'Medium', 'Normal']
    vehicle_options = ['Bike', 'Car', 'Truck', 'Van']
    area_options = ['Metropolitan', 'Urban', 'Rural']
    category_options = ['Electronics', 'Fashion', 'Food', 'Grocery', 'Home', 'Books']
    
    weather = st.sidebar.selectbox("Weather Condition", weather_options)
    traffic = st.sidebar.selectbox("Traffic Condition", traffic_options)
    vehicle = st.sidebar.selectbox("Vehicle Type", vehicle_options)
    area = st.sidebar.selectbox("Area Type", area_options)
    category = st.sidebar.selectbox("Product Category", category_options)
    
    # Prediction button
    if st.sidebar.button("🚀 Predict Delivery Time", type="primary"):
        
        try:
            if model is not None and label_encoders is not None:
                # ML-based prediction
                # Encode categorical variables safely
                def safe_encode(encoder, value, default=0):
                    try:
                        if value in encoder.classes_:
                            return encoder.transform([value])[0]
                        else:
                            return default
                    except:
                        return default
                
                weather_encoded = safe_encode(label_encoders.get('Weather'), weather, 0)
                traffic_encoded = safe_encode(label_encoders.get('Traffic'), traffic, 0)
                vehicle_encoded = safe_encode(label_encoders.get('Vehicle'), vehicle, 0)
                area_encoded = safe_encode(label_encoders.get('Area'), area, 0)
                category_encoded = safe_encode(label_encoders.get('Category'), category, 0)
                
                # Create input array
                input_features = np.array([[
                    agent_age, agent_rating, distance, order_hour,
                    weather_encoded, traffic_encoded, vehicle_encoded,
                    area_encoded, category_encoded
                ]])
                
                # Scale features if using Linear Regression
                model_name = type(model).__name__
                if model_name == 'LinearRegression' and scaler is not None:
                    input_features = scaler.transform(input_features)
                
                # Make prediction
                prediction = model.predict(input_features)[0]
                prediction_type = "🤖 ML Prediction"
                
            else:
                # Demo prediction
                prediction = demo_prediction(agent_age, agent_rating, distance, order_hour, 
                                           weather, traffic, vehicle, area, category)
                prediction_type = "📊 Demo Prediction"
            
            # Display prediction
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown(f"""
                <div class="prediction-box">
                    <h3 style="color: #666; text-align: center;">{prediction_type}</h3>
                    <h2 style="color: #FF9900; text-align: center;">🕒 Predicted Delivery Time</h2>
                    <h1 style="text-align: center; color: #232F3E;">{prediction:.2f} hours</h1>
                    <p style="text-align: center; color: #666;">
                        Approximately {int(prediction * 60)} minutes
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display input summary
            st.subheader("📊 Order Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Distance", f"{distance} km")
                st.metric("Agent Age", f"{agent_age} years")
                st.metric("Agent Rating", f"{agent_rating}/5.0")
            
            with col2:
                st.metric("Order Hour", f"{order_hour}:00")
                st.metric("Weather", weather)
                st.metric("Traffic", traffic)
            
            with col3:
                st.metric("Vehicle", vehicle)
                st.metric("Area Type", area)
                st.metric("Category", category)
            
            # Show confidence interval
            if df is not None:
                std_dev = df['Delivery_Time'].std() if 'Delivery_Time' in df.columns else 1.0
            else:
                std_dev = 1.0
                
            confidence_lower = max(0, prediction - std_dev * 0.5)
            confidence_upper = prediction + std_dev * 0.5
            
            st.info(f"📈 **Estimated Range:** {confidence_lower:.2f} - {confidence_upper:.2f} hours")
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")
            st.error("Please check that all required files are present and try again.")
    
    # Main content area
    st.markdown("---")
    
    # Tabs for additional information
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Insights", "📈 Model Performance", "🗺️ Distance Calculator", "ℹ️ About"])
    
    with tab1:
        st.subheader("📊 Dataset Insights")
        
        if df is not None and not df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_delivery = df['Delivery_Time'].mean()
                st.metric("Average Delivery Time", f"{avg_delivery:.2f} hrs")
            
            with col2:
                median_delivery = df['Delivery_Time'].median()
                st.metric("Median Delivery Time", f"{median_delivery:.2f} hrs")
            
            with col3:
                total_orders = len(df)
                st.metric("Total Orders", f"{total_orders:,}")
            
            with col4:
                avg_distance = df['Distance_km'].mean() if 'Distance_km' in df.columns else 0
                st.metric("Average Distance", f"{avg_distance:.2f} km")
            
            # Safe visualizations
            try:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Delivery time distribution
                    fig = px.histogram(df, x='Delivery_Time', nbins=30, 
                                     title="Delivery Time Distribution",
                                     color_discrete_sequence=['#FF9900'])
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Category-wise delivery time
                    if 'Category' in df.columns:
                        category_avg = df.groupby('Category')['Delivery_Time'].mean().reset_index()
                        fig = px.bar(category_avg, x='Category', y='Delivery_Time',
                                   title="Average Delivery Time by Category",
                                   color_discrete_sequence=['#232F3E'])
                        fig.update_layout(xaxis_tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating visualizations: {e}")
        else:
            st.info("📊 Load the dataset to see insights here.")
    
    with tab2:
        st.subheader("📈 Model Performance")
        
        try:
            results_df = pd.read_csv('model_comparison_results.csv')
            st.dataframe(results_df.round(3), use_container_width=True)
            
            # Safe model comparison visualization
            fig = px.bar(results_df, x='Model', y='Test_RMSE',
                        title="Model Comparison - Test RMSE",
                        color_discrete_sequence=['#FF9900'])
            st.plotly_chart(fig, use_container_width=True)
            
        except FileNotFoundError:
            st.info("📈 Model comparison results will appear here after running the model training notebook.")
        except Exception as e:
            st.error(f"Error loading model results: {e}")
    
    with tab3:
        st.subheader("🗺️ Distance Calculator")
        st.write("Calculate distance between store and delivery location")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Store Location**")
            store_lat = st.number_input("Store Latitude", value=12.9716, format="%.4f")
            store_lon = st.number_input("Store Longitude", value=77.5946, format="%.4f")
        
        with col2:
            st.write("**Delivery Location**")
            drop_lat = st.number_input("Delivery Latitude", value=12.9716, format="%.4f")
            drop_lon = st.number_input("Delivery Longitude", value=77.5946, format="%.4f")
        
        if st.button("Calculate Distance"):
            try:
                if GEOPY_AVAILABLE:
                    distance_calc = geodesic((store_lat, store_lon), (drop_lat, drop_lon)).kilometers
                    method = "using Geopy (accurate)"
                else:
                    distance_calc = simple_distance(store_lat, store_lon, drop_lat, drop_lon)
                    method = "using Haversine formula (approximate)"
                
                st.success(f"📏 Distance: {distance_calc:.2f} km {method}")
            except Exception as e:
                st.error(f"Error calculating distance: {e}")
    
    with tab4:
        st.subheader("ℹ️ About This Application")
        
        st.markdown("""
        ### 🚀 Amazon Delivery Time Prediction System
        
        This application predicts delivery times for Amazon orders based on various factors:
        
        **📊 Features Used:**
        - Agent characteristics (age, rating)
        - Distance between store and delivery location
        - Time factors (order hour)
        - Environmental conditions (weather, traffic)
        - Logistics factors (vehicle type, area type)
        - Product category
        
        **🤖 Models Available:**
        - Linear Regression
        - Random Forest Regressor
        - Gradient Boosting Regressor
        
        **📈 Technologies Used:**
        - **Python**: Data processing and modeling
        - **Scikit-learn**: Machine learning algorithms
        - **MLflow**: Model tracking and versioning
        - **Streamlit**: Web application framework
        - **Plotly**: Interactive visualizations
        
        **🎯 Business Benefits:**
        - Improved customer satisfaction through accurate delivery estimates
        - Better resource allocation and logistics planning
        - Agent performance evaluation and optimization
        - Dynamic adjustments based on real-time conditions
        
        **🔧 Current Mode:**
        """)
        
        if model is not None:
            st.success("✅ **ML Mode Active** - Using trained machine learning models for predictions")
        else:
            st.info("📊 **Demo Mode Active** - Using rule-based predictions. Run the model training notebook to enable ML mode.")
        
        st.markdown("""
        **📝 Note:** This is a demonstration application. For production use, ensure regular model retraining with fresh data.
        
        **🗂️ Required Files for Full ML Mode:**
        - `best_model.pkl` - Trained ML model
        - `feature_scaler.pkl` - Feature scaling parameters
        - `label_encoders.pkl` - Categorical encoding parameters  
        - `amazon_delivery_cleaned.csv` - Cleaned dataset
        """)

if __name__ == "__main__":
    main()