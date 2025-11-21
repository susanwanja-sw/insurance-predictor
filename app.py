import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="Insurance Predictor", page_icon="🏥", layout="centered")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return pickle.load(open('trainedmodel.sav', 'rb'))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# App title and description
st.title('🏥 Medical Insurance Cost Predictor')
st.markdown("""
Enter your details below to get an estimated insurance cost prediction.
""")

if model is not None:
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider('Age', 18, 100, 30, help="Your current age")
        sex = st.selectbox('Sex', ['Male', 'Female'])
        bmi = st.number_input('BMI (Body Mass Index)', min_value=10.0, max_value=50.0, value=25.0, step=0.1, help="Weight (kg) / Height (m)²")
    
    with col2:
        children = st.selectbox('Number of Children', [0, 1, 2, 3, 4, 5], help="Number of dependents")
        smoker = st.selectbox('Smoker', ['No', 'Yes'], help="Do you smoke?")
        region = st.selectbox('Region', ['Southwest', 'Southeast', 'Northwest', 'Northeast'], help="Your residential region")
    
    # Encoding based on training data
    # sex: male=0, female=1
    sex_encoded = 1 if sex == 'Female' else 0
    
    # smoker: yes=0, no=1
    smoker_encoded = 0 if smoker == 'Yes' else 1
    
    # region: southwest=0, northwest=1, southeast=2, northeast=3
    region_mapping = {
        'Southwest': 0,
        'Northwest': 1,
        'Southeast': 2,
        'Northeast': 3
    }
    region_encoded = region_mapping[region]
    
    # Predict button
    st.markdown("---")
    if st.button('🔮 Predict Insurance Charges', type='primary', use_container_width=True):
        try:
            # Prepare input data: age, sex, bmi, children, smoker, region
            input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
            
            # Make prediction
            prediction = model.predict(input_data)
            
            # Display result with styling
            st.markdown("### 💰 Prediction Result")
            st.success(f"## Estimated Annual Insurance Charges: ${prediction[0]:,.2f}")
            
            # Additional information in an expander
            with st.expander("📊 View Input Summary"):
                st.write(f"""
                - **Age:** {age} years
                - **Sex:** {sex}
                - **BMI:** {bmi}
                - **Children:** {children}
                - **Smoker:** {smoker}
                - **Region:** {region}
                """)
            
            # Add some context
            st.info("""
            💡 **Note:** This is an estimate based on historical data. 
            Actual insurance charges may vary based on additional factors and insurance provider policies.
            """)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            import traceback
            with st.expander("View error details"):
                st.code(traceback.format_exc())
    
    # Footer
    st.markdown("---")
    st.caption('Model R² Score: 0.74 | Built with Streamlit & scikit-learn')

else:
    st.error("Failed to load the model. Please check the deployment logs.")