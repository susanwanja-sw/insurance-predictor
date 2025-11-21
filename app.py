import streamlit as st

st.title('🏥 Medical Insurance Cost Predictor')

try:
    import pickle
    import numpy as np
    
    # Load model
    model = pickle.load(open('trainedmodel.sav', 'rb'))
    st.success("Model loaded successfully!")
    
    # Simple form
    age = st.slider('Age', 18, 100, 30)
    bmi = st.number_input('BMI', 10.0, 50.0, 25.0)
    
    if st.button('Predict'):
        # Test prediction
        test_input = np.array([[age, 0, bmi, 0, 1, 0]])
        prediction = model.predict(test_input)
        st.success(f'Estimated Charges: ${prediction[0]:,.2f}')
        
except Exception as e:
    st.error(f"Error: {e}")
    st.write("Full error details:")
    import traceback
    st.code(traceback.format_exc())