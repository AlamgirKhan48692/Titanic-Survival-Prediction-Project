import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("Titanic Survival Prediction App")

uploaded_file = st.file_uploader("Upload Titanic Dataset CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    data = data[['Pclass','Sex','Age','Fare','Survived']]

    data['Sex'] = data['Sex'].map({'male':0,'female':1})

    data = data.dropna()

    X = data[['Pclass','Sex','Age','Fare']]
    y = data['Survived']

    model = LogisticRegression()
    model.fit(X,y)

    st.subheader("Passenger Information")

    pclass = st.selectbox("Passenger Class", [1,2,3])
    sex = st.selectbox("Sex", ["male","female"])
    age = st.slider("Age",1,80,25)
    fare = st.slider("Fare",1,500,50)

    sex_val = 0 if sex=="male" else 1

    input_data = pd.DataFrame({
        'Pclass':[pclass],
        'Sex':[sex_val],
        'Age':[age],
        'Fare':[fare]
    })

    prediction = model.predict(input_data)

    if st.button("Predict Survival"):
        if prediction[0]==1:
            st.success("Passenger likely SURVIVED")
        else:
            st.error("Passenger likely DID NOT survive")