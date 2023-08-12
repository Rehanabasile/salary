import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
import io

# Read the CSV data
data = pd.read_csv("Salary_Data.csv")
x = np.array(data['Experience']).reshape(-1, 1)
lr = LinearRegression()
lr.fit(x, np.array(data['Salary']))

st.title("Salary prediction")
nav = st.sidebar.radio("Navigation", ["Home", "Prediction", "Contribute"])

if nav == "Home":
    st.image("your-new-salary-iStockphoto.jpg", width=700)

if st.checkbox("Show Table"):
    st.table(data)

graph = st.selectbox("What kind of Graph?", ["Non-interactive", "interactive"])
val = st.slider("Filter data using Experience", 0, 40)
data = data.loc[data["Experience"]]

if graph == "Non interactive":
    chart = alt.Chart(data).mark_circle(size=60).encode(
        x='Experience',
        y='Salary',
        tooltip=['Experience', 'Salary']
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(chart)

if graph == "interactive":
    layout = go.Layout(
        xaxis=dict(range=[0, 16]),
        yaxis=dict(range=[0, 200000])
    ) 
    fig = go.Figure(data=go.Scatter(x=data["Experience"], y=data["Salary"], mode="markers"),
                    layout=layout)
    st.plotly_chart(fig)

if nav == "Prediction":
    st.header("Know your salary")
    val = st.number_input("Enter your exp", 0.00, 20.00, step=0.25)
    val = np.array(val).reshape(1, -1)
    pred = lr.predict(val)[0]

    if st.button("Predict"):
        st.success(f"Your Predicted Salary is {round(pred)}")

if nav == "Contribute":
    st.write(" # Contribute to our Dataset")
    ex = st.number_input("Enter Your Experience", 0, 20)
    sal = st.number_input("Enter Your Salary", 0, 1000000, step=1000)
    if st.button("submit"):
        to_add = {"Experience": [ex], "Salary": [sal]}  # Creating a dictionary with lists
        to_add_df = pd.DataFrame(to_add)  # Create DataFrame from dictionary
        data = pd.concat([data, to_add_df], ignore_index=True)  # Concatenate existing data and new data
        data.to_csv("Salary_Data.csv", mode='a', header=False, index=False)
        st.success("Submitted")


