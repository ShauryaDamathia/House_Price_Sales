import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df = df[["GrLivArea", "TotalBsmtSF", "OverallQual", "YearBuilt", "GarageCars", "SalePrice"]]
    return df.dropna()

@st.cache_resource
def train_model(df):
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    model = LinearRegression().fit(X, y)
    return model

def main():
    st.title("🏠 House Price Predictor")

    df = load_data()
    model = train_model(df)

    st.header("Enter House Features")
    col1, col2 = st.columns(2)

    with col1:
        grlivarea = st.number_input("Living Area (sq ft)", 500, 4000, 1500)
        totalbsmt = st.number_input("Basement Area (sq ft)", 0, 2000, 800)
        overallqual = st.slider("Overall Quality (1-10)", 1, 10, 5)

    with col2:
        yearbuilt = st.number_input("Year Built", 1870, 2023, 2000)
        garagecars = st.slider("Garage Capacity (Cars)", 0, 4, 2)

    input_data = pd.DataFrame({
        "GrLivArea": [grlivarea],
        "TotalBsmtSF": [totalbsmt],
        "OverallQual": [overallqual],
        "YearBuilt": [yearbuilt],
        "GarageCars": [garagecars]
    })

    if st.button("Predict Price"):
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Price: ${prediction:,.0f}")

        fig = px.scatter(df, x="GrLivArea", y="SalePrice", title="Living Area vs Price")
        fig.add_scatter(x=[grlivarea], y=[prediction], mode="markers", marker=dict(color="red", size=12), name="Prediction")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
