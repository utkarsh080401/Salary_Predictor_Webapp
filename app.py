#Load Core Packages

import streamlit as st

#Import EDA packages

import pandas as pd
import numpy as np

#Data Viz Pkgs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




#ML Packages

import joblib

def main():
    "salary predict with ML"

    st.title("Salary Predictor")
    activity= ["eda", "prediction", "metrics"]
    choice= st.sidebar.selectbox("Choose An Activity", activity)
    #Load Data
    df= pd.read_csv("data/adult_salary.csv")
    #EDA
    if choice == 'eda':
        st.subheader("EDA Section")
        st.text("Exploratory Data Analysis")

        #Previews

        if st.checkbox("Preview Dataset"):
            number = st.number_input("Number to Show")
            st.dataframe(df.head(int(number)))

        #Showing Column/Rows

        if st.button("Column Names"):
            st.write(df.columns)



        #Description

        if st.checkbox("Show Description"):
            st.write(df.describe())

        #Shape

        if st.checkbox("Show Dataset Shape"):
            st.write(df.shape)
            data_dim= st.radio("Show Dimension by", ("Rows", "Columns"))

            if data_dim== 'Rows':
                st.text("Number of Rows")
                st.write(df.shape[0])

            elif data_dim== 'Columns':
                st.text("Number of Columns")
                st.write(df.shape[1])

            else:
                st.write(df.shape)

        #Selections

        if st.checkbox("Select Columns to Show"):
            all_columns = df.columns.tolist()
            selected_columns= st.multiselect("Select Columns", all_columns)
            new_df= df[selected_columns]
            st.dataframe(new_df)

        if st.checkbox("Select Rows to Show"):
            selected_index= st.multiselect("Select Rows", df.head(10).index)
            selected_rows= df.loc[selected_index]
            st.dataframe(selected_rows)

        #Value Count

        if st.button("Value Counts"):
            st.text("Value Counts By Class")
            st.write(df.iloc[:, -1].value_counts())

        #Plot



    #Prediction

    elif choice== 'prediction':
        st.subheader("Prediction Section")

    #Metrics

    elif choice== 'metrics':
        st.subheader("Metrics  Section")

if __name__== '__main__':
    main()