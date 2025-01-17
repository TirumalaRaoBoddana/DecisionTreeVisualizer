import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import dtreeviz
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from graphviz.backend import render
import numpy as np
#encoding the categorical input into numerical values using the standard scalar
def encode_input(df,input_columns,output_column):
    input_data=df[input_columns]
    output_data=df[output_column]
    le=LabelEncoder()
    for i in input_data.columns:
        if(input_data[i].dtype=="object"):
            input_data[i]=le.fit_transform(input_data[i])
    if(output_data.dtype=="object"):
        output_data=le.fit_transform(output_data)
    return input_data,output_data
# Sidebar Layout with Image and Title
col1, col2 = st.sidebar.columns([1, 6])  
with col1:
    st.image('tree.png', width=50)  
with col2:
    st.title("Decision Tree Visualizer")

# Type of Decision Tree (Classifier or Regressor)
type = st.sidebar.selectbox("Select Type of the Decision Tree", options=["Decision Tree Classifier", "Decision Tree Regressor"])

# Uploading Dataset
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head(),use_container_width=True)
    n_rows, n_cols = df.shape
    if type == "Decision Tree Classifier":
        input_columns=st.multiselect("Select input columns: ",options=df.columns)
        output_column=st.selectbox("select output column: ",options=df.columns)
        criterion = st.selectbox("Select Splitting Criteria", options=["gini", "entropy", "log_loss"])
        splitter = st.selectbox("Select Type of Splitter", options=["best", "random"])
        max_depth =st.selectbox("Select Depth of the Tree:", options=[None]+list(range(1, min(n_cols + 1, 21))))
        min_samples_split = st.selectbox("Select Min Samples to Split the Node:", options=[2,3,4,5,6,7,8,9,10])
        min_samples_leaf = st.selectbox("Select Min Samples in the Leaf Node:", options=[1,2,3,4,5,6,7,8,9,10])
        max_leaf_nodes = st.selectbox("Select Max Leaf Nodes:", options=[None,2,3,4,5,6,7,8,9,10])
        min_impurity_decrease=st.number_input("Enter the min impurity decrease at node: ",min_value=0.0,max_value=1.0,step=0.1,value=0.0)
        option=st.selectbox("Selecy any one of these: ",options=["Decision Tree with Distributions","Without Distributions","Show path for instance in decision tree","only path"])
        features=input_columns
        target=output_column
        class_names=list(df[target].unique())
        X,y=encode_input(df,input_columns,output_column)
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.20)
        if(option=="Show path for instance in decision tree" or option=="only path"):
            index=st.slider("Select any row from the dataset: ",min_value=0,max_value=X.shape[0]-1,step=1)
        if(st.button("Visualize Decision Tree")):
            if(len(output_column)==0):
                st.error("Select output column")
            elif(len(input_columns)==0):
                st.error("select atleast 1 input column..")
            elif(output_column in input_columns):
                st.error("output column should not be in input columns..")
            else:
                #fiiting the model using the above parameters and options
                dtc=DecisionTreeClassifier(criterion=criterion,
                                            splitter=splitter,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            max_leaf_nodes=max_leaf_nodes,
                                            )
                dtc.fit(X_train,y_train)
                predicted_values=dtc.predict(X_test)
                st.write("accuracy score of the model is: {}".format(accuracy_score(y_test,predicted_values)))
                viz_model = dtreeviz.model(dtc,
                                X_train, y_train,
                                feature_names=features,
                                target_name=target,
                                class_names=class_names,
                                )
                if(option=="Decision Tree with Distributions"):
                    #visualizing the decision tree
                    v=viz_model.view(scale=2)
                    png_file = "tree.png"
                    v.save(png_file)  # Save the visualization as a PNG
                    
                    # Display PNG in Streamlit
                    st.image(png_file, caption="Decision Tree Visualization", use_column_width=True)

                elif(option=="Without Distributions"):
                    #visualizing the decision tree
                    v=viz_model.view(scale=2,fancy=False)
                    v.show()
                elif(option=="Show path for instance in decision tree"):
                    col1,col2,col3=st.columns(3)
                    with col1:
                        st.write("input Vector..")
                        st.write(df[input_columns].iloc[index].to_dict())
                    with col2:
                        st.write("Actual value: ")
                        st.write(df[output_column].iloc[index])
                    with col3:
                        st.write("Predicted Value")
                        st.write(dtc.predict([np.array(X.iloc[index,:])])[0])
                    v=viz_model.view(scale=2,x=X.iloc[index,:])
                    v.show()
                elif(option=="only path"):
                    col1,col2,col3=st.columns(3)
                    with col1:
                        st.write("input Vector..")
                        st.write(df[input_columns].iloc[index].to_dict())
                    with col2:
                        st.write("Actual value: ")
                        st.write(df[output_column].iloc[index])
                    with col3:
                        st.write("Predicted Value")
                        st.write(dtc.predict([np.array(X.iloc[index,:])])[0])
                    v=viz_model.view(scale=2,x=X.iloc[index,:],show_just_path=True)
                    v.show()
    elif type == "Decision Tree Regressor":
        input_columns=st.multiselect("Select input columns: ",options=df.columns)
        output_column=st.selectbox("select output column: ",options=df.columns)
        criterion = st.selectbox("Select Splitting Criteria", options=["squared_error", "friedman_mse", "absolute_error","poisson"])
        splitter = st.selectbox("Select Type of Splitter", options=["best", "random"])
        max_depth =st.selectbox("Select Depth of the Tree:", options=[None]+list(range(1, min(n_cols + 1, 21))))
        min_samples_split = st.selectbox("Select Min Samples to Split the Node:", options=[2,3,4,5,6,7,8,9,10])
        min_samples_leaf = st.selectbox("Select Min Samples in the Leaf Node:", options=[1,2,3,4,5,6,7,8,9,10])
        max_leaf_nodes = st.selectbox("Select Max Leaf Nodes:", options=[None,2,3,4,5,6,7,8,9,10])
        min_impurity_decrease=st.number_input("Enter the min impurity decrease at node: ",min_value=0.0,max_value=1.0,step=0.1,value=0.0)
        option=st.selectbox("Selecy any one of these: ",options=["Decision Tree with Distributions","Without Distributions","Show path for instance in decision tree","only path"])
        features=input_columns
        target=output_column
        X,y=encode_input(df,input_columns,output_column)
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.20)
        if(option=="Show path for instance in decision tree" or option=="only path"):
            index=st.slider("Select any row from the dataset: ",min_value=0,max_value=X.shape[0]-1,step=1)
        if(st.button("Visualize Decision Tree")):
            if(len(output_column)==0):
                st.error("Select output column")
            elif(len(input_columns)==0):
                st.error("select atleast 1 input column..")
            elif(output_column in input_columns):
                st.error("output column should not be in input columns..")
            else:
                #fiiting the model using the above parameters and options
                dtr=DecisionTreeRegressor(criterion=criterion,
                                            splitter=splitter,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            max_leaf_nodes=max_leaf_nodes,
                                            )
                dtr.fit(X_train,y_train)
                predicted_values=dtr.predict(X_test)
                st.write("r2 score of the model is: {}".format(r2_score(y_test,predicted_values)))
                viz_model = dtreeviz.model(dtr,
                                X_train, y_train,
                                feature_names=features,
                                target_name=target,
                                )
                if(option=="Decision Tree with Distributions"):
                    #visualizing the decision tree
                    v=viz_model.view(scale=2)
                    v.show()
                elif(option=="Without Distributions"):
                    #visualizing the decision tree
                    v=viz_model.view(scale=2,fancy=False)
                    v.show()
                elif(option=="Show path for instance in decision tree"):
                    col1,col2,col3=st.columns(3)
                    with col1:
                        st.write("input Vector..")
                        st.write(df[input_columns].iloc[index].to_dict())
                    with col2:
                        st.write("Actual value: ")
                        st.write(df[output_column].iloc[index])
                    with col3:
                        st.write("Predicted Value")
                        st.write(dtr.predict([np.array(X.iloc[index,:])])[0])
                    v=viz_model.view(scale=2,x=X.iloc[index,:])
                    v.show()
                elif(option=="only path"):
                    col1,col2,col3=st.columns(3)
                    with col1:
                        st.write("input Vector..")
                        st.write(df[input_columns].iloc[index].to_dict())
                    with col2:
                        st.write("Actual value: ")
                        st.write(df[output_column].iloc[index])
                    with col3:
                        st.write("Predicted Value")
                        st.write(dtr.predict([np.array(X.iloc[index,:])])[0])
                    v=viz_model.view(scale=2,x=X.iloc[index,:],show_just_path=True)
                    v.show()
else:
    st.title("Decision Tree Visualizer")
    st.markdown("""
    This Streamlit app allows you to interactively visualize and explore decision tree models. You can upload your own dataset, select the target variable, and choose between a classification or regression decision tree model. The app will then train the model, display key metrics (such as accuracy or R² score), and visualize the decision tree itself.

    Key features:
    - Upload a CSV or Excel dataset.
    - Select the target variable for classification or regression.
    - Train and evaluate a decision tree model.
    - Visualize the decision tree structure with interactive plots.

    Whether you're learning about decision trees or need a quick way to explore them, this app provides an easy-to-use and informative interface for understanding the decision-making process of decision trees.
    """)
