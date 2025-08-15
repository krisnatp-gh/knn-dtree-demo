import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.ticker as ticker

# Set page config
st.set_page_config(page_title="Disease Risk ML Analysis", layout="wide")

# Title
st.title("🏥 Disease Risk ML Analysis")
st.markdown("Interactive ML modeling for disease risk prediction")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Disease_Risk_No_Duplicates.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'Disease_Risk_No_Duplicates.csv' is in the folder.")
        return None

df = load_data()

if df is not None:
    # Display basic info about the dataset
    st.sidebar.header("📊 Dataset Info")
    st.sidebar.write(f"**Shape:** {df.shape}")
    st.sidebar.write(f"**Features:** Age, Annual_Income_IDR")
    st.sidebar.write(f"**Target:** Disease_Risk")
    
    # Scale the features for display
    df_display = df.copy()
    df_display['Age_scaled'] = df_display['Age'] / 10  # Scale in tens
    df_display['Income_scaled'] = df_display['Annual_Income_IDR'] / 1_000_000  # Scale in millions
    df_display['Income in Juta'] = df_display["Income_scaled"].apply(lambda x: f'{x:,.0f} Juta')
    
    # Show scatter plot
    st.header("📈 Data Visualization")

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create scatter plot with seaborn
    sns.scatterplot(data=df_display, 
                    x='Age', 
                    y='Annual_Income_IDR',
                    hue='Disease_Risk',
                    palette='Set1',
                    s=100,
                    ax=ax)

    # Set labels and title
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income (IDR)')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.0f} Juta'))
    ax.set_title('Disease Risk by Age and Income', fontsize=16)

    # Move legend outside the plot
    ax.legend(title='Disease Risk', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Display in streamlit
    st.pyplot(fig)

    # Prepare data for ML
    X = df[['Age', 'Annual_Income_IDR']].copy()
    y = df['Disease_Risk'].copy()
    
    # Train-test split (not displayed, but used for ML)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    
    # ML Configuration Section
    st.header("🤖 Machine Learning Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Scaling Options")
        scaling_option = st.selectbox(
            "Choose scaling method:",
            ["No Scaling", "Min-Max Scaling", "Standard Scaling", "Robust Scaling"]
        )
        
        # Apply scaling
        scaler = None
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        if scaling_option == "Min-Max Scaling":
            scaler = MinMaxScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
        elif scaling_option == "Standard Scaling":
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
        elif scaling_option == "Robust Scaling":
            scaler = RobustScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    with col2:
        st.subheader("Model Selection")
        model_option = st.selectbox(
            "Choose model:",
            ["K-Nearest Neighbors", "Decision Tree"]
        )
    
    # Model Parameters
    st.subheader("🔧 Model Parameters")
    
    if model_option == "K-Nearest Neighbors":
        n_neighbors = st.slider("Number of Neighbors (n_neighbors)", 1, 100, 3)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        
    else:  # Decision Tree
        col1, col2, col3 = st.columns(3)
        with col1:
            random_state = st.number_input("Random State", value=0, step=1)
        with col2:
            max_depth = st.slider("Max Depth", 1, 20, 5)
        with col3:
            min_samples_split = st.slider("Min Samples Split", 2, 100, 2)
        
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 100, 1)
        
        model = DecisionTreeClassifier(
            random_state=int(random_state),
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
    
    # Train model and show results
    if st.button("🚀 Train Model", type="primary"):
        # Fit model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate accuracies
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Display results
        st.header("📊 Model Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Train Accuracy", f"{train_accuracy:.4f}")
        with col2:
            st.metric("Test Accuracy", f"{test_accuracy:.4f}")
        
        
        # Model-specific visualizations
        if model_option == "K-Nearest Neighbors":
            st.header("🎯 KNN Decision Boundary")
            
            # Always use original non-scaled data for visualization bounds
            x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
            y_min, y_max = X_train.iloc[:, 1].min() - 1e7, X_train.iloc[:, 1].max() + 1e7  # Add padding in millions
            
            # Calculate appropriate step size based on data range
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # Use adaptive step size to ensure reasonable grid size
            h_x = max(x_range / 100, 0.1)  # Adjusted for age scale (tens)
            h_y = max(y_range / 100, 1e6)  # Adjusted for income scale (100 millions)
            
            # Create meshgrid with safe step sizes
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x),
                                np.arange(y_min, y_max, h_y))
            
            # Limit grid size for safety
            if xx.size > 10000:  # If still too large, subsample
                n_points = int(np.sqrt(10000))
                x_points = np.linspace(x_min, x_max, n_points)
                y_points = np.linspace(y_min, y_max, n_points)
                xx, yy = np.meshgrid(x_points, y_points)
            
            # Make predictions on meshgrid
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            
            # If scaling was used, transform mesh points before prediction
            if scaling_option != "No Scaling":
                # Assuming you have the scaler object available
                mesh_points_scaled = scaler.transform(mesh_points)
                Z = model.predict(mesh_points_scaled)
            else:
                Z = model.predict(mesh_points)
            
            Z = Z.reshape(xx.shape)
            
            # Create boundary plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot decision boundary
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
            
            # Plot training points using original non-scaled data
            sns.scatterplot(x=X_train.iloc[:,0], y=X_train.iloc[:,1], hue=y_train, ax=ax,palette="Set1", s=100)
            # scatter = ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], 
            #                    hue=y_train)
            
            ax.set_xlabel('Age')
            ax.set_ylabel('Annual Income (IDR)')
            ax.set_title(f'KNN Decision Boundary (k={n_neighbors})')
            
            # Format y-axis for better readability of large numbers
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f} Juta'))
            ax.legend(title='Disease Risk', bbox_to_anchor=(1.05, 1), loc='upper left')
                        
            st.pyplot(fig)

        else:  # Decision Tree
            st.header("🌳 Decision Tree Visualization")
            
            # Plot tree
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # Convert class names to strings to avoid the error
            class_names = [str(cls) for cls in model.classes_]
            
            plot_tree(model, 
                     feature_names=['Age', 'Annual_Income_IDR'],
                     class_names=class_names,  # Fixed: convert to strings
                     filled=False,
                     rounded=True,
                     fontsize=15,
                     ax=ax)
            
            ax.set_title("Decision Tree Structure")
            st.pyplot(fig)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                st.subheader("📊 Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': ['Age', 'Annual_Income_IDR'],
                    'Importance': model.feature_importances_
                })
                
                fig_imp = px.bar(importance_df, x='Feature', y='Importance',
                                title="Feature Importance in Decision Tree")
                
                                # Update layout for better font styling
                fig_imp.update_layout(
                    font=dict(color='black', size=15),  # Black font color, larger size
                    title_font=dict(color='black', size=18),  # Title styling
                    xaxis_title_font=dict(color='black', size=15),  # X-axis title
                    yaxis_title_font=dict(color='black', size=15),  # Y-axis title,
                    xaxis=dict(tickfont=dict(color='black', size=15)),
                    yaxis=dict(tickfont=dict(color='black', size=15))
                )
                
                st.plotly_chart(fig_imp, use_container_width=True)
                
    # Display sample data
    with st.expander("📋 Sample Data"):
        st.write("First 10 rows of the dataset:")
        display_df = df_display[['Age', 'Income in Juta', 'Disease_Risk']].head(10)
        display_df.columns = ['Age', 'Annual Income (IDR)', 'Disease Risk']
        st.dataframe(display_df)

else:
    st.error("Unable to load the dataset. Please check the file path and try again.")
