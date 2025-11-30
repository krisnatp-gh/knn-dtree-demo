import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.ticker as ticker
from numerize import numerize


def format_number(value):
    """Format number nicely: use scientific notation if too small, otherwise 4 decimals."""
    abs_val = abs(value)
    # If the number would display as 0.0000 or very small (less than 0.00005), use scientific notation
    if abs_val < 0.00005 and abs_val != 0:
        return f"{value:.2e}"
    else:
        return f"{value:.4f}"

def format_ci(lower, upper):
    """Format confidence interval with smart notation."""
    return f"[{format_number(lower)}, {format_number(upper)}]"

def format_number_for_latex(value):
    """Format number for LaTeX: convert scientific notation to proper LaTeX format."""
    formatted = format_number(value)
    # If scientific notation, convert to LaTeX format
    if 'e' in formatted:
        mantissa, exponent = formatted.split('e')
        exponent = int(exponent)  # Remove leading zeros/plus signs
        return f"{mantissa} \\times 10^{{{exponent}}}"
    else:
        return formatted

def format_operator_operand(value):
    """
    Format a number to be enclosed in parentheses if negative.
    Used when displaying numbers on the right side of arithmetic operators.

    Args:
        value: Numeric value to format

    Returns:
        Formatted string, with parentheses if negative

    Example:
        format_operator_operand(5.2) -> "5.2000"
        format_operator_operand(-0.3) -> "(-0.3000)"
    """
    formatted = format_number_for_latex(value)
    # If the formatted string starts with a minus sign, wrap in parentheses
    if formatted.startswith('-'):
        return f"({formatted})"
    else:
        return formatted

# Set page config
st.set_page_config(page_title="Disease Risk ML Analysis", layout="wide")

# Initialize session state for Logistic Regression model persistence
if 'logreg_model' not in st.session_state:
    st.session_state.logreg_model = None
if 'logreg_scaler' not in st.session_state:
    st.session_state.logreg_scaler = None
if 'logreg_scaling_used' not in st.session_state:
    st.session_state.logreg_scaling_used = None
if 'logreg_type' not in st.session_state:
    st.session_state.logreg_type = None  # 'regularized' or 'unregularized'
if 'logreg_trained' not in st.session_state:
    st.session_state.logreg_trained = False
# Additional session state for results persistence
if 'logreg_train_acc' not in st.session_state:
    st.session_state.logreg_train_acc = None
if 'logreg_test_acc' not in st.session_state:
    st.session_state.logreg_test_acc = None
if 'logreg_conf_int' not in st.session_state:
    st.session_state.logreg_conf_int = None
if 'logreg_llr_pvalue' not in st.session_state:
    st.session_state.logreg_llr_pvalue = None
# Session state for tracking training parameters (for change detection)
if 'logreg_trained_scaling' not in st.session_state:
    st.session_state.logreg_trained_scaling = None
if 'logreg_trained_reg_type' not in st.session_state:
    st.session_state.logreg_trained_reg_type = None
if 'logreg_trained_penalty' not in st.session_state:
    st.session_state.logreg_trained_penalty = None
if 'logreg_trained_c_value' not in st.session_state:
    st.session_state.logreg_trained_c_value = None
# Session state for odds ratio analysis
if 'or_feature_selection' not in st.session_state:
    st.session_state.or_feature_selection = "Age"
if 'or_age_a' not in st.session_state:
    st.session_state.or_age_a = 30.0
if 'or_age_b' not in st.session_state:
    st.session_state.or_age_b = 20.0
if 'or_income_a' not in st.session_state:
    st.session_state.or_income_a = 50.0
if 'or_income_b' not in st.session_state:
    st.session_state.or_income_b = 75.0
# Session state for sigmoid curve visualization
if 'sigmoid_age_param' not in st.session_state:
    st.session_state.sigmoid_age_param = 43
if 'sigmoid_income_param' not in st.session_state:
    st.session_state.sigmoid_income_param = 75.0

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
    df_display['Income_formatted'] = df_display['Annual_Income_IDR'].apply(lambda x: f"{x:_.0f}".replace("_", " "))

    # Display sample data
    st.header("📋 Sample Data")
    st.write("First 10 rows of the dataset:")
    display_df = df_display[['Age', 'Income_formatted', 'Disease_Risk']].head(10)
    display_df.columns = ['Age', 'Annual Income (IDR)', 'Disease Risk']
    st.dataframe(display_df)

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

    with col2:
        st.subheader("Model Selection")
        model_option = st.selectbox(
            "Choose model:",
            ["K-Nearest Neighbors", "Decision Tree", "Logistic Regression"]
        )

    with col1:
        st.subheader("Scaling Options")
        # Show scaling options for all models
        if model_option in ["K-Nearest Neighbors", "Decision Tree"]:
            scaling_option = st.selectbox(
                "Choose scaling method:",
                ["No Scaling", "Min-Max Scaling", "Standard Scaling", "Robust Scaling"]
            )
        else:  # Logistic Regression - scaling is shown in its own configuration
            scaling_option = "No Scaling"  # Default, will be overridden below
            st.info("Scaling options available in Logistic Regression configuration below.")

    # Apply scaling for KNN and Decision Tree
    scaler = None
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if model_option in ["K-Nearest Neighbors", "Decision Tree"]:
        if scaling_option == "Min-Max Scaling":
            scaler = MinMaxScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
        elif scaling_option == "Standard Scaling":
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
        elif scaling_option == "Robust Scaling":
            scaler = RobustScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    # Model Parameters
    st.subheader("🔧 Model Parameters")

    if model_option == "K-Nearest Neighbors":
        n_neighbors = st.slider("Number of Neighbors (n_neighbors)", 1, 100, 7)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)

    elif model_option == "Decision Tree":
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

    else:  # Logistic Regression
        # Display logistic regression model formula
        st.markdown("**Logistic Regression Model:**")
        # st.latex(r"P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_{\text{Age}} + \beta_2 X_{\text{Income}})}}")
        st.latex(r"P(y=1) =\frac{\text{exp}({\beta_0 + \beta_{\text{Age}} \cdot x_{\text{Age}} + \beta_{\text{Annual Income}} \cdot x_{\text{Annual Income}} })}{1 + \text{exp}({\beta_0 + \beta_{\text{Age}} \cdot x_{\text{Age}} + \beta_{\text{Annual Income}} \cdot x_{\text{Annual Income}} })}")
        

        col1, col2 = st.columns(2)

        with col1:
            logreg_scaling = st.selectbox(
                "Choose scaling method:",
                ["No Scaling", "Min-Max Scaling", "Standard Scaling", "Robust Scaling"],
                key="logreg_scaling"
            )

        with col2:
            regularization_type = st.selectbox(
                "Regularization Type:",
                ["Unregularized", "Regularized"],
                key="reg_type"
            )

        # Apply scaling for Logistic Regression
        if logreg_scaling == "Min-Max Scaling":
            scaler = MinMaxScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
        elif logreg_scaling == "Standard Scaling":
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
        elif logreg_scaling == "Robust Scaling":
            scaler = RobustScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

        # Regularization parameters
        if regularization_type == "Regularized":
            col1, col2 = st.columns(2)
            with col1:
                penalty = st.selectbox(
                    "Regularization Method:",
                    ["L1", "L2"],
                    key="penalty"
                )
            with col2:
                # C values as powers of 10
                c_options = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                c_value = st.select_slider(
                    "C (Inverse Regularization Strength):",
                    options=c_options,
                    value=1,
                    key="c_value"
                )

            # Set solver based on penalty
            if penalty == "L1":
                solver = 'saga'
            else:
                solver = 'lbfgs'

            model = LogisticRegression(
                penalty=penalty.lower(),
                C=c_value,
                solver=solver,
                random_state=0,
                max_iter=1000
            )
        else:
            # Unregularized - will use statsmodels
            model = None  # Will be handled separately with statsmodels

        # Check if parameters have changed since last training
        if st.session_state.logreg_trained:
            params_changed = False

            # Check scaling method
            if st.session_state.logreg_trained_scaling != logreg_scaling:
                params_changed = True

            # Check regularization type
            if st.session_state.logreg_trained_reg_type != regularization_type:
                params_changed = True

            # Check regularization parameters (only if both current and trained are regularized)
            if regularization_type == "Regularized" and st.session_state.logreg_trained_reg_type == "Regularized":
                if st.session_state.logreg_trained_penalty != penalty:
                    params_changed = True
                if st.session_state.logreg_trained_c_value != c_value:
                    params_changed = True

            if params_changed:
                st.warning("⚠️ Model parameters have changed since last training. Please re-train the model to see updated results.")

    # Train model and show results
    if st.button("🚀 Train Model", type="primary"):

        if model_option == "Logistic Regression" and regularization_type == "Unregularized":
            # Use statsmodels for unregularized logistic regression
            X_train_with_const = sm.add_constant(X_train_scaled)
            X_test_with_const = sm.add_constant(X_test_scaled)

            # Fit model
            logit_model = sm.Logit(y_train, X_train_with_const)
            result = logit_model.fit(disp=0)

            # Make predictions
            y_train_pred_prob = result.predict(X_train_with_const)
            y_test_pred_prob = result.predict(X_test_with_const)

            y_train_pred = (y_train_pred_prob >= 0.5).astype(int)
            y_test_pred = (y_test_pred_prob >= 0.5).astype(int)

            # Calculate accuracies
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            # Store model and results in session state for persistence
            st.session_state.logreg_model = result
            st.session_state.logreg_scaler = scaler
            st.session_state.logreg_scaling_used = logreg_scaling
            st.session_state.logreg_type = 'unregularized'
            st.session_state.logreg_trained = True
            st.session_state.logreg_train_acc = train_accuracy
            st.session_state.logreg_test_acc = test_accuracy
            st.session_state.logreg_conf_int = result.conf_int(alpha=0.05)
            st.session_state.logreg_llr_pvalue = result.llr_pvalue
            # Store training parameters for change detection
            st.session_state.logreg_trained_scaling = logreg_scaling
            st.session_state.logreg_trained_reg_type = regularization_type
            st.session_state.logreg_trained_penalty = None
            st.session_state.logreg_trained_c_value = None
            # Rerun to clear any parameter change warning
            st.session_state.pop("or_feature_selection")
            st.session_state.pop("or_age_a")
            st.session_state.pop("or_age_b")
            st.session_state.pop("or_income_a")
            st.session_state.pop("or_income_b")
            st.session_state.pop("sigmoid_age_param")
            st.session_state.pop("sigmoid_income_param")
            st.rerun()

        else:
            # For KNN, Decision Tree, and Regularized Logistic Regression
            # Fit model
            model.fit(X_train_scaled, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            # Calculate accuracies
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            # Store regularized logistic regression model in session state
            if model_option == "Logistic Regression":
                st.session_state.logreg_model = model
                st.session_state.logreg_scaler = scaler
                st.session_state.logreg_scaling_used = logreg_scaling
                st.session_state.logreg_type = 'regularized'
                st.session_state.logreg_trained = True
                st.session_state.logreg_train_acc = train_accuracy
                st.session_state.logreg_test_acc = test_accuracy
                st.session_state.logreg_conf_int = None  # Not available for regularized
                st.session_state.logreg_llr_pvalue = None  # Not available for regularized
                # Store training parameters for change detection
                st.session_state.logreg_trained_scaling = logreg_scaling
                st.session_state.logreg_trained_reg_type = regularization_type
                st.session_state.logreg_trained_penalty = penalty
                st.session_state.logreg_trained_c_value = c_value
                # Rerun to clear any parameter change warning
                st.session_state.pop("or_feature_selection")
                st.session_state.pop("or_age_a")
                st.session_state.pop("or_age_b")
                st.session_state.pop("or_income_a")
                st.session_state.pop("or_income_b")
                st.session_state.pop("sigmoid_age_param")
                st.session_state.pop("sigmoid_income_param")
                st.rerun()

            # Display results for KNN and Decision Tree only
            # Logistic Regression display is handled by session state section below
            if model_option != "Logistic Regression":
                st.header("📊 Model Performance")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Train Accuracy", numerize.numerize(train_accuracy, 4))
                with col2:
                    st.metric("Test Accuracy", numerize.numerize(test_accuracy, 4))

            # Model-specific visualizations
            if model_option == "K-Nearest Neighbors":
                st.header("🎯 KNN Decision Boundary on Train Data")

                # Determine which dataset to use for visualization based on scaling
                if scaling_option != "No Scaling":
                    X_train_plot = X_train_scaled
                    padding_x = 0.05  # Proportional padding for scaled data
                    padding_y = 0.05
                    step_x = 0.001    # Fine step for normalized range
                    step_y = 0.001
                else:
                    X_train_plot = X_train
                    padding_x = 1     # Fixed padding for original scale
                    padding_y = 1e7
                    step_x = 0.1      # Coarse step for large ranges
                    step_y = 1e6

                # Calculate bounds using appropriate dataset
                x_min, x_max = X_train_plot.iloc[:, 0].min() - padding_x, X_train_plot.iloc[:, 0].max() + padding_x
                y_min, y_max = X_train_plot.iloc[:, 1].min() - padding_y, X_train_plot.iloc[:, 1].max() + padding_y

                # Calculate appropriate step size based on data range
                x_range = x_max - x_min
                y_range = y_max - y_min

                # Use adaptive step size to ensure reasonable grid size
                h_x = max(x_range / 100, step_x)
                h_y = max(y_range / 100, step_y)

                # Create meshgrid with safe step sizes
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x),
                                    np.arange(y_min, y_max, h_y))

                # Limit grid size for safety
                if xx.size > 10000:  # If still too large, subsample
                    n_points = int(np.sqrt(10000))
                    x_points = np.linspace(x_min, x_max, n_points)
                    y_points = np.linspace(y_min, y_max, n_points)
                    xx, yy = np.meshgrid(x_points, y_points)

                # Make predictions on meshgrid (already in correct space)
                mesh_points = np.c_[xx.ravel(), yy.ravel()]
                Z = model.predict(mesh_points)
                Z = Z.reshape(xx.shape)

                # Create boundary plot
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot decision boundary
                ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

                # Plot training points using appropriate dataset
                sns.scatterplot(x=X_train_plot.iloc[:,0], y=X_train_plot.iloc[:,1], hue=y_train, ax=ax, palette="Set1", s=100)

                # Set labels based on scaling
                if scaling_option != "No Scaling":
                    ax.set_xlabel('Age (Scaled)')
                    ax.set_ylabel('Annual Income (Scaled)')
                else:
                    ax.set_xlabel('Age')
                    ax.set_ylabel('Annual Income (IDR)')

                ax.set_title(f'KNN Decision Boundary (k={n_neighbors})')

                # Format y-axis only for non-scaled data
                if scaling_option == "No Scaling":
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f} Juta'))

                ax.legend(title='Disease Risk', bbox_to_anchor=(1.05, 1), loc='upper left')

                st.pyplot(fig)

            elif model_option == "Decision Tree":
                st.header("🌳 Decision Tree Visualization on Train Data")

                # Plot tree
                fig, ax = plt.subplots(figsize=(15, 10))

                # Convert class names to strings to avoid the error
                class_names = [str(cls) for cls in model.classes_]

                if scaling_option == "No Scaling":
                    feature_names=['Age', 'Annual Income']
                else:
                    feature_names=['Age (Scaled)', 'Annual Income (Scaled)']



                plot_tree(model,
                         feature_names=feature_names,
                         class_names=class_names,  # Fixed: convert to strings
                         filled=False,
                         rounded=True,
                         fontsize=15,
                         ax=ax)

                # Format large numbers with comma separators for readability when no scaling
                if scaling_option == "No Scaling":
                    import re
                    for text_obj in ax.texts:
                        original_text = text_obj.get_text()

                        def format_number(match):
                            num_str = match.group(0)
                            try:
                                num = float(num_str)
                                # Only format large numbers (likely income values)
                                if abs(num) >= 1000:
                                    if num == int(num):
                                        return f"{int(num):,}"
                                    else:
                                        return f"{num:,.1f}"
                            except ValueError:
                                pass
                            return num_str

                        # Match numbers including decimals
                        formatted_text = re.sub(r'-?\d+\.?\d*', format_number, original_text)
                        text_obj.set_text(formatted_text)

                ax.set_title("Decision Tree Structure")
                st.pyplot(fig)

                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    st.subheader("📊 Feature Importance in Decision Tree")
                    importance_df = pd.DataFrame({
                        'Feature': ['Age', 'Annual Income'],
                        'Importance': model.feature_importances_
                    })

                    fig_imp, ax_imp = plt.subplots(figsize=(10, 4))

                    # Create horizontal bar plot with seaborn
                    sns.barplot(data=importance_df, x='Importance', y='Feature',
                               color='green', ax=ax_imp)

                    # Annotate the bars
                    for i, (idx, row) in enumerate(importance_df.iterrows()):
                        value = row['Importance']
                        ax_imp.annotate(f'{value:.4f}',
                                       xy=(value, i),
                                       xytext=(5, 0),
                                       textcoords='offset points',
                                       ha='left', va='center',
                                       fontsize=12, fontweight='bold')

                    ax_imp.set_xlabel('Importance', fontsize=14)
                    ax_imp.set_ylabel('Feature', fontsize=14)
                    # ax_imp.set_title('Feature Importance in Decision Tree', fontsize=16)
                    ax_imp.tick_params(labelsize=12)

                    # Remove top and right spines
                    ax_imp.spines['top'].set_visible(False)
                    ax_imp.spines['right'].set_visible(False)

                    # Adjust x-axis limit to give room for annotations
                    xmax = ax_imp.get_xlim()[1]
                    ax_imp.set_xlim(0, xmax * 1.15)

                    plt.tight_layout()
                    st.pyplot(fig_imp)

            # Regularized Logistic Regression display is handled by session state section below

    # Display Logistic Regression results from session state (persists across reruns)
    if model_option == "Logistic Regression" and st.session_state.logreg_trained:
        cached_model = st.session_state.logreg_model
        model_type = st.session_state.logreg_type

        # Display results
        st.header("📊 Model Performance")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Train Accuracy", numerize.numerize(st.session_state.logreg_train_acc, 4))
        with col2:
            st.metric("Test Accuracy", numerize.numerize(st.session_state.logreg_test_acc, 4))

        # Beta coefficients bar plot
        st.header("📊 β Coefficient Values")

        coef_names = [r'$\beta_0$', r'$\beta_{\text{\Age}}$', r'$\beta_{\text{Annual Income}}$']
        if model_type == 'unregularized':
            coef_values = cached_model.params.values
        else:  # regularized
            coef_values = [cached_model.intercept_[0], cached_model.coef_[0][0], cached_model.coef_[0][1]]

        coef_df = pd.DataFrame({
            'Coefficient': coef_names,
            'Value': coef_values
        })

        fig_coef, ax_coef = plt.subplots(figsize=(12, 6))

        # Create bar plot with seaborn
        bars = sns.barplot(data=coef_df, x='Coefficient', y='Value', color='blue', ax=ax_coef)

        # Annotate bars with values
        for i, (idx, row) in enumerate(coef_df.iterrows()):
            value = row['Value']
            # Position annotation above positive bars, below negative bars
            va = 'bottom' if value >= 0 else 'top'
            ax_coef.annotate(format_number(value),
                        xy=(i, value),
                        ha='center', va=va,
                        fontsize=12, fontweight='bold')

        ax_coef.tick_params(labelsize=14)
        ax_coef.set_xlabel('')
        ax_coef.set_ylabel('Value', fontsize=14)
        ax_coef.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)

        ax_coef.spines['top'].set_visible(False)
        ax_coef.spines['right'].set_visible(False)

        # Extend y-axis limits to give room for annotations
        ymin, ymax = ax_coef.get_ylim()
        y_range = ymax - ymin
        ax_coef.set_ylim(ymin - 0.1 * y_range, ymax + 0.1 * y_range)

        st.pyplot(fig_coef)

        # LLR and Wald tests for unregularized only
        if model_type == 'unregularized':
            with st.expander("📊 Statistical Tests", expanded=True):
                tab1, tab2 = st.tabs(["LLR Test", "Wald Test"])

                with tab1:
                    # LLR-Ratio Test
                    st.header("📈 Likelihood Ratio (LLR) Test")

                    st.markdown("**Hypotheses:**")
                    st.latex(r"""
                    \begin{aligned}
                    H_0 &: \beta_{\text{Age}} = \beta_{\text{Annual Income}} = 0 \\
                    H_A &: \text{At least one } \beta_i \neq 0
                    \end{aligned}
                    """)

                    llr_pvalue = st.session_state.logreg_llr_pvalue

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("LLR Test p-value", f"{llr_pvalue:.2e}")
                    with col2:
                        if llr_pvalue < 0.05:
                            st.success("**Conclusion:** Reject H₀ at α = 0.05. The model is statistically significant.")
                        else:
                            st.warning("**Conclusion:** Fail to reject H₀ at α = 0.05. The model is not statistically significant.")

                with tab2:
                    # Wald Test for each coefficient
                    st.header("📊 Wald Test for Individual Coefficients")

                    # Get confidence intervals from session state
                    conf_int = st.session_state.logreg_conf_int

                    # Create three columns for each coefficient
                    col1, col2, col3 = st.columns(3)

                    param_names = ['const', 'Age', 'Annual_Income_IDR']
                    display_names = [r'$\beta_0$', r'$\beta_{\text{\Age}}$', r'$\beta_{\text{Annual Income}}$']
                    latex_names = [r'\beta_0', r'\beta_{\text{Age}}', r'\beta_{\text{Annual Income}}']

                    columns = [col1, col2, col3]

                    for i, (param, display_name, latex_name, col) in enumerate(zip(param_names, display_names, latex_names, columns)):
                        with col:
                            st.subheader(f"${latex_name}$")

                            # Beta value
                            beta_value = cached_model.params[param]
                            st.metric("Value", format_number(beta_value))

                            # Hypotheses
                            st.markdown("**Wald Test Hypotheses:**")
                            st.latex(f"H_0: {latex_name} = 0")
                            st.latex(f"H_A: {latex_name} \\neq 0")

                            # P-value
                            p_value = cached_model.pvalues[param]
                            st.metric("p-value", f"{p_value:.2e}")

                            # Confidence interval
                            ci_lower = conf_int.loc[param, 0]
                            ci_upper = conf_int.loc[param, 1]
                            st.metric("95% CI", format_ci(ci_lower, ci_upper))

                            # Conclusion
                            if p_value < 0.05:
                                st.success("Significant at α = 0.05")
                            else:
                                st.warning("Not significant at α = 0.05")

                    components.html("<br>", height=100) # free space to write on whiteboard/zoom
        else:
            # Regularized model
            st.info("Note: Statistical tests (LLR, Wald) are only available for **unregularized logistic regression**")

    # Odds Ratio Analysis section - outside button block, uses session state
    if model_option == "Logistic Regression" and st.session_state.logreg_trained:
        with st.expander("📊 Odds Analysis & Sigmoid Visualization"):
            # Get model and scaler from session state
            cached_model = st.session_state.logreg_model
            cached_scaler = st.session_state.logreg_scaler
            cached_scaling = st.session_state.logreg_scaling_used
            model_type = st.session_state.logreg_type

            # Feature selection - placed ABOVE tabs
            st.markdown("**Select Feature:**")
            feature_options = ["Age", "Annual_Income_IDR"]
            default_index = 0

            or_feature = st.selectbox(
                "Feature to compare (Odds Ratio) / Feature to hold constant (Sigmoid Curve):",
                feature_options,
                index=default_index,
                key="or_feature_selection"
            )

            # # Update session state when selection changes
            # if or_feature != st.session_state.or_feature_selection:
            #     st.session_state.or_feature_selection = or_feature



            st.markdown("---")
    
            # Create tabs
            tab1, tab2 = st.tabs(["Odds Ratio", "Sigmoid Curve"])

            # TAB 1: ODDS RATIO
            with tab1:
                # Two columns: inputs on left, formulas on right
                col_input, col_formula = st.columns([1, 2])

                with col_input:
                    st.markdown("**Input Values:**")

                    # Get reasonable defaults based on feature
                    if or_feature == "Age":
                        step_val = 1.0

                        or_a = st.number_input(
                            "Numerator (a):",
                            step=step_val,
                            format="%.1f",
                            key="or_age_a"
                        )

                        or_b = st.number_input(
                            "Denominator (b):",
                            step=step_val,
                            format="%.1f",
                            key="or_age_b"
                        )


                    else:
                        # Input in millions (Juta)
                        step_val = 1.0

                        # Numerator and denominator inputs with Juta in label
                        or_a_millions = st.number_input(
                            "Numerator (a) - Juta:",
                            step=step_val,
                            value=50.0,
                            format="%.1f",
                            key="or_income_a"
                        )
                        st.caption("(in millions IDR)")

                        or_b_millions = st.number_input(
                            "Denominator (b) - Juta:",
                            step=step_val,
                            value=75.0,
                            format="%.1f",
                            key="or_income_b"
                        )
                        st.caption("(in millions IDR)")

                        # Convert back to actual values
                        or_a = or_a_millions * 1e6
                        or_b = or_b_millions * 1e6
                    
                    st.markdown("**β Coefficient Values:**")

                    # Get coefficient values
                    if model_type == 'unregularized':
                        beta_0_val = cached_model.params['const']
                        beta_age_val = cached_model.params['Age']
                        beta_income_val = cached_model.params['Annual_Income_IDR']
                    else:  # regularized
                        beta_0_val = cached_model.intercept_[0]
                        beta_age_val = cached_model.coef_[0][0]
                        beta_income_val = cached_model.coef_[0][1]

                    # Build HTML table
                    html_content = f"""
                    <style>
                    .beta-table {{
                        font-size: 18px !important;
                        border-collapse: collapse !important;
                        border: 1px solid #ddd !important;
                    }}
                    .beta-table th {{
                        font-size: 18px !important;
                        font-weight: bold !important;
                        padding: 10px 15px !important;
                        text-align: left !important;
                        border: 1px solid #ddd !important;
                        background-color: #f8f9fa !important;
                    }}
                    .beta-table td {{
                        font-size: 18px !important;
                        padding: 10px 15px !important;
                        border: 1px solid #ddd !important;
                    }}
                    </style>
                    <table class="beta-table">
                        <thead>
                            <tr>
                                <th>Coefficient</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>β<sub>0</sub></td>
                                <td>{format_number(beta_0_val)}</td>
                            </tr>
                            <tr>
                                <td>β<sub>Age</sub></td>
                                <td>{format_number(beta_age_val)}</td>
                            </tr>
                            <tr>
                                <td>β<sub>Annual Income</sub></td>
                                <td>{format_number(beta_income_val)}</td>
                            </tr>
                        </tbody>
                    </table>
                    """
                    components.html(html_content, height=200)

                with col_formula:
                    st.markdown("**Odds Ratio Calculation:**")

                    # Get the beta coefficient for selected feature
                    feature_idx = 0 if or_feature == "Age" else 1

                    if model_type == 'unregularized':
                        beta_val = cached_model.params[or_feature]
                    else:  # regularized
                        beta_val = cached_model.coef_[0][feature_idx]

                    
                    # Handle scaling
                    if cached_scaling != "No Scaling" and cached_scaler is not None:
                        # Create input arrays for scaling
                        input_a_arr = np.zeros((1, 2))
                        input_b_arr = np.zeros((1, 2))
                        input_a_arr[0, feature_idx] = or_a
                        input_b_arr[0, feature_idx] = or_b

                        # Scale the inputs
                        scaled_a = cached_scaler.transform(input_a_arr)[0, feature_idx]
                        scaled_b = cached_scaler.transform(input_b_arr)[0, feature_idx]
                    else:
                        scaled_a = or_a
                        scaled_b = or_b

                    # Calculate odds ratio
                    odds_ratio = np.exp(beta_val * (scaled_a - scaled_b))

                    # Feature display name
                    feature_display = r"\text{Age}" if or_feature == "Age" else r"\text{Annual Income}"

                    # Display aligned formulas based on scaling
                    if cached_scaling != "No Scaling" and cached_scaler is not None:
                        st.latex(r"""
                        \begin{aligned}
                        \text{Odds Ratio} &= \frac{\text{Odds}(x_{%s}=a)}{\text{Odds}(x_{%s}=b)} \\[8pt]
                        &= \text{exp}(\beta \cdot (a_{\text{scaled}} - b_{\text{scaled}})) \\[8pt]
                        &= \text{exp}(\beta_{%s} \cdot (%s - %s)) \\[8pt]
                        &= \text{exp}(%s \cdot %s) \\[8pt]
                        &= %s
                        \end{aligned}
                        """ % (
                            feature_display,
                            feature_display,
                            feature_display,
                            format_number(scaled_a),
                            format_number(scaled_b),
                            format_number(beta_val),
                            format_number(scaled_a - scaled_b),
                            format_number(odds_ratio)
                        ))
                        # Format caption based on feature type
                        if or_feature == "Age":
                            a_display = f"{or_a:.0f}"
                            b_display = f"{or_b:.0f}"
                        else:  # Annual_Income_IDR
                            # Display in Juta (millions) for readability
                            a_display = f"{or_a:,.0f}"
                            b_display = f"{or_b:,.0f}"

                        st.caption(f":red[**{cached_scaling} applied: a={a_display} → {format_number(scaled_a)}, b={b_display} → {format_number(scaled_b)}**]")

                        st.info(f"""
                        **Note:**
                        - {r"$\text{Odds Ratio} = $"} {r"$\text{exp}(\beta \cdot (a - b))$"} is obtained from and is valid for **Logistic Regression**.
                        - The formula assumes that all other variables in the model are held constant (i.e., have the same values when comparing scenarios {"$a$"} and {"$b$"}).
                        """)
                    else:
                        # Format values nicely for display
                        if or_feature == "Age":
                            # Age: show as integers
                            a_display_formula = f"{int(scaled_a)}"
                            b_display_formula = f"{int(scaled_b)}"
                            diff_display = f"{int(scaled_a - scaled_b)}"

                            # Format beta coefficient - use scientific notation for very small values
                            beta_display = format_number(beta_val)
                        else:
                            # Annual Income: show with coma separator for readability
                            # a_juta = scaled_a
                            # b_juta = scaled_b
                            # diff_juta = (scaled_a - scaled_b) / 1e6
                            a_display_formula = f"{scaled_a:,.0f}" # + r"\text{ Juta}"
                            b_display_formula = f"{scaled_b:,.0f}" # + r"\text{ Juta}"
                            # Format beta coefficient - use scientific notation for very small values
                            beta_display = format_number(beta_val)

                            # Only convert to LaTeX scientific notation if 'e' is present
                            if 'e' in beta_display:
                                # Split on 'e' to get mantissa and exponent
                                mantissa, exponent = beta_display.split('e')
                                exponent = int(exponent)  # Remove leading zeros
                                beta_display = f"{mantissa} \\times 10^{{{exponent}}}"
                            # else: keep beta_display as is (regular decimal format)

                            diff_display = f"{scaled_a - scaled_b:,.0f}" # + r"\text{ Juta}"

                        # Calculate the product for display
                        product_val = beta_val * (scaled_a - scaled_b)
                        product_display = format_number(product_val)

                        st.latex(r"""
                        \begin{aligned}
                        \text{Odds Ratio} &= \frac{\text{Odds}(x_{%s}=a)}{\text{Odds}(x_{%s}=b)} \\[8pt]
                        &= \text{exp}(\beta \cdot (a - b)) \\[8pt]
                        &= \text{exp}(\beta_{%s} \cdot (%s - %s)) \\[8pt]
                        &= \text{exp}(%s \cdot %s) \\[8pt]
                        &= %s
                        \end{aligned}
                        """ % (
                            feature_display,
                            feature_display,
                            feature_display,
                            a_display_formula,
                            b_display_formula,
                            beta_display,
                            diff_display,
                            format_number(odds_ratio)
                        ))

                        st.info(f"""
                        **Note:**
                        - {r"$\text{Odds Ratio} = $"} {r"$\text{exp}(\beta \cdot (a - b))$"} is obtained from and is valid for **Logistic Regression**.
                        - The formula assumes that all other variables in the model are held constant (i.e., have the same values when comparing scenarios {"$a$"} and {"$b$"}).
                        """)


                # Interpretation text box
                st.markdown("---")
                st.markdown("**Interpretation:**")

                feature_name = "Age" if or_feature == "Age" else "Annual Income"

                if or_feature == "Age":
                    a_display = f"{or_a:.0f} years"
                    b_display = f"{or_b:.0f} years"
                else:
                    a_display = f"IDR {or_a:,.0f}"
                    b_display = f"IDR {or_b:,.0f}"

                if odds_ratio > 1:
                    change_text = f"**{format_number(odds_ratio)}** times higher"
                    direction = "increases"
                elif odds_ratio < 1:
                    change_text = f"**{format_number(1/odds_ratio)}** times lower"
                    direction = "decreases"
                else:
                    change_text = "the same"
                    direction = "does not change"

                interpretation = f"""
                The odds of Disease Risk at {feature_name} = {a_display} are
                {change_text} than the odds at {feature_name} = {b_display}.
                """

                st.info(interpretation)

            # TAB 2: SIGMOID CURVE
            with tab2:
                # Two columns: parameter input on left, plot on right
                col_param, col_plot = st.columns([0.40, 1])

                with col_param:
                    st.markdown("**Parameter Settings:**")

                    # Slider for parameter feature
                    if or_feature == "Age":
                        # Age is the parameter, Income is the x-axis variable
                        param_value = st.slider(
                            "Age (held constant):",
                            min_value=int(X['Age'].min()),
                            max_value=int(X['Age'].max()),
                            step=1,
                            key="sigmoid_age_param"
                        )

                        # Determine variable feature details
                        var_feature_name = "Annual Income"
                        var_feature_col = "Annual_Income_IDR"
                        var_feature_idx = 1
                        param_idx = 0
                    else:
                        # Income is the parameter, Age is the x-axis variable
                        param_value_millions = st.slider(
                            "Annual Income (held constant) - Juta:",
                            min_value=float(X['Annual_Income_IDR'].min() / 1e6),
                            max_value=float(X['Annual_Income_IDR'].max() / 1e6),
                            step=1.0,
                            value=75.0,
                            key="sigmoid_income_param"
                        )
                        st.caption("(in millions IDR)")

                        param_value = param_value_millions * 1e6

                        # Determine variable feature details
                        var_feature_name = "Age"
                        var_feature_col = "Age"
                        var_feature_idx = 0
                        param_idx = 1

                    st.markdown("**β Coefficient Values:**")

                    # Get coefficient values
                    if model_type == 'unregularized':
                        beta_0_val = cached_model.params['const']
                        beta_age_val = cached_model.params['Age']
                        beta_income_val = cached_model.params['Annual_Income_IDR']
                    else:  # regularized
                        beta_0_val = cached_model.intercept_[0]
                        beta_age_val = cached_model.coef_[0][0]
                        beta_income_val = cached_model.coef_[0][1]

                    # Build HTML table
                    html_content = f"""
                    <style>
                    .beta-table {{
                        font-size: 18px !important;
                        border-collapse: collapse !important;
                        border: 1px solid #ddd !important;
                    }}
                    .beta-table th {{
                        font-size: 18px !important;
                        font-weight: bold !important;
                        padding: 10px 15px !important;
                        text-align: left !important;
                        border: 1px solid #ddd !important;
                        background-color: #f8f9fa !important;
                    }}
                    .beta-table td {{
                        font-size: 18px !important;
                        padding: 10px 15px !important;
                        border: 1px solid #ddd !important;
                    }}
                    </style>
                    <table class="beta-table">
                        <thead>
                            <tr>
                                <th>Coefficient</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>β<sub>0</sub></td>
                                <td>{format_number(beta_0_val)}</td>
                            </tr>
                            <tr>
                                <td>β<sub>Age</sub></td>
                                <td>{format_number(beta_age_val)}</td>
                            </tr>
                            <tr>
                                <td>β<sub>Annual Income</sub></td>
                                <td>{format_number(beta_income_val)}</td>
                            </tr>
                        </tbody>
                    </table>
                    """
                    components.html(html_content, height=200)

                    # Get coefficients
                    if model_type == 'unregularized':
                        beta_0 = cached_model.params['const']
                        beta_age = cached_model.params['Age']
                        beta_income = cached_model.params['Annual_Income_IDR']
                    else:
                        beta_0 = cached_model.intercept_[0]
                        beta_age = cached_model.coef_[0][0]
                        beta_income = cached_model.coef_[0][1]

                    # Handle scaling for parameter value
                    if cached_scaling != "No Scaling" and cached_scaler is not None:
                        # Create input array for scaling
                        temp_input = np.zeros((1, 2))
                        temp_input[0, param_idx] = param_value
                        scaled_param = cached_scaler.transform(temp_input)[0, param_idx]
                    else:
                        scaled_param = param_value

                    # Calculate new intercept based on which feature is parameter
                    if or_feature == "Age":
                        constant_term = beta_0 + beta_age * scaled_param
                        var_beta = beta_income
                        var_latex_name = r"\text{Annual Income}"
                        var_subscript = r"x_\text{Annual Income}"
                        if cached_scaling != "No Scaling" and cached_scaler is not None:
                            var_subscript_scaled = r"x_\text{Annual Income (scaled)}"
                        else:
                            var_subscript_scaled = var_subscript
                    else:
                        constant_term = beta_0 + beta_income * scaled_param
                        var_beta = beta_age
                        var_latex_name = r"\text{Age}"
                        var_subscript = r"x_\text{Age}"
                        if cached_scaling != "No Scaling" and cached_scaler is not None:
                            var_subscript_scaled = r"x_\text{Age (scaled)}"
                        else:
                            var_subscript_scaled = var_subscript

                            

                with col_plot:
                    st.markdown("**Sigmoid Curve:**")

                    # Display simplified formula
                    # st.markdown("**Formula:**")
                    if or_feature == "Age":
                        st.latex(rf"""
                        P(y=1|x_\text{{Age}}={param_value:.0f}) = \frac{{\exp({format_number_for_latex(constant_term)} + {format_operator_operand(var_beta)} \cdot {var_subscript_scaled})}}{{1 + \exp({format_number_for_latex(constant_term)} + {format_operator_operand(var_beta)} \cdot {var_subscript_scaled})}}
                        """)
                    else:
                        st.latex(rf"""
                        P(y=1|x_\text{{Annual Income}}={param_value / 1e6:,.0f}\  \text{{Juta}}) = \frac{{\exp({format_number_for_latex(constant_term)} + {format_operator_operand(var_beta)} \cdot {var_subscript_scaled})}}{{1 + \exp({format_number_for_latex(constant_term)} + {format_operator_operand(var_beta)} \cdot {var_subscript_scaled})}}
                        """)

                    # Display compact explanatory note
                    if or_feature == "Age":
                        param_display_text = f"{param_value:.0f}"
                    else:
                        param_display_text = f"IDR {param_value:,.0f}"

                    if cached_scaling != "No Scaling" and cached_scaler is not None:
                        scaling_note = f" (scaled to {format_number(scaled_param)})"
                    else:
                        scaling_note = ""

                    st.info(f"**New Intercept:** {format_number(constant_term)} — calculated with {or_feature.replace('_', ' ')} held constant at {param_display_text}{scaling_note}")

                    st.markdown("---")

                    # Generate x values for the variable feature
                    x_min = X[var_feature_col].min()
                    x_max = X[var_feature_col].max()
                    x_range = x_max - x_min
                    x_values = np.linspace(x_min - 0.05 * x_range, x_max + 0.05 * x_range, 200)

                    # Create input arrays for prediction
                    prediction_inputs = np.zeros((len(x_values), 2))
                    prediction_inputs[:, var_feature_idx] = x_values
                    prediction_inputs[:, param_idx] = param_value

                    # Apply scaling if needed
                    if cached_scaling != "No Scaling" and cached_scaler is not None:
                        prediction_inputs_scaled = cached_scaler.transform(prediction_inputs)
                        # Extract scaled x-values for plotting
                        x_values_plot = prediction_inputs_scaled[:, var_feature_idx]
                    else:
                        prediction_inputs_scaled = prediction_inputs
                        x_values_plot = x_values

                    # Calculate probabilities
                    if model_type == 'unregularized':
                        # Add constant term for statsmodels
                        prediction_inputs_with_const = sm.add_constant(prediction_inputs_scaled, has_constant='add')
                        probabilities = cached_model.predict(prediction_inputs_with_const)
                    else:
                        # Use sklearn predict_proba
                        probabilities = cached_model.predict_proba(prediction_inputs_scaled)[:, 1]

                    # Create the plot
                    fig_sigmoid, ax_sigmoid = plt.subplots(figsize=(10, 6))

                    # Plot sigmoid curve
                    ax_sigmoid.plot(x_values_plot, probabilities, 'b-', linewidth=2, label='Sigmoid Curve')

                    # Add horizontal line at P=0.5
                    ax_sigmoid.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Decision Threshold (P=0.5)')

                    # Format axes
                    if var_feature_col == "Annual_Income_IDR":
                        if cached_scaling != "No Scaling" and cached_scaler is not None:
                            ax_sigmoid.set_xlabel('Annual Income (Scaled)', fontsize=12)
                            # No formatter needed for scaled values
                        else:
                            ax_sigmoid.set_xlabel('Annual Income (IDR)', fontsize=12)
                            ax_sigmoid.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f} Juta'))
                    else:
                        if cached_scaling != "No Scaling" and cached_scaler is not None:
                            ax_sigmoid.set_xlabel('Age (Scaled)', fontsize=12)
                        else:
                            ax_sigmoid.set_xlabel('Age', fontsize=12)

                    ax_sigmoid.set_ylabel('P(Disease Risk = 1)', fontsize=12)

                    # Format parameter value for title
                    if or_feature == "Age":
                        param_display = f"{param_value:.0f}"
                    else:
                        param_display = f"{param_value:,.0f}"

                    ax_sigmoid.set_title(f'Probability of Disease Risk vs {var_feature_name}\n({or_feature.replace("_", " ")} held constant at {param_display})', fontsize=14)
                    ax_sigmoid.set_ylim(-0.05, 1.05)
                    ax_sigmoid.grid(True, alpha=0.3)
                    ax_sigmoid.legend(loc='best')

                    st.pyplot(fig_sigmoid)

                
                # Display expander with detailed calculation steps
                with st.expander("Show how the new intercept is calculated"):
                    # STEP 1: Logistic Regression Formula
                    st.markdown("**> STEP 1**")
                    st.markdown("**Logistic Regression Formula:**")
                    st.latex(r"""
                    \begin{aligned}
                    u(x_\text{Age},\ x_\text{Annual Income}) &= \beta_0 + \beta_\text{Age} \cdot x_\text{Age} + \beta_\text{Annual Income} \cdot x_\text{Annual Income} \\\\
                    P(y=1) &= \frac{\text{exp}(u)}{1 + \text{exp}(u)}
                    \end{aligned}
                    """)

                    st.markdown("---")

                    # STEP 2: Plug in ALL numbers
                    st.markdown("**> STEP 2**")
                    st.markdown("**Plug in ALL numbers:**")

                    if or_feature == "Age":
                        # Age is parameter, Income is variable
                        if cached_scaling != "No Scaling" and cached_scaler is not None:
                            # With scaling
                            st.latex(rf"""
                            u(x_\text{{Age}} = {param_value:.0f},\ x_\text{{Annual Income}}) = {format_number_for_latex(beta_0)} + {format_operator_operand(beta_age)} \cdot {format_operator_operand(scaled_param)} + {format_operator_operand(beta_income)} \cdot x_\text{{Annual Income (scaled)}}
                            """)
                            st.caption(f":red[**{cached_scaling} applied: Age {param_value:.0f} → {format_number(scaled_param)}**]")

                            # Rearrangement step (for Age)
                            st.markdown("**Rearrange (Group Constant Terms):**")
                            st.latex(rf"""
                            u(x_\text{{Age}} = {param_value:.0f},\ x_\text{{Annual Income}}) = ({format_number_for_latex(beta_0)} + {format_operator_operand(beta_age)} \cdot {format_operator_operand(scaled_param)}) + {format_operator_operand(beta_income)} \cdot x_\text{{Annual Income (scaled)}}
                            """)
                        else:
                            # No scaling
                            st.latex(rf"""
                            u(x_\text{{Age}} = {param_value:.0f},\ x_\text{{Annual Income}}) = {format_number_for_latex(beta_0)} + {format_operator_operand(beta_age)} \cdot {param_value:.0f} + {format_operator_operand(beta_income)} \cdot x_\text{{Annual Income}}
                            """)

                            # Rearrangement step (for Age)
                            st.markdown("**Rearrange (Group Constant Terms):**")
                            st.latex(rf"""
                            u(x_\text{{Age}} = {param_value:.0f},\ x_\text{{Annual Income}}) = ({format_number_for_latex(beta_0)} + {format_operator_operand(beta_age)} \cdot {param_value:.0f}) + {format_operator_operand(beta_income)} \cdot x_\text{{Annual Income}}
                            """)
                    else:
                        # Income is parameter, Age is variable
                        if cached_scaling != "No Scaling" and cached_scaler is not None:
                            # With scaling - show rearrangement
                            st.latex(rf"""
                            u(x_\text{{Age}},\ x_\text{{Annual Income}} = {param_value:,.0f}) = {format_number_for_latex(beta_0)} + {format_operator_operand(beta_age)} \cdot x_\text{{Age (scaled)}} + {format_operator_operand(beta_income)} \cdot {format_operator_operand(scaled_param)}
                            """)
                            st.caption(f":red[**{cached_scaling} applied: Income {param_value:,.0f} → {format_number(scaled_param)}**]")

                            # Rearrangement step (only for Annual Income)
                            # st.markdown("**Since** $\\beta_{\\text{Annual Income}} \\cdot x_{\\text{Annual Income}}$ **is constant and separated from** $\\beta_{\\text{Age}} \\cdot x_{\\text{Age}}$**, rearrange to put** $\\beta_0$ **and the constant term together:**")
                            st.markdown("**Rearrange (Group Constant Terms):**")
                            st.latex(rf"""
                            u(x_\text{{Age}},\ x_\text{{Annual Income}} = {param_value:,.0f}) = ({format_number_for_latex(beta_0)} + {format_operator_operand(beta_income)} \cdot {format_operator_operand(scaled_param)}) + {format_operator_operand(beta_age)} \cdot x_\text{{Age (scaled)}}
                            """)
                        else:
                            # No scaling - show rearrangement
                            st.latex(rf"""
                            u(x_\text{{Age}},\ x_\text{{Annual Income}} = {param_value:,.0f}) = {format_number_for_latex(beta_0)} + {format_operator_operand(beta_age)} \cdot x_\text{{Age}} + {format_operator_operand(beta_income)} \cdot {param_value:,.0f}
                            """)

                            # Rearrangement step (only for Annual Income)
                            # st.markdown("**Since** $\\beta_{\\text{Annual Income}} \\cdot x_{\\text{Annual Income}}$ **is constant and separated from** $\\beta_{\\text{Age}} \\cdot x_{\\text{Age}}$**, rearrange to put** $\\beta_0$ **and the constant term together:**")
                            st.markdown("**Rearrange (Group Constant Terms):**")
                            st.latex(rf"""
                            u(x_\text{{Age}},\ x_\text{{Annual Income}} = {param_value:,.0f}) = ({format_number_for_latex(beta_0)} + {format_operator_operand(beta_income)} \cdot {param_value:,.0f}) + {format_operator_operand(beta_age)} \cdot x_\text{{Age}}
                            """)

                    st.markdown("---")

                    # STEP 3: Calculate New Intercept
                    st.markdown("**> STEP 3**")
                    st.markdown("**Calculate New Intercept:**")

                    if or_feature == "Age":
                        if cached_scaling != "No Scaling" and cached_scaler is not None:
                            st.latex(rf"""
                            \text{{New Intercept}} = {format_number_for_latex(beta_0)} + {format_operator_operand(beta_age)} \cdot {format_operator_operand(scaled_param)} = {format_number_for_latex(constant_term)}
                            """)
                        else:
                            st.latex(rf"""
                            \text{{New Intercept}} = {format_number_for_latex(beta_0)} + {format_operator_operand(beta_age)} \cdot {param_value:.0f} = {format_number_for_latex(constant_term)}
                            """)
                    else:
                        if cached_scaling != "No Scaling" and cached_scaler is not None:
                            st.latex(rf"""
                            \text{{New Intercept}} = {format_number_for_latex(beta_0)} + {format_operator_operand(beta_income)} \cdot {format_operator_operand(scaled_param)} = {format_number_for_latex(constant_term)}
                            """)
                        else:
                            st.latex(rf"""
                            \text{{New Intercept}} = {format_number_for_latex(beta_0)} + {format_operator_operand(beta_income)} \cdot {param_value:,.0f} = {format_number_for_latex(constant_term)}
                            """)

                    st.markdown("---")

                    # STEP 4: Final Formula
                    st.markdown("**> STEP 4**")

                    if or_feature == "Age":
                        # Age is parameter, Income is variable
                        if cached_scaling != "No Scaling" and cached_scaler is not None:
                            st.latex(rf"""
                            \begin{{aligned}}
                            u(x_\text{{Age}} = {param_value:.0f},\ x_\text{{Annual Income}}) &= {format_number_for_latex(constant_term)} + {format_operator_operand(beta_income)} \cdot x_\text{{Annual Income (scaled)}} \\\\[8pt]
                            P(y=1) &= \frac{{\text{{exp}}(u)}}{{1 + \text{{exp}}(u)}} \\\\[8pt]
                            P(y=1 \mid x_\text{{Age}} = {param_value:.0f}) &= \frac{{\text{{exp}}({format_number_for_latex(constant_term)} + {format_operator_operand(beta_income)} \cdot x_\text{{Annual Income (scaled)}})}}{{1 + \text{{exp}}({format_number_for_latex(constant_term)} + {format_operator_operand(beta_income)} \cdot x_\text{{Annual Income (scaled)}})}}
                            \end{{aligned}}
                            """)
                        else:
                            st.latex(rf"""
                            \begin{{aligned}}
                            u(x_\text{{Age}} = {param_value:.0f},\ x_\text{{Annual Income}}) &= {format_number_for_latex(constant_term)} + {format_operator_operand(beta_income)} \cdot x_\text{{Annual Income}} \\\\[8pt]
                            P(y=1) &= \frac{{\text{{exp}}(u)}}{{1 + \text{{exp}}(u)}} \\\\[8pt]
                            P(y=1 \mid x_\text{{Age}} = {param_value:.0f}) &= \frac{{\text{{exp}}({format_number_for_latex(constant_term)} + {format_operator_operand(beta_income)} \cdot x_\text{{Annual Income}})}}{{1 + \text{{exp}}({format_number_for_latex(constant_term)} + {format_operator_operand(beta_income)} \cdot x_\text{{Annual Income}})}}
                            \end{{aligned}}
                            """)
                    else:
                        # Income is parameter, Age is variable
                        if cached_scaling != "No Scaling" and cached_scaler is not None:
                            st.latex(rf"""
                            \begin{{aligned}}
                            u(x_\text{{Age}},\ x_\text{{Annual Income}} = {param_value:,.0f}) &= {format_number_for_latex(constant_term)} + {format_operator_operand(beta_age)} \cdot x_\text{{Age (scaled)}} \\\\[8pt]
                            P(y=1) &= \frac{{\text{{exp}}(u)}}{{1 + \text{{exp}}(u)}} \\\\[8pt]
                            P(y=1 \mid x_\text{{Annual Income}} = {param_value:,.0f}) &= \frac{{\text{{exp}}({format_number_for_latex(constant_term)} + {format_operator_operand(beta_age)} \cdot x_\text{{Age (scaled)}})}}{{1 + \text{{exp}}({format_number_for_latex(constant_term)} + {format_operator_operand(beta_age)} \cdot x_\text{{Age (scaled)}})}}
                            \end{{aligned}}
                            """)
                        else:
                            st.latex(rf"""
                            \begin{{aligned}}
                            u(x_\text{{Age}},\ x_\text{{Annual Income}} = {param_value:,.0f}) &= {format_number_for_latex(constant_term)} + {format_operator_operand(beta_age)} \cdot x_\text{{Age}} \\\\[8pt]
                            P(y=1) &= \frac{{\text{{exp}}(u)}}{{1 + \text{{exp}}(u)}} \\\\[8pt]
                            P(y=1 \mid x_\text{{Annual Income}} = {param_value:,.0f}) &= \frac{{\text{{exp}}({format_number_for_latex(constant_term)} + {format_operator_operand(beta_age)} \cdot x_\text{{Age}})}}{{1 + \text{{exp}}({format_number_for_latex(constant_term)} + {format_operator_operand(beta_age)} \cdot x_\text{{Age}})}}
                            \end{{aligned}}
                            """)


else:
    st.error("Unable to load the dataset. Please check the file path and try again.")
