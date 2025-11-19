import streamlit as st
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
        n_neighbors = st.slider("Number of Neighbors (n_neighbors)", 1, 100, 3)
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
        # st.latex(r"P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_{\text{Age}} + \beta_2 X_{\text{Income}})}}")
        st.latex(r"P(Y=1|X) =\frac{\text{exp}({\beta_0 + \beta_{\text{Age}} \cdot x_{\text{Age}} + \beta_{\text{Annual Income}} \cdot x_{\text{Annual Income}} })}{1 + \text{exp}({\beta_0 + \beta_{\text{Age}} \cdot x_{\text{Age}} + \beta_{\text{Annual Income}} \cdot x_{\text{Annual Income}} })}")
        

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

            # Display results
            st.header("📊 Model Performance")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Train Accuracy", numerize.numerize(train_accuracy, 4))
            with col2:
                st.metric("Test Accuracy", numerize.numerize(test_accuracy, 4))

            # Beta coefficients bar plot
            st.header("📊 β Coefficient Values")

            coef_names = ['β₀ (Intercept)', 'β_Age', 'β_Annual_Income']
            coef_values = result.params.values

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

            # ax_coef.set_title('Logistic Regression Coefficients', fontsize=16)
            # ax_coef.set_xlabel('Coefficient', fontsize=14)
            ax_coef.set_ylabel('Value', fontsize=14)
            ax_coef.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)

            ax_coef.spines['top'].set_visible(False)
            ax_coef.spines['right'].set_visible(False)

            # Extend y-axis limits to give room for annotations
            ymin, ymax = ax_coef.get_ylim()
            y_range = ymax - ymin
            ax_coef.set_ylim(ymin - 0.1 * y_range, ymax + 0.1 * y_range)  # Add 10% padding on each side

            st.pyplot(fig_coef)

            # Odds Ratio Section
            with st.expander("📊 Odds Ratio Analysis"):
                # Display beta coefficients table
                st.markdown("**β Coefficient Values:**")
                beta_table = pd.DataFrame({
                    'Coefficient': ['β₀ (Intercept)', 'β_Age', 'β_Annual_Income'],
                    'Value': [format_number(result.params['const']),
                             format_number(result.params['Age']),
                             format_number(result.params['Annual_Income_IDR'])]
                })

                # Display as styled HTML table
                st.markdown(
                    beta_table.to_html(index=False, classes='beta-table'),
                    unsafe_allow_html=True
                )
                st.markdown("""
                <style>
                .beta-table {
                    font-size: 18px !important;
                    margin-bottom: 10px;
                }
                .beta-table th {
                    font-size: 18px !important;
                    font-weight: bold !important;
                    padding: 10px 15px !important;
                    text-align: left !important;
                }
                .beta-table td {
                    font-size: 18px !important;
                    padding: 10px 15px !important;
                }
                </style>
                """, unsafe_allow_html=True)

                st.markdown("---")

                # Two columns: inputs on left, formulas on right
                col_input, col_formula = st.columns([1, 2])

                with col_input:
                    st.markdown("**Input Values:**")

                    # Feature selection
                    or_feature = st.selectbox(
                        "Select Feature:",
                        ["Age", "Annual_Income_IDR"],
                        key="or_feature_unreg"
                    )

                    # Get reasonable defaults based on feature
                    if or_feature == "Age":
                        default_a, default_b = 50.0, 40.0
                        step_val = 1.0
                    else:
                        default_a, default_b = 100000000.0, 50000000.0
                        step_val = 1000000.0

                    # Numerator and denominator inputs
                    or_a = st.number_input(
                        "Numerator (a):",
                        value=default_a,
                        step=step_val,
                        key="or_a_unreg"
                    )
                    or_b = st.number_input(
                        "Denominator (b):",
                        value=default_b,
                        step=step_val,
                        key="or_b_unreg"
                    )

                with col_formula:
                    st.markdown("**Odds Ratio Formula:**")

                    # Get the beta coefficient for selected feature
                    beta_val = result.params[or_feature]
                    feature_idx = 0 if or_feature == "Age" else 1

                    # Handle scaling
                    if logreg_scaling != "No Scaling":
                        # Create input arrays for scaling
                        input_a_arr = np.zeros((1, 2))
                        input_b_arr = np.zeros((1, 2))
                        input_a_arr[0, feature_idx] = or_a
                        input_b_arr[0, feature_idx] = or_b

                        # Scale the inputs
                        scaled_a = scaler.transform(input_a_arr)[0, feature_idx]
                        scaled_b = scaler.transform(input_b_arr)[0, feature_idx]
                    else:
                        scaled_a = or_a
                        scaled_b = or_b

                    # Calculate odds ratio
                    odds_ratio = np.exp(beta_val * (scaled_a - scaled_b))

                    # Feature display name
                    feature_display = r"\text{Age}" if or_feature == "Age" else r"\text{Annual Income}"

                    # Display aligned formulas based on scaling
                    if logreg_scaling != "No Scaling":
                        st.latex(r"""
                        \begin{aligned}
                        \text{Odds Ratio} &= \frac{\text{Odds}(X=a)}{\text{Odds}(X=b)} \\[8pt]
                        &= \text{exp}(\beta \cdot (a_{\text{scaled}} - b_{\text{scaled}})) \\[8pt]
                        &= \text{exp}(\beta_{%s} \cdot (%s - %s)) \\[8pt]
                        &= \text{exp}(%s \cdot %s) \\[8pt]
                        &= %s
                        \end{aligned}
                        """ % (
                            feature_display,
                            format_number(scaled_a),
                            format_number(scaled_b),
                            format_number(beta_val),
                            format_number(scaled_a - scaled_b),
                            format_number(odds_ratio)
                        ))
                        st.caption(f"*Scaling applied ({logreg_scaling}): a={format_number(or_a)} → {format_number(scaled_a)}, b={format_number(or_b)} → {format_number(scaled_b)}*")
                    else:
                        st.latex(r"""
                        \begin{aligned}
                        \text{Odds Ratio} &= \frac{\text{Odds}(X=a)}{\text{Odds}(X=b)} \\[8pt]
                        &= \text{exp}(\beta \cdot (a - b)) \\[8pt]
                        &= \text{exp}(\beta_{%s} \cdot (%s - %s)) \\[8pt]
                        &= \text{exp}(%s \cdot %s) \\[8pt]
                        &= %s
                        \end{aligned}
                        """ % (
                            feature_display,
                            format_number(scaled_a),
                            format_number(scaled_b),
                            format_number(beta_val),
                            format_number(scaled_a - scaled_b),
                            format_number(odds_ratio)
                        ))

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

            # LLR-Ratio Test
            st.header("📈 Likelihood Ratio (LLR) Test")

            st.markdown("**Hypotheses:**")
            st.latex(r"H_0: \beta_{\text{Age}} = \beta_{\text{Annual Income}} = 0")
            st.latex(r"H_A: \text{At least one } \beta_i \neq 0")

            llr_pvalue = result.llr_pvalue

            col1, col2 = st.columns(2)
            with col1:
                st.metric("LLR Test p-value", f"{llr_pvalue:.2e}")
            with col2:
                if llr_pvalue < 0.05:
                    st.success("**Conclusion:** Reject H₀ at α = 0.05. The model is statistically significant.")
                else:
                    st.warning("**Conclusion:** Fail to reject H₀ at α = 0.05. The model is not statistically significant.")

            # Wald Test for each coefficient
            st.header("📊 Wald Test for Individual Coefficients")

            # Get confidence intervals
            conf_int = result.conf_int(alpha=0.05)

            # Create three columns for each coefficient
            col1, col2, col3 = st.columns(3)

            param_names = ['const', 'Age', 'Annual_Income_IDR']
            display_names = ['β₀ (Intercept)', 'β_Age', 'β_Annual_Income']
            latex_names = [r'\beta_0', r'\beta_{\text{Age}}', r'\beta_{\text{Annual Income}}']

            columns = [col1, col2, col3]

            for i, (param, display_name, latex_name, col) in enumerate(zip(param_names, display_names, latex_names, columns)):
                with col:
                    st.subheader(f"${latex_name}$")

                    # Beta value
                    beta_value = result.params[param]
                    st.metric("Value", format_number(beta_value))

                    # Hypotheses
                    st.markdown("**Wald Test Hypotheses:**")
                    st.latex(f"H_0: {latex_name} = 0")
                    st.latex(f"H_A: {latex_name} \\neq 0")

                    # P-value
                    p_value = result.pvalues[param]
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

            # Display results
            st.header("📊 Model Performance")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Train Accuracy", numerize.numerize(train_accuracy, 4))
            with col2:
                st.metric("Test Accuracy", numerize.numerize(test_accuracy, 4))

            # Model-specific visualizations
            if model_option == "K-Nearest Neighbors":
                st.header("🎯 KNN Decision Boundary on Train Data")

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

                ax.set_xlabel('Age')
                ax.set_ylabel('Annual Income (IDR)')
                ax.set_title(f'KNN Decision Boundary (k={n_neighbors})')

                # Format y-axis for better readability of large numbers
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f} Juta'))
                ax.legend(title='Disease Risk', bbox_to_anchor=(1.05, 1), loc='upper left')

                st.pyplot(fig)

            elif model_option == "Decision Tree":
                st.header("🌳 Decision Tree Visualization on Train Data")

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

            else:  # Regularized Logistic Regression
                # Beta coefficients bar plot
                st.header("📊 β Coefficient Values")

                coef_names = ['β₀ (Intercept)', 'β_Age', 'β_Annual_Income']
                coef_values = [model.intercept_[0], model.coef_[0][0], model.coef_[0][1]]

                coef_df = pd.DataFrame({
                    'Coefficient': coef_names,
                    'Value': coef_values
                })

                fig_coef, ax_coef = plt.subplots(figsize=(10, 6))

                # Create bar plot with seaborn
                bars = sns.barplot(data=coef_df, x='Coefficient', y='Value',
                                 color='blue', ax=ax_coef)

                # Annotate bars with values
                # for i, (idx, row) in enumerate(coef_df.iterrows()):
                #     value = row['Value']
                #     # Position annotation above positive bars, below negative bars
                #     va = 'bottom' if value >= 0 else 'top'
                #     ax_coef.annotate(format_number(value),
                #                 xy=(i, value),
                #                 ha='center', va=va,
                #                 fontsize=12, fontweight='bold')

                # ax_coef.set_title(f'Logistic Regression Coefficients ({penalty} Regularization, C={c_value})', fontsize=16)
                # ax_coef.set_xlabel('Coefficient', fontsize=14)
                # ax_coef.set_ylabel('Value', fontsize=14)
                # ax_coef.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

                # st.pyplot(fig_coef)

                # Annotate bars with values
                for i, (idx, row) in enumerate(coef_df.iterrows()):
                    value = row['Value']
                    # Position annotation above positive bars, below negative bars
                    va = 'bottom' if value >= 0 else 'top'
                    ax_coef.annotate(format_number(value),
                                xy=(i, value),
                                ha='center', va=va,
                                fontsize=12, fontweight='bold')

                ax_coef.set_ylabel('Value', fontsize=14)
                ax_coef.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)

                ax_coef.spines['top'].set_visible(False)
                ax_coef.spines['right'].set_visible(False)

                # Extend y-axis limits to give room for annotations
                ymin, ymax = ax_coef.get_ylim()
                y_range = ymax - ymin
                ax_coef.set_ylim(ymin - 0.1 * y_range, ymax + 0.1 * y_range)  # Add 10% padding on each side

                st.pyplot(fig_coef)

                # Odds Ratio Section
                with st.expander("📊 Odds Ratio Analysis"):
                    # Display beta coefficients table
                    st.markdown("**β Coefficient Values:**")
                    beta_table = pd.DataFrame({
                        'Coefficient': ['β₀ (Intercept)', 'β_Age', 'β_Annual_Income'],
                        'Value': [format_number(model.intercept_[0]),
                                 format_number(model.coef_[0][0]),
                                 format_number(model.coef_[0][1])]
                    })

                    # Display as styled HTML table
                    st.markdown(
                        beta_table.to_html(index=False, classes='beta-table'),
                        unsafe_allow_html=True
                    )
                    st.markdown("""
                    <style>
                    .beta-table {
                        font-size: 18px !important;
                        margin-bottom: 10px;
                    }
                    .beta-table th {
                        font-size: 18px !important;
                        font-weight: bold !important;
                        padding: 10px 15px !important;
                        text-align: left !important;
                    }
                    .beta-table td {
                        font-size: 18px !important;
                        padding: 10px 15px !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    st.markdown("---")

                    # Two columns: inputs on left, formulas on right
                    col_input, col_formula = st.columns([1, 2])

                    with col_input:
                        st.markdown("**Input Values:**")

                        # Feature selection
                        or_feature_reg = st.selectbox(
                            "Select Feature:",
                            ["Age", "Annual_Income_IDR"],
                            key="or_feature_reg"
                        )

                        # Get reasonable defaults based on feature
                        if or_feature_reg == "Age":
                            default_a, default_b = 50.0, 40.0
                            step_val = 1.0
                        else:
                            default_a, default_b = 100000000.0, 50000000.0
                            step_val = 1000000.0

                        # Numerator and denominator inputs
                        or_a_reg = st.number_input(
                            "Numerator (a):",
                            value=default_a,
                            step=step_val,
                            key="or_a_reg"
                        )
                        or_b_reg = st.number_input(
                            "Denominator (b):",
                            value=default_b,
                            step=step_val,
                            key="or_b_reg"
                        )

                    with col_formula:
                        st.markdown("**Odds Ratio Formula:**")

                        # Get the beta coefficient for selected feature
                        feature_idx = 0 if or_feature_reg == "Age" else 1
                        beta_val = model.coef_[0][feature_idx]

                        # Handle scaling
                        if logreg_scaling != "No Scaling":
                            # Create input arrays for scaling
                            input_a_arr = np.zeros((1, 2))
                            input_b_arr = np.zeros((1, 2))
                            input_a_arr[0, feature_idx] = or_a_reg
                            input_b_arr[0, feature_idx] = or_b_reg

                            # Scale the inputs
                            scaled_a = scaler.transform(input_a_arr)[0, feature_idx]
                            scaled_b = scaler.transform(input_b_arr)[0, feature_idx]
                        else:
                            scaled_a = or_a_reg
                            scaled_b = or_b_reg

                        # Calculate odds ratio
                        odds_ratio = np.exp(beta_val * (scaled_a - scaled_b))

                        # Feature display name
                        feature_display = r"\text{Age}" if or_feature_reg == "Age" else r"\text{Annual Income}"

                        # Display aligned formulas based on scaling
                        if logreg_scaling != "No Scaling":
                            st.latex(r"""
                            \begin{aligned}
                            \text{Odds Ratio} &= \frac{\text{Odds}(X=a)}{\text{Odds}(X=b)} \\[8pt]
                            &= \text{exp}(\beta \cdot (a_{\text{scaled}} - b_{\text{scaled}})) \\[8pt]
                            &= \text{exp}(\beta_{%s} \cdot (%s - %s)) \\[8pt]
                            &= \text{exp}(%s \cdot %s) \\[8pt]
                            &= %s
                            \end{aligned}
                            """ % (
                                feature_display,
                                format_number(scaled_a),
                                format_number(scaled_b),
                                format_number(beta_val),
                                format_number(scaled_a - scaled_b),
                                format_number(odds_ratio)
                            ))
                            st.caption(f":red[**Scaling applied ({logreg_scaling}): a={format_number(or_a_reg)} → {format_number(scaled_a)}, b={format_number(or_b_reg)} → {format_number(scaled_b)}**]")
                        else:
                            st.latex(r"""
                            \begin{aligned}
                            \text{Odds Ratio} &= \frac{\text{Odds}(X=a)}{\text{Odds}(X=b)} \\[8pt]
                            &= \text{exp}(\beta \cdot (a - b)) \\[8pt]
                            &= \text{exp}(\beta_{%s} \cdot (%s - %s)) \\[8pt]
                            &= \text{exp}(%s \cdot %s) \\[8pt]
                            &= %s
                            \end{aligned}
                            """ % (
                                feature_display,
                                format_number(scaled_a),
                                format_number(scaled_b),
                                format_number(beta_val),
                                format_number(scaled_a - scaled_b),
                                format_number(odds_ratio)
                            ))

                    # Interpretation text box
                    st.markdown("---")
                    st.markdown("**Interpretation:**")

                    feature_name = "Age" if or_feature_reg == "Age" else "Annual Income"

                    if or_feature_reg == "Age":
                        a_display = f"{or_a_reg:.0f} years"
                        b_display = f"{or_b_reg:.0f} years"
                    else:
                        a_display = f"IDR {or_a_reg:,.0f}"
                        b_display = f"IDR {or_b_reg:,.0f}"

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

                st.info("Note: Statistical tests (LLR, Wald) are only available for **unregularized logistic regression**")

else:
    st.error("Unable to load the dataset. Please check the file path and try again.")
