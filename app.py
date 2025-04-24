import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Set page configuration
st.set_page_config(
    page_title="Should I Rent or Buy?",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 Should I Rent or Buy?")
st.markdown("""
This app helps you decide whether renting or buying a home makes more financial sense based on your inputs.
Toggle between India and UK to see country-specific defaults and recommendations.
""")


# Function to calculate EMI (Equated Monthly Installment)
def calculate_emi(principal, rate, tenure_years):
    """
    Calculate monthly EMI for a loan

    Parameters:
    - principal: Loan amount
    - rate: Annual interest rate (in %)
    - tenure_years: Loan tenure in years

    Returns:
    - Monthly EMI amount
    """
    # Convert annual rate to monthly rate and percentage to decimal
    rate_monthly = (rate / 100) / 12
    tenure_months = tenure_years * 12

    # EMI formula: P * r * (1 + r)^n / ((1 + r)^n - 1)
    if rate_monthly > 0:
        emi = principal * rate_monthly * (1 + rate_monthly) ** tenure_months / ((1 + rate_monthly) ** tenure_months - 1)
    else:
        emi = principal / tenure_months

    return emi


# City data with default values
city_data = {
    "India": {
        "Mumbai": {"rent": 50000, "price": 15000000, "tax": 2000},
        "Delhi": {"rent": 35000, "price": 10000000, "tax": 1500},
        "Bangalore": {"rent": 30000, "price": 9000000, "tax": 1200},
        "Hyderabad": {"rent": 25000, "price": 7000000, "tax": 1000},
        "Chennai": {"rent": 22000, "price": 6500000, "tax": 900}
    },
    "UK": {
        "London": {"rent": 2000, "price": 550000, "tax": 200},
        "Manchester": {"rent": 1000, "price": 250000, "tax": 150},
        "Birmingham": {"rent": 900, "price": 220000, "tax": 140},
        "Edinburgh": {"rent": 1100, "price": 300000, "tax": 160},
        "Bristol": {"rent": 1200, "price": 320000, "tax": 170}
    }
}

# Default values for each country
country_defaults = {
    "India": {
        "down_payment_percent": 20,
        "interest_rate": 7.5,
        "loan_tenure": 20,
        "stay_years": 10,
        "currency": "₹",
        "appreciation_rate": 5  # Yearly appreciation rate in %
    },
    "UK": {
        "down_payment_percent": 15,
        "interest_rate": 4.0,
        "loan_tenure": 25,
        "stay_years": 7,
        "currency": "£",
        "appreciation_rate": 3  # Yearly appreciation rate in %
    }
}


# Function to generate synthetic historical data for ML model
def generate_historical_data(country):
    """
    Generate synthetic historical housing data for training ML model

    Parameters:
    - country: Selected country (India or UK)

    Returns:
    - DataFrame with synthetic historical data
    """
    # Generate 100 synthetic data points
    np.random.seed(42)  # For reproducibility

    # Base appreciation rate for selected country
    base_rate = country_defaults[country]["appreciation_rate"]

    # Generate random years (from 1 to 20 years ago)
    years = np.random.randint(1, 21, size=100)

    # Features that might affect appreciation
    interest_rates = np.random.uniform(2.0, 10.0, size=100)
    inflation = np.random.uniform(2.0, 8.0, size=100)
    gdp_growth = np.random.uniform(1.0, 8.0, size=100)

    # Generate appreciation rates with some noise
    appreciation = base_rate + 0.2 * interest_rates - 0.3 * inflation + 0.5 * gdp_growth + np.random.normal(0, 1,
                                                                                                            size=100)
    # Clip to reasonable range (0% to 15%)
    appreciation = np.clip(appreciation, 0, 15)

    # Create DataFrame
    data = pd.DataFrame({
        'years_ago': years,
        'interest_rate': interest_rates,
        'inflation': inflation,
        'gdp_growth': gdp_growth,
        'appreciation_rate': appreciation
    })

    return data


# Function to train ML model for price appreciation prediction
def train_appreciation_model(historical_data):
    """
    Train a simple linear regression model to predict housing price appreciation

    Parameters:
    - historical_data: DataFrame with synthetic historical data

    Returns:
    - Trained model and feature list
    """
    # Features and target
    X = historical_data[['years_ago', 'interest_rate', 'inflation', 'gdp_growth']]
    y = historical_data['appreciation_rate']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Calculate score
    score = model.score(X_test, y_test)

    return model, ['years_ago', 'interest_rate', 'inflation', 'gdp_growth'], score


# Function to predict future appreciation rate
def predict_appreciation(model, features, interest_rate, country):
    """
    Predict future appreciation rate using the trained model

    Parameters:
    - model: Trained ML model
    - features: List of feature names
    - interest_rate: Current interest rate
    - country: Selected country (for default inflation and GDP values)

    Returns:
    - Predicted yearly appreciation rate
    """
    # Default values based on country
    if country == "India":
        inflation = 5.0
        gdp_growth = 6.5
    else:  # UK
        inflation = 2.5
        gdp_growth = 2.0

    # Create feature vector for prediction
    # Years ago is 0 for current prediction
    X_pred = pd.DataFrame([[0, interest_rate, inflation, gdp_growth]], columns=features)

    # Predict
    prediction = model.predict(X_pred)[0]

    # Clip to reasonable range
    prediction = np.clip(prediction, 0, 15)

    return prediction


# Sidebar: Country and City Selection
with st.sidebar:
    st.header("Location Settings")

    # Country selection
    selected_country = st.selectbox("Select Country", ["India", "UK"])
    currency = country_defaults[selected_country]["currency"]

    # City selection
    cities = list(city_data[selected_country].keys())
    selected_city = st.selectbox("Select City", cities)

    # Get default values for selected city
    city_rent = city_data[selected_country][selected_city]["rent"]
    city_price = city_data[selected_country][selected_city]["price"]
    city_tax = city_data[selected_country][selected_city]["tax"]

    # Advanced settings
    st.header("Advanced Settings")
    enable_ml = st.checkbox("Enable ML Price Prediction", value=False)

    # If ML is enabled, show additional inputs for the model
    if enable_ml:
        st.subheader("Economic Indicators")
        current_inflation = st.slider(
            "Current Inflation Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=5.0 if selected_country == "India" else 2.5,
            step=0.1
        )
        current_gdp = st.slider(
            "GDP Growth Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=6.5 if selected_country == "India" else 2.0,
            step=0.1
        )


# Main inputs
st.header("Financial Inputs")

col1, col2 = st.columns(2)

with col1:
    # Renting inputs
    st.subheader("Renting Details")
    monthly_rent = st.number_input(
        f"Monthly Rent ({currency})",
        min_value=0.0,
        max_value=1000000.0,
        value=float(city_rent),
        step=100.0
    )

    stay_years = st.slider(
        "Years Planning to Stay",
        min_value=1,
        max_value=30,
        value=country_defaults[selected_country]["stay_years"]
    )

with col2:
    # Buying inputs
    st.subheader("Buying Details")
    house_price = st.number_input(
        f"Property Price ({currency})",
        min_value=0.0,
        max_value=100000000.0,
        value=float(city_price),
        step=10000.0
    )

    down_payment_percent = st.slider(
        "Down Payment (%)",
        min_value=5,
        max_value=90,
        value=country_defaults[selected_country]["down_payment_percent"]
    )

    interest_rate = st.slider(
        "Loan Interest Rate (%)",
        min_value=1.0,
        max_value=15.0,
        value=country_defaults[selected_country]["interest_rate"],
        step=0.1
    )

    loan_tenure = st.slider(
        "Loan Tenure (years)",
        min_value=5,
        max_value=30,
        value=country_defaults[selected_country]["loan_tenure"]
    )

    property_tax = st.number_input(
        f"Monthly Property/Council Tax ({currency})",
        min_value=0.0,
        max_value=100000.0,
        value=float(city_tax),
        step=100.0
    )

# Calculations
st.header("Results")

# Calculate down payment amount
down_payment = house_price * (down_payment_percent / 100)

# Calculate loan amount
loan_amount = house_price - down_payment

# Calculate EMI
monthly_emi = calculate_emi(loan_amount, interest_rate, loan_tenure)

# Initialize ML model if enabled
if enable_ml:
    historical_data = generate_historical_data(selected_country)
    model, features, model_score = train_appreciation_model(historical_data)

    # Use model to predict future appreciation
    predicted_appreciation = predict_appreciation(model, features, interest_rate, selected_country)

    # Display the predicted appreciation rate
    st.info(
        f"ML Model predicts a yearly appreciation rate of {predicted_appreciation:.2f}% (Model accuracy: {model_score:.2f})")
else:
    # Default appreciation rate from country defaults
    predicted_appreciation = country_defaults[selected_country]["appreciation_rate"]

# Calculate costs over time
years = list(range(1, max(stay_years, loan_tenure) + 1))
rent_costs = [monthly_rent * 12 * year for year in years]
buy_costs_no_appreciation = [(monthly_emi * 12 * min(year, loan_tenure)) +
                             (property_tax * 12 * year) +
                             down_payment for year in years]

# Calculate buy costs with appreciation (if enabled)
buy_costs_with_appreciation = []
for year in years:
    # Initial buying cost
    cost = (monthly_emi * 12 * min(year, loan_tenure)) + (property_tax * 12 * year) + down_payment

    # Calculate property value after appreciation
    appreciated_value = house_price * (1 + predicted_appreciation / 100) ** year

    # Subtract the gain in value from the cost
    if year == stay_years:  # Only consider the gain at the end of the stay period
        cost_with_appreciation = cost - (appreciated_value - house_price)
    else:
        cost_with_appreciation = cost

    buy_costs_with_appreciation.append(max(0, cost_with_appreciation))

# Use the appropriate buy costs based on ML toggle
buy_costs = buy_costs_with_appreciation if enable_ml else buy_costs_no_appreciation

# Find break-even point
break_even_year = None
for i, (rent_cost, buy_cost) in enumerate(zip(rent_costs, buy_costs)):
    if buy_cost <= rent_cost:
        break_even_year = i + 1
        break

# Create DataFrame for plotting
df_costs = pd.DataFrame({
    'Year': years,
    'Rent': rent_costs,
    'Buy': buy_costs
})

# Visualization
st.subheader("Cost Comparison Over Time")

# Use plotly for interactive chart
fig = px.line(
    df_costs,
    x='Year',
    y=['Rent', 'Buy'],
    title=f'Cumulative Cost: Rent vs Buy ({currency})',
    labels={'value': f'Cumulative Cost ({currency})'},
    color_discrete_sequence=['#ff7f0e', '#1f77b4']
)

# Add a vertical line at the planned stay duration
fig.add_vline(
    x=stay_years,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Planned Stay: {stay_years} years",
    annotation_position="top right"
)

# Add break-even point if it exists and within the stay period
if break_even_year and break_even_year <= max(years):
    fig.add_vline(
        x=break_even_year,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Break-even: {break_even_year} years",
        annotation_position="bottom right"
    )

st.plotly_chart(fig, use_container_width=True)

# Recommendation
rent_total = monthly_rent * 12 * stay_years
buy_total_at_end = buy_costs[stay_years - 1] if stay_years <= len(buy_costs) else buy_costs[-1]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Total Cost of Renting",
        value=f"{currency} {rent_total:,.2f}",
        delta=None
    )

with col2:
    st.metric(
        label="Total Cost of Buying",
        value=f"{currency} {buy_total_at_end:,.2f}",
        delta=f"{(rent_total - buy_total_at_end):,.2f}" if buy_total_at_end < rent_total else f"-{(buy_total_at_end - rent_total):,.2f}"
    )

# Final recommendation
st.subheader("Recommendation")
if buy_total_at_end < rent_total:
    recommendation = "Better to Buy"
    color = "green"
    explanation = f"Buying would save you approximately {currency} {(rent_total - buy_total_at_end):,.2f} over {stay_years} years."
else:
    recommendation = "Better to Rent"
    color = "orange"
    explanation = f"Renting would save you approximately {currency} {(buy_total_at_end - rent_total):,.2f} over {stay_years} years."

st.markdown(f"<h1 style='text-align: center; color: {color};'>{recommendation}</h1>", unsafe_allow_html=True)
st.write(explanation)

if break_even_year:
    if break_even_year <= stay_years:
        st.write(f"You will break even on buying after {break_even_year} years.")
    else:
        st.write(f"You would need to stay for {break_even_year} years to break even on buying.")
else:
    st.write("Buying does not break even within the displayed time frame.")

# Additional details
with st.expander("View Detailed Calculations"):
    st.write(f"**Down Payment:** {currency} {down_payment:,.2f}")
    st.write(f"**Loan Amount:** {currency} {loan_amount:,.2f}")
    st.write(f"**Monthly EMI:** {currency} {monthly_emi:,.2f}")
    st.write(f"**Monthly Property/Council Tax:** {currency} {property_tax:,.2f}")

    if enable_ml:
        st.write(f"**Predicted Annual Property Appreciation:** {predicted_appreciation:.2f}%")
        st.write(
            f"**Property Value after {stay_years} years:** {currency} {house_price * (1 + predicted_appreciation / 100) ** stay_years:,.2f}")
