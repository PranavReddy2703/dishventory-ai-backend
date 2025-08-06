from io import BytesIO
import base64
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet import Prophet
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000",
     "https://dishventory-ai.vercel.app"])


# Ingredient mapping for one pepperoni pizza
pizza_recipe = {
    'flour_g': 200,
    'water_ml': 130,
    'yeast_g': 3,
    'sugar_g': 5,
    'salt_g': 4,
    'olive_oil_ml': 10,
    'tomato_sauce_g': 80,
    'garlic_g': 2,
    'oregano_g': 1,
    'basil_g': 1,
    'mozzarella_g': 100,
    'parmesan_g': 10,
    'pepperoni_slices': 20
}

# Define common holidays
holidays = pd.DataFrame({
    'holiday': ['new_year', 'christmas'],
    'ds': pd.to_datetime(['2016-01-01', '2015-12-25']),
    'lower_window': 0,
    'upper_window': 1
})


def calculate_ingredients(predicted_sales):
    return {
        ingredient: int(round(qty * predicted_sales))
        for ingredient, qty in pizza_recipe.items()
    }


def forecast_pepperoni(df):
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df = df.dropna(subset=['order_date', 'pizza_id', 'quantity'])
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df = df.dropna(subset=['quantity'])

    pizza_df = df[df['pizza_id'] == 'pepperoni_m']
    daily_sales = pizza_df.groupby('order_date')[
        'quantity'].sum().reset_index()
    daily_sales.rename(columns={'order_date': 'ds',
                       'quantity': 'y'}, inplace=True)

    if len(daily_sales) < 10:
        return {'error': 'Not enough data for pepperoni_m'}

    # Prophet Forecast with holidays
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        holidays=holidays
    )
    model.fit(daily_sales)
    future = model.make_future_dataframe(
        periods=7, freq='D', include_history=False)
    forecast = model.predict(future)

    # Predicted sales for next 7 days (sum)
    forecast_7_days = forecast[['ds', 'yhat']].copy()
    forecast_7_days['yhat'] = forecast_7_days['yhat'].apply(
        lambda x: max(0, round(x)))
    predicted_sales = forecast_7_days['yhat'].sum()

    # Plot only the 7-day forecast
    plt.figure(figsize=(10, 5))
    plt.plot(forecast_7_days['ds'],
             forecast_7_days['yhat'], marker='o', linestyle='-')
    plt.title('7-Day Forecast for Pepperoni Pizza')
    plt.xlabel('Date')
    plt.ylabel('Predicted Quantity')
    plt.grid(True)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    graph_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    ingredients = calculate_ingredients(predicted_sales)

    return {
        'predicted_sales': int(predicted_sales),
        'ingredients_needed': ingredients,
        'graph_base64': graph_base64
    }


@app.route('/predict', methods=['POST'])
def predict():
    if 'data-file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['data-file']
    filename = file.filename

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        result = forecast_pepperoni(df)

        if 'error' in result:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Failed to process file. Make sure it has order_date, pizza_id, quantity columns.'}), 500


if __name__ == '__main__':
    app.run(debug=True)
