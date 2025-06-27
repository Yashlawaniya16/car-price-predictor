from flask import Flask,render_template,request,redirect, url_for, flash
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np


app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearRegression.pkl', 'rb'))
car=pd.read_csv('Cleaned_Car_data.csv')

app.secret_key = 'your_secret_key_here'


def retrain_model():
    import pandas as pd
    import pickle
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    # Load and clean dataset
    df = pd.read_csv('Cleaned_Car_data.csv')

    # Drop rows with any missing values
    df.dropna(inplace=True)

    # ✅ Convert numeric fields safely
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # Drop rows that couldn't be converted
    df.dropna(inplace=True)  # ✅ important: drop from full df (X + y stay aligned)

    # Define features and target
    X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
    y = df['Price']

    # Categorical columns
    cat_cols = ['name', 'company', 'fuel_type']

    # One-hot encode only those
    ct = ColumnTransformer([
        ('encoder', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ], remainder='passthrough')

    model_pipeline = Pipeline(steps=[
        ('preprocessor', ct),
        ('regressor', LinearRegression())
    ])

    model_pipeline.fit(X, y)

    with open('LinearRegression.pkl', 'wb') as f:
        pickle.dump(model_pipeline, f)

    print("✅ Model retrained and saved.")


@app.route('/retrain', methods=['GET'])
def retrain():
    retrain_model()
    global model
    model = pickle.load(open('LinearRegression.pkl', 'rb'))

    flash("✅ Model retrained successfully!")  # Show success message
    return redirect(url_for('index'))

@app.route('/', methods=['GET', 'POST'])
def index():
    car = pd.read_csv('Cleaned_Car_data.csv')  # ✅ moved here

    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')

    return render_template('index.html',
                           companies=companies,
                           car_models=car_models,
                           years=year,
                           fuel_types=fuel_type)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    company=request.form.get('company')

    car_model=request.form.get('car_models')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    driven=request.form.get('kilo_driven')

    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0],2))

@app.route('/add_car', methods=['GET', 'POST'])
def add_car():
    if request.method == 'POST':
        name = request.form['name']
        company = request.form['company']
        year = int(request.form['year'])
        kms = int(request.form['kms_driven'])
        fuel = request.form['fuel_type']
        price = float(request.form['price'])

        # 🔍 Debug output
        print("Adding:", [name, company, year, kms, fuel, price])

        new_data = pd.DataFrame([[name, company, year, price, kms, fuel]],
                                columns=['name', 'company', 'year', 'price', 'kms_driven', 'fuel_type'])
        new_data.to_csv('Cleaned_Car_data.csv', mode='a', header=False, index=False)

        retrain_model()
        global model
        model = pickle.load(open('LinearRegression.pkl', 'rb'))

        flash("✅ Car added and model retrained!")
        return redirect(url_for('index'))

    return render_template('add_car.html')



@app.route('/delete_car', methods=['GET', 'POST'])
def delete_car():
    if request.method == 'POST':
        model_to_delete = request.form['car_model'].strip().lower()

        df = pd.read_csv('Cleaned_Car_data.csv')
        original_len = len(df)

        # Delete matching rows
        df = df[df['name'].str.lower() != model_to_delete]

        # Save updated CSV
        df.to_csv('Cleaned_Car_data.csv', index=False)

        deleted = original_len - len(df)

        if deleted > 0:
            flash(f"🗑️ Deleted {deleted} entries for model: {model_to_delete}")
        else:
            flash(f"⚠️ No matching entries found for: {model_to_delete}")

        return redirect(url_for('index'))  # ✅ Redirect to homepage after deleting

    return render_template('delete_car.html')


if __name__=='__main__':
    app.run(debug=True)