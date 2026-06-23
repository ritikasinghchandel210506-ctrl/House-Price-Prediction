from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# ==========================================
# PREDICT
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------
        # GET FORM DATA
        # ----------------------------------

        area = float(
            request.form["area"]
        )

        bedrooms = int(
            request.form["bedrooms"]
        )

        bathrooms = int(
            request.form["bathrooms"]
        )

        stories = int(
            request.form["stories"]
        )

        parking = int(
            request.form["parking"]
        )

        mainroad = request.form["mainroad"]

        guestroom = request.form["guestroom"]

        basement = request.form["basement"]

        hotwaterheating = request.form[
            "hotwaterheating"
        ]

        airconditioning = request.form[
            "airconditioning"
        ]

        prefarea = request.form[
            "prefarea"
        ]

        furnishingstatus = request.form[
            "furnishingstatus"
        ]

        # ----------------------------------
        # FEATURE ENGINEERING
        # SAME AS train.py
        # ----------------------------------

        area_per_bedroom = (
            area / bedrooms
        )

        total_rooms = (
            bedrooms + bathrooms
        )

        luxury_score = (
            area +
            bathrooms * 1000 +
            parking * 500
        )

        # ----------------------------------
        # CREATE DATAFRAME
        # ----------------------------------

        input_df = pd.DataFrame({

            "area":
                [area],

            "bedrooms":
                [bedrooms],

            "bathrooms":
                [bathrooms],

            "stories":
                [stories],

            "parking":
                [parking],

            "mainroad":
                [mainroad],

            "guestroom":
                [guestroom],

            "basement":
                [basement],

            "hotwaterheating":
                [hotwaterheating],

            "airconditioning":
                [airconditioning],

            "prefarea":
                [prefarea],

            "furnishingstatus":
                [furnishingstatus],

            "area_per_bedroom":
                [area_per_bedroom],

            "total_rooms":
                [total_rooms],

            "luxury_score":
                [luxury_score]
        })

        # ----------------------------------
        # PREDICT
        # ----------------------------------

        prediction_log = model.predict(
            input_df
        )[0]

        prediction = np.expm1(
            prediction_log
        )

        prediction = round(
            prediction
        )

        # ----------------------------------
        # RETURN RESULT
        # ----------------------------------

        return render_template(

            "index.html",

            prediction_text=
            f"₹ {prediction:,.0f}",

            area=area,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            stories=stories,
            parking=parking,

            mainroad=mainroad,
            guestroom=guestroom,
            basement=basement,

            hotwaterheating=
            hotwaterheating,

            airconditioning=
            airconditioning,

            prefarea=prefarea,

            furnishingstatus=
            furnishingstatus
        )

    except Exception as e:

        return render_template(

            "index.html",

            error=
            f"Error: {str(e)}"
        )

# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )