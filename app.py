from flask import render_template, request, Flask
import pandas as pd
import joblib
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

VALID_TEAMS = ["RR", "RCB", "GT", "PBKS", "MI", "SRH", "LSG", "CSK", "DC", "KKR"]

TEAM_MAP = {
    "RR": 1,
    "RCB": 0.9,
    "GT": 0.8,
    "PBKS": 0.7,
    "MI": 0.6,
    "SRH": 0.5,
    "LSG": 0.4,
    "CSK": 0.3,
    "DC": 0.2,
    "KKR": 0.1
}

try:
    pipe = joblib.load('pipelin3e.pkl')
except FileNotFoundError:
    logging.error("pipeline.pkl not found. Place it in the app's root directory.")
    pipe = None
except Exception as e:
    logging.error(f"Failed to load pipeline.pkl: {e}")
    pipe = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    if pipe is None:
        return render_template(
            "index.html",
            error="Model is unavailable right now. Please try again later."
        )

    # --- Validate presence and types of all fields ---
    required_fields = ["batting_team", "bowling_team", "target", "curr_run", "curr_wick", "ball_number"]
    for field in required_fields:
        if not request.form.get(field):
            return render_template("index.html", error=f"Missing value for {field.replace('_', ' ')}.")

    batting_team = request.form["batting_team"].strip().upper()
    bowling_team = request.form["bowling_team"].strip().upper()

    if batting_team not in VALID_TEAMS or bowling_team not in VALID_TEAMS:
        return render_template("index.html", error="Please select valid teams from the list.")

    if batting_team == bowling_team:
        return render_template("index.html", error="Batting and bowling team can't be the same.")

    try:
        target = int(request.form["target"])
        curr_run = int(request.form["curr_run"])
        curr_wick = int(request.form["curr_wick"])
        ball_number = int(request.form["ball_number"])
    except ValueError:
        return render_template("index.html", error="Target, score, wickets, and balls must be whole numbers.")

    # --- Sanity-check ranges ---
    if target < 0 or target > 320:
        return render_template("index.html", error="Target must be between 0 and 320.")
    if curr_run < 0 or curr_run > target:
        return render_template("index.html", error="Current score must be between 0 and the target.")
    if curr_wick < 0 or curr_wick > 9:
        return render_template("index.html", error="Wickets fallen must be between 0 and 9.")
    if ball_number < 1 or ball_number > 119:
        return render_template("index.html", error="Balls bowled must be between 1 and 119.")

    try:
        data = pd.DataFrame({
            "batting_team": [batting_team],
            "bowling_team": [bowling_team],
            "target": [target],
            "curr_run": [curr_run],
            "curr_wick": [curr_wick],
            "ball_number": [ball_number]
        })

        data["batting_team"] = data["batting_team"].map(TEAM_MAP)
        data["bowling_team"] = data["bowling_team"].map(TEAM_MAP)

        data["cr"] = data["curr_run"] * 6 / data["ball_number"]
        data["req_run"] = data["target"] - data["curr_run"]
        data["balls_left"] = 120 - data["ball_number"]
        data["wick_left"] = 10 - data["curr_wick"]

        balls_left_val = data["balls_left"].iloc[0]
        data["rr"] = 0 if balls_left_val == 0 else data["req_run"] * 6 / data["balls_left"]

        data["rpw"] = data["curr_run"] / max(data["curr_wick"].iloc[0], 1)
        data["rrpw"] = data["req_run"] / max(data["wick_left"].iloc[0], 1)

        features = data[["batting_team", "bowling_team", "target", "curr_run",
                          "curr_wick", "ball_number", "cr", "req_run",
                          "balls_left", "wick_left", "rr", "rrpw", "rpw"]]

        probability = pipe.predict_proba(features)[0][1]

        if probability > 0.75:
            result = "easy chase for batting team"
        elif data["rr"] > 36:
            result = "easy win for bowling team"
            probability = 0.01
        elif probability > 0.55:
            result = "close but chasable"
        elif probability > 0.25:
            result = "close but can't chasable"
        else:
            result = "easy win for bowling team"

        probability = round(probability * 100, 2)

        return render_template("index.html", prediction=result, proba=probability)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return render_template("index.html", error="Something went wrong while predicting. Please check your inputs and try again.")


if __name__ == "__main__":
    app.run(debug=True)
