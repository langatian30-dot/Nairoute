import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================================
# LOAD DATASET
# ==========================================================

print("Loading dataset...")

data = pd.read_csv(
    "road_risk_dataset.csv"
)

print(f"Dataset Size: {data.shape}")

# ==========================================================
# FEATURES
# ==========================================================

X = data[

    [

        "highway",

        "length",

        "maxspeed",

        "lanes",

        "crime_score",

        "time_period"

    ]

]

y = data["risk"]

# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)

# ==========================================================
# PREPROCESSING
# ==========================================================

categorical_features = [

    "highway",

    "time_period"

]

numeric_features = [

    "length",

    "maxspeed",

    "lanes",

    "crime_score"

]

preprocessor = ColumnTransformer(

    transformers=[

        (

            "cat",

            OneHotEncoder(handle_unknown="ignore"),

            categorical_features

        ),

        (

            "num",

            "passthrough",

            numeric_features

        )

    ]

)

# ==========================================================
# MODEL
# ==========================================================

model = Pipeline(

    steps=[

        (

            "preprocessor",

            preprocessor

        ),

        (

            "regressor",

            RandomForestRegressor(

                n_estimators=150,

                random_state=42,

                n_jobs=-1

            )

        )

    ]

)

# ==========================================================
# TRAIN
# ==========================================================

print("Training AI model...")

model.fit(

    X_train,

    y_train

)

# ==========================================================
# EVALUATION
# ==========================================================

predictions = model.predict(

    X_test

)

mae = mean_absolute_error(

    y_test,

    predictions

)

r2 = r2_score(

    y_test,

    predictions

)

print()

print(f"Mean Absolute Error : {mae:.3f}")

print(f"R² Score            : {r2:.3f}")

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(

    model,

    "risk_model.joblib"

)

print()

print("risk_model.joblib saved successfully.")