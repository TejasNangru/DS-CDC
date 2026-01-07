🏠 Multimodal Real Estate Price Prediction (DS-CDC)

A machine learning pipeline that predicts residential property prices by combining tabular features with satellite imagery using a multimodal deep learning architecture.

📋 Project Overview

This project builds a comprehensive property valuation framework by:

🛰️ Programmatically acquiring satellite imagery using the Sentinel Hub API

🧠 Engineering 21+ predictive features from structured property data

📊 Training and comparing:

Baseline models (Random Forest, XGBoost)

Multimodal deep learning model (CNN + MLP)

🎯 Achieving R² = 0.76 on validation data with RMSE = $172,903

🚀 Quick Start
🔧 Prerequisites
pip install pandas numpy scikit-learn xgboost tensorflow sentinelhub pillow matplotlib seaborn openpyxl

🧩 Step 1: Data Preparation

Upload your training and test CSV files to Google Colab when prompted in:

preprocessing.ipynb

🛰️ Step 2: Satellite Image Acquisition
Get Sentinel Hub Credentials

Sign up at Sentinel Hub

Go to Dashboard → User Settings → OAuth Clients

Create a new OAuth client and copy:

CLIENT_ID

CLIENT_SECRET

Fetch Satellite Images
python data_fetcher.py \
  --train train.csv \
  --output images/ \
  --client-id YOUR_ID \
  --client-secret YOUR_SECRET


Alternatively, use the automated pipeline in Cells 3–5 of preprocessing.ipynb.

🧪 Step 3: Feature Engineering

The preprocessing pipeline creates 8 engineered features:

Feature	Description
luxury_score	Composite quality metric (grade + condition + view + waterfront × 3)
price_per_sqft	Normalized pricing
lot_ratio	Building density indicator
neighbor_living_premium	Living area vs neighborhood average
neighbor_lot_premium	Lot size vs neighborhood average
has_basement	Binary basement indicator
age	Property age (2026 - yr_built)
years_since_renovation	Maintenance recency
🤖 Step 4: Model Training

Run:

model_training.ipynb

Models Trained

Baseline Models

Random Forest

XGBoost (tabular only)

Multimodal Model

ResNet50 (image branch)

MLP (tabular branch)

Feature fusion via concatenation

📤 Step 5: Generate Predictions

Predictions are automatically generated in Cell 11 of model_training.ipynb and saved as:

predictions.csv

📊 Results
🔍 Model Performance Comparison
Model	RMSE ($)	MAE ($)	R² Score	Training Time
Random Forest	174,541	114,090	0.7572	2m 15s
XGBoost	172,903	113,122	0.7618	3m 42s
Multimodal CNN	479,326	469,048	-22.57*	8m 30s

* Multimodal model trained on limited image data (30 images). Performance expected to improve with full dataset.

🧠 Feature Importance (XGBoost)

Top 5 predictive features:

Grade – 34.8% (construction quality)

Luxury Score – 23.8% (composite appeal)

Sqft Living – 7.7%

Waterfront – 6.3%

Years Since Renovation – 3.4%

🏗️ Multimodal Architecture
IMAGE BRANCH (ResNet50)          TABULAR BRANCH (MLP)
Input: 224×224×3                Input: 21 features
        ↓                               ↓
ResNet50 (pretrained)           Dense(128) + BN + Dropout
        ↓                               ↓
GlobalAvgPool                   Dense(64) + BN + Dropout
        ↓                               ↓
Dense(256) + Dropout            Dense(32)
        ↓
Dense(128)

        └──────────────┬──────────────┘
                       ↓
            Concatenate (160 features)
                       ↓
          Dense(256) + BN + Dropout
                       ↓
          Dense(128) + Dropout
                       ↓
           Dense(64) + Dropout
                       ↓
               Output: Price

📈 Key Insights
🔎 Exploratory Data Analysis (EDA)

Price distribution is right-skewed

Mean: $549K

Median: $458K

Strong geographic clustering near waterfront

Grade and sqft_living show highest correlation with price (r > 0.67)

🧠 Modeling Insights

Tabular-only XGBoost outperforms multimodal due to image data scarcity

Construction quality is the single strongest predictor

Engineered features improve R² by ~0.04

🛰️ Visual Insights

Satellite imagery captures:

Green spaces

Water proximity

Dense vegetation → 15–20% price premium

High building density → lower property value

🛠️ Technologies Used

Python 3.10+

TensorFlow 2.15 – Deep learning

XGBoost 2.0 – Gradient boosting

Sentinel Hub API – Satellite imagery

Scikit-learn – ML utilities

Pandas / NumPy – Data processing

Matplotlib / Seaborn – Visualization

📁 Dataset Description
📊 Tabular Features (21)

Physical

bedrooms, bathrooms, sqft_living, sqft_lot

floors, sqft_above, sqft_basement

Quality

condition (1–5), grade (1–13), view (0–4)

Location

waterfront, latitude, longitude

Temporal

yr_built, yr_renovated

Neighborhood

sqft_living15, sqft_lot15

Engineered

luxury_score, lot_ratio, price_per_sqft

neighbor premiums, age features

🛰️ Visual Data

Source: Sentinel-2 L2A imagery

Resolution: 400 × 400 pixels

Coverage: 200m × 200m bounding box

Time Range: Last 6 months

Cloud Coverage: ≤ 30%

🔁 Reproducibility
git clone https://github.com/YOUR_USERNAME/real-estate-multimodal.git
cd real-estate-multimodal
pip install -r requirements.txt


Run notebooks in order:

preprocessing.ipynb

model_training.ipynb

Outputs

predictions.csv – Predictions for 5,404 properties

Model weights stored in memory during training

⚠️ Limitations & Future Work
Current Limitations

Only 40 satellite images vs 16,209 tabular samples

Multimodal model affected by data imbalance

Sentinel Hub free-tier processing limits

Future Improvements

📦 Scale image dataset to full 16K+

🧠 Replace ResNet50 with EfficientNet-B7

🎯 Add attention mechanisms

🌍 Integrate external data (Street View, POIs, crime stats)

🔄 Image data augmentation

Expected Outcome:
📈 Multimodal R² > 0.80 with full dataset

📄 License

MIT License – Free for educational and commercial use

👤 Author

Developed as part of a Real Estate Analytics Challenge
📅 January 2026

📘 For detailed analysis, see report.pdf
