# Stock-Market-Forcasting
🔋Energy Stock Market Forcasting using ML Models
A machine learning-based web application that predicts stock prices for selected energy sector companies using LSTM (Long Short-Term Memory) and Linear Regression. Built using Streamlit and powered by Alpha Vantage API, this tool helps users make informed investment decisions with real-time predictions, volatility assessments, and trend analysis.

📌 Project Objective
Predict daily closing prices for energy sector stocks (XOM, CVX, COP, SLB, MRO).
Assess short-term volatility and 5-month trend direction.
Generate Buy/Sell/Hold recommendations using AI.
Enable interactive forecasting through a Streamlit web interface.

🧠 Models Used
🔹 Linear Regression
Uses sklearn.linear_model.LinearRegression
Predicts next-day price based on time index
Simple but lacks deep temporal understanding

🔹 LSTM (Long Short-Term Memory)
Built using Keras and TensorFlow
Learns from 100-day historical windows
Captures temporal patterns for more adaptive predictions
RMSE (Root Mean Squared Error) calculated for last 30 days

📊 Features
Live Prediction for tomorrow's price
Volatility Classification: Low / Medium / High
5-Month Price Trend Forecast
Recommendation Engine: BUY / SELL / HOLD
Interactive Graphs: Real prices, forecasts & future trend

🧰 Tech Stack
Tool	Usage
Python	Core programming language
Streamlit	Web interface
Alpha Vantage	Stock data source
scikit-learn	Linear Regression model
TensorFlow/Keras	LSTM model
Pandas, Numpy	Data manipulation
Matplotlib	Visualization

📦 Installation & Usage
Clone the repository
bash
Copy
Edit
git clone https://github.com/yourusername/energy-stock-predictor.git
cd energy-stock-predictor

Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Set Your Alpha Vantage API Key

Replace the API key in streamlit_app_energy_alpha.py:
python
Copy
Edit
ALPHA_VANTAGE_API_KEY = "YOUR_KEY_HERE"

Run the App
bash
Copy
Edit
streamlit run streamlit_app_energy_alpha.py

⚠️ Limitations
Only uses daily close prices (no high/low/open/sentiment).
LSTM requires long sequences and longer training time.
Linear Regression is naive for time-series dynamics.
No macroeconomic or news input considered yet.

🔮 Future Enhancements
Add Open/High/Low/Sentiment/News inputs
Integrate advanced models (GRU, Transformers)
Deploy as cloud or mobile dashboard
Include reinforcement learning for trading

🧑‍💻 Author
Miss Rutuja Borkar
Date: 13-May-2025

📄 License
This project is for academic and educational use only. Please cite appropriately if reused.
