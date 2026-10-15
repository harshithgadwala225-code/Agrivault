# 🌾 AgriVault — AI Crop Storage & Sell Optimizer

AgriVault is an AI-powered decision support system designed to help farmers, agricultural traders, and distributors decide whether to **sell crops immediately** or **store them for future profit**.

By evaluating wholesale market price trends, monthly storage overheads, and spoilage rates, AgriVault provides real-time, data-backed financial recommendations.

## ✨ Features

* **AI-Powered Recommendation:** Generates an actionable recommendation (**SELL TODAY** vs. **STORE CROP**) using historical price dynamics.

* **Financial Projections:** Displays immediate revenue, expected stored revenue, storage overheads, and net gain/loss forecasts.

* **Risk-Adjusted Decision Making:** Factors in expected commodity spoilage percentages and holding costs.

* **Lightweight & Fast:** Responsive UI built with Tailwind CSS backed by a lightweight FastAPI server.

## 📊 Model & Dataset Information

* **Dataset:** Historical Agmarknet wholesale agricultural commodity price data (**2022–2025**).

* **Core Input Variables:**

  * `Current Price (₹/kg)`: Wholesale market rate at harvest/decision time.

  * `30-Day Ago Price (₹/kg)`: Price benchmark to measure 30-day market momentum.

  * `Stock Quantity (kg)`: Total crop volume under consideration.

  * `Storage Cost (₹/kg/month)`: Cold storage or facility rental fee per month.

  * `Spoilage Rate (%)`: Expected monthly crop decay or quality degradation percentage.

## 🛠️ Tech Stack

* **Frontend:** HTML5, JavaScript (Fetch API), Tailwind CSS

* **Backend Framework:** Python, FastAPI, Uvicorn

* **Machine Learning:** Scikit-Learn, NumPy, Pandas

## 🚀 Quickstart (Run Locally)

### Prerequisites

* Python 3.9 or higher installed

* Git installed

### Installation & Setup

1. **Clone the repository:**

   ```
   git clone https://github.com/harshithgadwala225-code/AgriVault.git
   cd AgriVault
   
   ```

2. **Create and activate a virtual environment:**

   * **Windows:**

     ```
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     
     ```

   * **macOS / Linux:**

     ```
     python3 -m venv .venv
     source .venv/bin/activate
     
     ```

3. **Install dependencies:**

   ```
   pip install -r requirements.txt
   
   ```

4. **Start the backend server:**

   ```
   uvicorn backend.main:app --reload
   
   ```

   The backend API will run locally at `http://127.0.0.1:8000`. You can test interactive API documentation at `http://127.0.0.1:8000/docs`.

5. **Open the frontend:**
   Open `frontend/index.html` in your web browser (or run using VS Code Live Server).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the repository issues page if you'd like to contribute.
