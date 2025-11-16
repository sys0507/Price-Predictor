# 💰 XoX Price Prediction Model

<div align="center">

A comprehensive machine learning project for predicting XoX product prices using multiple regression algorithms and ensemble methods.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=Matplotlib&logoColor=black)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white)

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![GitHub stars](https://img.shields.io/github/stars/sys0507/Price-Predictor?style=social)
![GitHub forks](https://img.shields.io/github/forks/sys0507/Price-Predictor?style=social)

</div>

---

## 📑 Table of Contents

- [📊 Project Overview](#-project-overview)
- [💼 Business Context](#-business-context)
- [📁 Dataset](#-dataset)
- [🤖 Machine Learning Models](#-machine-learning-models)
- [📂 Project Structure](#-project-structure)
- [✨ Key Features](#-key-features)
- [🚀 Installation](#-installation)
- [💻 Usage](#-usage)
- [📈 Model Performance](#-model-performance)
- [⚙️ Technical Implementation](#-technical-implementation)
- [📊 Results](#-results)
- [🔮 Future Enhancements](#-future-enhancements)
- [🤝 Contributing](#-contributing)
- [📝 License](#-license)
- [👤 Author](#-author)
- [🙏 Acknowledgments](#-acknowledgments)

---

## 📊 Project Overview

As an agency helping customers purchase XoX products from various makers, price estimation is critical for making informed purchasing decisions. This project develops and compares multiple machine learning models to accurately predict XoX prices based on product characteristics, dimensions, and other features.

## 💼 Business Context

Our agency needs to estimate XoX prices before purchase to:
- 🎯 Recommend products at optimal price points
- 📈 Identify pricing trends across different makers
- 💡 Make data-driven purchasing decisions
- 🔍 Understand which features most influence pricing

---

## 📁 Dataset

The project uses sales data containing the following features:

### 📊 Numerical Features
- `cost` - Production cost
- `weight` - Product weight
- `height`, `width`, `depth` - Product dimensions
- `volume` - Calculated as height × width × depth

### 🏷️ Categorical Features
- `product_type` - Type classification (can be multi-valued)
- `product_level` - Product tier/level
- `maker` - Manufacturer name
- `ingredients` - Product composition (can be multi-valued)

### 📅 Temporal Features
- `purchase_date` - Date of purchase
- Derived: `year`, `month`, `weekday`, `day`

### 🎯 Target Variable
- `price` - XoX product price

---

## 🤖 Machine Learning Models

This project implements and compares **8 regression models** with comprehensive evaluation:

| # | Model | Type | Key Feature |
|---|-------|------|-------------|
| 1️⃣ | Linear Regression | Baseline | Simple linear relationships |
| 2️⃣ | Ridge Regression | Regularized | L2 regularization |
| 3️⃣ | Lasso Regression | Regularized | L1 + feature selection |
| 4️⃣ | PLS Regression | Dimensionality Reduction | Partial Least Squares |
| 5️⃣ | Random Forest | Ensemble | Decision tree ensemble |
| 6️⃣ | Gradient Boosting | Ensemble | Sequential boosting |
| 7️⃣ | XGBoost | Ensemble | Optimized gradient boosting |
| 8️⃣ | **Stacking Regressor** | **Meta-Ensemble** | **Combines top performers** ⭐ |

---

## 📂 Project Structure

```
Price_predictor/
├── 📄 README.md                           # Project documentation
├── 📋 requirements.txt                    # Python dependencies
├── 📓 regression modeling.ipynb           # Main analysis notebook
├── 📁 data/
│   └── 📊 sample_data.csv                # Sample dataset
├── 📁 results/                            # Model performance visualizations
│   ├── 📈 performance.png                # Radar plots of all models
│   └── 📉 stacking regressor.png         # Stacking Regressor predictions
└── 📽️ XoX_Price_Prediction_Model.pptx    # Project presentation
```

---

## ✨ Key Features

### 🔧 1. Data Preprocessing & Cleaning
- Custom transformation functions for price, cost, and dimension conversions
- Handling of multi-valued categorical features
- Missing value imputation strategies
- Feature engineering including volume calculation

### 🎨 2. Feature Engineering
- Temporal feature extraction (year, month, weekday, day)
- Volume calculation from dimensions
- Numerical feature scaling using MinMaxScaler
- Categorical encoding with OneHotEncoder
- Custom text processing for multi-valued categories

### 🎯 3. Model Training & Evaluation
- Cross-validation for robust performance estimation
- Hyperparameter tuning using GridSearchCV
- Comprehensive metrics: MAE, MSE, RMSE, R²
- Composite scoring system for model comparison

### 📊 4. Visualization
- Distribution analysis of features and target
- Correlation heatmaps
- Model performance comparison charts
- Radar plots for multi-metric evaluation
- Prediction vs actual plots

---

## 🚀 Installation

### 📋 Prerequisites
- Python 3.8 or higher
- Jupyter Notebook

### ⚡ Setup

1. **Clone the repository:**
```bash
git clone https://github.com/sys0507/Price-Predictor.git
cd Price-Predictor
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook:**
```bash
jupyter notebook
```

4. **Open and run:**
   - Open `regression modeling.ipynb`
   - Run all cells ▶️

---

## 💻 Usage

### 🎬 Running the Analysis

1. **Data Loading** 📥 - The notebook loads data from `data/sample_data.csv`
2. **Preprocessing** 🔧 - Automatic data cleaning and transformation
3. **Model Training** 🏋️ - All models are trained with optimized hyperparameters
4. **Evaluation** 📊 - Comprehensive performance metrics are calculated
5. **Visualization** 📈 - Results are visualized in multiple formats

### 🔄 Modifying for Your Data

To use your own sales data:
1. Format your data to match the expected schema (see [Dataset](#-dataset) section)
2. Place your CSV file in the `data/` folder
3. Update the file path in the notebook (Cell 1)
4. Run all cells

---

## 📈 Model Performance

### 📊 Quick Stats

<div align="center">

| Metric | Value |
|--------|-------|
| **Total Models** | 8 (7 individual + 1 ensemble) |
| **Best Performer** | Stacking Regressor ⭐ |
| **Features Analyzed** | 10+ (numerical, categorical, temporal) |
| **Evaluation Metrics** | MAE, MSE, RMSE, R² |

</div>

### 📏 Evaluation Metrics

The notebook includes detailed performance comparisons across all models using:
- **Train/Test MAE** - Mean Absolute Error
- **Train/Test MSE** - Mean Squared Error
- **Train/Test R²** - Coefficient of Determination
- **Composite Score** - Weighted metric combining all measures

### 📊 Performance Visualizations

Performance visualizations include:
- 📊 Bar charts comparing metrics across models
- 🗺️ Normalized heatmaps for multi-metric view
- 🎯 Stacked and individual radar plots
- 📈 Actual vs Predicted scatter plots

---

## ⚙️ Technical Implementation

### 🏗️ Pipeline Architecture

```python
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('numerical', MinMaxScaler(), numerical_features),
        ('categorical', OneHotEncoder(), categorical_features),
        ('temporal', StandardScaler(), temporal_features)
    ])),
    ('regressor', Model())
])
```

### 🎯 Hyperparameter Tuning

Each model undergoes GridSearchCV optimization with model-specific parameter grids to find optimal configurations.

### 🏗️ Stacking Strategy

The Stacking Regressor combines top-performing base models with a meta-learner to achieve superior prediction accuracy.

---

## 📊 Results

### 🔍 Model Performance Comparison

The following radar plots show the normalized performance metrics (MAE, MSE, R²) for all models across both training and test sets:

![Model Performance Radar Plots](results/performance.png)

**🔑 Key Observations:**
- ⭐ XGBoost and Gradient Boosting show the most balanced performance across all metrics
- 🌲 Random Forest demonstrates strong R² scores on both train and test sets
- 📉 Lasso Regression shows signs of underfitting with lower overall performance
- 🔄 Models with PCA show different metric patterns compared to non-PCA versions

### 📊 Stacking Regressor Performance

The Stacking Regressor combines the best-performing models to achieve superior prediction accuracy:

![Stacking Regressor - Predicted vs Actual](results/stacking%20regressor.png)

**🎯 Performance Highlights:**
- ✅ Strong correlation between predicted and actual prices on both train and test sets
- ✅ Good generalization with similar performance patterns across train/test splits
- ✅ Effective handling of the full price range from low to high-cost products

### 💡 Model Insights

The notebook provides comprehensive analysis including:
- 📊 Feature importance rankings
- 📈 Temporal price trends
- 🏷️ Categorical feature impact analysis
- ⚖️ Model strengths and weaknesses
- 🚀 Recommendations for production deployment

---

## 🔮 Future Enhancements

Potential improvements include:
- 🧠 Deep learning models (Neural Networks)
- 🔧 Additional feature engineering
- 📅 Time series forecasting components
- 🎯 Automated feature selection
- 🔍 Model interpretability tools (SHAP, LIME)
- 🌐 Deployment as REST API

---

## 🛠️ Technologies Used

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **ML Libraries** | scikit-learn, XGBoost |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Development** | Jupyter Notebook |
| **Version Control** | Git, GitHub |

</div>

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available for educational and commercial use under the MIT License.

---

## 👤 Author

**Created by [sys0507](https://github.com/sys0507)**

Feel free to reach out for questions, suggestions, or collaborations!

---

## 🙏 Acknowledgments

- 🎓 **Techlent ML Camp** for project guidance
- 🔬 **Scikit-learn and XGBoost teams** for excellent ML libraries
- 💻 **The open-source community** for tools and inspiration

---

<div align="center">

### ⭐ If you find this project helpful, please consider giving it a star!

**Made with ❤️ for the ML community**

---

> **📌 Note**: The current `sample_data.csv` is placeholder data. For actual price predictions, please use real XoX sales data formatted according to the schema described above.

[⬆ Back to Top](#-xox-price-prediction-model)

</div>
