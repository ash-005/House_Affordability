### README: House Affordability Classification Project

---

#### **Project Title**:  
**House Affordability Classification**

---

#### **Project Description**:  
This project aims to classify houses into four categories of affordability based on a dataset containing housing attributes. The classification categories are as follows:  

1. **Least Expensive**  
2. **Affordable**  
3. **Luxury**  
4. **Expensive**

Various machine learning models were implemented and evaluated, with **XGBoost** achieving the highest accuracy score of **76.6%**.
---
#### Deployed Site: [House Genie](https://housegenie.streamlit.app/)
---
#### **Dataset:** [California Housing](https://www.kaggle.com/datasets/camnugent/california-housing-prices/data)
#### **Dataset Features**:  
The dataset includes the following attributes:  
- **Longitude**: Geographic coordinate for the house's location.  
- **Latitude**: Geographic coordinate for the house's location.  
- **Housing Median Age**: Median age of houses in the area.  
- **Total Rooms**: Total number of rooms in the house.  
- **Total Bedrooms**: Total number of bedrooms in the house.  
- **Population**: Number of people in the housing block.  
- **Households**: Number of households in the area.  
- **Median Income**: Median income of residents in the area.  
- **Median House Value**: Median value of the house in the area.  
- **Ocean Proximity**: Proximity of the house to the ocean (e.g., Near Ocean, Inland).  

Feature engineering was applied to enhance classification performance.

---

#### **Models Explored**:  
The following machine learning models were trained and evaluated:  
1. **Support Vector Classifier (SVC)**  
2. **Logistic Regression**  
3. **Random Forest Classifier**  
4. **XGBoost Classifier**  

Among these, **XGBoost** emerged as the most accurate, with a classification accuracy of **76.6%**.

---

#### **Project Workflow**:  

1. **Data Preprocessing**:  
   - Handled missing values in `total_bedrooms` by imputing the median.  
   - Normalized numerical features for better model performance.  
   - Encoded the categorical feature `ocean_proximity` using one-hot encoding.  

2. **Feature Engineering**:  
   - Created derived features such as **rooms_per_household**, **population_per_household**, and **bedrooms_per_room**.  

3. **Model Training and Hyperparameter Tuning**:  
   - Implemented the models and fine-tuned hyperparameters using grid search and cross-validation techniques.

4. **Model Evaluation**:  
   - Evaluated all models on accuracy, precision, recall, and F1-score metrics.  

5. **Results**:  
   - The **XGBoost** classifier provided the best results, making it the recommended model for deployment.

---

#### **Dependencies**:  
The project requires the following Python libraries:  
- `numpy`  
- `pandas`  
- `scikit-learn`  
- `xgboost`  
- `matplotlib`  
- `seaborn`

Install all dependencies using:  
```bash
pip install -r requirements.txt
```

---

#### **How to Run the Project**:  

1. Clone the repository:  
   ```bash
   git clone <repository_url>
   cd HouseAffordabilityClassification
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the classification script:  
   ```bash
   python classify_houses.py
   ```

4. The model's accuracy and visualizations of the results will be displayed.

---

#### **Key Results**:  
- **Best-performing Model**: XGBoost  
- **Accuracy**: 76.6%  
- The model effectively categorizes houses into four affordability levels based on input features.

---

#### **Future Scope**:  
- Incorporate additional geographic or demographic features for improved classification.  
- Experiment with deep learning models to explore performance improvements.  
- Deploy the model as a web application for user-friendly access.

---
