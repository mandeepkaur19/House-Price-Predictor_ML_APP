# Salary Predictor App

This repository contains a **Salary Predictor App** that uses a Simple Linear Regression model to predict employee salaries based on their years of experience. The project includes both a backend for training the model and a frontend built with Streamlit for user interaction.

## Project Structure

- `app.py`: The Streamlit application for user input and salary prediction.
- `Salary_Data.csv`: The dataset containing employee salaries and years of experience.
- `linear_regression_model.pkl`: The trained linear regression model saved for predictions.
- `backend.py`: The script that trains the linear regression model.

## Getting Started

### Prerequisites

Make sure you have Python installed on your machine. You will also need to install the required libraries. You can do this by running:


``` pip install pandas numpy scikit-learn matplotlib streamlit ```

### Running the App
1. Clone the repository to your local machine:

    ``git clone <your-repo-url> ``
   
    ``  cd <your-repo-name> ``

3. Run the Streamlit app:

    `` streamlit run app.py ``

4. Open the provided URL in your web browser to access the app.

### How to Use
1. Enter the number of years of experience in the input field.
2. Click the "Predict Salary" button.
3. The app will display the predicted salary based on the input.
