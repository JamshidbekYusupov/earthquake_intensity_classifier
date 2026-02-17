# Earthquake Intensity Classification Model

## Description

This project leverages machine learning to classify the intensity of earthquakes based on scraped data from the [Korean Meteorological Administration](https://www.weather.go.kr/neng/earthquake/earthquake-korea.do). The model utilizes various classification algorithms to predict the magnitude or intensity of an earthquake from the scraped data.

The data is automatically scraped from the given URL, preprocessed, and used to train a variety of machine learning models to predict earthquake intensity.

## Features

- **Data Scraping**: Automatically scrapes the latest earthquake data from the official KMA website.
- **Data Preprocessing**: Cleans and preprocesses raw data for training.
- **Model Training**: Multiple classifiers are used, including Logistic Regression, Decision Trees, XGBoost, and Random Forests.
- **Evaluation Metrics**: Models are evaluated using metrics like accuracy, precision, recall, and F1-score.
- **Model Saving**: The best-performing model is saved for later use, and metrics are logged.
