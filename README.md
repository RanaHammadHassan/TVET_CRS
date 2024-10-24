Welcome to the guidelines for the Technical and Vocational Education and Training (TVET) Course Recommendation System. To utilize this source code, please follow the steps outlined below.

TVET-CRS Python Code: TVET-CRS.ipynb
Dataset: Tvet_data.xlsx

To execute follow python Code, follow below steps:-

Step 1: Import Python Libraries: 
Imports necessary Python libraries and modules for data manipulation, model building, evaluation, and visualization.

Step 2: Remove the existing Keras Tuner project directory (if exists)

Step 3: Load Data: 
Reads data from an Excel file (Tvet_data.xlsx) into a Pandas DataFrame (df).

Step 4:	Feature Engineering:
-	Displays basic information about the dataset (head(), describe(), info()).
- Visualizes the correlation matrix of numerical features using a heatmap.
- Visualizes the distribution of numerical features using histograms.

5.	Data Transformation and Normalization:
o	Encodes categorical variables ('Gender', 'Trade') using LabelEncoder.
o	Normalizes numerical features using StandardScaler.
6.	Data Splitting:
o	Splits the dataset into training and testing sets using train_test_split.
7.	Model Building (Hyperparameter Tuning):
o	Defines a function build_model(hp) to construct a neural network model with hyperparameters tuned using Keras Tuner (kt.BayesianOptimization).
o	Sets up the tuner to search for the best hyperparameters based on validation accuracy.
8.	Search for Best Hyperparameters:
o	Executes the hyperparameter search (tuner.search) using training data (train_data).
9.	Get and Save Best Model:
o	Retrieves the best model (best_model) found by the tuner and saves it as best_model.h5.
10.	Load and Fine-tune Best Model:
o	Loads the saved best_model.h5 and compiles it.
o	Fine-tunes the best model on the training data with callbacks for early stopping and learning rate reduction (EarlyStopping, ReduceLROnPlateau).
11.	Make Predictions:
o	Uses the fine-tuned model to predict classes (y_pred) and probabilities (y_pred_proba) on the test data (test_data).
12.	Evaluate the Model:
o	Computes various evaluation metrics (accuracy, F1 score, precision, recall, ROC AUC, MCC, Cohen's kappa, MAE, NMAE, RMSE, MRR, NDCG) based on predicted and actual labels (y_test, y_pred).
13.	Save Predictions and Evaluation Metrics:
o	Saves the predicted vs. actual labels to an Excel file (DL-Model-Two(M2).xlsx).
o	Saves the computed evaluation metrics to another Excel file (DLM2Eval_Met.xlsx).
