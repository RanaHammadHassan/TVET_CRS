Welcome to the guidelines for the Technical and Vocational Education and Training (TVET) Course Recommendation System. To utilize this source code, please follow the steps outlined below.

- TVET-CRS Python Code: TVET-CRS.ipynb
- Dataset: Tvet_data.xlsx

Step 1: Import Python Libraries: 
Imports necessary Python libraries and modules for data manipulation, model building, evaluation, and visualization.

Step 2: Remove the existing Keras Tuner project directory (if exists)

Step 3: Load Data: 
Reads data from an Excel file (Tvet_data.xlsx) into a Pandas DataFrame (df).

Step 4:	Feature Engineering:
-	Displays basic information about the dataset (head(), describe(), info()).
- Visualizes the correlation matrix of numerical features using a heatmap.
- Visualizes the distribution of numerical features using histograms.

Step 5: Data Transformation and Normalization:
- Encodes categorical variables ('Gender', 'Trade') using LabelEncoder.
- Normalizes numerical features using StandardScaler.

Step 6: Data Splitting:
Splits the dataset into training and testing sets using train_test_split.

Step 7: Model Building (Hyperparameter Tuning):
- Defines a function build_model(hp) to construct a neural network model with hyperparameters tuned using Keras Tuner (kt.BayesianOptimization).
- Sets up the tuner to search for the best hyperparameters based on validation accuracy.

Step 8: Search for Best Hyperparameters:
Executes the hyperparameter search (tuner.search) using training data (train_data).

Step 9: Get and Save Best Model:
Retrieves the best model (best_model) found by the tuner and saves it as best_model.h5.

Step 10: Load and Fine-tune Best Model:
- Loads the saved best_model.h5 and compiles it.
- Fine-tunes the best model on the training data with callbacks for early stopping and learning rate reduction.
  
Step 11: Make Predictions:
Uses the fine-tuned model to predict classes (y_pred) and probabilities (y_pred_proba) on the test data (test_data).

Step 12: Evaluate the Model:
Computes various evaluation metrics (accuracy, F1 score, precision, recall, ROC AUC, MCC, Cohen's kappa, MAE, NMAE, RMSE, MRR, NDCG) based on predicted and actual labels (y_test, y_pred).

Step 13: Save Predictions and Evaluation Metrics:
- Saves the predicted vs. actual labels to an Excel file (DL-Model-Two(M2).xlsx).
- Saves the computed evaluation metrics to another Excel file (DLM2Eval_Met.xlsx).
