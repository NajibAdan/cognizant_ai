import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# K is used to define the number of folds that will be used for cross-validation
K = 10

# Split defines the % of data that will be used in the training sample
# 1 - SPLIT = the % used for testing
SPLIT = 0.75

# Loads Data
def load_data(path: str ="data\sales.csv"):
    '''
    This function takes a path string to a CSV file and loads it into
    a Pandas DataFrame.

    Input: path to the csv file
    Return: Pandas Dataframe
    '''
    df = pd.read_csv(path)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df


def create_target_predictor(data: pd.DataFrame = None,target: str = "estimated_stock_pct"):
    '''
    This function takes a pandas dataframe and creates target variables and predictor variables.
    The `target` column becomes the target variables.
    The remaning columns becomes the predictor variables.

    Input:      data-> Pandas DataFrame to create target and predictor variables from
                target-> str which specifies the column to use as the target variable

    Returns:    X-> Predictor variables
                y-> target variables
    '''
    X = data.drop(columns=['estimated_stock_pct'])
    y = data['estimated_stock_pct']
    return X,y

# Trains Model
def train_algorithm_with_cross_validation(
        X: pd.DataFrame = None,
        y: pd.Series = None):
    '''
    This function takes the predictor and target variables and
    trains a Random Forest Regressor model across K folds. Using
    cross-validation, performance metrics will be output for each
    fold during training.

    Inputs: X-> Pandas Dataframe, predictor variables
            y-> Pandas series, target variable
    '''
    
    # list for holding the MAE scores across the K folds
    accuracy = []

    for fold in range(0, K):

        # Instantiate RandomForest algorithm and StandardScaler
        scaler = StandardScaler()
        model = RandomForestRegressor()

        # Create training and test samples
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT, random_state=42)

        # Scale X data, we scale the data because it helps the algorithm to converge
        # and helps the algorithm to not be greedy with large values
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model
        trained_model = model.fit(X_train, y_train)

        # Generate predictions on test sample
        y_pred = trained_model.predict(X_test)

        # Compute accuracy, using mean absolute error
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracy.append(mae)
        
        # Displays the MAE score for the current fold
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")
    
    # Computes average MAE
    print(f"Average MAE : {(sum(accuracy) / len(accuracy)):.2f}")

def run():
    # Load the data first
    df = load_data()
    # Now split the data into predictors and target variables
    X,y = create_target_predictor(df,'estimated_stock_pct')
    # Finally, train the machine learning model
    train_algorithm_with_cross_validation(X,y)

