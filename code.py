#Loading the necessary Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from collections import Counter
import shap
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import mode
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Loading csv dataset
# Original path has been replaced
combined_matches = pd.read_csv('data/combined_matches.csv')

# Exploratory Data Analysis

# Filtering out relegated clubs BEFORE further processing
Relegated = ["Leicester City", "Ipswich Town", "Southampton"]        
mask = ~(
    combined_matches["home_team_name"].isin(Relegated) |
    combined_matches["away_team_name"].isin(Relegated)
)
combined_matches = combined_matches[mask].reset_index(drop=True)
print(f"Dataset size after drop: {combined_matches.shape[0]} matches")

# Displaying first 5 rows of a dataset
combined_matches.head()

# Displaying last 5 rows of that dataset
combined_matches.tail()

# Displaying concise information about a dataset
combined_matches.info()

# Displaying number of rows and columns
combined_matches.shape

# Checking for duplicates using the function
# Function to check for duplicates
def check_duplicates(combined_matches):
    duplicates = combined_matches[combined_matches.duplicated()]
    if not duplicates.empty:
        print("Duplicate rows found:")
        print(duplicates)
    else:
        print("No duplicate rows found.")
        
# Call the function with EPL_Matches DataFrame
check_duplicates(combined_matches)       

# Counting missing values as boolean 1 for missing values 0 for non missing values
combined_matches.isnull().sum()

# Displaying decriptive statistics about a dataset
combined_matches.describe()

# Displaying outliers by calcuating kurtosis
# High kurtosis (>3) indicates a large number of outliers and low kurtosis (<3) a lack of outliers
# Specifying the columns for which you want to calculate kurtosis
columns_of_interest = ['home_team_goal_count', 'away_team_goal_count', 'total_goal_count', 'total_goals_at_half_time', 'home_team_goal_count_half_time', 'away_team_goal_count_half_time']
kurtosis_values = combined_matches[columns_of_interest].kurtosis()
print(kurtosis_values)

# Group by home and away teams to calculate total goals scored and plot Bar Graph
home_team_goals = df.groupby('home_team_name')['home_team_goal_count'].sum()
away_team_goals = df.groupby('away_team_name')['away_team_goal_count'].sum()
# Combine the goals into a single DataFrame
team_goals = pd.DataFrame({
    'Home Goals': home_team_goals,
    'Away Goals': away_team_goals
})

# Add a Total Goals column
team_goals['Total Goals'] = team_goals['Home Goals'] + team_goals['Away Goals']
# Sort by total goals for better visualization
team_goals = team_goals.sort_values(by='Total Goals', ascending=False)
# Plot the goals
team_goals[['Home Goals', 'Away Goals']].plot(kind='bar', stacked=True, figsize=(12, 8), color=['skyblue', 'salmon'])
plt.title('Goals Scored by Teams (Home and Away)')
plt.xlabel('Team Name')
plt.ylabel('Number of Goals')
plt.xticks(rotation=90)
plt.legend(title='Goal Type')
plt.tight_layout()
plt.show()

# Calculate recent performance (last 5 matches) and visualise in a graph
def calculate_recent_form(data, team_column, goals_column, recent_n=5):
    recent_form = (
        data.groupby(team_column)[goals_column]
        .rolling(recent_n)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return recent_form

# Add recent form stats to dataset
df['recent_home_goals'] = calculate_recent_form(
    df, 'home_team_name', 'home_team_goal_count', recent_n=5
)
df['recent_away_goals'] = calculate_recent_form(
    df, 'away_team_name', 'away_team_goal_count', recent_n=5
)

df['recent_home_conceded'] = calculate_recent_form(
    df, 'home_team_name', 'away_team_goal_count', recent_n=5
)
df['recent_away_conceded'] = calculate_recent_form(
    df, 'away_team_name', 'home_team_goal_count', recent_n=5
)

# Filter recent performance for upcoming matches
upcoming_analysis_df['Recent Home Avg Goals Scored'] = upcoming_analysis_df['Home Team'].map(
    df.groupby('home_team_name')['recent_home_goals'].mean()
)
upcoming_analysis_df['Recent Away Avg Goals Scored'] = upcoming_analysis_df['Away Team'].map(
    df.groupby('away_team_name')['recent_away_goals'].mean()
)
upcoming_analysis_df['Recent Home Avg Goals Conceded'] = upcoming_analysis_df['Home Team'].map(
    df.groupby('home_team_name')['recent_home_conceded'].mean()
)
upcoming_analysis_df['Recent Away Avg Goals Conceded'] = upcoming_analysis_df['Away Team'].map(
    df.groupby('away_team_name')['recent_away_conceded'].mean()
)

# Sort the DataFrame by index
upcoming_analysis_df_sorted = upcoming_analysis_df.sort_index()

# Display sorted DataFrame
print("\nUpcoming Matches with Recent Performance Stats (Sorted by Index):")
print(upcoming_analysis_df_sorted)

# Bar graph visualization for sorted recent performance
plt.figure(figsize=(12, 6))

# Extracting relevant columns for visualization from the sorted DataFrame
home_teams_sorted = upcoming_analysis_df_sorted['Home Team']
away_teams_sorted = upcoming_analysis_df_sorted['Away Team']
home_avg_goals_sorted = upcoming_analysis_df_sorted['Recent Home Avg Goals Scored']
away_avg_goals_sorted = upcoming_analysis_df_sorted['Recent Away Avg Goals Scored']
home_avg_conceded_sorted = upcoming_analysis_df_sorted['Recent Home Avg Goals Conceded']
away_avg_conceded_sorted = upcoming_analysis_df_sorted['Recent Away Avg Goals Conceded']

# Plot for Home Team Performance
plt.barh(
    home_teams_sorted + " (H)",
    home_avg_goals_sorted,
    color='skyblue',
    edgecolor='black',
    label='Home Avg Goals Scored'
)
plt.barh(
    home_teams_sorted + " (H)",
    -home_avg_conceded_sorted,
    color='lightcoral',
    edgecolor='black',
    label='Home Avg Goals Conceded'
)

# Plot for Away Team Performance
plt.barh(
    away_teams_sorted + " (A)",
    away_avg_goals_sorted,
    color='green',
    edgecolor='black',
    label='Away Avg Goals Scored'
)
plt.barh(
    away_teams_sorted + " (A)",
    -away_avg_conceded_sorted,
    color='orange',
    edgecolor='black',
    label='Away Avg Goals Conceded'
)

# Add titles and labels
plt.title('Recent Performance of Upcoming Matches (Sorted by Index)', fontsize=16)
plt.xlabel('Goals (Positive = Scored, Negative = Conceded)', fontsize=12)
plt.ylabel('Teams', fontsize=12)
plt.axvline(0, color='black', linewidth=0.8)  # Add a vertical line at 0
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
# Show the plot
plt.show()


# Data Preprocessing and Feature Engineering
# Function to determine match results numerically
def determine_result_numeric(row):
    if row['home_team_goal_count'] > row['away_team_goal_count']:
        return 1, 0  # Home team wins
    elif row['home_team_goal_count'] < row['away_team_goal_count']:
        return 0, 1  # Away team wins
    else:
        return 2, 2  # Draw
# Applying the function to each row
combined_matches['home_team_result_numeric'], combined_matches['away_team_result_numeric'] = zip(*combined_matches.apply(determine_result_numeric, axis=1))
# Displaying the DataFrame
print(combined_matches.head())

# Creating and encoding over 2.5 column as binary
combined_matches['home_team_goal_count'] = pd.to_numeric(combined_matches['home_team_goal_count'], errors='coerce')
combined_matches['away_team_goal_count'] = pd.to_numeric(combined_matches['away_team_goal_count'], errors='coerce')
combined_matches['total_goal_count'] = pd.to_numeric(combined_matches['total_goal_count'], errors='coerce')
# Add a new column to indicate if the total goals were over 1.5
combined_matches['over_2_5_goals'] = combined_matches['total_goal_count'].apply(lambda x: 1 if x > 2.5 else 0)
# Display the updated DataFrame
print(combined_matches.head())

# Creating goal_difference column
combined_matches['goal_difference'] = combined_matches['home_team_goal_count'] - combined_matches['away_team_goal_count']
# Display the updated DataFrame
print(combined_matches.head())

# Function to calculate BTS Goals as Binary
combined_matches['home_team_goal_count'] = pd.to_numeric(combined_matches['home_team_goal_count'], errors='coerce')
combined_matches['away_team_goal_count'] = pd.to_numeric(combined_matches['away_team_goal_count'], errors='coerce')
combined_matches['total_goal_count'] = pd.to_numeric(combined_matches['total_goal_count'], errors='coerce')
# Adding a new column to indicate if both teams scored
def both_teams_to_score(row):
    home_goals = row['home_team_goal_count']
    away_goals = row['away_team_goal_count']
    return 1 if home_goals > 0 and away_goals > 0 else 0
combined_matches['both_teams_to_score'] = combined_matches.apply(both_teams_to_score, axis=1)
# Displaying the updated DataFrame
print(combined_matches.head())

# Creating half_time_goal_difference column
combined_matches['half_time_goal_difference'] = combined_matches['home_team_goal_count_half_time'] - combined_matches['away_team_goal_count_half_time']
# Display the updated DataFrame
print(combined_matches.head())

# Creating corner_difference column
combined_matches['corner_difference'] = combined_matches['home_team_corner_count'] - combined_matches['away_team_corner_count']
# Display the updated DataFrame
print(combined_matches.head())

# Creating shots_difference column
combined_matches['shots_on_target_difference'] = combined_matches['home_team_shots_on_target'] - combined_matches['away_team_shots_on_target']
# Display the updated DataFrame
print(combined_matches.head())

# Creating interaction terms
combined_matches['home_team_possession'] = combined_matches['home_team_possession'] /100
combined_matches['away_team_possession'] = combined_matches['away_team_possession'] /100
# Creating interaction terms for the home team
combined_matches['home_shots_possession_interaction'] = combined_matches['home_team_shots_on_target'] * combined_matches['home_team_possession']
# Creating interaction terms for the away team
combined_matches['away_shots_possession_interaction'] = combined_matches['away_team_shots_on_target'] * combined_matches['away_team_possession']
# Printing the first few rows to check the new interaction columns
print(combined_matches[['home_shots_possession_interaction', 'away_shots_possession_interaction']].head())

# Creating shots_possession_interaction_difference column
combined_matches['shots_possession_interaction_difference'] = combined_matches['home_shots_possession_interaction'] - combined_matches['away_shots_possession_interaction']
# Display the updated DataFrame
print(combined_matches.head())

# Creating possession_difference column
combined_matches['possession_difference'] = combined_matches['home_team_possession'] - combined_matches['away_team_possession']
# Display the updated DataFrame
print(combined_matches.head())

# Creating interaction terms for the home team
combined_matches['home_shots_corner_interaction'] = combined_matches['home_team_shots_on_target'] * combined_matches['home_team_corner_count']
# Creating interaction terms for the away team
combined_matches['away_shots_corner_interaction'] = combined_matches['away_team_shots_on_target'] * combined_matches['home_team_corner_count']
# Printing the first few rows to check the new interaction columns
print(combined_matches[['home_shots_corner_interaction', 'away_shots_corner_interaction']].head())

# Creating shots_corner_interaction_difference column
combined_matches['shots_corner_interaction_difference'] = combined_matches['home_shots_corner_interaction'] - combined_matches['away_shots_corner_interaction']
# Display the updated DataFrame
print(combined_matches.head())

# Save the modified DataFrame with all columns to a new CSV file
# Previous combined_matches dataset have been saved as modified_combined_matches
# Going forward  modified_combined_matches dataset will be used
# Function to convert string representation to numeric lists
def convert_to_numeric_lists(series):
    return series.apply(lambda x: pd.Series(eval(x)))

# Convert home_team_name_recent_result column
modified_combined_matches[['home_recent_result_0', 'home_recent_result_1', 'home_recent_result_2', 'home_recent_result_3', 'home_recent_result_4']] = convert_to_numeric_lists(modified_combined_matches['home_team_name_recent_result'])

# Convert away_team_name_recent_result column
modified_combined_matches[['away_recent_result_0', 'away_recent_result_1', 'away_recent_result_2', 'away_recent_result_3', 'away_recent_result_4']] = convert_to_numeric_lists(modified_combined_matches['away_team_name_recent_result'])

# Drop original columns from the Dataframe
modified_combined_matches.drop(['home_team_name_recent_result', 'away_team_name_recent_result'], axis=1, inplace=True)
# Display the updated DataFrame
print(modified_combined_matches.head())

# Drop unwated or categorical columns from the Dataframe
columns_to_temporarily_drop = [
    'timestamp', 'status', 'home_team_yellow_cards', 'away_team_yellow_cards', 'home_team_red_cards', 'away_team_red_cards',
    'total_goals_at_half_time', 'home_team_shots_off_target', 'away_team_shots_off_target', 'away_team_corner_count', 
    'stadium_name', 'possession_difference', 'home_team_corner_count', 'total_goal_count', 'corner_difference',
    'home_team_possession', 'away_team_possession'
]
# Drop columns
modified_combined_matches = modified_combined_matches.drop(columns=[col for col in columns_to_temporarily_drop if col in modified_combined_matches.columns])
# === Aggregate Head-to-Head & Form Features ===
h2h_columns = [f'h2h_{i}' for i in range(5)]
home_recent_columns = [f'home_recent_result_{i}' for i in range(5)]
away_recent_columns = [f'away_recent_result_{i}' for i in range(5)]
modified_combined_matches['h2h_home_wins'] = modified_combined_matches[h2h_columns].apply(lambda row: (row == 1).sum(), axis=1)
modified_combined_matches['h2h_home_losses'] = modified_combined_matches[h2h_columns].apply(lambda row: (row == 0).sum(), axis=1)
modified_combined_matches['h2h_draws'] = modified_combined_matches[h2h_columns].apply(lambda row: (row == 2).sum(), axis=1)
modified_combined_matches['home_recent_wins'] = modified_combined_matches[home_recent_columns].apply(lambda row: (row == 1).sum(), axis=1)
modified_combined_matches['home_recent_losses'] = modified_combined_matches[home_recent_columns].apply(lambda row: (row == 0).sum(), axis=1)
modified_combined_matches['home_recent_draws'] = modified_combined_matches[home_recent_columns].apply(lambda row: (row == 2).sum(), axis=1)
modified_combined_matches['away_recent_wins'] = modified_combined_matches[away_recent_columns].apply(lambda row: (row == 1).sum(), axis=1)
modified_combined_matches['away_recent_losses'] = modified_combined_matches[away_recent_columns].apply(lambda row: (row == 0).sum(), axis=1)
modified_combined_matches['away_recent_draws'] = modified_combined_matches[away_recent_columns].apply(lambda row: (row == 2).sum(), axis=1)
modified_combined_matches.drop(h2h_columns + home_recent_columns + away_recent_columns, axis=1, inplace=True)
print(modified_combined_matches.head())

# Save the modified DataFrame with all columns to a new CSV file
# Going forward will use feature_modified_combined_matches as our dataset and renamed it as df
# Assuming 'feature_modified_combined_matches' is already loaded as a DataFrame
df = feature_modified_combined_matches.copy()

#Training Machine Learning and Evaluating the Models
# Training mulitinomial logistic regression, support vector machines, and random forest
# Evaluating the model using test accuracy, cross validation, and confussion matrix

# --- Drop Unnecessary Columns ---
columns_to_drop = [
    'timestamp', 'date_GMT', 'status', 'home_team_name', 'away_team_name',
    'home_team_yellow_cards', 'away_team_yellow_cards', 'home_team_red_cards',
    'away_team_red_cards', 'total_goals_at_half_time', 'home_team_shots_off_target',
    'away_team_shots_off_target', 'away_team_goal_count', 'away_team_corner_count',
    'stadium_name', 'possession_difference', 'home_team_corner_count',
    'total_goal_count', 'corner_difference', 'home_team_goal_count',
    'home_team_possession', 'away_team_possession'
]
df_temp = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# --- Use Selected Features ---
top_features = [
    'home_team_goal_count_half_time', 'away_team_goal_count_half_time',
    'shots_on_target_difference', 'home_team_shots_on_target', 'away_team_shots_on_target',
    'half_time_goal_difference', 'home_shots_possession_interaction',
    'away_shots_possession_interaction', 'shots_possession_interaction_difference',
    'shots_corner_interaction_difference', 'h2h_home_wins', 'h2h_home_losses',
    'h2h_draws', 'home_recent_wins', 'home_recent_losses', 'home_recent_draws',
    'away_recent_wins', 'away_recent_losses', 'away_recent_draws'
]
X_selected = df_temp[[col for col in top_features if col in df_temp.columns]]

# --- Targets ---
targets = {
    'unified_result': df_temp['unified_result'],
    'over_2_5_goals': df_temp['over_2_5_goals'],
    'both_teams_to_score': df_temp['both_teams_to_score']
}

# --- Train-Test Split ---
X_train, X_test, y_train_dict, y_test_dict = {}, {}, {}, {}
for target, y in targets.items():
    X_train[target], X_test[target], y_train_dict[target], y_test_dict[target] = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nClass distribution for {target}:")
    print(y_train_dict[target].value_counts())

# --- Apply SMOTE selectively ---
def apply_smote(X_train, y_train, threshold=0.3):
    class_distribution = Counter(y_train)
    min_class = min(class_distribution.values())
    max_class = max(class_distribution.values())
    imbalance_ratio = min_class / max_class
    if imbalance_ratio < threshold:
        print(f"Applying SMOTE: Before balancing: {class_distribution}")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {Counter(y_train_res)}")
        return X_train_res, y_train_res
    else:
        print(f"No SMOTE applied. Class distribution: {class_distribution}")
        return X_train, y_train

for target in y_train_dict:
    print(f"\nProcessing target: {target}")
    X_train[target], y_train_dict[target] = apply_smote(X_train[target], y_train_dict[target])

# --- Scale Features ---
scaler = StandardScaler()
for target in X_train:
    X_train[target] = scaler.fit_transform(X_train[target])
    X_test[target] = scaler.transform(X_test[target])

# --- Parameter Grids ---
param_grid_logistic = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs']}
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'probability': [True]}

best_params = {}

# --- Logistic Regression (multiclass) ---
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_logistic,
                    cv=StratifiedKFold(n_splits=10), scoring='accuracy')
grid.fit(X_train['unified_result'], y_train_dict['unified_result'])
best_params['unified_result'] = grid.best_params_

# --- Binary targets Logistic Regression ---
for target in ['over_2_5_goals', 'both_teams_to_score']:
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_logistic,
                        cv=StratifiedKFold(n_splits=10), scoring='accuracy')
    grid.fit(X_train[target], y_train_dict[target])
    best_params[target] = grid.best_params_

# --- Unified Result: add SVM & Random Forest ---
grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
grid_svm.fit(X_train['unified_result'], y_train_dict['unified_result'])
best_params_svm = grid_svm.best_params_

grid_rf_unified = GridSearchCV(RandomForestClassifier(random_state=42),
                               param_grid_rf, cv=StratifiedKFold(n_splits=5),
                               scoring='accuracy')
grid_rf_unified.fit(X_train['unified_result'], y_train_dict['unified_result'])
best_params_rf_unified = grid_rf_unified.best_params_

# --- Train all unified_result models ---
models_unified = {
    'LogReg': LogisticRegression(**best_params['unified_result']),
    'RandomForest': RandomForestClassifier(**best_params_rf_unified, random_state=42),
    'SVM': SVC(**best_params_svm, random_state=42)
}

for name, model in models_unified.items():
    scores = cross_val_score(model, X_train['unified_result'], y_train_dict['unified_result'], cv=5)
    print(f"{name} | CV Mean: {scores.mean():.2%} | Std: {scores.std():.2%}")
    model.fit(X_train['unified_result'], y_train_dict['unified_result'])
    joblib.dump(model, f"unified_result_{name}_model.joblib")

# --- Compare test predictions ---
predictions = {}
for name, model in models_unified.items():
    y_pred = model.predict(X_test['unified_result'])
    predictions[name] = y_pred
    print(f"\n{name} Test Accuracy: {accuracy_score(y_test_dict['unified_result'], y_pred):.2%}")
    print(classification_report(y_test_dict['unified_result'], y_pred, zero_division=0))

# --- Consensus Voting ---
preds_array = np.array(list(predictions.values()))
final_preds, _ = mode(preds_array, axis=0)
final_preds = final_preds.flatten()

print("\nConsensus (Majority Vote) Results:")
print(f"Accuracy: {accuracy_score(y_test_dict['unified_result'], final_preds):.2%}")
print(classification_report(y_test_dict['unified_result'], final_preds, zero_division=0))

# --- Save all models and preprocessing ---
joblib.dump(models_unified, 'unified_result_all_models.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(top_features, 'top_features.joblib')
print("\nAll models and artifacts saved successfully.")

# Plot class distribution for each target variable
fig, axes = plt.subplots(3, 2, figsize=(12, 12))  # 3 rows, 2 columns
target_names = ['unified_result', 'over_2_5_goals', 'both_teams_to_score']
titles = ["Unified Result", "Over 2.5 Goals", "Both Teams to Score"]
for i, target in enumerate(target_names):
    # Training data
    sns.barplot(x=y_train_dict[target].value_counts().index, 
                y=y_train_dict[target].value_counts().values, 
                ax=axes[i, 0], palette="coolwarm")
    axes[i, 0].set_title(f"Training Data - {titles[i]}")
    axes[i, 0].set_ylabel("Count")
    
    # Test data
    sns.barplot(x=y_test_dict[target].value_counts().index, 
                y=y_test_dict[target].value_counts().values, 
                ax=axes[i, 1], palette="coolwarm")
    axes[i, 1].set_title(f"Test Data - {titles[i]}")
    axes[i, 1].set_ylabel("Count")
plt.tight_layout()
plt.show()


# SHAP explainer
# === 1. Target key (changed from binary to multi-class)
target_key = 'unified_result'

# === 2. Get selected features from SelectKBest
selected_features = [feat for feat, score in selected_features_dict[target_key]]
X_selected = X[selected_features]

# === 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, targets[target_key], test_size=0.2, random_state=42
)

# === 4. Scale and preserve column names
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=selected_features)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=selected_features)

# === 5. Train logistic regression model (multi-class)
model = LogisticRegression(max_iter=1000, multi_class='ovr')  # One-vs-Rest for multi-class
model.fit(X_train_scaled, y_train)


# === 6. SHAP explainer
explainer = shap.Explainer(model, X_train_scaled)
shap_values = explainer(X_test_scaled)

# === 7. Create segmented SHAP bar chart for multi-class classification
def create_multiclass_segmented_shap_bar(shap_values, feature_names, class_names):
    """
    Create a segmented bar chart showing SHAP contributions for each class
    """
    # shap_values.values shape: (n_samples, n_features, n_classes)
    shap_vals = shap_values.values
    n_classes = shap_vals.shape[2]
    
    # Calculate mean absolute SHAP values for each feature and class
    class_contributions = []
    for class_idx in range(n_classes):
        class_shap = shap_vals[:, :, class_idx]  # All samples, all features, specific class
        # Mean absolute SHAP value for each feature for this class
        mean_abs_shap = np.mean(np.abs(class_shap), axis=0)
        class_contributions.append(mean_abs_shap)
    
    class_contributions = np.array(class_contributions)  # Shape: (n_classes, n_features)
    
    # Calculate total impact per feature (sum across all classes)
    total_impact = np.sum(class_contributions, axis=0)
    
    # Sort features by total impact (most important first)
    sorted_idx = np.argsort(total_impact)[::-1]
    
    # Create the plot with reversed y-axis order (most important at top)
    y_pos = np.arange(len(feature_names))
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Colors for each class
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Create stacked horizontal bars
    left_positions = np.zeros(len(feature_names))
    
    for class_idx in range(n_classes):
        class_contrib = class_contributions[class_idx][sorted_idx]
        ax.barh(y_pos, class_contrib, 
                left=left_positions,
                label=class_names[class_idx], 
                color=colors[class_idx % len(colors)], 
                alpha=0.8)
        left_positions += class_contrib
    
    # Customize the plot - reverse y-axis to put most important features at top
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.invert_yaxis()  # This puts the most important features at the top
    ax.set_xlabel('mean(|SHAP value|) (average impact on model output magnitude)')
    ax.set_title('SHAP Feature Importance: Unified Result Multi-class Classification')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

# === 8. Define result mappings for readability
result_mappings = {
    'unified_result': {'Home Win': 'Home Win', 'Away Win': 'Away Win', 'Draw': 'Draw'}
}

# Get class names from the model or data
class_names = ['Home Win', 'Away Win', 'Draw']  # Based on your mapping

# === 9. Show the chart
fig = create_multiclass_segmented_shap_bar(shap_values, selected_features, class_names)
plt.show()

# === 10. Optional: Create individual class SHAP plots
def create_individual_class_shap_plots(shap_values, feature_names, class_names):
    """
    Create separate SHAP summary plots for each class
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    for class_idx, class_name in enumerate(class_names):
        # Extract SHAP values for this specific class
        class_shap_values = shap_values.values[:, :, class_idx]
        
        # Calculate mean absolute SHAP values for sorting
        mean_abs_shap = np.mean(np.abs(class_shap_values), axis=0)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        
        # Create bar plot for this class with most important features at top
        y_pos = np.arange(len(feature_names))
        axes[class_idx].barh(y_pos, mean_abs_shap[sorted_idx], 
                           color=f'C{class_idx}', alpha=0.7)
        axes[class_idx].set_yticks(y_pos)
        axes[class_idx].set_yticklabels([feature_names[i] for i in sorted_idx])
        axes[class_idx].invert_yaxis()  # Most important features at top
        axes[class_idx].set_xlabel('mean(|SHAP value|)')
        axes[class_idx].set_title(f'{class_name} - Feature Importance')
        axes[class_idx].grid(True, alpha=0.3, axis='x')
        axes[class_idx].spines['top'].set_visible(False)
        axes[class_idx].spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

# === 11. Show individual class plots
fig_individual = create_individual_class_shap_plots(shap_values, selected_features, class_names)
plt.show()
# === 12. Optional: Print feature importance summary
print("Feature Importance Summary for Unified Result:")
print("=" * 50)

# Calculate overall feature importance (sum across all classes)
shap_vals = shap_values.values
overall_importance = np.mean(np.sum(np.abs(shap_vals), axis=2), axis=0)
sorted_idx = np.argsort(overall_importance)[::-1]

for i, feat_idx in enumerate(sorted_idx):
    feature_name = selected_features[feat_idx]
    importance = overall_importance[feat_idx]
    print(f"{i+1:2d}. {feature_name:<30} | Importance: {importance:.4f}")

# Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# And the last step is to make predictions on future matches to predict their outcomes



    
    






