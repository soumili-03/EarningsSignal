from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_model(df):
    features = [
        'avg_evasiveness', 'avg_sentiment', 'avg_readability', 'avg_QA_similarity',
        'avg_answer_length', 'avg_numeric_density', 'n_questions'
    ]
    X = df[features]
    y = df['beat_miss']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    log_reg_model = LogisticRegression(class_weight='balanced', random_state=42).fit(X_train_scaled, y_train)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_scaled, y_train)

    return {
        "model": log_reg_model,
        "scaler": scaler,
        "features": features,
        "X_test": X_test,
        "y_test": y_test,
        "feature_importances": rf_model.feature_importances_
    }