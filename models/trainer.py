from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score


def train_model(df):
    features = [
        'avg_evasiveness', 'avg_sentiment', 'avg_sentiment_gap',
        'avg_readability_kincaid', 'avg_readability_ease', 'avg_QA_similarity',
        'avg_answer_length', 'avg_numeric_density', 'avg_lm_sentiment',
        'avg_lexical_diversity', 'avg_complex_word_ratio', 'avg_hedge_to_modal_ratio',
        'avg_dale_chall_score', 'avg_sentiment_polarity', 'avg_modal_ratio_verbs',
        'avg_coleman_liau_index', 'avg_filler_freq', 'avg_hedge_freq',
        'avg_passive_rate', 'n_questions'
    ]
    
    features = [f for f in features if f in df.columns]
    X = df[features].values
    y = df['beat_miss'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_f1 = 0
    best_model = None
    best_scaler = None
    best_X_val = None
    best_y_val = None


    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale features
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        
        # Train model
        model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        model.fit(X_train_res, y_train_res)
        
        # Evaluate
        y_val_pred = model.predict(X_val_scaled)
        f1 = f1_score(y_val, y_val_pred)


        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_scaler = scaler
            best_X_val = X_val
            best_y_val = y_val
    
    # Train Random Forest for feature importance
    scaler_full = StandardScaler().fit(X)
    X_scaled = scaler_full.transform(X)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_res, y_res)

    return {
        "model": best_model,
        "scaler": best_scaler,
        "features": features,
        "X_test": best_X_val,
        "y_test": best_y_val,
        "feature_importances": rf_model.feature_importances_
    }
