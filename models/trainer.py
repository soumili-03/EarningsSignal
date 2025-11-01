from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


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

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)

    # Track best models for both algorithms
    best_lr_f1 = 0
    best_lr_model = None
    best_lr_scaler = None
    best_lr_X_val = None
    best_lr_y_val = None
    lr_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    best_rf_f1 = 0
    best_rf_model = None
    best_rf_scaler = None
    best_rf_X_val = None
    best_rf_y_val = None
    rf_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    # Cross-validation loop
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale features
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=40)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        
        # ====== Train Logistic Regression ======
        lr_model = LogisticRegression(class_weight='balanced', random_state=40, max_iter=1000)
        lr_model.fit(X_train_res, y_train_res)
        
        y_val_pred_lr = lr_model.predict(X_val_scaled)
        lr_f1 = f1_score(y_val, y_val_pred_lr, zero_division=0)
        lr_acc = accuracy_score(y_val, y_val_pred_lr)
        lr_prec = precision_score(y_val, y_val_pred_lr, zero_division=0)
        lr_rec = recall_score(y_val, y_val_pred_lr, zero_division=0)
        
        lr_metrics['accuracy'].append(lr_acc)
        lr_metrics['precision'].append(lr_prec)
        lr_metrics['recall'].append(lr_rec)
        lr_metrics['f1'].append(lr_f1)
        
        if lr_f1 > best_lr_f1:
            best_lr_f1 = lr_f1
            best_lr_model = lr_model
            best_lr_scaler = scaler
            best_lr_X_val = X_val
            best_lr_y_val = y_val
        
        # ====== Train Random Forest ======
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            random_state=40,
            class_weight='balanced',
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        rf_model.fit(X_train_res, y_train_res)
        
        y_val_pred_rf = rf_model.predict(X_val_scaled)
        rf_f1 = f1_score(y_val, y_val_pred_rf, zero_division=0)
        rf_acc = accuracy_score(y_val, y_val_pred_rf)
        rf_prec = precision_score(y_val, y_val_pred_rf, zero_division=0)
        rf_rec = recall_score(y_val, y_val_pred_rf, zero_division=0)
        
        rf_metrics['accuracy'].append(rf_acc)
        rf_metrics['precision'].append(rf_prec)
        rf_metrics['recall'].append(rf_rec)
        rf_metrics['f1'].append(rf_f1)
        
        if rf_f1 > best_rf_f1:
            best_rf_f1 = rf_f1
            best_rf_model = rf_model
            best_rf_scaler = scaler
            best_rf_X_val = X_val
            best_rf_y_val = y_val
    
    # Calculate average metrics across all folds
    lr_avg_metrics = {k: sum(v) / len(v) for k, v in lr_metrics.items()}
    rf_avg_metrics = {k: sum(v) / len(v) for k, v in rf_metrics.items()}
    
    # Determine which model is better based on F1 score
    if lr_avg_metrics['f1'] >= rf_avg_metrics['f1']:
        best_model = best_lr_model
        best_scaler = best_lr_scaler
        best_X_val = best_lr_X_val
        best_y_val = best_lr_y_val
        best_model_name = "Logistic Regression"
        print(f"✅ Selected Model: Logistic Regression (F1: {lr_avg_metrics['f1']:.4f})")
    else:
        best_model = best_rf_model
        best_scaler = best_rf_scaler
        best_X_val = best_rf_X_val
        best_y_val = best_rf_y_val
        best_model_name = "Random Forest"
        print(f"✅ Selected Model: Random Forest (F1: {rf_avg_metrics['f1']:.4f})")
    
    # Train final Random Forest for feature importance (always needed)
    scaler_full = StandardScaler().fit(X)
    X_scaled = scaler_full.transform(X)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    
    rf_for_importance = RandomForestClassifier(n_estimators=100, random_state=40)
    rf_for_importance.fit(X_res, y_res)

    return {
        "model": best_model,
        "scaler": best_scaler,
        "features": features,
        "X_test": best_X_val,
        "y_test": best_y_val,
        "feature_importances": rf_for_importance.feature_importances_,
        
        # Model comparison data
        "best_model_name": best_model_name,
        "lr_model": best_lr_model,
        "rf_model": best_rf_model,
        "lr_scaler": best_lr_scaler,
        "rf_scaler": best_rf_scaler,
        "lr_metrics": lr_avg_metrics,
        "rf_metrics": rf_avg_metrics,
        "lr_X_test": best_lr_X_val,
        "lr_y_test": best_lr_y_val,
        "rf_X_test": best_rf_X_val,
        "rf_y_test": best_rf_y_val
    }