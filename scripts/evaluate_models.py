"""Evaluate BERT model vs classical baselines (SVM, RandomForest).

Produces accuracy, precision, recall, f1 for each model and saves a short
report to `models/eval_report.json`.
"""
import json
from pathlib import Path

def main():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from moderation_model import ModerationModel
    from utils.preprocess import preprocess_text

    df = pd.read_csv('dataset/labeled.csv')
    df = df.dropna(subset=['text'])[:5000]
    X = df['text'].astype(str).tolist()
    y = df['label'].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Baseline: TF-IDF + SVM
    vect = TfidfVectorizer(max_features=20000, preprocessor=preprocess_text)
    Xtr = vect.fit_transform(X_train)
    Xte = vect.transform(X_test)

    svm = SVC(probability=True)
    svm.fit(Xtr, y_train)
    preds_svm = svm.predict(Xte)
    p_svm, r_svm, f1_svm, _ = precision_recall_fscore_support(y_test, preds_svm, average='binary')

    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(Xtr, y_train)
    preds_rf = rf.predict(Xte)
    p_rf, r_rf, f1_rf, _ = precision_recall_fscore_support(y_test, preds_rf, average='binary')

    # BERT/HF model via ModerationModel wrapper (uses HF if available)
    bert = ModerationModel()
    preds_bert = [1 if bert.predict(t)['is_harmful'] else 0 for t in X_test]
    p_bert, r_bert, f1_bert, _ = precision_recall_fscore_support(y_test, preds_bert, average='binary')

    report = {
        'svm': {'precision': p_svm, 'recall': r_svm, 'f1': f1_svm},
        'rf': {'precision': p_rf, 'recall': r_rf, 'f1': f1_rf},
        'bert': {'precision': p_bert, 'recall': r_bert, 'f1': f1_bert},
    }

    Path('models').mkdir(parents=True, exist_ok=True)
    Path('models/eval_report.json').write_text(json.dumps(report, indent=2))
    print('Saved evaluation report to models/eval_report.json')


if __name__ == '__main__':
    main()
