"""Evaluate multiple models on the held-out test split saved in `processed/`.

Produces accuracy, precision, recall, f1 for each model and saves a report
to `models/eval_report.json`.
"""
import json
from pathlib import Path


def main():
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / 'utils'))
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
    from utils.preprocess import preprocess_text
    from moderation_model import ModerationModel

    processed_dir = Path(__file__).resolve().parent.parent / 'processed'
    test_path = processed_dir / 'test.csv'
    train_path = processed_dir / 'train.csv'

    if not test_path.exists() or not train_path.exists():
        raise FileNotFoundError("Processed train/test splits not found. Run scripts/train_hf.py to create them.")

    df_train = pd.read_csv(train_path).dropna(subset=['text', 'label'])
    df_test = pd.read_csv(test_path).dropna(subset=['text', 'label'])

    X_train = df_train['text'].astype(str).tolist()
    y_train = df_train['label'].astype(int).tolist()
    X_test = df_test['text'].astype(str).tolist()
    y_test = df_test['label'].astype(int).tolist()

    # TF-IDF vectorizer
    vect = TfidfVectorizer(max_features=20000, preprocessor=preprocess_text)
    Xtr = vect.fit_transform(X_train)
    Xte = vect.transform(X_test)

    results = {}

    # Logistic Regression baseline
    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xtr, y_train)
    preds_lr = lr.predict(Xte)
    p_lr, r_lr, f1_lr, _ = precision_recall_fscore_support(y_test, preds_lr, average='binary')
    results['logistic'] = {'precision': p_lr, 'recall': r_lr, 'f1': f1_lr, 'accuracy': accuracy_score(y_test, preds_lr)}

    # SVM
    svm = SVC(probability=True)
    svm.fit(Xtr, y_train)
    preds_svm = svm.predict(Xte)
    p_svm, r_svm, f1_svm, _ = precision_recall_fscore_support(y_test, preds_svm, average='binary')
    results['svm'] = {'precision': p_svm, 'recall': r_svm, 'f1': f1_svm, 'accuracy': accuracy_score(y_test, preds_svm)}

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(Xtr, y_train)
    preds_rf = rf.predict(Xte)
    p_rf, r_rf, f1_rf, _ = precision_recall_fscore_support(y_test, preds_rf, average='binary')
    results['random_forest'] = {'precision': p_rf, 'recall': r_rf, 'f1': f1_rf, 'accuracy': accuracy_score(y_test, preds_rf)}

    # Hugging Face / ModerationModel wrapper
    bert = ModerationModel()
    preds_bert = [1 if bert.predict(t)['is_harmful'] else 0 for t in X_test]
    p_bert, r_bert, f1_bert, _ = precision_recall_fscore_support(y_test, preds_bert, average='binary')
    results['bert'] = {'precision': p_bert, 'recall': r_bert, 'f1': f1_bert, 'accuracy': accuracy_score(y_test, preds_bert)}

    # Save report
    Path('models').mkdir(parents=True, exist_ok=True)
    report_path = Path('models') / 'eval_report.json'
    report = {'results': results}
    report_path.write_text(json.dumps(report, indent=2))

    print('Evaluation complete. Summary:')
    for name, metrics in results.items():
        print(f"- {name}: acc={metrics['accuracy']:.4f}, p={metrics['precision']:.4f}, r={metrics['recall']:.4f}, f1={metrics['f1']:.4f}")
    print(f"Full report written to {report_path}")


if __name__ == '__main__':
    main()
