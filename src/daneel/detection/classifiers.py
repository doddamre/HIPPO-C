"""Classifier interfaces reserved for the detection course exercises."""



from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # figures are only saved to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as SkRandomForest
from sklearn.metrics import (accuracy_score, average_precision_score,
      classification_report, confusion_matrix,
      precision_recall_curve, precision_score,
      recall_score, roc_auc_score, roc_curve)
import joblib
from daneel.detection.data_process import create_balanced_dataset

class RandomForestClassifier:
    """Random-forest transit detector for TESS light curves.

    Parameters
    ----------
    params : dict
        Settings from the ``detection`` section of the YAML file. Keys:
        ``data_path``, ``n_bins``, ``test_size``, ``random_state``,
        ``samples_per_class``, ``use_scaler``, ``output_dir`` and a
        ``random_forest`` dictionary (``n_estimators``, ``max_depth``,
        ``min_samples_leaf``, ``max_features``, ``bootstrap``,
        ``class_weight``, ``oob_score``). All have defaults except
        ``data_path``.
    """
    def __init__(self, params=None):
        params = params or {}
        self.data_path = params.get("data_path", "tess_data.csv")
        self.n_bins = params.get("n_bins", 1000)
        self.test_size = params.get("test_size", 0.2)
        self.random_state = params.get("random_state", 42)
        self.samples_per_class = params.get("samples_per_class", 350)
        self.use_scaler = params.get("use_scaler", False)
        self.output_dir = Path(params.get("output_dir", "rf_results"))
        self.rf_params = {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "class_weight": None,
            "oob_score": True,
        }
        self.rf_params.update(params.get("random_forest", {}) or {})
        self.model = None
        self.scaler = None

    def load_data(self):
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
    
        df = pd.read_csv(self.data_path)
        print(f"Dataset: {df.shape[0]} samples")
    
        flux_cols = [f'flux_{i:04d}' for i in range(self.n_bins)]
        flux_err_cols = [f'flux_err_{i:04d}' for i in range(self.n_bins)]
        X = df[flux_cols].values
        y = df['label'].values
    
        print(f"  Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state, stratify=y)

        np.random.seed(self.random_state)  # augmentation uses numpy's RNG
        X_train, y_train = create_balanced_dataset(
            X_train, y_train, samples_per_class=self.samples_per_class)
        
        if self.use_scaler:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def build_model(self):
        return SkRandomForest(
            random_state=self.random_state, n_jobs=-1, **self.rf_params)

    def fit(self, X, y):
        self.model = self.build_model()
        self.model.fit(X, y)
        if getattr(self.model, "oob_score_", None) is not None:
            print(f"OOB score: {self.model.oob_score_:.4f}")
        return self
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    def evaluate(self, X_test, y_test):
        """Statistics at the default (0.5) and the Youden-J threshold."""
        proba = self.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, proba)
        best = int(np.argmax(tpr - fpr))
        threshold = float(thresholds[best])

        results = {"proba": proba, "threshold": threshold,
                   "auc": roc_auc_score(y_test, proba)}
        for name, thr in (("default", 0.5), ("optimal", threshold)):
            pred = (proba >= thr).astype(int)
            results[name] = {
                "threshold": thr,
                "pred": pred,
                "confusion_matrix": confusion_matrix(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred),
                "accuracy": accuracy_score(y_test, pred),
                "report": classification_report(
                    y_test, pred, target_names=["Non-Planet", "Planet"],
                    digits=4, zero_division=0),
            }
        return results
    
    def format_report(self, results):
        lines = ["Random Forest - TESS transit detection", "",
                 f"Hyperparameters: {self.rf_params}",
                 f"Train samples per class (augmented): {self.samples_per_class}",
                 f"AUC-ROC: {results['auc']:.4f}"]
        if getattr(self.model, "oob_score_", None) is not None:
            lines.append(f"OOB score: {self.model.oob_score_:.4f}")
        for name in ("default", "optimal"):
            r = results[name]
            lines += ["", f"--- {name} threshold = {r['threshold']:.4f} ---",
                      "Confusion matrix [[TN FP] [FN TP]]:",
                      str(r["confusion_matrix"]),
                      f"Precision (planet) = {r['precision']:.4f}",
                      f"Recall (planet)    = {r['recall']:.4f}",
                      f"Accuracy           = {r['accuracy']:.4f}",
                      "", r["report"]]
        return "\n".join(lines)

    def plot_all(self, y_test, results):
        out = self.output_dir
        proba, thr = results["proba"], results["threshold"]
        cm = results["optimal"]["confusion_matrix"]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest")
        fig.colorbar(im, ax=ax)
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=["Non-Planet", "Planet"],
               yticklabels=["Non-Planet", "Planet"],
               xlabel="Predicted", ylabel="True",
               title=f"Confusion Matrix (threshold={thr:.3f})")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm.sum() * 100:.1f}%)",
                        ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out / "confusion_matrix_rf.png", dpi=300)
        plt.close(fig)

        fpr, tpr, _ = roc_curve(y_test, proba)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title=f"ROC Curve (AUC={results['auc']:.4f})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "roc_curve_rf.png", dpi=300)
        plt.close(fig)

        prec, rec, _ = precision_recall_curve(y_test, proba)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rec, prec, linewidth=2)
        ax.set(xlabel="Recall", ylabel="Precision",
               title="Precision-Recall Curve "
                     f"(AP={average_precision_score(y_test, proba):.4f})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "pr_curve_rf.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(proba[y_test == 0], bins=30, alpha=0.6, density=True,
                label="Non-Planet")
        ax.hist(proba[y_test == 1], bins=30, alpha=0.6, density=True,
                label="Planet")
        ax.set(xlabel="Predicted probability (class=1)", ylabel="Density",
               title="Predicted Probability Distributions by Class")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "probability_hist_rf.png", dpi=300)
        plt.close(fig)

    def run(self):
        """Full pipeline: load, train, evaluate, save report/figures/model."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        X_train, X_test, y_train, y_test = self.load_data()
        self.fit(X_train, y_train)
        results = self.evaluate(X_test, y_test)

        text = self.format_report(results)
        print(text)
        (self.output_dir / "report_random_forest.txt").write_text(text)

        self.plot_all(y_test, results)
        joblib.dump(self.model, self.output_dir / "tess_rf_model.joblib")
        np.save(self.output_dir / "rf_optimal_threshold.npy",
                results["threshold"])
        print(f"\nResults saved in {self.output_dir}/")
        return results

class CNNClassifier:
    """Placeholder for the CNN exercise."""

    def fit(self, X, y):
        raise NotImplementedError("CNN training is a course exercise")
