import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, fbeta_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix, average_precision_score
from sklearn.base import clone
import numpy as np
import warnings
import pandas as pd

class RepeatedNestedCV:
    def __init__(self, estimators, paramSpaces, nRounds=10, nTrials=3, nOuter=5, nInner=3, randomState=42):
        self.estimators = estimators
        self.paramSpaces = paramSpaces
        self.nRounds = nRounds
        self.nTrials = nTrials
        self.nOuter = nOuter
        self.nInner = nInner
        self.randomState = randomState
        self.results = {}

    def _computeMetrics(self, yTrue, yPred, yProb):
        tn, fp, fn, tp = confusion_matrix(yTrue, yPred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics = {
            'MCC': matthews_corrcoef(yTrue, yPred),
            'AUC': roc_auc_score(yTrue, yProb),
            'BalancedAccuracy': balanced_accuracy_score(yTrue, yPred),
            'F1': f1_score(yTrue, yPred),
            'F2': fbeta_score(yTrue, yPred, beta=2),
            'Recall': recall_score(yTrue, yPred),
            'Specificity': specificity,
            'Precision': precision_score(yTrue, yPred),
            'PRAUC': average_precision_score(yTrue, yPred),
            'NPV': npv
        }
        return metrics

    def _optunaObjective(self, trial, estimator, paramSpace, XTrain, yTrain):
        params = {}
        for k, v in paramSpace.items():
            if v['type'] == 'categorical':
                params[k] = trial.suggest_categorical(k, v['values'])
            elif v['type'] == 'float':
                params[k] = trial.suggest_float(k, v['low'], v['high'], log=v.get('log', False))
            elif v['type'] == 'int':
                params[k] = trial.suggest_int(k, v['low'], v['high'])
        clf = clone(estimator).set_params(**params)
        cv = StratifiedKFold(n_splits=self.nInner, shuffle=True, random_state=self.randomState)
        scores = []
        for trainIdx, valIdx in cv.split(XTrain, yTrain):
            clf.fit(XTrain[trainIdx], yTrain[trainIdx])
            yPred = clf.predict(XTrain[valIdx])
            scores.append(matthews_corrcoef(yTrain[valIdx], yPred))
        return np.mean(scores)

    def fit(self, X, y):
        from collections import defaultdict
        for estName, estimator in self.estimators.items():
            self.results[estName] = defaultdict(list)
        for rnd in range(self.nRounds):
            print(f"Now starting round {rnd}.\n\n\n")
            outerCV = StratifiedKFold(n_splits=self.nOuter, shuffle=True, random_state=self.randomState + rnd)
            for trainIdx, testIdx in outerCV.split(X, y):
                XTrain, XTest = X[trainIdx], X[testIdx]
                yTrain, yTest = y[trainIdx], y[testIdx]
                for estName, estimator in self.estimators.items():
                    paramSpace = self.paramSpaces[estName]
                    study = optuna.create_study(direction='maximize')
                    func = lambda trial: self._optunaObjective(trial, estimator, paramSpace, XTrain, yTrain)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        study.optimize(func, n_trials=self.nTrials, show_progress_bar=False)
                    bestParams = study.best_params
                    bestClf = clone(estimator).set_params(**bestParams)
                    bestClf.fit(XTrain, yTrain)
                    yPred = bestClf.predict(XTest)
                    yProb = bestClf.predict_proba(XTest)[:, 1] if hasattr(bestClf, 'predict_proba') else bestClf.decision_function(XTest)
                    metrics = self._computeMetrics(yTest, yPred, yProb)
                    for m, v in metrics.items():
                        self.results[estName][m].append(v)
        return self

    def getResults(self):
        summary = {}
        for estName, metrics in self.results.items():
            summary[estName] = {m: (np.median(v), np.percentile(v, [2.5, 97.5])) for m, v in metrics.items()}
        return pd.DataFrame(summary)
