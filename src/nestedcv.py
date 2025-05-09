import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import (matthews_corrcoef, roc_auc_score, balanced_accuracy_score,
                             f1_score, fbeta_score, recall_score, precision_score, confusion_matrix,
                             precision_recall_curve, auc)
from sklearn.impute import SimpleImputer
import optuna
import warnings

class RepeatedNestedCV:
    def __init__(self, x, y, estimators, paramSpaces, repeats=10, outerLoopsN=5, innerLoopsN=3, seed=42, nTrials=30, scoringMetric="mcc"):
        self.x = x
        self.y = y
        self.estimators = estimators
        self.paramSpaces = paramSpaces
        self.repeats = repeats
        self.outerLoopsN = outerLoopsN
        self.innerLoopsN = innerLoopsN
        self.seed = seed
        self.nTrials = nTrials
        self.scoringMetric = scoringMetric
        self.results = {name: [] for name in estimators}

    def fit(self):
        rskf = RepeatedStratifiedKFold(
            n_splits=self.outerLoopsN, n_repeats=self.repeats, random_state=self.seed
        )
        for foldIdx, (trainIdx, testIdx) in enumerate(rskf.split(self.x, self.y)):
            xTrain, xTest = self.x[trainIdx], self.x[testIdx]
            yTrain, yTest = self.y[trainIdx], self.y[testIdx]
            for name, estimator in self.estimators.items():
                # Inner CV with Optuna for hyperparameter tuning
                def objective(trial):
                    params = {}
                    for paramName, paramSpace in self.paramSpaces[name].items():
                        if isinstance(paramSpace, list):
                            params[paramName] = trial.suggest_categorical(paramName, paramSpace)
                        elif isinstance(paramSpace, tuple) and len(paramSpace) == 2:
                            params[paramName] = trial.suggest_float(paramName, paramSpace[0], paramSpace[1])
                    clf = estimator.set_params(**params)
                    innerCv = StratifiedKFold(n_splits=self.innerLoopsN, shuffle=True, random_state=self.seed)
                    scores = []
                    for innerTrainIdx, innerValIdx in innerCv.split(xTrain, yTrain):
                        xInnerTrain, xInnerVal = xTrain[innerTrainIdx], xTrain[innerValIdx]
                        yInnerTrain, yInnerVal = yTrain[innerTrainIdx], yTrain[innerValIdx]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            clf.fit(xInnerTrain, yInnerTrain)
                        yPred = clf.predict(xInnerVal)
                        if self.scoringMetric == "mcc":
                            score = matthews_corrcoef(yInnerVal, yPred)
                        elif self.scoringMetric == "roc_auc":
                            yProb = clf.predict_proba(xInnerVal)[:, 1]
                            score = roc_auc_score(yInnerVal, yProb)
                        else:
                            score = f1_score(yInnerVal, yPred)
                        scores.append(score)
                    return np.mean(scores)
                study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed))
                study.optimize(objective, n_trials=self.nTrials, show_progress_bar=False)
                bestParams = study.best_params
                bestClf = estimator.set_params(**bestParams)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    bestClf.fit(xTrain, yTrain)
                yPred = bestClf.predict(xTest)
                if hasattr(bestClf, "predict_proba"):
                    yProb = bestClf.predict_proba(xTest)[:, 1]
                elif hasattr(bestClf, "decision_function"):
                    yProb = bestClf.decision_function(xTest)
                else:
                    yProb = yPred
                # Compute metrics
                mcc = matthews_corrcoef(yTest, yPred)
                rocAuc = roc_auc_score(yTest, yProb)
                ba = balanced_accuracy_score(yTest, yPred)
                f1 = f1_score(yTest, yPred)
                recall = recall_score(yTest, yPred)
                precision = precision_score(yTest, yPred)
                # f2 = f1_score(yTest, yPred, beta=2)
                f2 = fbeta_score(yTest, yPred, beta=2)
                tn, fp, fn, tp = confusion_matrix(yTest, yPred).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                pr, re, _ = precision_recall_curve(yTest, yProb)
                prauc = auc(re, pr)
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                self.results[name].append({
                    "mcc": mcc,
                    "roc_auc": rocAuc,
                    "balanced_accuracy": ba,
                    "f1": f1,
                    "f2": f2,
                    "recall": recall,
                    "specificity": specificity,
                    "precision": precision,
                    "prauc": prauc,
                    "npv": npv,
                    "best_params": bestParams
                })

    def getResults(self):
        return self.results

    def getSummary(self):
        summary = {}
        for name, resList in self.results.items():
            metrics = pd.DataFrame(resList)
            summary[name] = {
                "mcc_median": metrics["mcc"].median(),
                "roc_auc_median": metrics["roc_auc"].median(),
                "ba_median": metrics["balanced_accuracy"].median(),
                "f1_median": metrics["f1"].median(),
                "f2_median": metrics["f2"].median(),
                "recall_median": metrics["recall"].median(),
                "specificity_median": metrics["specificity"].median(),
                "precision_median": metrics["precision"].median(),
                "prauc_median": metrics["prauc"].median(),
                "npv_median": metrics["npv"].median(),
            }
        return pd.DataFrame(summary).T
