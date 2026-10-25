"""Minimal stand-in for lightgbm, used by the tests only when lightgbm is not installed."""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier


class LGBMClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, num_leaves=31, learning_rate=0.1, n_estimators=100, reg_alpha=0.0, reg_lambda=0.0,
                 class_weight=None, random_state=None, n_jobs=None, device="cpu", verbose=-1):
        self.num_leaves, self.learning_rate, self.n_estimators = num_leaves, learning_rate, n_estimators
        self.reg_alpha, self.reg_lambda, self.class_weight = reg_alpha, reg_lambda, class_weight
        self.random_state, self.n_jobs, self.device, self.verbose = random_state, n_jobs, device, verbose

    def fit(self, X, y):
        self._m = HistGradientBoostingClassifier(max_leaf_nodes=max(2, self.num_leaves), learning_rate=self.learning_rate,
                                                 max_iter=self.n_estimators, l2_regularization=self.reg_lambda,
                                                 class_weight=self.class_weight, random_state=self.random_state)
        self._m.fit(X, y)
        self.classes_ = self._m.classes_
        return self

    def predict_proba(self, X):
        return self._m.predict_proba(X)

    def predict(self, X):
        return self._m.predict(X)
