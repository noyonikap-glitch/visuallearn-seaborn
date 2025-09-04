from sklearn.datasets import make_moons
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from visuallearn.combinedplot import CombinedPlotCoordinator
import matplotlib.pyplot as plt

X, y = make_moons(noise=0.3, random_state=42)
clf = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=0.1, max_iter=1, warm_start=True)
plotter = CombinedPlotCoordinator(X, y)

for epoch in range(50):
    clf.fit(X, y)
    loss = log_loss(y, clf.predict_proba(X))
    plotter.update(clf, epoch, loss)
