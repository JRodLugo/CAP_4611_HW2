"""Generate only the missing figures; preserve the submitted source and answers.

Run from the repository root with Python, numpy and matplotlib installed.
Local optional dependencies can be placed in .plot_dependencies.
"""
from pathlib import Path
import sys
import pickle
import json

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.plot_dependencies'))
sys.path.insert(0, str(ROOT / 'a2' / 'code'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from knn import KNN
from kmeans import Kmeans
from utils import plot_classifier

OUT = ROOT / 'a2' / 'figs'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'figure.figsize': (7, 4.4), 'font.size': 11, 'savefig.dpi': 180})
results = {}

def data(name):
    with (ROOT / 'a2' / 'data' / (name + '.pkl')).open('rb') as f:
        return pickle.load(f)

def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches='tight')
    plt.close()

def error(model, X, y):
    return float(np.mean(model.predict(X) != y))

ds = data('citiesSmall')
X, y = ds['X'], ds['y']
results['q1'] = []
for k in [1, 3, 10]:
    model = KNN(k)
    model.fit(X, y)
    results['q1'].append({'k': k, 'train_error': error(model, X, y),
                          'test_error': error(model, ds['Xtest'], ds['ytest'])})
model = KNN(1)
model.fit(X, y)
plot_classifier(model, X, y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('citiesSmall: 1-nearest-neighbour classifier')
save('q1_knn_boundary.png')

ds = data('ccdebt')
X, y = ds['X'], ds['y']
ks = list(range(1, 30, 4))
cv, test, train = [], [], []
assert len(X) % 10 == 0, 'Assignment requires ten equal consecutive folds'
fold_size = len(X) // 10
for k in ks:
    accs = []
    for fold in range(10):
        mask = np.ones(len(X), dtype=bool)
        mask[fold*fold_size:(fold+1)*fold_size] = False
        model = KNN(k)
        model.fit(X[mask], y[mask])
        accs.append(1 - error(model, X[~mask], y[~mask]))
    cv.append(float(np.mean(accs)))
    model = KNN(k)
    model.fit(X, y)
    test.append(1 - error(model, ds['Xtest'], ds['ytest']))
    train.append(error(model, X, y))
results['q2'] = {'ks': ks, 'cv_accuracy': cv, 'test_accuracy': test, 'train_error': train}
plt.figure()
plt.plot(ks, cv, 'o-', label='10-fold cross-validation')
plt.plot(ks, test, 's-', label='Test')
plt.xticks(ks)
plt.xlabel('Number of neighbours, k')
plt.ylabel('Accuracy')
plt.title('Credit-card debt: cross-validation and test accuracy')
plt.legend()
plt.grid(alpha=.25)
save('q2_cv_test_accuracy.png')
plt.figure()
plt.plot(ks, train, 'o-')
plt.xticks(ks)
plt.xlabel('Number of neighbours, k')
plt.ylabel('Training error (fraction misclassified)')
plt.title('Credit-card debt: training error')
plt.grid(alpha=.25)
save('q2_training_error.png')

# Use the assignment fitting implementation and its completed SSE method.
X = data('clusterData')['X']
np.random.seed(4611)
minimum_errors = []
for k in range(1, 11):
    best = float('inf')
    for repeat in range(50):
        model = Kmeans(k)
        model.fit(X)
        labels = model.predict(X)
        sse = model.error(X, labels, model.means)
        if sse < best:
            best, best_labels, best_means = sse, labels.copy(), model.means.copy()
    minimum_errors.append(best)
    if k == 4:
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=best_labels, cmap='tab10', s=15)
        plt.scatter(best_means[:, 0], best_means[:, 1], c='black', marker='X', s=110, label='Centroids')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Best of 50 initializations: k = 4, SSE = {best:.6f}')
        plt.legend()
        save('q5_best_clustering.png')
plt.figure()
plt.plot(range(1, 11), minimum_errors, 'o-')
plt.xticks(range(1, 11))
plt.xlabel('Number of clusters, k')
plt.ylabel('Minimum sum of squared distances')
plt.title('k-means: best of 50 random initializations per k')
plt.grid(alpha=.25)
save('q5_elbow.png')
results['q5'] = {'seed': 4611, 'runs_per_k': 50, 'minimum_errors': minimum_errors}
ds = data('newsgroups')
results['q3_2'] = {'word_73': str(ds['wordlist'][72]),
                    'words_803': [str(w) for w in np.asarray(ds['wordlist'])[ds['X'][802].astype(bool)]],
                    'group_803': str(ds['groupnames'][ds['y'][802]])}
(OUT / 'generated_results.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
