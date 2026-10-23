"""Write submission.csv from solutions/ and score it with the official metric (pandas float64 round trip)."""
import sys, os, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import write_submission
import metric

ROOT = os.path.join(os.path.dirname(__file__), '..')
sol = {int(os.path.basename(p)[1:-4]): np.load(p) for p in glob.glob(os.path.join(ROOT, 'solutions', 'n*.npy'))}
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'submission.csv')
write_submission(out, sol)
sub = pd.read_csv(out)  # float64 round trip, as Kaggle does
solution = pd.DataFrame({'id': range(1000), 'n': range(1, 1001)})
print('official score:', metric.score(solution, sub, 'id'))
