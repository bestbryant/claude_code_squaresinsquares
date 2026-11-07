"""List competition notebooks via the authenticated Kaggle CLI and print refs whose lastRunTime
changed since the cached stamp (cache dir given as argv[1]). Writes stamps for printed refs to
<cache>/<slug>/.stamp only after download (done by caller via --mark)."""
import csv, io, os, subprocess, sys
cache = sys.argv[1]
rows = []
for page in range(1, 6):
    out = subprocess.run(['kaggle', 'kernels', 'list', '--competition', 'squares-in-squares', '--sort-by', 'dateRun',
                          '--page-size', '100', '--page', str(page), '--csv'], capture_output=True, text=True, timeout=180).stdout
    r = list(csv.DictReader(io.StringIO(out)))
    if not r:
        break
    rows += r
for r in rows:
    ref = r['ref']; slug = ref.split('/')[1]
    st = os.path.join(cache, slug, '.stamp')
    old = open(st).read().strip() if os.path.exists(st) else ''
    if old != r['lastRunTime']:
        print(ref, r['lastRunTime'].replace(' ', 'T'))
