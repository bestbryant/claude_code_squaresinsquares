"""Download a public Kaggle notebook's output files (unauthenticated), with backoff on HTTP 429."""
import json, sys, subprocess, os, time
ref = sys.argv[1]; u, s = ref.split('/')
STAMP = sys.argv[2].replace('T', ' ') if len(sys.argv) > 2 else None
os.makedirs(s, exist_ok=True)
d = None
# fast path: authenticated Kaggle CLI (no anonymous rate limit)
try:
    r = subprocess.run(['kaggle', 'kernels', 'output', ref, '-p', s, '-o'], capture_output=True, text=True, timeout=600)
    if r.returncode == 0:
        for fn in os.listdir(s):
            if fn.endswith('.log') or (not fn.endswith('.csv') and not fn.startswith('.')):
                try:
                    os.remove(os.path.join(s, fn))
                except OSError:
                    pass
        if STAMP:
            open(os.path.join(s, '.stamp'), 'w').write(STAMP)
        print(ref, 'cli ok')
        sys.exit(0)
except Exception:
    pass
for attempt in range(6):
    raw = subprocess.check_output(['curl', '-sS', '-m', '60', f'https://www.kaggle.com/api/v1/kernels/output?userName={u}&kernelSlug={s}'])
    try:
        d = json.loads(raw)
    except Exception:
        d = None
    if d and d.get('code') != 429:
        break
    time.sleep(30 * (attempt + 1))
for f in (d or {}).get('files', []):
    fn = f['fileName']
    if fn.startswith('.') or not fn.endswith('.csv'):
        continue
    p = os.path.join(s, fn.replace('/', '__'))
    subprocess.run(['curl', '-sS', '-L', '-m', '300', '-o', p, f['url']])
    print(ref, fn, os.path.getsize(p))
    time.sleep(2)

if STAMP and (d or {}).get('files') is not None:
    open(os.path.join(s, '.stamp'), 'w').write(STAMP)
