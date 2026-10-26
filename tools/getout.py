"""Download a public Kaggle notebook's output files (unauthenticated), with backoff on HTTP 429."""
import json, sys, subprocess, os, time
ref = sys.argv[1]; u, s = ref.split('/')
os.makedirs(s, exist_ok=True)
d = None
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
