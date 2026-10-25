import json,sys,subprocess,os,urllib.request
ref=sys.argv[1]; u,s=ref.split('/')
os.makedirs(s,exist_ok=True)
d=json.loads(subprocess.check_output(['curl','-sS','-m','60',f'https://www.kaggle.com/api/v1/kernels/output?userName={u}&kernelSlug={s}']))
for f in d.get('files',[]):
    fn=f['fileName']
    if fn.startswith('.'): continue
    p=os.path.join(s,fn.replace('/','__'))
    subprocess.run(['curl','-sS','-L','-m','300','-o',p,f['url']])
    print(ref,fn,os.path.getsize(p))
