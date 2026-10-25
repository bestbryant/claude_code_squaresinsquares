import sys,re
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b=p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome', args=['--no-sandbox'])
    pg=b.new_page()
    pg.goto(sys.argv[1], wait_until='networkidle', timeout=120000); pg.wait_for_timeout(10000)
    for f in pg.frames:
        print('FRAME',f.url[:200])
        try:
            h=f.content()
            m=re.search(r'const DATA = (\[.*?\]);',h,re.S)
            if m: open(sys.argv[2],'w').write(m.group(1)); print('GOT DATA',len(m.group(1)))
        except Exception as e: print(e)
    b.close()
