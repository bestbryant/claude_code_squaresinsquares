from playwright.sync_api import sync_playwright
import re
refs=set()
with sync_playwright() as p:
    b=p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome', args=['--no-sandbox'])
    pg=b.new_page()
    for sort in ['hotness','dateRun','scoreAscending','voteCount','dateCreated']:
        pg.goto(f'https://www.kaggle.com/competitions/squares-in-squares/code?sortBy={sort}', wait_until='networkidle', timeout=120000)
        for i in range(25):
            pg.mouse.wheel(0,20000); pg.wait_for_timeout(1200)
        for a in pg.query_selector_all('a'):
            h=a.get_attribute('href') or ''
            m=re.match(r'^/code/([^/]+/[^/?#]+)',h)
            if m: refs.add(m.group(1))
        print(sort,len(refs))
    b.close()
open('refs.txt','w').write('\n'.join(sorted(refs)))
