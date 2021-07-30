import re
import json
import numpy as np
import pandas as pd


def parse_paper(line):
    instance = {}
    content = line.split('*, ')
    title = venue = code = None
    if len(content) == 3:
        title, venue, code = content
    else:
        title, venue = content
        
    end = venue.find(']')
    year = re.findall(r"\d+", venue[max(end-5, 0):end])
    
    if year:
        year = int(year[0])
        if year < 2000:
            year += 2000
            
    instance['Title'] = title
    instance['Venue'] = venue
    instance['Year'] = year
    instance['Code'] = code
    
    return instance
    

if __name__ == '__main__':
    with open("README.md", "r", encoding='UTF-8') as f:
        content = f.readlines()
        
    # parsing
    i = 0
    papers = []
    titles = []

    while i < len(content):
        line = content[i]
        if line.strip() == '# 🔗Resource': break
        
        if line.startswith("# "):
            this_level = 0
            titles = []
            titles.append(line[this_level+2:].strip())
        elif line.startswith("## "):
            this_level = 1   
            while len(titles)>this_level:
                titles.pop()
            titles.append(line[this_level+2:].strip())

        elif line.startswith("### "):
            this_level = 2   
            while len(titles)>this_level:
                titles.pop()
            titles.append(line[this_level+2:].strip())

        elif line.startswith("#### "):
            this_level = 3
            while len(titles)>this_level:
                titles.pop()
            titles.append(line[this_level+2:].strip())    
            
        is_paper = line.startswith("+ ")

        if is_paper:
            line = line[2:].strip()
            paper = parse_paper(line)
            paper['belong'] = titles[:]
            if not paper['Year'] and len(titles) > 1:
                paper['Year'] = int(titles[-1])
            paper["Type"] = titles[0].strip()
            papers.append(paper)
        i += 1

    # To pandas DataFrame
    tb = []
    columns = ["Title", "Type", "Venue", "Code", "Year"]
    for paper in papers:
        tmp = []
        for col in columns:
            if isinstance(paper[col], str):
                tmp.append(paper[col].strip('*'))
            else:
                tmp.append(paper[col])
                
        tb.append(tmp)
    tb = pd.DataFrame(tb, columns=columns)

    # Sort by Title
    tb_sort = tb.drop_duplicates(subset=['Title']).sort_values('Title').reset_index(drop=True)
    with open('Sorted/sort_by_alphabet.md', 'w', encoding='utf-8') as f:
        tb_sort.to_markdown(f)

    # Sort by Year
    tb_sort = tb.drop_duplicates(subset=['Title']).sort_values('Title').reset_index(drop=True)
    tb_sort = tb_sort.sort_values('Year', kind='mergesort', ascending=False).reset_index(drop=True)

    with open('Sorted/sort_by_year.md', 'w', encoding='utf-8') as f:
        tb_sort.to_markdown(f)

    # Sort by venue
    arr = tb.to_numpy()
    confs = ['AAAI', 'IJCAI', 'ICLR', 'WWW', 'KDD', 'ICML', 'TKDE', 'CIKM', 'WSDM', 'NeurIPS', 'ICSE', 'USENIX', 'ICDM', 'ECAI', 'Arxiv', 'UAI', 'Others']
    arr_out = []
    for line in arr:
        title, types, venue, code, year = line
        pubs = None
        for c in confs:
            if c in venue:
                pubs = c
                break
        pubs = pubs or 'Others'
        arr_out.append(np.array([title, types, venue, code, year, pubs]))
        
    tb_pubs = pd.DataFrame(np.array(arr_out), columns=columns+['Pubs'])
    tb_pubs = tb_pubs.sort_values('Pubs', kind='mergesort').reset_index(drop=True)

    for i, c in enumerate(confs):
        t = tb_pubs[tb_pubs['Pubs']==c]
        mod = 'w' if i==0 else 'a'
        with open('Sorted/sort_by_venue.md', mod, encoding='utf-8') as f:
            f.writelines('# ' + c + '\n')
            t.to_markdown(f)
            f.writelines('\n')
            

    # Write README.md
    content[0] = content[0][:40] + f'(Updating {len(tb)} papers)'
    with open('README.md', 'w', encoding='utf-8') as f:
        for line in content:
            f.writelines(line)
            f.writelines('\n')
