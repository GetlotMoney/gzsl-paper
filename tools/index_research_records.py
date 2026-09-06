"""Read-only inventory of Git records; output catalogs contain pointers, not research verdicts.

Requires Python standard library and Git. Queries remote refs without fetching or pushing.
Run from the canonical checkout; pass --workspace for additional unsaved Idea variants.
"""
from pathlib import Path
import subprocess, json, csv, hashlib, re
from concurrent.futures import ThreadPoolExecutor

import argparse
from datetime import datetime, timezone

parser = argparse.ArgumentParser(description="Index existing research records and branch snapshots; never change source records or refs.")
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--workspace', type=Path, action='append', default=[], help='Additional checkout whose uncommitted Idea cards should be indexed, without importing content.')
parser.add_argument('--output-dir', type=Path, help='Defaults to <repo>/research; writes only RECORD_INDEX.csv and CODE_SNAPSHOTS.csv.')
args=parser.parse_args()
ROOT=args.repo.resolve()
OUT=(args.output_dir or ROOT/'research').resolve()
OUT.mkdir(parents=True, exist_ok=True)
observed_at=datetime.now(timezone.utc).isoformat(timespec='seconds')

def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])

refs = [line.split(' ',1) for line in git('for-each-ref','--format=%(objectname) %(refname:short)','refs/heads').decode().splitlines()]
remote = dict((ref, sha) for sha, ref in (line.split() for line in git('ls-remote','--heads','--tags','origin').decode().splitlines()))
main_sha = git('rev-parse','main').decode().strip()
remote_sha = remote['refs/heads/main']

def scan(item):
    sha, ref = item
    records=[]
    for entry in git('ls-tree','-r','-z',sha,'research','experiments').split(b'\0'):
        if not entry: continue
        meta, raw_path=entry.split(b'\t',1)
        path=raw_path.decode('utf-8')
        blob=meta.decode().split()[2]
        name=Path(path).name
        kind = ('idea' if re.match(r'IDEA-\d+.*\.md$',name) and '/ideas/' in path else
                'queue' if name=='EXPERIMENT_QUEUE.csv' else
                'experiment' if name=='EXPERIMENT.yaml' else
                'run_matrix' if name=='PARAMETER_MATRIX.csv' else
                'result' if name.lower()=='result.md' else
                'framework' if name=='FRAMEWORK.yaml' else
                'proposal' if '/proposals/' in path and name!='README.md' and name.endswith('.md') else None)
        if kind: records.append((kind,path,blob))
    return sha,ref,records

snapshots=list(ThreadPoolExecutor(max_workers=4).map(scan,refs))
snapshots.sort(key=lambda x: (x[1]!='main',x[1]))
records={}
branch_rows=[]
for sha,ref,items in snapshots:
    remote_head=remote.get('refs/heads/'+ref,'')
    status='same_tip' if remote_head==sha else 'different_tip' if remote_head else 'no_remote_branch'
    branch_rows.append(dict(branch=ref,snapshot_commit=sha,remote_commit=remote_head,remote_ref_state=status,
                            idea_cards=sum(x[0]=='idea' for x in items),ledger_files=sum(x[0]!='idea' for x in items)))
    for kind,path,blob in items:
        key=(path,blob)
        if key not in records:
            records[key]=dict(kind=kind,record_id=(re.search(r'IDEA-\d+',path).group() if kind=='idea' else ''),
                              path=path,blob_id=blob,source_commit=sha,source_ref=ref,ref_count=0,
                              content_state='committed',title='',declared_status='')
        records[key]['ref_count']+=1

# Read each distinct light record blob once. Never read model/data/checkpoint contents.
blobs=list(dict.fromkeys(row['blob_id'] for row in records.values()))
raw=subprocess.run(['git','-C',str(ROOT),'cat-file','--batch'],input=('\n'.join(blobs)+'\n').encode(),stdout=subprocess.PIPE,check=True).stdout
pos=0; contents={}
for blob in blobs:
    end=raw.index(b'\n',pos); header=raw[pos:end].split(); size=int(header[2]); pos=end+1
    contents[blob]=raw[pos:pos+size].decode('utf-8-sig',errors='replace'); pos+=size+1
for row in records.values():
    content=contents[row['blob_id']]
    title=re.search(r'^#\s+(.+)',content,re.M)
    status=re.search(r'^\s*status:\s*(.+)',content,re.M)
    row['title']=title.group(1).strip() if title else ''
    row['declared_status']=status.group(1).strip() if status else ''

# Index unsaved Idea variants only from explicitly selected checkouts.
# Records retain the exact checkout path; no draft text is copied into the catalog.
for workspace in dict.fromkeys([ROOT, *[p.resolve() for p in args.workspace]]):
    common=subprocess.check_output(['git','-C',str(workspace),'rev-parse','--path-format=absolute','--git-common-dir']).decode().strip()
    own=git('rev-parse','--path-format=absolute','--git-common-dir').decode().strip()
    if Path(common).resolve()!=Path(own).resolve():
        raise ValueError('Workspace must belong to the same repository: '+str(workspace))
    for path in sorted((workspace/'research/ideas').glob('IDEA-*.md')):
        data=path.read_bytes(); rel=path.relative_to(workspace).as_posix()
        blob=subprocess.check_output(['git','-C',str(workspace),'hash-object','--path='+rel,str(path)]).decode().strip()
        if (rel,blob) in records: continue
        content=data.decode('utf-8-sig'); title=re.search(r'^#\s+(.+)',content,re.M)
        records[(rel,blob)]=dict(kind='idea',record_id=re.search(r'IDEA-\d+',rel).group(),path=rel,blob_id=blob,
            source_commit='',source_ref='workspace:'+workspace.as_posix(),ref_count=0,
            content_state='local_uncommitted',title=title.group(1) if title else '',declared_status='not_reconciled')

rows=sorted(records.values(),key=lambda x:(x['kind'],x['path'],x['source_ref']))
for name,values in [('RECORD_INDEX.csv',rows),('CODE_SNAPSHOTS.csv',branch_rows)]:
    for row in values:
        row['observed_at_utc']=observed_at
    target=OUT/name
    with target.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)

ideas=sorted({r['record_id'] for r in rows if r['kind']=='idea'},key=lambda x:int(x.split('-')[1]))
summary={'main_before':main_sha,'remote_main_before':remote_sha,'branches':len(refs),'records':len(rows),
         'idea_ids':len(ideas),'idea_max':ideas[-1],
         'missing_ids':[f'IDEA-{i:03}' for i in range(1,int(ideas[-1].split('-')[1])+1) if f'IDEA-{i:03}' not in ideas],
         'uncommitted_cards':[r['path'] for r in rows if r['content_state']=='local_uncommitted'],
         'remote_refs':remote,'kind_counts':{k:sum(r['kind']==k for r in rows) for k in sorted({r['kind'] for r in rows})}}
print(json.dumps({k:v for k,v in summary.items() if k!='remote_refs'},ensure_ascii=False,indent=2))
