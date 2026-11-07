"""Recompute supplied evidence on CPU, or reproduce training/inference on CUDA."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
def read(path):return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run(args,cwd):subprocess.run([sys.executable]+args,cwd=cwd,check=True)

def verify():
    manifest=json.loads((ROOT/'SHA256SUMS.json').read_text())
    for name,expected in manifest.items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or digest(path)!=expected:raise RuntimeError('Hash mismatch: '+name)
    sys.path.insert(0,str(ROOT/'training'))
    from datos import metrics,validate
    rows=read(ROOT/'training/pares.jsonl');controls=read(ROOT/'training/controles.jsonl');validate(rows,controls)
    refs=read(ROOT/'evidence/training/referencias.jsonl')
    for stage in ['baseline','post_N2','post_N3']:
        path=ROOT/'evidence/training'/stage
        report,details=metrics(refs,read(path/'predicciones.jsonl'))
        cfg=json.loads((ROOT/'training/config.json').read_text())
        report['generation']=dict(do_sample=False,num_beams=1,repetition_penalty=1.0,max_new_tokens=cfg['max_new_tokens'])
        if report!=json.loads((path/'metricas.json').read_text()):raise RuntimeError('Score mismatch: '+stage)
        if details!=read(path/'detalle.jsonl'):raise RuntimeError('Per-case mismatch: '+stage)
    sys.path.insert(0,str(ROOT/'manual_eval'))
    from estructura import score
    refs=read(ROOT/'manual_eval/referencias.jsonl')
    if refs!=read(ROOT/'evidence/manual/datos/referencias_fijadas.jsonl'):raise RuntimeError('Manual references changed')
    paired=[]
    for stage in ['base','adaptador']:
        path=ROOT/'evidence/manual/datos'
        report,details=score(refs,read(path/f'predicciones_{stage}.jsonl'))
        if report!=json.loads((path/f'metricas_{stage}.json').read_text()):raise RuntimeError('Manual score mismatch')
        if details!=read(path/f'detalle_{stage}.jsonl'):raise RuntimeError('Manual per-case mismatch')
        paired.append(details)
    old={r['input'] for r in rows+controls}
    if old & {r['input'] for r in refs}:raise RuntimeError('Exact input overlap in new evaluation')
    for folder in ['training','manual_eval']:
        run(['-m','unittest','discover','-p','test_*.py','-v'],ROOT/folder)
    print('Verified hashes, training/development isolation, new-input non-overlap and all recorded scores.')
    for stage in ['base','adaptador']:
        report=json.loads((ROOT/f'evidence/manual/datos/metricas_{stage}.json').read_text())
        print(stage,report['global_records']['exact'],'/',report['global_records']['n'])

def train():
    before=set((ROOT/'training').glob('resultados_*'))
    run(['ejecutar.py'],ROOT/'training')
    new=set((ROOT/'training').glob('resultados_*'))-before
    if len(new)!=1:raise RuntimeError('Cannot identify the new training run')
    return new.pop()/'piloto/adapter_N3'

def evaluate_new_checkpoint(checkpoint):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    work=ROOT/'reruns'/stamp;work.mkdir(parents=True)
    for path in (ROOT/'manual_eval').iterdir():
        if path.is_file():shutil.copyfile(path,work/path.name)
    shutil.copytree(checkpoint,work/'adapter_congelado')
    config=json.loads((work/'config.json').read_text())
    config['adapter_sha256']=digest(work/'adapter_congelado/adapter_model.safetensors')
    config['adapter_config_sha256']=digest(work/'adapter_congelado/adapter_config.json')
    (work/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    run(['ejecutar.py'],work)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['verify','train','evaluate','full'],default='verify',nargs='?')
    args=parser.parse_args();verify()
    if args.mode=='train':train()
    elif args.mode=='evaluate':run(['ejecutar.py'],ROOT/'manual_eval')
    elif args.mode=='full':evaluate_new_checkpoint(train())
