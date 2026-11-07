"""Una ejecución: comprobaciones, dependencias, evaluación y ZIP incluso si falla."""
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from zipfile import ZipFile,ZIP_DEFLATED

def main():
    root=Path(__file__).resolve().parent
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    out=root/('resultados_'+stamp);out.mkdir()
    target=root.parent/('RESULTADOS_EVALUACION_MANUAL_12_'+stamp+'.zip')
    cfg=json.loads((root/'config.json').read_text());start=time.monotonic();deadline=start+cfg['wall_time_seconds']
    status=dict(status='running',training_performed=False,examples_in_prompt=False)
    def run(cmd):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Límite global alcanzado')
        with (out/'consola.log').open('a',encoding='utf-8') as log:
            p=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                raise TimeoutError('Tiempo máximo alcanzado; se conserva lo obtenido')
            if code:raise RuntimeError('Un paso ha fallado; el diagnóstico se incluirá en el ZIP')
    try:
        print('1/3 — Comprobando textos, métricas y adaptador.',flush=True)
        run([sys.executable,'-m','unittest','discover','-p','test_*.py','-v'])
        print('2/3 — Comprobando dependencias.',flush=True)
        required={'transformers':'5.22.1','accelerate':'1.15.0','safetensors':'0.8.0','huggingface-hub':'2.4.0','peft':'0.21.2'}
        missing=[]
        for name,version in required.items():
            try:current=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:current=None
            if current!=version:missing.append(name+'=='+version)
        if missing:run([sys.executable,'-m','pip','install','--disable-pip-version-check']+missing)
        print('3/3 — Comparando modelo base y adaptador. No se entrena.',flush=True)
        run([sys.executable,'-u','evaluar.py','--out',str(out/'datos'),
             '--candidates',str(root/'referencias.jsonl'),'--review',str(root/'revision.json'),
             '--budget-seconds',str(max(1,int(deadline-time.monotonic()-15)))])
        status['status']=json.loads((out/'datos/evaluacion.json').read_text())['status']
        print((out/'datos/RESUMEN.txt').read_text(),flush=True)
    except Exception as exc:
        status.update(status='failed',error=str(exc));(out/'error.txt').write_text(traceback.format_exc())
        print(str(exc),flush=True)
    finally:
        status.update(elapsed_seconds=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat())
        (out/'estado_global.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
        with ZipFile(target,'w',ZIP_DEFLATED) as z:
            for p in sorted(root.iterdir()):
                if p.is_file():z.write(p,'codigo_y_datos/'+p.name)
            for p in sorted(out.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(root))
        print('ESTADO: '+status['status'],flush=True)
        print('ZIP PARA COMPARTIR: '+str(target),flush=True)
    if status['status']=='failed':sys.exit(1)

if __name__=='__main__':main()
