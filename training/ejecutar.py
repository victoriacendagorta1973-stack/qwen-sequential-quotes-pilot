"""Instala PEFT fijado, comprueba, entrena y empaqueta, incluso si hay fallo."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from zipfile import ZipFile,ZIP_DEFLATED


def main():
    root=Path(__file__).resolve().parent
    cfg=json.loads((root/'config.json').read_text())
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out=root/f'resultados_{stamp}'
    out.mkdir()
    started=time.monotonic();deadline=started+cfg['wall_time_seconds']
    status=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat())
    target=root.parent/'RESULTADOS_PILOTO_SECUENCIAL_V1.zip'
    if target.exists():
        target=root.parent/f'RESULTADOS_PILOTO_SECUENCIAL_V1_{stamp}.zip'
    def run(command,timeout):
        remaining=min(timeout,deadline-time.monotonic())
        if remaining<=0:
            raise TimeoutError('Límite global alcanzado')
        with (out/'consola.log').open('a',encoding='utf-8') as log:
            log.write('\nCOMANDO: '+repr(command)+'\n');log.flush()
            process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                result=process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise TimeoutError('Límite de tiempo: proceso detenido')
            if result!=0:
                raise RuntimeError(f'Proceso fallido con código {result}; consultar consola.log')
    try:
        print('1/3 — Instalando PEFT fijado y comprobando datos.',flush=True)
        # No actualizar torch, transformers o las demás dependencias del pod.
        run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps',f"peft=={cfg['peft_version']}"],180)
        run([sys.executable,'-m','unittest','discover','-p','test_*.py','-v'],60)
        print('2/3 — Baseline, entrenamiento N2/N3 y evaluaciones. Puedes dejarlo ejecutándose.',flush=True)
        budget=max(1,int(deadline-time.monotonic()-30))
        run([sys.executable,'-u','entrenar.py','--out',str(out/'piloto'),'--budget-seconds',str(budget)],budget+10)
        status['status']='complete'
        summary=out/'piloto/RESUMEN.txt'
        if summary.exists():print(summary.read_text(),flush=True)
    except Exception as exc:
        status.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        (out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('La ejecución se detuvo; el diagnóstico se incluirá en el ZIP.',flush=True)
    finally:
        status.update(finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
        (out/'estado_global.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
        for name in ['config.json','pares.jsonl','controles.jsonl','LEEME.md','datos.py','entrenar.py','ejecutar.py','test_piloto.py']:
            if (root/name).exists():shutil.copyfile(root/name,out/name)
        print('3/3 — Guardando resultados y adaptadores.',flush=True)
        with ZipFile(target,'w',ZIP_DEFLATED) as z:
            for path in sorted(out.rglob('*')):
                if path.is_file():z.write(path,path.relative_to(root))
        print('\nZIP FINAL: '+str(target),flush=True)
        print('Estado: '+status['status'],flush=True)
    # Devolver error después de guardar el ZIP, nunca antes.
    if status['status']!='complete':
        sys.exit(1)


if __name__=='__main__':
    main()
