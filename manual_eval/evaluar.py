"""Evalúa base y adaptador congelado tras recibir un acta de revisión de referencias."""
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from estructura import read,score


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def main(out,candidates,review,budget_seconds):
    root=Path(__file__).resolve().parent;out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((root/'config.json').read_text())
    act=json.loads(Path(review).read_text())
    if act.get('candidates_sha256')!=sha(candidates):raise RuntimeError('El acta no corresponde a los candidatos')
    if act.get('reviewed_without_predictions') is not True:raise RuntimeError('Falta revisión previa sin predicciones')
    all_rows=read(candidates);ids=act.get('accepted_ids',[])
    if not ids or len(set(ids))!=len(ids) or not set(ids)<={r['id'] for r in all_rows}:
        raise RuntimeError('Selección de IDs inválida')
    refs=[r for r in all_rows if r['id'] in set(ids)]
    for r in refs:
        if r.get('source_type')!='human_authored' or r.get('source_author')!='Victoria Cendagorta Villalba':
            raise RuntimeError('Origen distinto del lote manual autorizado')
    if sha(candidates)!=cfg['references_file_sha256']:
        raise RuntimeError('Las referencias difieren del lote congelado')
    if sha(root/'adapter_congelado/adapter_model.safetensors')!=cfg['adapter_sha256']:
        raise RuntimeError('El adaptador ha cambiado')
    if sha(root/'adapter_congelado/adapter_config.json')!=cfg['adapter_config_sha256']:
        raise RuntimeError('La configuración del adaptador ha cambiado')
    if (out/'referencias_fijadas.jsonl').exists():raise RuntimeError('Las referencias ya existen')
    reference_bytes=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in refs).encode()
    (out/'referencias_fijadas.jsonl').write_bytes(reference_bytes)
    save(out/'acta_revision.json',act)
    start=time.monotonic();deadline=start+budget_seconds
    def timecheck():
        if time.monotonic()>deadline:raise TimeoutError('Presupuesto de evaluación agotado')
    manifest=dict(status='running',model=cfg['model'],revision=cfg['revision'],
        adapter_sha256=cfg['adapter_sha256'],references_sha256=hashlib.sha256(reference_bytes).hexdigest(),
        training_performed=False,examples_in_prompt=False,
        started_utc=datetime.now(timezone.utc).isoformat(),n=len(refs))
    save(out/'evaluacion.json',manifest)
    try:
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from peft import PeftModel
        if not torch.cuda.is_available():raise RuntimeError('Se necesita GPU CUDA')
        manifest['versions']={p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate','peft','safetensors','huggingface-hub']}
        manifest['gpu']=torch.cuda.get_device_name(0)
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        tokenizer=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'])
        model=AutoModelForCausalLM.from_pretrained(cfg['model'],revision=cfg['revision'],dtype='auto',
            device_map={'':0},attn_implementation='sdpa')
        if getattr(model,'peft_config',None):raise RuntimeError('La primera condición debe ser el modelo base')
        generation=dict(do_sample=False,num_beams=1,repetition_penalty=1.0,max_new_tokens=1024)
        manifest['generation']=generation;save(out/'evaluacion.json',manifest)
        reports={};details_by_stage={}
        for stage in ['base','adaptador']:
            timecheck()
            if stage=='adaptador':
                model=PeftModel.from_pretrained(model,str(root/'adapter_congelado'),is_trainable=False)
            model.eval()
            if stage=='adaptador' and any(p.requires_grad for p in model.parameters()):
                raise RuntimeError('La evaluación no debe tener parámetros entrenables')
            predictions=[]
            with (out/f'predicciones_{stage}.jsonl').open('w',encoding='utf-8') as f:
                for index,row in enumerate(refs,1):
                    timecheck()
                    messages=[dict(role='system',content=cfg['instruction']),dict(role='user',content=row['input'])]
                    prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                    inputs=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to('cuda')
                    if inputs['input_ids'].shape[-1]+1024>model.config.max_position_embeddings:
                        raise RuntimeError('Entrada demasiado larga; no se trunca')
                    with torch.inference_mode():output=model.generate(**inputs,**generation,pad_token_id=tokenizer.eos_token_id,use_cache=True)
                    tokens=output[0,inputs['input_ids'].shape[-1]:]
                    eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                    pred=dict(id=row['id'],prediction=tokenizer.decode(tokens,skip_special_tokens=True,
                        clean_up_tokenization_spaces=False),generated_tokens=len(tokens),
                        hit_token_limit=len(tokens)>=1024 and int(tokens[-1]) not in eos)
                    predictions.append(pred);f.write(json.dumps(pred,ensure_ascii=False)+'\n');f.flush()
                    print(f'{stage}: {index}/{len(refs)}',flush=True)
            report,details=score(refs,predictions);reports[stage]=report;details_by_stage[stage]=details
            save(out/f'metricas_{stage}.json',report)
            (out/f'detalle_{stage}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in details),encoding='utf-8')
        paired=dict(improved=sum(not b['exact'] and a['exact'] for b,a in zip(details_by_stage['base'],details_by_stage['adaptador'])),
                    regressed=sum(b['exact'] and not a['exact'] for b,a in zip(details_by_stage['base'],details_by_stage['adaptador'])))
        save(out/'comparacion.json',dict(reports=reports,paired=paired))
        lines=['Evaluación manual nueva: doce textos de la autora, cuatro familias; sin ejemplos en el prompt.']
        for cat in reports['base']['by_category']:
            b=reports['base']['by_category'][cat];a=reports['adaptador']['by_category'][cat]
            lines.append(f"{cat}: base {b['exact']}/{b['n']} → adaptador {a['exact']}/{a['n']}")
        lines.append('No se han medido aquí GSM8K, IFEval ni MMLU.')
        (out/'RESUMEN.txt').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines),flush=True)
        manifest['status']='complete'
    except Exception as exc:
        manifest.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
        save(out/'evaluacion.json',manifest)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--candidates',required=True)
    p.add_argument('--review',required=True);p.add_argument('--budget-seconds',type=int,default=2400)
    a=p.parse_args();main(a.out,a.candidates,a.review,a.budget_seconds)
