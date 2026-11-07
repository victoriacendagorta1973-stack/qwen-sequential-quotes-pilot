"""Piloto N2→N3, atención LoRA en índices 8..19. Ejecutado por ejecutar.py."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import time

from datos import read, validate, schedule, metrics


def save(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def assert_trainable(model, cfg):
    found=set(); names=[]
    for name,param in model.named_parameters():
        if not param.requires_grad:
            continue
        match=re.search(r'\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.lora_([AB])\.',name)
        if not match or int(match[1]) not in cfg['layers']:
            raise RuntimeError('Parámetro entrenable no autorizado: '+name)
        found.add((int(match[1]),match[2],match[3]));names.append(name)
    expected={(i,module,ab) for i in cfg['layers'] for module in cfg['target_modules'] for ab in ['A','B']}
    if found!=expected or len(names)!=len(expected):
        raise RuntimeError('El conjunto entrenable no coincide con los 12 bloques y 4 proyecciones')
    return dict(names=names,tensors=len(names),parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))


def make_lora(cfg):
    from peft import LoraConfig,TaskType
    return LoraConfig(task_type=TaskType.CAUSAL_LM,r=cfg['r'],lora_alpha=cfg['alpha'],
        lora_dropout=cfg['dropout'],bias='none',target_modules=cfg['target_modules'],
        layers_to_transform=cfg['layers'],layers_pattern='layers',revision=cfg['revision'])


def preflight(cfg):
    # Compatibilidad real entre PEFT/Transformers y la selección de capas, sin descargar pesos.
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import get_peft_model
    tiny=Qwen2ForCausalLM(Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=28,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64))
    tiny=get_peft_model(tiny,make_lora(cfg))
    result=assert_trainable(tiny,cfg)
    ids=torch.tensor([[1,2,3,4,5,6]])
    loss=tiny(input_ids=ids,labels=ids,use_cache=False).loss
    if not torch.isfinite(loss):
        raise RuntimeError('Loss no finita en comprobación de compatibilidad')
    loss.backward()
    if any(p.grad is None for p in tiny.parameters() if p.requires_grad):
        raise RuntimeError('Algún adaptador no recibe gradiente')
    del tiny
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--budget-seconds',type=int,default=2400)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'ejecucion.json').exists():
        raise RuntimeError('Ya hay una ejecución en esa carpeta')
    cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    rows=read(root/'pares.jsonl');controls=read(root/'controles.jsonl');validate(rows,controls)
    refs=[dict(r,type=r.get('type','reconstruccion')) for r in rows+controls]
    (out/'referencias.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in refs),encoding='utf-8')
    start=time.monotonic();deadline=start+args.budget_seconds
    def check_time():
        if time.monotonic()>=deadline:
            raise TimeoutError('Límite de tiempo del piloto alcanzado')
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import get_peft_model
    if not torch.cuda.is_available():
        raise RuntimeError('Se necesita GPU CUDA')
    random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    manifest=dict(status='running',config=cfg,training_performed=False,
        started_utc=datetime.now(timezone.utc).isoformat(),
        versions={p:importlib.metadata.version(p) for p in ['torch','transformers','peft','accelerate','safetensors']},
        gpu=torch.cuda.get_device_name(0),input_hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['pares.jsonl','controles.jsonl','config.json']},
        scope='Piloto de desarrollo, sin M1 del cliente, sin GSM8K/IFEval/MMLU',completed_stages=[])
    save(out/'ejecucion.json',manifest)
    model=None
    try:
        manifest['preflight']=preflight(cfg);save(out/'ejecucion.json',manifest)
        tokenizer=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'])
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model=AutoModelForCausalLM.from_pretrained(cfg['model'],revision=cfg['revision'],
            dtype=dtype,device_map={'':0},attn_implementation='sdpa')
        if model.config.num_hidden_layers!=28:
            raise RuntimeError('Arquitectura inesperada')
        if getattr(model,'peft_config',None):
            raise RuntimeError('El baseline debe carecer de adaptadores')
        resolved=getattr(model.config,'_commit_hash',None)
        # Transformers puede no conservar _commit_hash: los pesos y el tokenizer se cargan por SHA completo.
        if resolved is not None and resolved!=cfg['revision']:
            raise RuntimeError('La revisión cargada no coincide')
        manifest['requested_full_revision']=cfg['revision'];manifest['dtype']=str(dtype)
        manifest['resolved_config_revision']=resolved
        save(out/'ejecucion.json',manifest)

        def messages(row):
            return [dict(role='system',content=cfg['instruction']),dict(role='user',content=row['input'])]

        # Construir ejemplos con etiquetas SOLO en la respuesta del asistente; no truncar.
        encoded={}
        for row in refs:
            if row['split']!='train':
                continue
            prompt=tokenizer.apply_chat_template(messages(row),tokenize=False,add_generation_prompt=True)
            full=tokenizer.apply_chat_template(messages(row)+[dict(role='assistant',content=row['output'])],
                tokenize=False,add_generation_prompt=False)
            prefix=tokenizer.encode(prompt,add_special_tokens=False)
            ids=tokenizer.encode(full,add_special_tokens=False)
            if ids[:len(prefix)]!=prefix or len(ids)<=len(prefix):
                raise RuntimeError('No se puede delimitar la respuesta sin ambigüedad')
            if len(ids)>cfg['max_sequence_length']:
                raise RuntimeError('Ejemplo demasiado largo; no se trunca: '+row['id'])
            encoded[row['id']]=(ids,[-100]*len(prefix)+ids[len(prefix):])
        manifest['max_training_tokens']=max(len(x[0]) for x in encoded.values())
        reports={}

        def evaluate(stage):
            check_time();model.eval();model.gradient_checkpointing_disable()
            dest=out/stage;dest.mkdir()
            predictions=[];cache={}
            with (dest/'predicciones.jsonl').open('w',encoding='utf-8') as f:
                for index,row in enumerate(refs,1):
                    check_time()
                    # Cache por texto de entrada, nunca por referencia ni nivel.
                    if row['input'] not in cache:
                        prompt=tokenizer.apply_chat_template(messages(row),tokenize=False,add_generation_prompt=True)
                        inputs=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to('cuda')
                        with torch.inference_mode():
                            output=model.generate(**inputs,do_sample=False,num_beams=1,repetition_penalty=1.0,
                                max_new_tokens=cfg['max_new_tokens'],pad_token_id=tokenizer.eos_token_id,use_cache=True)
                        tokens=output[0,inputs['input_ids'].shape[-1]:]
                        eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                        cache[row['input']]=dict(prediction=tokenizer.decode(tokens,skip_special_tokens=True,
                            clean_up_tokenization_spaces=False),generated_tokens=len(tokens),
                            hit_token_limit=len(tokens)>=cfg['max_new_tokens'] and int(tokens[-1]) not in eos)
                    pred=dict(id=row['id'],**cache[row['input']]);predictions.append(pred)
                    f.write(json.dumps(pred,ensure_ascii=False)+'\n');f.flush()
            report,details=metrics(refs,predictions)
            report['generation']=dict(do_sample=False,num_beams=1,repetition_penalty=1.0,max_new_tokens=cfg['max_new_tokens'])
            save(dest/'metricas.json',report)
            (dest/'detalle.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in details),encoding='utf-8')
            reports[stage]=report
            save(out/'comparacion.json',reports)
            print(stage,json.dumps(report['unique_pairs']),flush=True)

        evaluate('baseline')
        model=get_peft_model(model,make_lora(cfg))
        manifest['trainable']=assert_trainable(model,cfg)
        model.enable_input_require_grads()
        model.config.use_cache=False
        save(out/'ejecucion.json',manifest)
        tokenizer.save_pretrained(out/'tokenizer')
        for stage in ['N2','N3']:
            check_time()
            data=schedule(rows,controls,cfg,stage)
            (out/f'muestreo_{stage}.json').write_text(json.dumps(dict(ids=[r['id'] for r in data],
                counts_by_level=dict(Counter(r['level'] for r in data)),note='Duplicados y repeticiones muestreados explícitamente; no son casos independientes'),ensure_ascii=False,indent=2),encoding='utf-8')
            model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            trainable=[p for p in model.parameters() if p.requires_grad]
            optimizer=torch.optim.AdamW(trainable,lr=cfg['lr'],weight_decay=0.0)
            scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
            optimizer.zero_grad(set_to_none=True)
            warmup=max(1,round(cfg['steps_per_stage']*cfg['warmup_ratio']))
            checkpoint=out/f'adapter_{stage}'
            with (out/f'loss_{stage}.jsonl').open('w',encoding='utf-8') as log:
                for step in range(cfg['steps_per_stage']):
                    check_time()
                    if step<warmup:
                        factor=(step+1)/warmup
                    else:
                        factor=.5*(1+math.cos(math.pi*(step-warmup)/max(1,cfg['steps_per_stage']-warmup)))
                    for group in optimizer.param_groups:group['lr']=cfg['lr']*factor
                    losses=[]
                    for row in data[step*cfg['gradient_accumulation']:(step+1)*cfg['gradient_accumulation']]:
                        check_time()
                        ids,labels=encoded[row['id']]
                        x=torch.tensor([ids],device='cuda');y=torch.tensor([labels],device='cuda')
                        with torch.autocast('cuda',dtype=dtype):
                            loss=model(input_ids=x,attention_mask=torch.ones_like(x),labels=y,use_cache=False).loss
                        if not torch.isfinite(loss):
                            raise RuntimeError('Loss no finita')
                        losses.append(float(loss.detach()))
                        scaler.scale(loss/cfg['gradient_accumulation']).backward()
                        del x,y,loss
                    scaler.unscale_(optimizer)
                    norm=torch.nn.utils.clip_grad_norm_(trainable,1.0)
                    if not torch.isfinite(norm):
                        raise RuntimeError('Gradiente no finito')
                    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                    manifest['training_performed']=True
                    record=dict(step=step+1,loss=sum(losses)/len(losses),lr=optimizer.param_groups[0]['lr'],
                        grad_norm=float(norm),elapsed_seconds=time.monotonic()-start)
                    log.write(json.dumps(record)+'\n');log.flush()
                    if (step+1)%10==0:
                        print(stage,record,flush=True)
                        # Checkpoint recuperable: contiene el mismo adaptador en su estado actual.
                        model.save_pretrained(checkpoint,safe_serialization=True)
                model.save_pretrained(checkpoint,safe_serialization=True)
            del optimizer,scaler
            torch.cuda.empty_cache()
            manifest['completed_stages'].append(stage);save(out/'ejecucion.json',manifest)
            evaluate('post_'+stage)
        assert_trainable(model,cfg)
        manifest['status']='complete'
        lines=['Piloto secuencial terminado. Pares únicos; desarrollo, no test independiente.','']
        for stage,report in reports.items():
            lines.append(stage)
            for split,levels in report['by_split'].items():
                for level,m in levels.items():
                    lines.append(f"  {split} N{level}: {m['exact']}/{m['n']} exactos")
        lines+=['','No se han medido GSM8K, IFEval o MMLU. No se certifica conservación general.',
                'Para inferencia final se usa únicamente adapter_N3 con el modelo base exacto.']
        (out/'RESUMEN.txt').write_text('\n'.join(lines),encoding='utf-8')
        print('\n'.join(lines),flush=True)
    except Exception as exc:
        manifest.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
            max_allocated_gpu_gb=torch.cuda.max_memory_allocated()/1024**3)
        save(out/'ejecucion.json',manifest)


if __name__=='__main__':
    main()
