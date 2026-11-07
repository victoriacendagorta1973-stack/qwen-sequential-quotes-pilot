import json
from pathlib import Path
import random

OPEN, CLOSE = '«“‘', '»”’'


def read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l]


def depth(text):
    stack, maximum = [], 0
    for c in text:
        if c in OPEN:
            if len(stack)>=3 or c!=OPEN[len(stack)]:
                raise ValueError('anidamiento incorrecto')
            stack.append(c); maximum=max(maximum,len(stack))
        elif c in CLOSE:
            if not stack or stack.pop()!=OPEN[CLOSE.index(c)]:
                raise ValueError('cierre incorrecto')
        elif c in '\"\'':
            raise ValueError('comilla recta en objetivo')
    if stack:
        raise ValueError('apertura sin cierre')
    return maximum


def neutral(text):
    return text.translate(str.maketrans({c:'"' for c in OPEN+CLOSE}))


def skeleton(text):
    return ''.join(c for c in text if c not in OPEN+CLOSE+'"\'')


def validate(rows, controls):
    all_rows=rows+controls
    if len({r['id'] for r in all_rows})!=len(all_rows):
        raise ValueError('IDs duplicados')
    by_pattern={}
    input_splits={}
    for r in rows:
        if r['pattern_id']=='P1':
            raise ValueError('P1 excluido por procedencia mixta')
        if depth(r['output'])!=r['level'] or neutral(r['output'])!=r['input']:
            raise ValueError('par inválido')
        by_pattern.setdefault(r['pattern_id'],set()).add(r['split'])
    if any(len(s)>1 for s in by_pattern.values()):
        raise ValueError('una plantilla cruza particiones')
    for r in all_rows:
        input_splits.setdefault(r['input'],set()).add(r['split'])
    if any(len(s)>1 for s in input_splits.values()):
        raise ValueError('entrada compartida entre train y dev')
    docs={}
    for r in controls:
        if r['input']!=r['output'] or depth(r['output'])!=0:
            raise ValueError('control inválido')
        docs.setdefault(r['source_document'],set()).add(r['split'])
    if any(len(s)>1 for s in docs.values()):
        raise ValueError('documento de controles cruza particiones')


def schedule(rows, controls, cfg, stage):
    rng=random.Random(cfg['seed']+(2 if stage=='N2' else 3))
    n=cfg['steps_per_stage']*cfg['gradient_accumulation']
    control_n=round(n*cfg['controls_fraction'])
    quotes_n=n-control_n
    mix=cfg['replay'][stage]
    counts={key:int(quotes_n*weight) for key,weight in mix.items()}
    counts[list(mix)[-1]]+=quotes_n-sum(counts.values())
    result=[]
    for key,count in counts.items():
        pool=[r for r in rows if r['split']=='train' and r['level']==int(key[1:])]
        if not pool:
            raise ValueError('pool de entrenamiento vacío')
        result.extend(rng.choices(pool,k=count))
    pool=[r for r in controls if r['split']=='train']
    result.extend(rng.choices(pool,k=control_n))
    rng.shuffle(result)
    return result


def metrics(refs, predictions):
    if len(refs)!=len(predictions) or [r['id'] for r in refs]!=[p['id'] for p in predictions]:
        raise ValueError('predicciones sin correspondencia')
    details=[]
    for ref,pred in zip(refs,predictions):
        text=pred['prediction']
        try: valid=depth(text)==ref['level']
        except ValueError: valid=False
        details.append(dict(id=ref['id'],level=ref['level'],split=ref['split'],
            exact=text==ref['output'],hierarchy_valid=valid,
            non_quote_preserved=skeleton(text)==skeleton(ref['input']),
            delimiters_and_content_preserved=neutral(text)==ref['input'],
            control=ref['level']==0,unnecessary_change=ref['level']==0 and text!=ref['input']))
    def summarize(items):
        n=len(items)
        return dict(n=n,exact=sum(x['exact'] for x in items),
            exact_rate=sum(x['exact'] for x in items)/n if n else None,
            hierarchy_valid_rate=sum(x['hierarchy_valid'] for x in items)/n if n else None,
            non_quote_preserved_rate=sum(x['non_quote_preserved'] for x in items)/n if n else None)
    unique={}
    for ref,detail in zip(refs,details):
        unique.setdefault((ref['input'],ref['output']),detail)
    result=dict(global_records=summarize(details),unique_pairs=summarize(list(unique.values())),by_split={})
    for split in ['train','dev']:
        result['by_split'][split]={str(level):summarize([r for r in unique.values() if r['split']==split and r['level']==level]) for level in range(4)}
    return result,details
