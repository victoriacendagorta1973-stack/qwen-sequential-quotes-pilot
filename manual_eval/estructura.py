"""Construcción determinista de delimitadores; no contiene frases objetivo."""
import json
from pathlib import Path
import re
from collections import Counter

OPEN, CLOSE = '«“‘', '»”’'


def neutral(text):
    return text.translate(str.maketrans({c:'"' for c in OPEN+CLOSE}))


def parse_json(raw):
    raw=raw.strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw=raw.split('\n',1)[1].rsplit('```',1)[0].strip()
    obj=json.loads(raw)
    if not isinstance(obj,dict) or set(obj)!={'texto'} or not isinstance(obj['texto'],str):
        raise ValueError('se requiere un objeto con la única clave texto')
    return obj['texto']


def render(marked):
    """Sustituye <q>/</q> por los signos según profundidad; ninguna palabra cambia."""
    if marked!=marked.strip() or '\n' in marked:
        raise ValueError('espacios exteriores o saltos de línea')
    if any(c in marked for c in OPEN+CLOSE+'"\''):
        raise ValueError('el texto debe usar solo etiquetas q, sin comillas literales')
    if re.search(r'<(?!/?q>)|(?<!q)>|\[|\]',marked):
        raise ValueError('etiquetas o marcadores no permitidos')
    stack=[];max_depth=0;counts=Counter();output=[];trees=[]
    for chunk in re.split(r'(<q>|</q>)',marked):
        if chunk=='<q>':
            if len(stack)>=3:
                raise ValueError('profundidad mayor que tres')
            node=[]
            if stack:stack[-1].append(node)
            else:trees.append(node)
            stack.append(node);max_depth=max(max_depth,len(stack));counts[len(stack)]+=1
            output.append(OPEN[len(stack)-1])
        elif chunk=='</q>':
            if not stack:
                raise ValueError('cierre sin apertura')
            output.append(CLOSE[len(stack)-1]);stack.pop()
        else:
            output.append(chunk)
    if stack:
        raise ValueError('etiqueta sin cerrar')
    result=''.join(output)
    # No reparar espacios silenciosamente: rechazarlos y solicitar otro candidato.
    for i,c in enumerate(result):
        if c in OPEN:
            if i+1>=len(result) or result[i+1].isspace() or result[i+1] in CLOSE:
                raise ValueError('cita vacía o espacio tras apertura')
            if i>0 and result[i-1].isalnum():
                raise ValueError('apertura pegada a una palabra')
        if c in CLOSE:
            if i==0 or result[i-1].isspace():
                raise ValueError('espacio antes de cierre')
            if i+1<len(result) and result[i+1].isalnum():
                raise ValueError('cierre pegado a una palabra')
    if re.search(r'\.[»”’]',result):
        raise ValueError('punto inmediatamente antes del cierre; evitar esta construcción en el lote')
    if result.count('¿')!=result.count('?') or result.count('¡')!=result.count('!'):
        raise ValueError('interrogación/exclamación desequilibrada')
    if result.count('(')!=result.count(')'):
        raise ValueError('paréntesis desequilibrados')
    return result,max_depth,dict(counts),trees


def check(marked,recipe):
    text,depth,counts,trees=render(marked)
    if depth!=recipe['level']:
        raise ValueError('nivel distinto del solicitado')
    if len(text.split())<recipe.get('min_words',12) or len(text.split())>recipe.get('max_words',85):
        raise ValueError('longitud fuera del intervalo')
    if not text.endswith(('.', '»', '?', '!')):
        raise ValueError('fragmento incompleto')
    if len(trees)<recipe.get('roots',0):
        raise ValueError('faltan citas exteriores separadas')
    if counts.get(2,0)<recipe.get('english',0) or counts.get(3,0)<recipe.get('simple',0):
        raise ValueError('faltan citas interiores hermanas')
    if recipe.get('every_root_child') and any(not tree for tree in trees):
        raise ValueError('cada cita exterior debe incluir una subcita')
    if recipe.get('every_english_child'):
        english=[child for root in trees for child in root]
        if not english or any(not child for child in english):
            raise ValueError('cada subcita inglesa debe contener un tercer nivel')
    return text,dict(depth=depth,counts=counts,tree=trees)


def read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l]


def score(refs,predictions):
    by_id={p['id']:p for p in predictions}
    if len(by_id)!=len(predictions) or set(by_id)!={r['id'] for r in refs}:
        raise ValueError('IDs faltantes, extra o repetidos')
    details=[]
    for r in refs:
        p=by_id[r['id']]['prediction']
        stack=[];maximum=0;valid=True
        for c in p:
            if c in OPEN:
                if len(stack)>=3 or OPEN[len(stack)]!=c:
                    valid=False;break
                stack.append(c);maximum=max(maximum,len(stack))
            elif c in CLOSE:
                if not stack or stack.pop()!=OPEN[CLOSE.index(c)]:
                    valid=False;break
            elif c in '\"\'':
                valid=False;break
        valid=valid and not stack and maximum==r['level']
        details.append(dict(id=r['id'],category=r['category'],level=r['level'],group=r['group'],
            exact=p==r['output'],hierarchy_valid=valid,
            delimiter_positions_preserved=neutral(p)==neutral(r['input']),
            non_quote_preserved=''.join(c for c in p if c not in OPEN+CLOSE+'"\'')==
                ''.join(c for c in r['input'] if c not in OPEN+CLOSE+'"\''),
            hit_token_limit=by_id[r['id']].get('hit_token_limit',False)))
    def summarize(items):
        n=len(items)
        return dict(n=n,exact=sum(r['exact'] for r in items),exact_rate=sum(r['exact'] for r in items)/n if n else None,
            non_quote_preserved=sum(r['non_quote_preserved'] for r in items),
            hierarchy_valid=sum(r['hierarchy_valid'] for r in items),
            delimiter_positions_preserved=sum(r['delimiter_positions_preserved'] for r in items),
            token_limits=sum(r['hit_token_limit'] for r in items))
    result=dict(global_records=summarize(details),by_category={})
    for cat in ['N1','N2','N3','control_sin_comillas','control_comillas_correctas']:
        result['by_category'][cat]=summarize([r for r in details if r['category']==cat])
    return result,details
