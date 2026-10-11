#!/usr/bin/env python3
"""Primary answer metric v2, fixed before primary-outcome generation.

Correct if, after normalization, either
  (a) prediction == gold, or numbers are equal after canonicalization, or
  (b) the gold answer appears as a whole-token span in the prediction AND the
      prediction has at most MAX_EXTRA_TOKENS tokens beyond the gold.

Compatible with this repo's answer rows (`answer`, `gold`, `source`) and with
older rows (`prediction`, `condition`).
"""
import argparse
import json
import re
import string

MAX_EXTRA_TOKENS = 4
_PUNCT = set(string.punctuation) - {'.', '-'}
_DASHES = str.maketrans({
    '\u2010': '-',
    '\u2011': '-',
    '\u2012': '-',
    '\u2013': '-',
    '\u2014': '-',
    '\u2212': '-',
})


def _canon_number(tok):
    t = tok.replace(',', '')
    if re.fullmatch(r'[-+]?\d+(\.\d+)?%?', t):
        pct = t.endswith('%')
        v = float(t.rstrip('%'))
        s = ('%f' % v).rstrip('0').rstrip('.')
        return s + ('%' if pct else '')
    return tok


def normalize_v2(s):
    s = str(s).lower().replace('\u00a0', ' ').translate(_DASHES)
    s = re.sub(r'\s+category:articles with hcards\b', ' ', s)
    s = re.sub(r'(?<=\S)\s+details\s*$', ' ', s)
    s = re.sub(r'(?<=\S)\s+\([^)]*\)\s*$', ' ', s)
    s = re.sub(r'(?<=\d),(?=\d{3}\b)', '', s)
    while re.search(r'(?<=\d)\s(?=\d{3}\b)', s):
        s = re.sub(r'(?<=\d)\s(?=\d{3}\b)', '', s)
    s = re.sub(r'(?<=\d)\s*-\s*(?=\d)', '-', s)
    s = re.sub(r'(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])', ' ', s)
    s = ''.join(ch if ch not in _PUNCT else ' ' for ch in s)
    no_art = re.sub(r'\b(a|an|the)\b', ' ', s)
    base = no_art if no_art.split() else s
    toks = [t.strip('.-') if not re.fullmatch(r'[-+]?[\d,]*\.?\d+%?', t) else t
            for t in base.split()]
    return [_canon_number(t) for t in toks if t]


def exact_match(pred, gold):
    return normalize_v2(pred) == normalize_v2(gold)


def answer_match(pred, gold):
    p, g = normalize_v2(pred), normalize_v2(gold)
    if not g:
        return False
    if p == g:
        return True
    if len(p) - len(g) > MAX_EXTRA_TOKENS or len(p) < len(g):
        return False
    n = len(g)
    return any(p[i:i + n] == g for i in range(len(p) - n + 1))


def row_condition(row):
    return row.get('condition') or row.get('source')


def row_prediction(row):
    return row.get('prediction', row.get('answer', ''))


def row_gold(row, item):
    return row.get('gold', item.get('gold_answer', item.get('answer', '')))


def _selftest():
    cases = [
        ('Ghana', 'Ghana', True),
        ('The answer is Ghana.', 'Ghana', True),
        ('4.0', '4', True),
        ('1,234', '1234', True),
        ('1', '1999', False),
        ('Ghana, Togo, Benin, Mali, Niger, Chad', 'Ghana', False),
        ('A', 'A', True),
        ('The A', 'A', True),
        ("O'Brien", 'O Brien', True),
        ('2-1', '2–1', True),
        ('3 - 1', '3-1', True),
        ('−21', '-21', True),
        ('3 115', '3115', True),
        ('David Toms', 'David Toms (3)', True),
        ('Klaas-Jan Huntelaar', 'Klaas-Jan Huntelaar Category:Articles with hCards', True),
        ('1968', '1968 Details', True),
        ('1904', '(6)', False),
        ('February 18, 2005', 'February18,2005', True),
    ]
    for p, g, want in cases:
        got = answer_match(p, g)
        assert got == want, (p, g, got, want)
    print('selftest ok')


def main(a):
    items = {}
    with open(a.items, encoding='utf-8') as f:
        for l in f:
            if not l.strip():
                continue
            it = json.loads(l)
            items[it['item_id']] = it
    n = old = new = 0
    fails = []
    with open(a.answers, encoding='utf-8') as f:
        for l in f:
            if not l.strip():
                continue
            r = json.loads(l)
            if row_condition(r) != a.condition:
                continue
            if a.model and r.get('model') != a.model:
                continue
            it = items[r['item_id']]
            gold, pred = row_gold(r, it), row_prediction(r)
            n += 1
            from_old = exact_match(pred, gold)
            ok = answer_match(pred, gold)
            old += int(from_old)
            new += int(ok)
            if not ok:
                sel = it['sql']['sel']
                g = it['rows'][it['gold_row']]
                fails.append({'item_id': it['item_id'], 'model': r.get('model'),
                              'condition': row_condition(r), 'question': it['question'],
                              'gold': gold, 'pred': pred,
                              'asked_column': it['header'][sel],
                              'pred_matches_column': [it['header'][j] for j, c in enumerate(g)
                                                      if answer_match(pred, c)],
                              'gold_row': dict(zip(it['header'], g))})
    if n == 0:
        print(f'{a.condition}: n=0')
    else:
        print(f'{a.condition}: n={n}  v2-exact={old / n:.4f}  answer_match v2={new / n:.4f}  '
              f'remaining failures={len(fails)}')
    if a.dump:
        with open(a.dump, 'w', encoding='utf-8') as f:
            for x in fails:
                f.write(json.dumps(x, ensure_ascii=False, sort_keys=True) + '\n')


if __name__ == '__main__':
    _selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument('--items')
    ap.add_argument('--answers')
    ap.add_argument('--condition', default='sel:external:gold_only')
    ap.add_argument('--model')
    ap.add_argument('--dump')
    args = ap.parse_args()
    if args.items and args.answers:
        main(args)
