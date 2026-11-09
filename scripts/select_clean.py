#!/usr/bin/env python3
"""Select the ChartAudit-Fin items straight from the raw ChartQA files, by written rules only.

No model is involved anywhere in this script. Steps, per official split (train/val/test,
chart-disjoint):
  1. Read the human-written questions (<split>_human.json).
  2. Drop structural problems: missing data table, duplicate or malformed question, empty
     answer, unparseable table, a train/val table identical to one in a later split, and the
     rule-based checks in chartqa_rules.classify (e.g. a looked-up number must exist in the
     chart's own data table).
  3. Keep questions whose keyword domain tags include finance or economy.
  4. Stratified random sample (seed fixed) over source x chart type x answer type x
     question type, plus ~5% "unanswerable" items (a question paired with an unrelated chart).
  5. Copy the chart PNGs and write <out>/<split>.jsonl with the human answer as gold.
     No evidence or rationale is written here; those come from blind labelling.

Usage:
  python scripts/select_clean.py --chartqa "/path/to/ChartQA Dataset" --out data/v2.0/selection --images data/v2.0/images
"""
import argparse, collections, hashlib, json, os, random, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chartqa_rules as R

TARGET = {"train": (1100, 55), "val": (130, 7), "test": (171, 9)}   # (answerable, unanswerable)
KEEP_DOMAINS = {"finance", "economy"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chartqa", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", required=True, help="folder for chart PNGs; record paths are images/<split>/<chart>.png relative to its parent")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)

    raw, table_hash, manifest = {}, {}, {}
    for split in TARGET:
        qa = os.path.join(a.chartqa, split, f"{split}_human.json")
        manifest[f"raw {split}/{split}_human.json"] = R.sha256(qa)
        raw[split] = json.load(open(qa, encoding="utf-8"))
        for x in raw[split]:
            p = os.path.join(a.chartqa, split, "tables", x["imgname"][:-4] + ".csv")
            if os.path.exists(p):
                table_hash[(split, x["imgname"][:-4])] = hashlib.md5(open(p, "rb").read()).hexdigest()
    later = {"train": {"val", "test"}, "val": {"test"}, "test": set()}

    stats = {"seed": a.seed, "keep_domains": sorted(KEEP_DOMAINS), "splits": {}}
    for split, (n_ans, n_unans) in TARGET.items():
        reasons, seen, pool, charts = collections.Counter(), set(), [], {}
        block = {h for (s, _), h in table_hash.items() if s in later[split]}
        for qidx, x in enumerate(raw[split]):
            name = x["imgname"][:-4]
            p = os.path.join(a.chartqa, split, "tables", name + ".csv")
            q = x["query"].strip()
            if not os.path.exists(p):
                reasons["no_data_table"] += 1; continue
            if (name, q.lower()) in seen:
                reasons["duplicate"] += 1; continue
            seen.add((name, q.lower()))
            if len(q.split()) < 3:
                reasons["malformed_question"] += 1; continue
            if str(x["label"]).strip() == "":
                reasons["empty_answer"] += 1; continue
            if name not in charts:
                t = R.read_table(p)
                if t is None or len(R.numeric_cells(t[1])) < 2:
                    charts[name] = None
                else:
                    anp = os.path.join(a.chartqa, split, "annotations", name + ".json")
                    ann = R.load_annotation(anp, *t) if os.path.exists(anp) else None
                    charts[name] = (t[0], t[1], ann, R.linearise(t[0], t[1], R.source_of(name), ann))
            c = charts[name]
            if c is None:
                reasons["table_unparseable"] += 1; continue
            if table_hash[(split, name)] in block:
                reasons["same_table_in_later_split"] += 1; continue
            header, body, ann, text = c
            ans, atype = R.normalise_answer(x["label"])
            it = {"id": f"{split}-{name}-{qidx}", "chart": name, "qidx": qidx, "query": q, "label": x["label"],
                  "answer": ans, "answer_type": atype, "source": R.source_of(name),
                  "chart_type": (ann or {}).get("chart_type") or "unknown"}
            why, flags = R.classify(it, header, body, ann, text)
            if why:
                reasons[why] += 1; continue
            it["flags"], it["question_type"] = flags, flags["question_type"]
            it["domain_tags"] = R.domain_tags(header, body, q, ann)
            if not KEEP_DOMAINS & set(it["domain_tags"]):
                reasons["not_finance_or_economy"] += 1; continue
            pool.append(it)
        charts = {k: v for k, v in charts.items() if v is not None}

        chosen = R.stratified_sample(pool, n_ans, rng)
        used = {x["id"] for x in chosen}
        colours = {k: (R.chart_colours(v[2])[0] if v[2] else []) for k, v in charts.items()}
        unans = R.make_unanswerable(pool, {k: v[3] for k, v in charts.items()}, colours, n_unans, rng, used)

        img_dir = os.path.join(a.images, split)
        os.makedirs(img_dir, exist_ok=True)
        recs = []
        for it, other in [(i, None) for i in chosen] + list(unans):
            chart = other or it["chart"]
            shutil.copy(os.path.join(a.chartqa, split, "png", chart + ".png"), os.path.join(img_dir, chart + ".png"))
            ann = charts[chart][2]
            un = other is not None
            recs.append({
                "id": it["id"] + ("-unans" if un else ""), "split": split, "chart_id": chart,
                "question_source_chart": it["chart"], "source": R.source_of(chart),
                "chart_type": (ann or {}).get("chart_type") or "unknown",
                "question": it["query"],
                "gold_answer_raw": None if un else it["label"],
                "answer": None if un else it["answer"],
                "answer_type": "unanswerable" if un else it["answer_type"],
                "question_type": "unanswerable" if un else it["question_type"],
                "domain": "finance_economy", "domain_tags": it["domain_tags"],
                "flags": {} if un else it["flags"],
                "image": f"images/{split}/{chart}.png",
                "chart_data_table": charts[chart][3],
            })
        rng.shuffle(recs)
        with open(os.path.join(a.out, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats["splits"][split] = {
            "raw_human_questions": len(raw[split]), "excluded": dict(reasons.most_common()),
            "finance_economy_pool": len(pool), "selected_answerable": len(chosen),
            "selected_unanswerable": len(unans), "charts": len({r["chart_id"] for r in recs}),
            "by_answer_type": dict(collections.Counter(r["answer_type"] for r in recs)),
            "by_question_type": dict(collections.Counter(r["question_type"] for r in recs)),
            "by_source": dict(collections.Counter(r["source"] for r in recs)),
        }
    ids = {s: {json.loads(l)["chart_id"] for l in open(os.path.join(a.out, f"{s}.jsonl"))} for s in TARGET}
    stats["chart_overlap"] = {"train_val": len(ids["train"] & ids["val"]), "train_test": len(ids["train"] & ids["test"]),
                              "val_test": len(ids["val"] & ids["test"])}
    for s in TARGET:
        manifest[f"out {s}.jsonl"] = R.sha256(os.path.join(a.out, f"{s}.jsonl"))
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=2)
    with open(os.path.join(a.out, "MANIFEST.txt"), "w") as f:
        for k, v in manifest.items():
            f.write(f"{v}  {k}\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
