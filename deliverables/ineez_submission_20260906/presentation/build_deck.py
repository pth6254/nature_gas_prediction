"""Build the offline HTML presentation from saved evidence; never refit models."""
from __future__ import annotations

import csv
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SOURCES: dict[str, str] = {}


def rows(name: str) -> list[dict]:
    path = ROOT / name
    SOURCES[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def one(data: list[dict], **keys) -> dict:
    matches = [r for r in data if all(r[k] == str(v) for k, v in keys.items())]
    assert len(matches) == 1, (keys, len(matches))
    return matches[0]


def triplet(data: list[dict], field="mape_pct", **keys) -> list[float]:
    return [float(one(data, horizon=h, **keys)[field]) for h in (1, 2, 3)]


def main() -> None:
    submission_hash = hashlib.sha256((ROOT / "submission.csv").read_bytes()).hexdigest()
    final = rows("outputs_final_v1/final_vs_naive.csv")
    metrics = rows("outputs_final_v1/validation_metrics.csv")
    predictions = rows("outputs_final_v1/validation_predictions.csv")
    train = rows("★data_train.csv")
    assessment = rows("outputs_variable_suitability/variable_assessment.csv")
    hypotheses = rows("outputs_hypotheses/metrics.csv")
    stats = rows("outputs_statistics/metrics.csv")
    ml = rows("outputs_ml/metrics.csv")
    dl = rows("outputs_dl/seed_stability.csv")
    bolt = rows("outputs_chronos/metrics.csv")
    chronos = rows("outputs_chronos_covariates/metrics.csv")
    audit = rows("outputs_variable_suitability/metrics.csv")
    feature = rows("outputs_feature_audit/metrics.csv")
    incremental = rows("outputs_incremental_fundamentals/metrics.csv")
    cases = rows("outputs_final_v1/report_cases.csv")
    contributions = rows("outputs_final_v1/case_top_contributions.csv")
    submission = rows("submission.csv")
    config_path = ROOT / "outputs_final_v1/model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    SOURCES[str(config_path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    for source in ("REPORT.md", "(주)인이지+코딩테스트+안내.pdf"):
        SOURCES[source] = hashlib.sha256((ROOT / source).read_bytes()).hexdigest()

    # Independently recompute the final metrics from individual forecast rows.
    for h, expected_n in ((1, 83), (2, 82), (3, 81)):
        p = [r for r in predictions if r["horizon"] == str(h)]
        assert len(p) == expected_n
        mape = sum(abs(float(r["predicted"]) / float(r["actual"]) - 1) for r in p) / len(p) * 100
        naive = sum(abs(float(r["current_price"]) / float(r["actual"]) - 1) for r in p) / len(p) * 100
        assert abs(mape - float(one(final, horizon=h)["MAPE"])) < 1e-10
        assert abs(naive - float(one(final, horizon=h)["Naive_MAPE"])) < 1e-10
        assert all(int(r["train_last_feature"]) + h <= int(r["origin"]) for r in p)
    assert len(submission) == 48
    assert list(submission[0]) == ["date", "pred_1", "pred_2", "pred_3"]

    linear_rows = []
    for name, key in (("Naive", "naive"), ("SARIMA", "sarima_011_011")):
        linear_rows.append(dict(label=name, values=triplet(stats, experiment=key, year="all")))
    for name, family in (("Ridge", "ridge"), ("ElasticNet", "elastic_net"), ("LightGBM", "lightgbm"), ("CatBoost", "catboost")):
        best = [min((r for r in ml if r["year"] == "all" and r["horizon"] == str(h) and r["experiment"].startswith(family + "__")), key=lambda r: float(r["mape_pct"])) for h in (1, 2, 3)]
        linear_rows.append(dict(label=name, values=[float(r["mape_pct"]) for r in best], selected=[r["experiment"] for r in best]))
    deep_rows = []
    for name in ("MLP", "NBEATSx", "TFT"):
        best = [min((r for r in dl if r["model"] == name and r["horizon"] == str(h)), key=lambda r: float(r["mape_mean"])) for h in (1, 2, 3)]
        deep_rows.append(dict(label=name, values=[float(r["mape_mean"]) for r in best], errors=[float(r["mape_std"]) for r in best], selected=[r["input_group"] for r in best]))
    deep_rows.append(dict(label="Bolt · 가격", values=triplet(bolt, year="all")))
    for label, group in (("C2 · 가격", "price"), ("C2 · +수급", "fundamentals"), ("C2 · +시장", "market")):
        deep_rows.append(dict(label=label, values=triplet(chronos, year="all", group=group)))

    ablation = []
    for group in ("production", "storage", "consumption", "market"):
        worse = 0
        for family in ("ridge", "elastic_net", "lightgbm", "catboost"):
            for h in (1, 2, 3):
                base = float(one(feature, family=family, combination="current", horizon=h, year="all")["mape_pct"])
                dropped = float(one(feature, family=family, combination="drop_" + group, horizon=h, year="all")["mape_pct"])
                worse += dropped > base
        ablation.append(worse)

    data = dict(
        final=[dict(h=int(r["horizon"]), model=r["model"], mape=float(r["MAPE"]), naive=float(r["Naive_MAPE"]), mae=float(r["MAE"]), direction=float(r["direction_pct"]), f1=float(one(metrics, horizon=r["horizon"], period="all")["turning_f1_pct"])) for r in final],
        train=[dict(date=r["date"][:7], value=float(r["Natural_Gas_US_Henry_Hub_Gas"])) for r in train],
        hypotheses=[dict(label=label, values=triplet(hypotheses, experiment=key, year="all")) for label, key in (("가격 이력", "history"), ("개별 Z 입력", "core_z"), ("동일 가중 합성", "equal_index"), ("계수 부호 제한", "signed_core"))],
        eligible=sum(r["predictive_eligible"] == "True" for r in assessment),
        audit=[dict(label=label, values=[float(one(audit, family=fam, candidate=key, horizon=3, period="all")["mape_pct"]) for fam in ("ridge", "lightgbm")]) for label, key in (("가격 이력", "history"), ("+미국 수급 6개", "add_fundamentals"), ("+에너지 3개", "add_energy"), ("전체 78개", "all_groups"))],
        ablation=ablation, linear=linear_rows, deep=deep_rows,
        validation=[dict(date=r["target_date"][:7], actual=float(r["actual"]), predicted=float(r["predicted"]), naive=float(r["current_price"])) for r in predictions if r["horizon"] == "3"],
        cases=[dict(date=r["date"][:7], h=int(r["horizon"]), current=float(r["current_price"]), actual=float(r["actual"]), predicted=float(r["predicted"]), previous=float(one(predictions, date=r["date"], horizon=r["horizon"])["previous_price"])) for r in cases],
        contributions=[dict(feature=r["feature"], value=float(r["centered_log_return_contribution"])) for r in contributions if r["date"] == "2018-11-30" and r["horizon"] == "1"],
        increment=[dict(label=label, delta=float(one(incremental, family="ridge", horizon=2, policy="imports", period=period)["mape_pct"]) - float(one(incremental, family="ridge", horizon=2, policy="base", period=period)["mape_pct"])) for label, period in (("전체", "all"), ("전반", "early"), ("후반", "late"))],
        submission=submission[-3:], config=config,
    )
    assert data["eligible"] == 78 and len(assessment) == 112
    assert data["ablation"] == [8, 11, 7, 11]
    source_json = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = (HERE / "slides.template.html").read_text(encoding="utf-8")
    assert template.count("/*__DATA__*/") == 1
    output = HERE / "REPORT_SLIDES.html"
    output.write_text(template.replace("/*__DATA__*/", source_json), encoding="utf-8")
    class LinkCheck(HTMLParser):
        links: list[str] = []
        slides = 0

        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            if tag == "a" and "href" in attributes:
                self.links.append(attributes["href"])
            if tag == "section" and "slide" in attributes.get("class", "").split():
                self.slides += 1

    checker = LinkCheck()
    checker.feed(output.read_text(encoding="utf-8"))
    assert checker.slides == 10
    missing = [link for link in checker.links if not link.startswith(("http", "#")) and link != "evidence_manifest.json" and not (HERE / link).exists()]
    assert not missing, missing
    manifest = dict(sources=SOURCES, html_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), checks=dict(final_mape_recomputed=True, final_naive_recomputed=True, validation_counts=[83, 82, 81], slide_count=10, source_links_checked=len(checker.links), submission_rows=48, submission_unchanged=submission_hash == hashlib.sha256((ROOT / "submission.csv").read_bytes()).hexdigest()), selected_chart_data=data)
    (HERE / "evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built {output}; {output.stat().st_size:,} bytes; evidence and submission integrity checked.")


if __name__ == "__main__":
    main()
