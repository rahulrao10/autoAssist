"""report.py -- step 8: save the decision log (JSON) and a short HTML report."""
import html
import json
from pathlib import Path


def save_reports(ai, output_dir):
    """Write decision_log.json and report.html into output_dir. Returns (json_path, html_path)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    chosen = ai.chosen.spec.key if ai.chosen else None

    payload = {
        "source": ai.source, "target": ai.target, "task": ai.task, "task_reason": ai.task_reason,
        "metric": ai.metric_label, "config": ai.config.to_dict(),
        "chosen_model": chosen, "script_path": str(ai.script_path) if ai.script_path else None,
        "dropped_columns": ai.plan.dropped,
        "models": [{"rank": r.rank, "key": r.spec.key, "cv_mean": r.cv_mean, "cv_ci95": r.cv_ci,
                    "cv_fold_scores": r.cv_scores, "train_mean": r.train_mean, "holdout": r.holdout,
                    "gap_cv_minus_holdout": r.gap, "gain_over_baseline": r.gain,
                    "rating": r.rating, "rating_reason": r.rating_reason} for r in ai.results],
        "decisions": ai.log.entries,
    }
    json_path = out / "decision_log.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    e = html.escape
    rows = "".join(
        "<tr><td>%d</td><td>%s</td><td>%.4f &plusmn; %.4f</td><td>%.4f</td><td>%+.3f</td><td>%+.3f</td><td><b>%s</b></td><td>%s</td></tr>"
        % (r.rank, e(r.spec.display), r.cv_mean, r.cv_ci, r.holdout, r.gap, r.gain, e(r.rating), e(r.rating_reason))
        for r in ai.results)
    decisions = "".join("<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % (e(d["stage"]), e(d["decision"]), e(d["reason"]))
                        for d in ai.log.entries)
    page = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>AutoAssist report</title>
<style>body{font-family:Arial,sans-serif;max-width:1000px;margin:2em auto;padding:0 1em}
table{border-collapse:collapse;width:100%%;margin:1em 0}td,th{border:1px solid #ccc;padding:5px 8px;text-align:left;font-size:14px}
th{background:#f0f0f0}</style></head><body>
<h1>AutoAssist report</h1>
<p><b>Data:</b> %s<br><b>Target:</b> %s<br><b>Task:</b> %s (%s)<br><b>Metric:</b> %s<br><b>Chosen model:</b> %s</p>
<h2>Models (ranked by cross-validation score)</h2>
<table><tr><th>#</th><th>Model</th><th>CV mean &plusmn; 95%% CI</th><th>Holdout</th><th>Gap</th><th>vs dummy</th><th>Rating</th><th>Reason</th></tr>%s</table>
<h2>Every decision AutoAssist made</h2>
<table><tr><th>Stage</th><th>Decision</th><th>Reason</th></tr>%s</table>
</body></html>""" % (e(str(ai.source)), e(str(ai.target)), e(ai.task), e(ai.task_reason),
                     e(ai.metric_label), e(str(chosen)), rows, decisions)
    html_path = out / "report.html"
    html_path.write_text(page, encoding="utf-8")
    return json_path, html_path
