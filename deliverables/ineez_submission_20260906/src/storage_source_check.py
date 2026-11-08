"""Compare provided storage with a cached official monthly series; never replace inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import urllib.request

import numpy as np
import pandas as pd

OUT = Path("outputs_storage_source")
URL = "https://www.eia.gov/dnav/ng/hist/n5030us2m.htm"
COLUMN = "Total_Natural_Gas_Underground_Storage_Volume_USA"


def parse_series(content):
    records = []
    for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", content, re.S | re.I):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", x)).strip() for x in
                 re.findall(r"<td\b[^>]*>(.*?)</td>", row, re.S | re.I)]
        if len(cells) != 13 or not re.fullmatch(r"\d{4}", cells[0]):
            continue
        for month, value in enumerate(cells[1:], 1):
            if re.fullmatch(r"[\d,]+", value):
                records.append(dict(month=f"{cells[0]}-{month:02}", eia_value=int(value.replace(",", ""))))
    result = pd.DataFrame(records)
    if result.empty or result.month.duplicated().any():
        raise ValueError("Unexpected EIA monthly table")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    snapshot = OUT/"n5030us2m.html"
    retrieval = OUT/"retrieval.json"
    if args.download and not snapshot.exists():
        payload = urllib.request.urlopen(URL, timeout=45).read()
        parse_series(payload.decode("utf-8"))
        snapshot.write_bytes(payload)
        retrieval.write_text(json.dumps(dict(url=URL, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                                             sha256=hashlib.sha256(payload).hexdigest()), indent=2), encoding="utf-8")
    if not snapshot.exists():
        raise FileNotFoundError("First run requires --download with network permission")
    provenance = json.loads(retrieval.read_text(encoding="utf-8"))
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == provenance["sha256"]
    official = parse_series(snapshot.read_text(encoding="utf-8"))
    train = next(Path('.').glob('*data_train.csv'))
    data = pd.read_csv(train, usecols=["date", COLUMN])
    data["month"] = pd.to_datetime(data.date).dt.strftime("%Y-%m")
    paired = data.merge(official, on="month", how="left", validate="one_to_one")
    assert len(paired) == 204 and paired.eia_value.notna().all()
    paired["difference_mmcf"] = paired[COLUMN]-paired.eia_value
    paired["matches_nearest10"] = paired[COLUMN].eq(np.round(paired.eia_value/10)*10)
    paired.to_csv(OUT/"matched_training_months.csv", index=False)
    summary = dict(**provenance, train_sha256=hashlib.sha256(train.read_bytes()).hexdigest(),
                   compared_months=len(paired), exact_matches=int(paired.difference_mmcf.eq(0).sum()),
                   within5_mmcf=int(paired.difference_mmcf.abs().le(5).sum()),
                   nearest10_matches=int(paired.matches_nearest10.sum()),
                   max_absolute_difference_mmcf=int(paired.difference_mmcf.abs().max()),
                   interpretation="Value alignment supports series identification, not original provider lineage or historical vintage identity.",
                   release_policy="Use monthly storage lag2 for new experiments; historical complete release calendar/vintage still unverified; do not infer availability from MAPE.")
    (OUT/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
