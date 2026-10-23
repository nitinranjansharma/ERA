"""Live queries against HuggingFace datasets-server. Failures are reported, not invented.

Class 1: sources whose size can be measured here. Gated sources carry no number.

Run:  python scripts/supply_audit.py
"""
from __future__ import annotations
import json
import urllib.parse
import urllib.request

HF = "https://datasets-server.huggingface.co/size?dataset="


def hf_size(dataset, timeout=90):
    url = HF + urllib.parse.quote(dataset, safe="")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 78)
    print("LIVE SUPPLY AUDIT")
    print("Every figure below is measured in this process, or the query failed.")
    print("=" * 78)
    for ds in ("ai4bharat/sangraha", "ai4bharat/IndicCorpV2", "ai4bharat/BPCC"):
        print(f"\n-- {ds}")
        j = hf_size(ds)
        if "error" in j:
            print(f"  QUERY FAILED: {j['error']}")
            print("  Ledger continues to use the stated inventory-widget / card figure.")
            continue
        size = j.get("size", {})
        dset = size.get("dataset", {})
        print(f"  partial={j.get('partial')}")
        print(f"  num_rows={dset.get('num_rows')}  estimated_num_rows={dset.get('estimated_num_rows')}")
        print(f"  num_bytes_memory={dset.get('num_bytes_memory')}")
        splits = size.get("splits") or []
        print(f"  n_splits={len(splits)}")
        if ds.endswith("IndicCorpV2") and dset.get("estimated_num_rows"):
            frac = dset["num_rows"] / dset["estimated_num_rows"]
            print(f"  converted fraction {frac*100:.1f}%")
            if dset.get("num_bytes_memory"):
                full = dset["num_bytes_memory"] / frac
                bpt = full / 20.9e9
                print(f"  extrapolated full {full/1e9:.0f}GB against published 20.9B tokens")
                print(f"  => {bpt:.1f} bytes/token (calibration, measured here)")


if __name__ == "__main__":
    main()
