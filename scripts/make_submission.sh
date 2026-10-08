#!/bin/bash
# Build the Gradescope zip: report PDF, code (git HEAD), modified XML, CleanRL diff, videos, final figures/tables,
# and the 10 trained policies. Run after editing report/report.typ.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain)" ]; then echo "commit your changes first (the code in the zip is git HEAD)" >&2; exit 1; fi
uvx --from typst python -c "import typst; typst.compile('report/report.typ', output='report/report.pdf', root='.')"
out=dist/submission; rm -rf dist; mkdir -p "$out"
git archive HEAD | tar -x -C "$out"                       # code, XML, diff, README, job lists
mv "$out/report/report.pdf" "$out/report.pdf"           # one copy, at the top level
rm -rf "$out/chtc/diag.sh" "$out/chtc/diag.sub" "$out/chtc/batches/smoke.txt" "$out/scripts/dev"  # debugging only
cp -R videos "$out/videos"
mkdir -p "$out/results"
cp -R results/final_plots "$out/results/final_plots"
cp -R results/final_eval "$out/results/final_eval"
for algo in sac ppo; do
    for f in results/final_$algo/runs/*/*.cleanrl_model; do
        mkdir -p "$out/$(dirname "$f")"; cp "$f" "$out/$f"
    done
done
(cd dist && zip -qr submission.zip submission)
ls -la dist/submission.zip
if grep -q "todo\[" report/report.typ; then echo "WARNING: report still contains TODO markers"; fi
