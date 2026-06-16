#!/usr/bin/env bash
# Smoke-test every paper figure script in the new repo.
set +e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

SCRIPTS=(
  fd_effect
  animal_vs_body
  fig3_semantic_territory_combined
  image_consensus
  dat_before_vs_after
  creativity_effects
  fig5cd_diversity_combined
  fig6_preferred_categories
  fig7_preferred_categories_fd
  eyetracking_contrasts
  image_features
  diversity_vs_post_dat
)

PASS=0; FAIL=0
declare -a FAILED

for s in "${SCRIPTS[@]}"; do
  if [ -f "analysis/figures/${s}.py" ]; then
    cmd="python -m analysis.figures.${s} --no-show"
  else
    echo "SKIP   $s  (not found)"
    continue
  fi

  echo "──────────────────────────────────────────────"
  echo "RUN    $s"
  printf "       "
  start=$(date +%s)
  out=$($cmd 2>&1)
  rc=$?
  dur=$(( $(date +%s) - start ))
  if [ $rc -eq 0 ]; then
    PASS=$((PASS+1))
    last=$(echo "$out" | tail -2 | head -1)
    echo "PASS  ($dur s)  $last"
  else
    FAIL=$((FAIL+1))
    FAILED+=("$s")
    echo "FAIL  ($dur s, exit $rc):"
    echo "$out" | tail -8
  fi
done

echo "──────────────────────────────────────────────"
echo "Summary: $PASS pass, $FAIL fail"
if [ $FAIL -gt 0 ]; then
  echo "FAILED:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
