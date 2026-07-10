#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: transfer_bundle.sh [--execute] SOURCE_REPO DESTINATION

Transfer the migration allowlist while preserving paths relative to SOURCE_REPO.
DESTINATION may be a local directory or an rsync remote such as user@host:/path.

The default is --dry-run. Data is written only when --execute is supplied.
Authentication is delegated to the caller's SSH agent/configuration.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

execute=false
declare -a positional=()

while (($# > 0)); do
  case "$1" in
    --execute)
      execute=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while (($# > 0)); do
        positional+=("$1")
        shift
      done
      break
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      positional+=("$1")
      ;;
  esac
  shift
done

((${#positional[@]} == 2)) || {
  usage >&2
  exit 2
}

command -v rsync >/dev/null 2>&1 || die "rsync is required"
command -v realpath >/dev/null 2>&1 || die "realpath is required"
command -v python3 >/dev/null 2>&1 || die "python3 is required"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"

source_repo=$(realpath "${positional[0]}")
destination=${positional[1]}

[[ -d "$source_repo/.git" ]] || die "SOURCE_REPO is not a git worktree root: $source_repo"
[[ -n "$destination" ]] || die "DESTINATION must not be empty"
[[ "$destination" != -* ]] || die "DESTINATION must not begin with '-'"

declare -ar relative_paths=(
  "checkpoints/migration_hf_eval"
  ".cache/migration/tier1_assets"
  ".cache/migration/rollout_selection"
)

for relative_path in "${relative_paths[@]}"; do
  case "$relative_path" in
    models|models/*|.cache/rollout_dump|.cache/rollout_dump/*)
      die "internal allowlist violation: $relative_path"
      ;;
  esac
  [[ -d "$source_repo/$relative_path" ]] || die "required bundle path is missing: $source_repo/$relative_path"
done

require_file() {
  [[ -f "$1" ]] || die "required file is missing: $1"
}

reject_partial_files() {
  local root=$1
  local partial
  partial=$(find "$root" -type f -name '*.part' -print -quit)
  [[ -z "$partial" ]] || die "incomplete .part file prevents transfer: $partial"
}

verify_hf_bundle() {
  local root=$1
  require_file "$root/MANIFEST.csv"
  require_file "$root/MANIFEST.json"
  require_file "$root/SHA256SUMS"
  reject_partial_files "$root"

  if ! (cd "$root" && sha256sum -c SHA256SUMS); then
    die "HF checksum verification failed: $root/SHA256SUMS"
  fi
}

verify_tier1_bundle() {
  local root=$1
  local archive
  local sidecar
  local sidecar_count
  local -a archives=()
  local -a sidecars=()

  require_file "$root/asset_manifest.json"
  require_file "$root/asset_manifest.json.sha256"
  reject_partial_files "$root"

  while IFS= read -r -d '' archive; do
    archives+=("$archive")
  done < <(find "$root" -maxdepth 1 -type f -name '*.tar.zst' -print0)
  ((${#archives[@]} > 0)) || die "no Tier-1 archives found under: $root"

  for archive in "${archives[@]}"; do
    sidecar="${archive}.sha256"
    require_file "$sidecar"
    sidecars+=("$sidecar")
  done
  sidecar_count=$(find "$root" -maxdepth 1 -type f -name '*.tar.zst.sha256' -printf '.\n' | wc -l)
  [[ "$sidecar_count" -eq "${#archives[@]}" ]] || die "orphan or missing Tier-1 archive sidecar under: $root"

  if ! (
    cd "$root"
    sha256sum -c asset_manifest.json.sha256
    sha256sum -c "${sidecars[@]}"
  ); then
    die "Tier-1 checksum verification failed under: $root"
  fi
}

verify_rollout_bundle() {
  local root=$1
  local partial
  local required

  reject_partial_files "$root"
  partial=$(find "$root" -type f -name 'census_*.partial.parquet' -print -quit)
  [[ -z "$partial" ]] || die "incomplete rollout census prevents transfer: $partial"

  for required in \
    selection_manifest.csv \
    selection_manifest.parquet \
    selection_summary.json \
    SHA256SUMS \
    rollout_selection_control.tar.zst \
    rollout_selection_control.tar.zst.sha256; do
    require_file "$root/$required"
  done

  if ! python3 - "$root" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path

selection_dir = Path(sys.argv[1])
manifest_path = selection_dir / "selection_manifest.csv"
expected = {"M5": 8, "nocispo": 8, "RLonly": 8}
archive_owners = {}

with manifest_path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    required = {"run", "archive_name"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise SystemExit(f"selection manifest missing columns: {sorted(missing)}")
    for row_number, row in enumerate(reader, start=2):
        run = (row.get("run") or "").strip()
        archive_name = (row.get("archive_name") or "").strip()
        if run not in expected:
            raise SystemExit(f"unexpected run at CSV row {row_number}: {run!r}")
        if not archive_name:
            raise SystemExit(f"empty archive_name at CSV row {row_number}")
        archive_path = Path(archive_name)
        if archive_path.is_absolute() or archive_path.name != archive_name:
            raise SystemExit(f"archive_name must be a basename: {archive_name!r}")
        previous = archive_owners.setdefault(archive_name, run)
        if previous != run:
            raise SystemExit(f"archive assigned to multiple runs: {archive_name!r}")

if len(archive_owners) != 24:
    raise SystemExit(f"unique archive_name count={len(archive_owners)}, expected=24")
actual = Counter(archive_owners.values())
if dict(actual) != expected:
    raise SystemExit(f"archive counts by run={dict(actual)}, expected={expected}")
missing_archives = sorted(name for name in archive_owners if not (selection_dir / name).is_file())
if missing_archives:
    raise SystemExit(f"manifest archives missing on disk: {missing_archives}")
print(f"rollout manifest gate passed: archives={len(archive_owners)} by_run={dict(actual)}")
PY
  then
    die "rollout manifest readiness verification failed under: $root"
  fi

  if ! (
    cd "$root"
    sha256sum -c SHA256SUMS
    sha256sum -c rollout_selection_control.tar.zst.sha256
  ); then
    die "rollout checksum verification failed under: $root"
  fi
}

verify_hf_bundle "$source_repo/checkpoints/migration_hf_eval"
verify_tier1_bundle "$source_repo/.cache/migration/tier1_assets"
verify_rollout_bundle "$source_repo/.cache/migration/rollout_selection"

# A trailing slash makes DESTINATION the common root below which --relative
# recreates checkpoints/... and .cache/.... It is safe for both '/' and host:/.
destination=${destination%/}/

declare -a rsync_args=(
  --archive
  --relative
  --human-readable
  --itemize-changes
  --partial
  --partial-dir=.rsync-partial
  --info=progress2,stats2
  --protect-args
)

if [[ "$execute" == false ]]; then
  rsync_args+=(--dry-run)
  printf 'Mode: DRY RUN (pass --execute to transfer)\n'
else
  printf 'Mode: EXECUTE\n'
fi

printf 'Source: %s\nDestination: %s\n' "$source_repo" "$destination"
printf 'Allowlist:\n'
printf '  %s\n' "${relative_paths[@]}"

(
  cd "$source_repo"
  rsync "${rsync_args[@]}" -- "${relative_paths[@]}" "$destination"
)
