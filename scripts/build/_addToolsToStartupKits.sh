#!/usr/bin/env bash
#
# Add the tools that NVFlare provisioning does not generate to the newest kits of a project:
#
#   client kits  <SITE>/scripts/client_node_setup/   host helpers (VPN service, Docker cgroup fix,
#                                                     GPU watchdog, health checks)
#   admin kits   <ADMIN>/startup/                     the consortium job builder and run watcher
#                                                     (scripts/admin/build_consortium_job.py,
#                                                     scripts/admin/watch_run.py) when present
#
# Runs after nvflare provision and _injectLiveSyncIntoStartupKits.sh, before the archives are
# made (see _buildStartupKits.sh).
#
# Why the client kits carry scripts/client_node_setup: the partner instructions and docker.sh's
# own pre-run checks name these scripts by that path (e.g. "run
# scripts/client_node_setup/fix_docker_cgroupfs.sh"), but sites only receive the kit, never the
# repository. RUMC reported in October 2026 that the 1.8.1 kit had no such folder. The folder
# keeps the repository's relative layout inside the kit, so every documented command works as
# written from the kit folder, and setup_vpntunnel.sh (which resolves the repository root two
# levels above itself) treats the kit folder as its root.

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: _addToolsToStartupKits.sh SWARM_PROJECT.yml"
  exit 1
fi

PROJECT_YML="$1"
# Tools come from the checkout this script belongs to; the kits from ./workspace, the
# provisioning output relative to the current directory, as in the other build steps.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_FOLDER="workspace/$(grep '^name: ' "$PROJECT_YML" | sed 's/name: //')"
TARGET_FOLDER="$(ls -d "$OUTPUT_FOLDER"/prod_* | tail -n 1)"

SETUP_SOURCE="$REPO_ROOT/scripts/client_node_setup"
# An allowlist of file types rather than the whole folder: the build runs in a working tree,
# and a stray file in it (a log, a site's .ovpn with VPN credentials) must never be copied
# into every site's kit. tests/unit_tests/test_kit_tools_and_manifest.py fails when a tracked
# file in the folder does not match, so a new kind of helper cannot be dropped silently.
SETUP_PATTERNS=("*.sh" "*.service" "*.timer" "*.md")

# Admin tools shipped by path when the checkout has them. Sibling modules they import from
# scripts/admin/ are shipped too, so a tool that splits out a helper keeps working.
ADMIN_TOOLS=("scripts/admin/build_consortium_job.py" "scripts/admin/watch_run.py")

_kit_kind() {
  if [ -f "$1/startup/fed_server.json" ]; then
    echo server
  elif [ -f "$1/startup/fl_admin.sh" ] || [ -f "$1/startup/fed_admin.json" ]; then
    echo admin
  elif [ -f "$1/startup/fed_client.json" ]; then
    echo client
  else
    echo unknown
  fi
}

_add_client_node_setup() {
  local kit="$1" dest pattern file copied=0
  dest="$kit/scripts/client_node_setup"
  mkdir -p "$dest"
  for pattern in "${SETUP_PATTERNS[@]}"; do
    for file in "$SETUP_SOURCE"/$pattern; do
      [ -f "$file" ] || continue
      cp "$file" "$dest/"
      case "$file" in
        *.sh) chmod 755 "$dest/$(basename "$file")" ;;
        *)    chmod 644 "$dest/$(basename "$file")" ;;
      esac
      copied=$((copied + 1))
    done
  done
  echo "Added scripts/client_node_setup ($copied files) to $kit"
}

# Print the scripts/admin/<module>.py files that a Python file refers to: in an import
# statement, or by its name in quotes. The second form matters: build_consortium_job.py loads
# its helpers by file path (_load_sibling("hocon_lite")), so they appear in no import line,
# and a kit without them fails at the builder's first line. Checking every sibling file
# against the tool, rather than parsing one loader's syntax, survives a renamed loader.
_sibling_modules() {
  local tool="$1" candidate module pattern
  for candidate in "$REPO_ROOT"/scripts/admin/*.py; do
    [ -f "$candidate" ] || continue
    [ "$candidate" -ef "$tool" ] && continue
    module="$(basename "$candidate" .py)"
    pattern="^[[:space:]]*(import|from)[[:space:]]+${module}([[:space:].,]|\$)"   # import line
    pattern+="|[\"']${module}(\\.py)?[\"']"                                     # name in quotes
    if grep -qE "$pattern" "$tool"; then
      echo "scripts/admin/$module.py"
    fi
  done
}

_add_admin_tools() {
  local kit="$1" tool queue=() shipped=" " found=0
  for tool in "${ADMIN_TOOLS[@]}"; do
    if [ -f "$REPO_ROOT/$tool" ]; then
      queue+=("$tool")
      found=$((found + 1))
    else
      echo "NOTE: $tool is not in this checkout; the admin kit $kit is built without it"
    fi
  done
  while [ "${#queue[@]}" -gt 0 ]; do
    tool="${queue[0]}"
    queue=("${queue[@]:1}")
    case "$shipped" in *" $tool "*) continue ;; esac
    shipped+="$tool "
    cp "$REPO_ROOT/$tool" "$kit/startup/"
    chmod 755 "$kit/startup/$(basename "$tool")"
    echo "Added $tool to $kit/startup"
    while read -r module; do
      [ -n "$module" ] && queue+=("$module")
    done < <(_sibling_modules "$REPO_ROOT/$tool")
  done
  if [ "$found" -eq 0 ]; then
    echo "NOTE: no consortium job builder in this checkout; $kit ships only the older helpers"
  fi
}

if [ ! -d "$SETUP_SOURCE" ]; then
  echo "Missing $SETUP_SOURCE"
  exit 1
fi

echo "Adding site and admin tools to the startup kits under $TARGET_FOLDER"
for kit in "$TARGET_FOLDER"/*/; do
  kit="${kit%/}"
  [ -d "$kit/startup" ] || continue
  case "$(_kit_kind "$kit")" in
    client) _add_client_node_setup "$kit" ;;
    admin)  _add_admin_tools "$kit" ;;
  esac
done
echo "Tools added"
