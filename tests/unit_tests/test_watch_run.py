"""scripts/admin/watch_run.py: per-round progress, slowest site and the #542 alarms.

Fixture: the lines the watcher reads, copied from the coordinator's log of the October
2026 seven-site MST run (job 7a678d9a; 20 rounds, FINISHED:COMPLETED). It holds only
timestamps, site names, round numbers and loopback-free server messages. The alarm cases
are cut from it, so they use the real message formats.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "watch_run" / "server_log_7a678d9a.txt"
JOB = "7a678d9a-5fec-473f-9b23-e43c41322f1f"
SITES = ["CAM_1", "MHA_1", "RSH_1", "RUMC_1", "UKA_1", "UMCU_1", "USZ_1"]


def _load():
    name = "_test_admin_watch_run"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "admin" / "watch_run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wr = _load()
LINES = FIXTURE.read_text().splitlines(keepends=True)


def run_watch(tmp_path, lines, *args, name="log.txt"):
    path = tmp_path / name
    path.write_text("".join(lines))
    code = wr.main([str(path), "--json", *args])
    return code


def report(capsys):
    return json.loads(capsys.readouterr().out)


def cut_after(*patterns):
    """Lines up to and including the last one that contains every pattern."""
    idx = max(i for i, line in enumerate(LINES) if all(p in line for p in patterns))
    return LINES[: idx + 1]


def test_fixture_is_the_october_run():
    assert len(LINES) > 250
    assert all(JOB in line for line in LINES if "run=" in line)


def test_finished_run(tmp_path, capsys):
    assert run_watch(tmp_path, LINES) == wr.EXIT_OK
    rep = report(capsys)
    assert rep["job_id"] == JOB and rep["status"] == "FINISHED"
    assert rep["sites"] == SITES and rep["starting_client"] == "CAM_1"
    assert rep["num_rounds"] == 20 and rep["rounds_complete"] == 20
    assert all(r["complete"] and r["finished"] == 7 for r in rep["rounds"])
    slow = rep["slowest_site"]
    # docs/ODELIA_RUN_2026-10_MST.md: UKA_1 gates every round (median ~41 min), UMCU_1 next (~31 min).
    assert slow["site"] == "UKA_1" and slow["rounds_last"] == 20
    assert 40 <= slow["median_train_min"]["UKA_1"] <= 43
    assert 30 <= slow["median_train_min"]["UMCU_1"] <= 32
    assert rep["finished"] == "2026-10-13T23:36:43"
    assert rep["all_rounds_done_at"] == "2026-10-13T23:32:53"


def test_a_missing_finish_report_belongs_to_the_aggregator_who_trained_last(tmp_path, capsys):
    """Rounds 1 and 11 have no finished_learn_task from UKA_1: it was the aggregator and
    its better_aggregation overwrote the finish before the next status report."""
    run_watch(tmp_path, LINES)
    rounds = {r["round"]: r for r in report(capsys)["rounds"]}
    for rnd in (1, 11):
        assert rounds[rnd]["slowest"] == "UKA_1"
        assert rounds[rnd]["train_min"]["UKA_1"] > max(
            v for site, v in rounds[rnd]["train_min"].items() if site != "UKA_1"
        )


def test_all_rounds_done_but_not_finished_is_its_own_alarm(tmp_path, capsys):
    lines = cut_after("UKA_1 on round 19: ")  # last round trained, nothing after it
    code = run_watch(tmp_path, lines, "--now", "2026-10-14 00:40:00", "--finish-grace", "60")
    rep = report(capsys)
    assert code == wr.EXIT_FINISH_OVERDUE == 3
    assert rep["status"] == "ALL ROUNDS DONE, NOT FINISHED" and rep["rounds_complete"] == 20
    # Within the grace period it is just finishing.
    assert run_watch(tmp_path, lines, "--now", "2026-10-13 23:50:00") == wr.EXIT_OK
    assert report(capsys)["status"] == "FINISHING"


def test_no_progress_is_its_own_alarm(tmp_path, capsys):
    lines = cut_after("on round 10: ", "start_learn_task")  # every site training round 10
    code = run_watch(tmp_path, lines, "--now", "2026-10-13 21:00:00", "--stall", "180")
    rep = report(capsys)
    assert code == wr.EXIT_NO_PROGRESS == 4
    assert rep["status"] == "NO PROGRESS" and rep["rounds_complete"] == 10
    current = rep["rounds"][-1]
    assert current["round"] == 10 and not current["complete"]
    assert set(current["training_min_so_far"]) == set(SITES)


def test_running_with_recent_progress(tmp_path, capsys):
    lines = cut_after("on round 10: ", "start_learn_task")
    assert run_watch(tmp_path, lines, "--now", "2026-10-13 16:40:00") == wr.EXIT_OK
    rep = report(capsys)
    assert rep["status"] == "RUNNING"
    assert rep["rounds"][-1]["round"] == 10


def test_now_cuts_the_log_for_post_mortems(tmp_path, capsys):
    """--now shows what a watcher would have said at that moment, ignoring later lines."""
    assert run_watch(tmp_path, LINES, "--now", "2026-10-13 16:40:00") == wr.EXIT_OK
    rep = report(capsys)
    assert rep["status"] == "RUNNING" and rep["finished"] is None and rep["rounds_complete"] == 10


def test_abort_before_the_last_round_completes(tmp_path, capsys):
    """Job 513c8988's shape: the server logs 'Workflow ... finished on all clients' on an
    abort too, which must not count as the last round completing."""
    lines = cut_after("RSH_1 on round 19: ")
    prefix = f"[identity=odelia_allsites, run={JOB}, wf=swarm_controller] - "
    lines += [
        f"2026-10-14 03:00:00,000 - ServerRunner - INFO - {prefix}asked to abort - triggered abort_signal to stop the RUN\n",
        f"2026-10-14 03:00:01,000 - FaultTolerantSwarmServerController - INFO - {prefix}"
        "Workflow swarm_controller finished on all clients\n",
        f"\x1b[31m2026-10-14 03:00:02,000 - ServerRunner - ERROR - {prefix}Aborting current RUN due to "
        "FATAL_SYSTEM_ERROR received: failed to end workflow swarm_controller: only 6/7 clients responded\x1b[0m\n",
        f"2026-10-14 03:00:03,000 - ServerRunner - INFO - {prefix}Server runner finished.\n",
    ]
    assert run_watch(tmp_path, lines) == wr.EXIT_ABORTED
    rep = report(capsys)
    assert rep["status"] == "ABORTED" and rep["rounds_complete"] == 19
    assert "only 6/7 clients responded" in rep["detail"]
    last = rep["rounds"][-1]
    assert last["round"] == 19 and not last["complete"]
    # Elapsed training of an ended run stops at the end of the run.
    assert max(last["training_min_so_far"].values()) < 60 * 5


def test_nohup_with_colour_codes_and_several_runs(tmp_path, capsys):
    other = "11111111-2222-3333-4444-555555555555"
    early = [ln.replace(JOB, other).replace("2026-10-13", "2026-10-01") for ln in LINES[:40]]
    early.append(
        f"2026-10-01 09:00:00,000 - ServerRunner - INFO - [identity=x, run={other}, wf=swarm_controller] - "
        "Server runner finished.\n"
    )
    noise = "2026-10-13 08:00:00,000 - Cell - INFO - unrelated chatter\nTraceback (most recent call last):\n"
    coloured = [f"\x1b[38m{ln.rstrip()}\x1b[0m\n" for ln in LINES]
    lines = early + [noise] + coloured
    assert run_watch(tmp_path, lines) == wr.EXIT_OK
    assert report(capsys)["job_id"] == JOB  # default: the last run in the log
    assert run_watch(tmp_path, lines, "--job-id", other[:8]) == wr.EXIT_ABORTED
    rep = report(capsys)
    assert rep["job_id"] == other and rep["status"] == "ENDED EARLY"


def test_unknown_run_and_bad_arguments(tmp_path, capsys):
    assert run_watch(tmp_path, LINES, "--job-id", "deadbeef") == wr.EXIT_USAGE
    assert wr.main([str(tmp_path / "missing.txt")]) == wr.EXIT_USAGE
    assert wr.main([str(FIXTURE), "--now", "yesterday"]) == wr.EXIT_USAGE
    capsys.readouterr()


def test_text_report_names_the_slowest_site(tmp_path, capsys):
    path = tmp_path / "log.txt"
    path.write_text("".join(LINES))
    assert wr.main([str(path)]) == 0
    text = capsys.readouterr().out
    assert "slowest site: UKA_1 finished last in 20 of 20 rounds" in text
    assert "STATUS FINISHED (exit 0)" in text


def test_slow_sites_get_a_stall_hint(tmp_path, capsys):
    run_watch(tmp_path, LINES, "--stall", "30")
    assert "UKA_1 trains" in report(capsys)["slowest_site"]["note"]
