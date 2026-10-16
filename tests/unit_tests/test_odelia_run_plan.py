"""scripts/admin/odelia_run_plan.py: the Q4 2026 consortium runs as named presets.

Every preset is built for the real sites with build_consortium_job.py from this checkout's job
code (``--source repo``; the presets themselves build from the release image), and the written
job is read back. The expected values are written out here, independently of the preset table,
so a change to a preset has to be made twice: once in the table, once in this review copy.

The warm start is checked by value only: the sha256 and the state digest of the August model are
constants, so the test does not need the 722 MB checkpoint.
"""

import importlib.util
import json
import shlex
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ADMIN_DIR = REPO_ROOT / "scripts" / "admin"


def _load(name):
    module_name = f"_test_admin_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, ADMIN_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


rp = _load("odelia_run_plan")
bcj = rp.bcj
hocon = bcj.hocon

SITES8 = ["CAM_1", "MHA_1", "RSH_1", "RUMC_1", "UKA_1", "UMCU_1", "USZ_1", "VHIO_1"]
SITES7 = [s for s in SITES8 if s != "VHIO_1"]
ODELIA = "ODELIA_ternary_classification"
CHALLENGE = "challenge_1DivideAndConquer"

# The August model (job 87c5bbee): sha256 of FL_global_model.pt (release SHA256SUMS) and its
# state digest (state_digest.checkpoint_state_digest, computed in jefftud/odelia:1.9.1).
AUGUST_PATH = "/scratch/odelia_87c5bbee_global.pt"
AUGUST_SHA256 = "ad26080c71161e6f8abb6a279c06a41d459b7bb237699cbe2d4522ddb942270d"
AUGUST_DIGEST = "cdbfab5748a025f3a88216747ac57208d764a93f9e416e90d9c1063f1ea6b51e"


def _d32(strategy, budget):
    return dict(
        base=CHALLENGE, model="1DivideAndConquer", config="unilateral", rounds=5, sites=SITES8, start="UMCU_1",
        warm=True, hooks=["al_subset"],
        env={"ODELIA_AL_STRATEGY": strategy, "ODELIA_AL_BUDGET": str(budget), "ODELIA_AL_POOL_SPLIT": "train",
             "ODELIA_AL_SEED": "0", "ODELIA_AL_SELECT_ROUND": "0"},
    )


# What each run of the schedule must be (docs/ODELIA_RUN_PLAN_2026-Q4.md).
EXPECTED = {
    "check": dict(base=ODELIA, model="MST", config="unilateral", rounds=1, sites=SITES8, start=None, warm=False,
                  hooks=[], env={}),
    "cam": dict(base=ODELIA, model="MST_SAMMed2D", config="unilateral", rounds=20, sites=SITES8, start=None,
                warm=False, hooks=["ms6_capture"], archive=True,
                env={"ODELIA_EXPORT_TRAIN_PREDICTIONS": "0", "ODELIA_MS6_CAPTURE_SITES": "CAM_1,RUMC_1",
                     "ODELIA_MS6_CAPTURE_ROUNDS": "all"}),
    "vhio": dict(base=ODELIA, model="MedicalNet", config="unilateral", rounds=20, sites=SITES8, start=None,
                 warm=False, hooks=[], env={}),
    "d25_dutch": dict(base=CHALLENGE, model="1DivideAndConquer", config="unilateral", rounds=5,
                      sites=["RUMC_1", "UMCU_1"], start="UMCU_1", warm=True, hooks=[], env={}),
    "d32_entropy_b20": _d32("entropy", 20),
    "d32_entropy_b40": _d32("entropy", 40),
    "d32_random_b20": _d32("random", 20),
    "d32_random_b40": _d32("random", 40),
    "dp_s24": dict(base=ODELIA, model="MST", config="unilateral", rounds=20, sites=SITES8, start=None, warm=False,
                   hooks=["dp_noise"], per_case="off",
                   env={"ODELIA_DP_SIGMA": "2.4", "ODELIA_DP_CLIP_NORM": "1.0", "ODELIA_DP_DELTA": "1e-05",
                        "ODELIA_DP_TARGET_EPSILON": "10.0", "ODELIA_RETURN_PER_CASE": "0"}),
    "bd": dict(base=CHALLENGE, model="1DivideAndConquer", config="unilateral_BD", rounds=20, sites=SITES8,
               start=None, warm=False, hooks=[], env={"ODELIA_STRICT_DATA_FORMAT": "1"}),
}
BUILDABLE = list(EXPECTED)
WITH_VHIO = [key for key, want in EXPECTED.items() if "VHIO_1" in want["sites"]]
FEATURE_PREFIXES = ("ODELIA_AL_", "ODELIA_MS6_", "ODELIA_DP_", "ODELIA_EXPORT_", "ODELIA_RETURN_PER_CASE",
                    "ODELIA_STRICT_DATA_FORMAT", "ODELIA_TRAIN_INCLUDE_LIST")


def build(tmp_path, *argv, expect=0):
    """Build a preset from this checkout into tmp_path/out and return the job directory."""
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    assert rp.main([*argv, "--source", "repo", "--out-dir", str(out)]) == expect
    jobs = sorted(p for p in out.iterdir() if not p.name.startswith("."))
    assert len(jobs) == (1 if expect == 0 else 0)
    return jobs[0] if jobs else None


def read_job(job):
    manifest = json.loads((job / "job_manifest.json").read_text())
    meta = hocon.loads((job / "meta.conf").read_text())
    server = hocon.loads((job / "app/config/config_fed_server.conf").read_text())
    client = hocon.loads((job / "app/config/config_fed_client.conf").read_text())
    return manifest, meta, server, client


def component(conf, comp_id):
    return [c for c in conf["components"] if c.get("id") == comp_id][0]


def swarm_args(server):
    return [w for w in server["workflows"] if w["id"] == "swarm_controller"][0]["args"]


def launcher_env(client):
    tokens, _ = bcj.split_launcher(component(client, "launcher")["args"]["script"])
    return dict(tokens)


def check_job(job, want, sites):
    manifest, meta, server, client = read_job(job)
    # The builder exits 0 only after its validation passed; the manifest records it.
    assert manifest["code_checks"] == "passed"
    passed = manifest["validation"]["passed"]
    assert "configure_min_clients (F8)" in passed and "MODEL_NAME pins agree" in passed

    assert job.name == manifest["job_name"] == meta["name"]
    assert job.name.endswith(f"_{len(sites)}site")
    assert manifest["base_job"] == want["base"]
    assert manifest["model"] == want["model"] and manifest["config"] == want["config"]
    assert manifest["training_rounds"] == want["rounds"] and manifest["num_rounds"] == want["rounds"] + 1
    assert manifest["final_eval_round"] is True
    assert manifest["sites"] == sites and manifest["starting_client"] == want["start"]
    assert manifest["round_hooks"] == want["hooks"]
    assert manifest["ms6_archive_globals"] is want.get("archive", False)
    assert manifest["per_case_return"] == want.get("per_case", "site")
    assert manifest["source"]["kind"] == "repo"

    # The written configs, read back.
    assert meta["deploy_map"]["app"] == ["server", *sites]
    assert meta["min_clients"] == len(sites) and meta["mandatory_clients"] == sites
    args = swarm_args(server)
    assert args["participating_clients"] == sites
    assert args["min_clients"] == args["configure_min_clients"] == len(sites)
    assert args["num_rounds"] == want["rounds"] + 1
    assert args.get("starting_client") == want["start"]
    env = launcher_env(client)
    assert env["MODEL_NAME"] == want["model"] and env["CONFIG"] == want["config"]
    assert env["ODELIA_FINAL_EVAL_ROUND"] == "1"
    assert env.get("ODELIA_ROUND_HOOKS", "") == ",".join(want["hooks"])
    features = {k: v for k, v in env.items() if k.startswith(FEATURE_PREFIXES)}
    assert features == want["env"]
    persistor = component(client, "persistor")["args"]
    assert persistor["model_name"] == want["model"] and persistor["data_config"] == want["config"]
    assert component(client, "ms6_global_streamer")["args"]["enabled"] is want.get("archive", False)
    assert component(server, "ms6_global_archive")["args"]["enabled"] is want.get("archive", False)

    warm = manifest["warm_start"]
    if want["warm"]:
        assert warm["mode"] == "source" and warm["warm_start_mode"] == "auto"
        assert warm["source_ckpt_file_full_name"] == AUGUST_PATH == persistor["source_ckpt_file_full_name"]
        assert warm["source_sha256"] == AUGUST_SHA256 == persistor["source_sha256"]
        assert warm["state_digest"] == AUGUST_DIGEST == env["ODELIA_WARM_START_SHA256"]
        # Each experiment continues nothing and writes its own mirror, never over the start model.
        assert persistor["latest_global_path"] == f"/scratch/mediswarm_latest_global_{job.name}.pt"
    else:
        assert warm["mode"] == "fresh" and persistor["warm_start_mode"] == "fresh"
        assert "ODELIA_WARM_START_SHA256" not in env and "source_sha256" not in persistor
    return manifest


# ── Every preset, for the real sites ──────────────────────────────────────


def test_the_table_has_exactly_the_scheduled_presets():
    assert list(rp.PRESETS) == [
        "check", "cam", "vhio", "d25_dutch", "d25_greek",
        "d32_entropy_b20", "d32_entropy_b40", "d32_random_b20", "d32_random_b40", "dp_s24", "bd",
    ]
    assert set(rp.PRESETS) == set(BUILDABLE) | {"d25_greek"}


def test_constants_of_the_schedule():
    assert list(rp.SITES) == SITES8
    assert rp.DEFAULT_SOURCE == "image:jefftud/odelia:1.9.1"
    assert rp.AUGUST_MODEL_PATH == AUGUST_PATH
    assert rp.AUGUST_MODEL_SHA256 == AUGUST_SHA256
    assert rp.AUGUST_MODEL_STATE_DIGEST == AUGUST_DIGEST


@pytest.mark.parametrize("key", BUILDABLE)
def test_every_preset_builds_for_the_real_sites(tmp_path, key):
    want = EXPECTED[key]
    job = build(tmp_path, "--build", key)
    manifest = check_job(job, want, want["sites"])
    if key == "dp_s24":
        assert manifest["dp"]["epsilon_after_training_rounds"] <= 10.0
        assert manifest["dp"]["max_training_rounds"] >= 20 and manifest["dp"]["delta"] == 1e-5


@pytest.mark.parametrize("key", WITH_VHIO)
def test_every_preset_builds_without_vhio(tmp_path, key):
    want = EXPECTED[key]
    job = build(tmp_path, "--build", key, "--exclude-sites", "VHIO_1")
    manifest = check_job(job, want, SITES7)
    assert "no VHIO_1 unless listed" in manifest["validation"]["passed"]
    assert "VHIO_1" not in (job / "meta.conf").read_text()


def test_greek_cohort_alone_is_refused_with_the_reason(tmp_path, capsys):
    # MHA_1 alone cannot be a swarm job: NVFlare's swarm server controller refuses a single
    # participant at deploy time, so the builder refuses it before anything is written.
    build(tmp_path, "--build", "d25_greek", expect=2)
    err = capsys.readouterr().err
    assert "at least 2 sites" in err and "Not enough participating_clients" in err


# ── --show prints the command --build runs ────────────────────────────────


@pytest.mark.parametrize("key", list(rp.PRESETS))
def test_show_prints_the_exact_builder_command(capsys, key):
    assert rp.main(["--show", key, "--exclude-sites", "VHIO_1", "--out-dir", "/kit/local/mediswarm_jobs"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith(f"# {key}: ")
    assert any(line.startswith("# before submitting: every site on the 1.9.0 kit") for line in out)
    words = shlex.split(out[-1])
    assert words[:2] == ["python3", "scripts/admin/build_consortium_job.py"]
    request = rp.builder_request(key, exclude_sites="VHIO_1")
    assert words[2:] == [*request.argv, "--out-dir", "/kit/local/mediswarm_jobs"]
    assert "--source" in words and words[words.index("--source") + 1] == "image:jefftud/odelia:1.9.1"
    # The printed arguments are the builder's own flags.
    args = bcj.build_arg_parser().parse_args(words[2:])
    assert args.rounds == rp.PRESETS[key].rounds and args.out_dir == Path("/kit/local/mediswarm_jobs")


def test_the_plan_doc_shows_the_command_of_every_preset():
    # The doc wraps commands with backslash-newlines; compare with whitespace collapsed.
    doc = (REPO_ROOT / "docs" / "ODELIA_RUN_PLAN_2026-Q4.md").read_text()
    flat = " ".join(doc.replace("\\\n", " ").split())
    for key in rp.PRESETS:
        assert rp.command_line(rp.builder_request(key)) in flat, key
    for value in (AUGUST_SHA256, AUGUST_DIGEST, AUGUST_PATH):
        assert value in doc


def test_list_names_every_preset(capsys):
    assert rp.main(["--list"]) == 0
    out = capsys.readouterr().out
    for key in rp.PRESETS:
        assert f"\n{key} " in "\n" + out
    assert "NOT A SWARM JOB" in out and "d25_local_finetune.sh" in out


# ── Overrides: the dl3 test bed and a site that cannot join ───────────────


def test_test_bed_build_of_an_al_arm(tmp_path):
    job = build(tmp_path, "--build", "d32_entropy_b20", "--sites", "TEST_A_1,TEST_B_1",
                "--starting-client", "TEST_A_1")
    want = dict(EXPECTED["d32_entropy_b20"], sites=["TEST_A_1", "TEST_B_1"], start="TEST_A_1")
    check_job(job, want, want["sites"])
    assert job.name == "odelia_d32_entropy_b20_5r_2site"


def test_test_bed_build_of_the_cam_run(tmp_path):
    job = build(tmp_path, "--build", "cam", "--sites", "TEST_A_1,TEST_B_1", "--ms6-capture-sites", "TEST_B_1")
    want = dict(EXPECTED["cam"], sites=["TEST_A_1", "TEST_B_1"])
    want["env"] = dict(want["env"], ODELIA_MS6_CAPTURE_SITES="TEST_B_1")
    check_job(job, want, want["sites"])


def test_name_override(tmp_path):
    job = build(tmp_path, "--build", "check", "--exclude-sites", "VHIO_1", "--name", "odelia_check_retry")
    assert job.name == "odelia_check_retry"


def test_excluding_a_site_the_preset_does_not_have_is_a_note(tmp_path, capsys):
    job = build(tmp_path, "--build", "d25_dutch", "--exclude-sites", "VHIO_1")
    assert "VHIO_1 not in the roster of d25_dutch" in capsys.readouterr().err
    check_job(job, EXPECTED["d25_dutch"], ["RUMC_1", "UMCU_1"])


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--build", "d32_random_b40", "--sites", "TEST_A_1,TEST_B_1"], "starting client UMCU_1 is not in the roster"),
        (["--build", "d25_dutch", "--exclude-sites", "UMCU_1"], "starting client UMCU_1 is not in the roster"),
        (["--build", "cam", "--sites", "TEST_A_1,TEST_B_1"], "MS6 capture site(s) CAM_1, RUMC_1 not in the roster"),
        (["--build", "cam", "--exclude-sites", "RUMC_1"], "MS6 capture site(s) RUMC_1 not in the roster"),
        (["--build", "vhio", "--ms6-capture-sites", "CAM_1"], "captures no MS6 models"),
        (["--build", "d25_dutch", "--exclude-sites", "RUMC_1,UMCU_1"], "removed every site"),
        (["--build", "nope"], "unknown preset"),
        # Refused by the builder, not here: an override that names a site outside the roster.
        (["--build", "d32_entropy_b20", "--starting-client", "TEST_A_1"], "not in --sites"),
        (["--build", "cam", "--ms6-capture-sites", "TEST_A_1"], "not in --sites"),
        (["--build", "d25_dutch", "--exclude-sites", "RUMC_1"], "at least 2 sites"),
    ],
)
def test_overrides_that_do_not_fit_are_refused(tmp_path, capsys, argv, message):
    build(tmp_path, *argv, expect=2)
    assert message in capsys.readouterr().err
