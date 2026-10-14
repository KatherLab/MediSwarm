"""scripts/admin/build_consortium_job.py: one command that stages a correct consortium job.

Most tests build from a small source tree in tmp_path: the job templates of THIS checkout
(so a template change that breaks the builder's anchors fails here) plus stub job code
that declares the 1.9.0 features. The stubs only answer the builder's "does the job's
code support this?" probes; the configs written are the real thing and are parsed back.

One regression test builds from frozen copies of the October 2026 templates and compares
with the job that was staged by hand for run 7a678d9a (odelia_mst_20r_7site).
"""

import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ADMIN_DIR = REPO_ROOT / "scripts" / "admin"
JOBS_DIR = REPO_ROOT / "application" / "jobs"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "build_consortium_job"


def _load(name):
    module_name = f"_test_admin_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, ADMIN_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


bcj = _load("build_consortium_job")
hocon = bcj.hocon

SHA_A = "a" * 64
SHA_B = "b" * 64

STUB_PERSISTOR = '''
class WarmStartablePTFileModelPersistor:
    def __init__(self, latest_global_path="", warm_start_mode="auto", model_name=None, data_config=None,
                 source_sha256=None, **kwargs):
        pass
'''
OLD_PERSISTOR = '''
class WarmStartablePTFileModelPersistor:
    def __init__(self, latest_global_path="", warm_start_mode="auto", **kwargs):
        pass
'''
STUB_MAIN = "from round_hooks import load_hooks\n"
STUB_TRAINER = "\n".join(
    f'os.environ.get("{key}")'
    for key in (
        "ODELIA_FINAL_EVAL_ROUND",
        "ODELIA_EXPORT_TRAIN_PREDICTIONS",
        "ODELIA_RETURN_PER_CASE",
        "ODELIA_SEED",
        "ODELIA_WARM_START_SHA256",
        "ODELIA_ROUND_HOOKS",
    )
)
# Shaped like models_config.py since #636: the builder reads these constants statically.
STUB_MODELS = """
CHALLENGE_MODELS = {"1DivideAndConquer": {"team_name": "1DivideAndConquer"}, "2BCN_AIM": {}}
RESNET_MODEL_NAMES = ("ResNet18", "ResNet101")
BUILTIN_MODEL_NAMES = ("MST", "MST_SAMMed2D", "MedicalNet") + RESNET_MODEL_NAMES
CHALLENGE_PREFIX = "challenge_"


def get_supported_model_names():
    return list(BUILTIN_MODEL_NAMES) + [CHALLENGE_PREFIX + name for name in CHALLENGE_MODELS]
"""
# models_config.py before #636: no list to read, only quoted names.
OLD_MODELS = "NAMES = ['MST', 'MST_SAMMed2D', '1DivideAndConquer']\n"
STUB_DATA = "DATA = {'original': 'data', 'unilateral': 'data_unilateral', 'unilateral_BD': 'data_unilateral_BD'}\n"

CLIENT_STREAMER = """
    {
      id = "ms6_global_streamer"
      path = "ms6_archive.GlobalModelStreamer"
      args {
        enabled = false
      }
    }"""
SERVER_ARCHIVE = """
  {
    id = "ms6_global_archive"
    path = "ms6_archive.GlobalModelArchive"
    args {
      enabled = false
      archive_root = "/archive"
    }
  }"""


def make_tree(
    tmp_path,
    persistor=STUB_PERSISTOR,
    models=STUB_MODELS,
    hooks=("round_hooks", "al_subset", "ms6_capture", "dp_noise"),
    ms6_components=True,
    data=STUB_DATA,
    templates=None,
    privacy_accounting=None,
):
    """A MediSwarm-shaped source tree with this checkout's templates and stub job code."""
    root = tmp_path / "src"
    jobs = root / "application" / "jobs"
    custom = jobs / "_shared" / "custom"
    (custom / "models").mkdir(parents=True)
    (custom / "data" / "datasets").mkdir(parents=True)
    (custom / "warm_continue.py").write_text(persistor)
    (custom / "main.py").write_text(STUB_MAIN)
    (custom / "threedcnn_ptl.py").write_text(STUB_TRAINER)
    (custom / "models" / "models_config.py").write_text(models)
    (custom / "data" / "datasets" / "dataset_3d_odelia.py").write_text(data)
    (custom / "env_config.py").write_text("CONFIG = 'unilateral'\n")
    for hook in hooks:
        (custom / f"{hook}.py").write_text("def create_hook(env, logger):\n    return None\n")
    shutil.copy(JOBS_DIR / "_shared" / "custom" / "privacy_accounting.py", custom / "privacy_accounting.py")
    if privacy_accounting is not None:
        (custom / "privacy_accounting.py").write_text(privacy_accounting)
    os.makedirs(custom / "__pycache__")
    (custom / "__pycache__" / "junk.pyc").write_bytes(b"\0")
    for base in ("ODELIA_ternary_classification", "challenge_1DivideAndConquer"):
        job = jobs / base
        (job / "app" / "config").mkdir(parents=True)
        src_dir = templates or {}
        meta = Path(src_dir.get("meta", JOBS_DIR / base / "meta.conf"))
        client = Path(src_dir.get("client", JOBS_DIR / base / "app" / "config" / "config_fed_client.conf"))
        server = Path(src_dir.get("server", JOBS_DIR / base / "app" / "config" / "config_fed_server.conf"))
        shutil.copy(meta, job / "meta.conf")
        client_text = client.read_text()
        server_text = server.read_text()
        if ms6_components and "ms6_global_streamer" not in client_text:
            client_text = client_text.replace("components = [", "components = [" + CLIENT_STREAMER, 1)
        if ms6_components and "ms6_global_archive" not in server_text:
            server_text = server_text.replace("components = [", "components = [" + SERVER_ARCHIVE, 1)
        (job / "app" / "config" / "config_fed_client.conf").write_text(client_text)
        (job / "app" / "config" / "config_fed_server.conf").write_text(server_text)
        (job / "app" / "custom").symlink_to("../../_shared/custom")
        for sub in ("tests", "scripts"):
            (job / "app" / sub).mkdir()
            (job / "app" / sub / "x.py").write_text("")
    (root / "odelia_image.version").write_text("# version of the image\n1.9.0\n")
    return root


def build(tmp_path, *extra, tree=None, expect=0, capsys=None):
    tree = tree or make_tree(tmp_path)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    argv = ["--source", f"dir:{tree}", "--out-dir", str(out), *extra]
    code = bcj.main(argv)
    assert code == expect
    name = extra[extra.index("--name") + 1]
    return out / name


def confs(job):
    return (
        hocon.loads((job / "meta.conf").read_text()),
        hocon.loads((job / "app/config/config_fed_server.conf").read_text()),
        hocon.loads((job / "app/config/config_fed_client.conf").read_text()),
    )


def component(conf, comp_id):
    return [c for c in conf["components"] if c.get("id") == comp_id][0]


def swarm_args(server):
    return [w for w in server["workflows"] if w["id"] == "swarm_controller"][0]["args"]


def launcher_env(client):
    tokens, _ = bcj.split_launcher(component(client, "launcher")["args"]["script"])
    return dict(tokens)


def plan(*argv):
    return bcj.make_plan(bcj.build_arg_parser().parse_args(list(argv)))


# ── The October recipe, reproduced ────────────────────────────────────────


def test_october_recipe_reproduces_the_hand_built_job(tmp_path):
    october = FIXTURES / "october"
    tree = make_tree(
        tmp_path,
        templates={
            "meta": october / "template" / "meta.conf",
            "client": october / "template" / "config_fed_client.conf",
            "server": october / "template" / "config_fed_server.conf",
        },
        ms6_components=False,
    )
    job = build(
        tmp_path,
        "--name", "odelia_mst_20r_7site", "--model", "MST", "--rounds", "20", "--no-final-eval",
        "--mirror-path", "/scratch/mediswarm_latest_global.pt", "--no-code-checks",
        tree=tree,
    )
    meta, server, client = confs(job)
    want_meta = hocon.loads((october / "handbuilt" / "meta.conf").read_text())
    want_server = hocon.loads((october / "handbuilt" / "config_fed_server.conf").read_text())
    want_client = hocon.loads((october / "handbuilt" / "config_fed_client.conf").read_text())
    assert meta == want_meta
    assert server == want_server

    # 1.9.0 adds exactly these on top of the October job: CONFIG and the final-eval
    # switch on the launcher, and the persistor's provenance pins (contract 3.6).
    env = launcher_env(client)
    assert env.pop("CONFIG") == "unilateral"
    assert env.pop("ODELIA_FINAL_EVAL_ROUND") == "0"
    assert env == launcher_env(want_client)
    persistor = component(client, "persistor")["args"]
    assert persistor.pop("model_name") == "MST"
    assert persistor.pop("data_config") == "unilateral"
    for comp in client["components"]:
        if comp["id"] == "launcher":
            comp["args"].pop("script")
    for comp in want_client["components"]:
        if comp["id"] == "launcher":
            comp["args"].pop("script")
    assert client == want_client


# ── Default build ─────────────────────────────────────────────────────────


def test_default_build_is_strict_pinned_and_recorded(tmp_path):
    job = build(tmp_path, "--name", "mst20", "--model", "MST", "--rounds", "20")
    meta, server, client = confs(job)
    sites = list(bcj.DEFAULT_SITES)
    assert "VHIO_1" not in sites
    assert meta["name"] == "mst20"
    assert meta["deploy_map"]["app"] == ["server", *sites]
    assert meta["min_clients"] == 7 and meta["mandatory_clients"] == sites
    args = swarm_args(server)
    assert args["num_rounds"] == 21  # 20 training rounds + the evaluation-only round
    assert args["min_clients"] == args["configure_min_clients"] == 7
    assert args["participating_clients"] == sites
    assert "starting_client" not in args
    assert bcj._swarm_client_args(client)["min_responses_required"] == 7

    env = launcher_env(client)
    assert env["MODEL_NAME"] == "MST" and env["CONFIG"] == "unilateral"
    assert env["ODELIA_FINAL_EVAL_ROUND"] == "1"
    assert env["KEY_METRIC"] == "val/AUC_ROC"  # template token kept
    assert "ODELIA_ROUND_HOOKS" not in env and "ODELIA_RETURN_PER_CASE" not in env
    persistor = component(client, "persistor")["args"]
    assert persistor["model_name"] == "MST" and persistor["data_config"] == "unilateral"
    assert persistor["model"]["args"]["model_name"] == "MST"
    assert persistor["warm_start_mode"] == "fresh"
    assert persistor["latest_global_path"] == "/scratch/mediswarm_latest_global_mst20.pt"
    assert component(client, "ms6_global_streamer")["args"]["enabled"] is False
    assert component(server, "ms6_global_archive")["args"]["enabled"] is False

    assert not (job / "app" / "tests").exists() and not (job / "app" / "scripts").exists()
    custom = job / "app" / "custom"
    assert custom.is_dir() and not custom.is_symlink()
    assert not list(custom.rglob("__pycache__"))

    manifest = json.loads((job / "job_manifest.json").read_text())
    assert manifest["schema"] == "mediswarm.job_manifest/1"
    assert manifest["expected_image"] == "jefftud/odelia:1.9.0"
    assert manifest["num_rounds"] == 21 and manifest["training_rounds"] == 20
    assert manifest["flags"]["rounds"] == 20 and "--rounds" in manifest["argv"]
    assert manifest["source"]["kind"] == "dir"
    assert manifest["code_checks"] == "passed"
    for rel, sha in manifest["custom_files_sha256"].items():
        assert bcj._sha256_file(job / rel) == sha
    assert set(manifest["custom_files_sha256"]) == {
        str(p.relative_to(job)) for p in custom.rglob("*") if p.is_file()
    }
    assert manifest["custom_tree_sha256"] == bcj.tree_digest(manifest["custom_files_sha256"])
    assert manifest["config_sha256"]["meta.conf"] == bcj._sha256_file(job / "meta.conf")


def test_challenge_base_pins_the_persistor_but_not_its_fixed_factory(tmp_path):
    job = build(
        tmp_path, "--name", "dc5", "--base-job", "challenge_1DivideAndConquer", "--rounds", "5",
        "--sites", "CAM_1,UKA_1", "--config", "unilateral_BD",
    )
    _, server, client = confs(job)
    env = launcher_env(client)
    assert env["MODEL_NAME"] == "1DivideAndConquer" and env["CONFIG"] == "unilateral_BD"
    persistor = component(client, "persistor")["args"]
    assert persistor["model_name"] == "1DivideAndConquer" and persistor["data_config"] == "unilateral_BD"
    # This factory builds 1DivideAndConquer only and takes no model_name argument.
    assert "model_name" not in persistor["model"]["args"]
    assert swarm_args(server)["num_rounds"] == 6

    with pytest.raises(bcj.BuildError, match="always builds 1DivideAndConquer"):
        plan("--name", "x", "--base-job", "challenge_1DivideAndConquer", "--model", "MST", "--rounds", "5")


def test_model_is_required_for_the_odelia_job():
    with pytest.raises(bcj.BuildError, match="--model is required"):
        plan("--name", "x", "--rounds", "5")


# ── Feature switches and the hook order ───────────────────────────────────


def test_features_set_contract_variables_and_hooks_in_order(tmp_path):
    job = build(
        tmp_path, "--name", "all3", "--model", "MST", "--rounds", "3", "--sites", "CAM_1,MHA_1,UKA_1",
        "--dp-sigma", "2.4", "--dp-clip-norm", "0.5", "--dp-equal-weights",
        "--ms6-capture-sites", "CAM_1", "--ms6-capture-rounds", "2,0", "--ms6-capture-optimizer",
        "--al-strategy", "entropy", "--al-budget", "40", "--seed", "7", "--export-train-predictions", "0",
        "--env", "EPOCHS_PER_ROUND=1",
    )
    _, _, client = confs(job)
    env = launcher_env(client)
    assert env["ODELIA_ROUND_HOOKS"] == "al_subset,ms6_capture,dp_noise"
    assert env["ODELIA_DP_SIGMA"] == "2.4" and env["ODELIA_DP_CLIP_NORM"] == "0.5"
    assert env["ODELIA_DP_DELTA"] == "1e-05" and env["ODELIA_DP_EQUAL_WEIGHTS"] == "1"
    assert env["ODELIA_RETURN_PER_CASE"] == "0"  # implied by DP
    assert env["ODELIA_MS6_CAPTURE_SITES"] == "CAM_1"
    assert env["ODELIA_MS6_CAPTURE_ROUNDS"] == "0,2"
    assert env["ODELIA_MS6_CAPTURE_OPTIMIZER"] == "1"
    assert env["ODELIA_AL_STRATEGY"] == "entropy" and env["ODELIA_AL_BUDGET"] == "40"
    assert env["ODELIA_AL_POOL_SPLIT"] == "train" and env["ODELIA_AL_SEED"] == "0"
    assert env["ODELIA_AL_SELECT_ROUND"] == "0"
    assert env["ODELIA_SEED"] == "7" and env["ODELIA_EXPORT_TRAIN_PREDICTIONS"] == "0"
    assert env["EPOCHS_PER_ROUND"] == "1"
    manifest = json.loads((job / "job_manifest.json").read_text())
    assert manifest["round_hooks"] == ["al_subset", "ms6_capture", "dp_noise"]
    assert manifest["dp"]["epsilon_after_training_rounds"] == round(bcj.local_epsilon(2.4, 3, 1e-5), 4)


def test_hook_order_keeps_dp_noise_last_with_extra_hooks():
    assert bcj.derive_hooks({"ODELIA_DP_SIGMA": "1", "ODELIA_TRAIN_INCLUDE_LIST": "/x"}, ["my_hook"]) == [
        "al_subset",
        "my_hook",
        "dp_noise",
    ]
    assert bcj.derive_hooks({"ODELIA_MS6_CAPTURE_DIR": "/x"}) == ["ms6_capture"]
    assert bcj.derive_hooks({"MODEL_NAME": "MST"}) == []


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--dp-sigma", "2"], "needs --dp-clip-norm"),
        (["--dp-clip-norm", "1"], "need --dp-sigma"),
        (["--dp-target-epsilon", "5", "--dp-measure-only"], "needs --dp-sigma"),
        (["--dp-sigma", "2", "--dp-clip-norm", "1", "--dp-target-epsilon", "5", "--dp-measure-only"], "no budget"),
        (["--dp-sigma", "2", "--dp-clip-norm", "1", "--per-case", "site"], "ODELIA_RETURN_PER_CASE=0"),
        (["--dp-sigma", "2", "--dp-clip-norm", "1", "--dp-delta", "2"], "between 0 and 1"),
        (["--dp-sigma", "2", "--dp-clip-norm", "1", "--warm-start", "continue", "--mirror-path", "/scratch/m.pt"],
         "under-state epsilon"),
        (["--dp-equal-weights"], "need --dp-sigma"),
    ],
)
def test_dp_requests_that_cannot_work_are_refused(argv, message):
    with pytest.raises(bcj.BuildError, match=message):
        plan("--name", "x", "--model", "MST", "--rounds", "5", *argv)


def test_dp_measure_only_needs_no_sigma():
    p = plan("--name", "x", "--model", "MST", "--rounds", "5", "--dp-measure-only")
    assert p.launcher_env["ODELIA_DP_MEASURE_ONLY"] == "1"
    assert p.launcher_env["ODELIA_RETURN_PER_CASE"] == "0"
    assert p.hooks == ["dp_noise"]


def test_local_accountant_matches_privacy_accounting():
    sys.path.insert(0, str(JOBS_DIR / "_shared" / "custom"))
    try:
        spec = importlib.util.spec_from_file_location(
            "_test_privacy_accounting", JOBS_DIR / "_shared" / "custom" / "privacy_accounting.py"
        )
        pa = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pa)
    finally:
        sys.path.pop(0)
    for sigma, rounds in ((2.4, 1), (2.4, 20), (1.0, 5), (4.0, 50)):
        assert math.isclose(bcj.local_epsilon(sigma, rounds, 1e-5), pa.compute_epsilon(sigma, rounds, 1e-5))
    # docs: epsilon 9.84 after 20 rounds at sigma 2.4, delta 1e-5
    assert round(bcj.local_epsilon(2.4, 20, 1e-5), 2) == 9.84
    cap = bcj.local_max_rounds(2.4, 9.0, 1e-5)
    assert bcj.local_epsilon(2.4, cap, 1e-5) <= 9.0 < bcj.local_epsilon(2.4, cap + 1, 1e-5)


def test_dp_budget_refuses_too_many_rounds(tmp_path, capsys):
    tree = make_tree(tmp_path)
    out = build(
        tmp_path, "--name", "dp20", "--model", "MST", "--rounds", "20",
        "--dp-sigma", "2.4", "--dp-clip-norm", "1", "--dp-target-epsilon", "9", tree=tree, expect=2,
    )
    err = capsys.readouterr().err
    assert "exceed the budget" in err and "at most" in err
    assert not out.exists()
    assert list((tmp_path / "out").iterdir()) == []  # no half-built staging directory left behind

    ok = build(
        tmp_path, "--name", "dp20b", "--model", "MST", "--rounds", "20",
        "--dp-sigma", "2.4", "--dp-clip-norm", "1", "--dp-target-epsilon", "10", tree=tree,
    )
    dp = json.loads((ok / "job_manifest.json").read_text())["dp"]
    assert dp["max_training_rounds"] >= 20 and dp["target_epsilon"] == 10.0
    assert "builder copy" in dp["accountant"]  # this checkout's privacy_accounting has no cap helper yet


def test_dp_budget_uses_the_jobs_own_accountant_when_present(tmp_path, capsys):
    accountant = (JOBS_DIR / "_shared" / "custom" / "privacy_accounting.py").read_text() + (
        "\n\ndef max_rounds_for_budget(noise_multiplier, epsilon_target, delta=1e-5):\n    return 3\n"
    )
    tree = make_tree(tmp_path, privacy_accounting=accountant)
    build(
        tmp_path, "--name", "dp5", "--model", "MST", "--rounds", "5",
        "--dp-sigma", "2.4", "--dp-clip-norm", "1", "--dp-target-epsilon", "50", tree=tree, expect=2,
    )
    assert "job code" in capsys.readouterr().err


def test_importing_the_jobs_accountant_leaves_no_bytecode_in_the_job(tmp_path):
    # dp_budget imports custom/privacy_accounting.py from the job being built; a __pycache__
    # written there would ship with the job and be hashed into the manifest as job code.
    job = build(
        tmp_path, "--name", "dp_clean", "--model", "MST", "--rounds", "2",
        "--dp-sigma", "2.4", "--dp-clip-norm", "1", "--dp-target-epsilon", "50",
    )
    manifest = json.loads((job / "job_manifest.json").read_text())
    assert "import failed" not in manifest["dp"]["accountant"]  # the job's module was really imported
    assert not list(job.rglob("__pycache__")) and not list(job.rglob("*.pyc"))
    assert not [name for name in manifest["custom_files_sha256"] if name.endswith(".pyc")]


# ── Warm start ────────────────────────────────────────────────────────────


def test_source_warm_start_pins_starting_client_and_digest(tmp_path):
    job = build(
        tmp_path, "--name", "d32", "--base-job", "challenge_1DivideAndConquer", "--rounds", "2",
        "--sites", "CAM_1,UKA_1", "--warm-start", f"source:/scratch/odelia_87c5bbee_global.pt:{SHA_A}",
        "--warm-start-digest", SHA_B, "--starting-client", "UKA_1",
    )
    _, server, client = confs(job)
    assert swarm_args(server)["starting_client"] == "UKA_1"
    persistor = component(client, "persistor")["args"]
    # Only the starting client holds the file; the others initialise fresh and are
    # overwritten by the round-0 global, which every site checks against the digest.
    assert persistor["warm_start_mode"] == "auto"
    assert persistor["source_ckpt_file_full_name"] == "/scratch/odelia_87c5bbee_global.pt"
    assert persistor["source_sha256"] == SHA_A
    assert persistor["latest_global_path"] == "/scratch/mediswarm_latest_global_d32.pt"
    assert launcher_env(client)["ODELIA_WARM_START_SHA256"] == SHA_B


@pytest.mark.parametrize(
    "argv, message",
    [
        ([f"--warm-start", f"source:/scratch/a.pt:{SHA_A}", "--warm-start-digest", SHA_B], "needs --starting-client"),
        ([f"--warm-start", f"source:/scratch/a.pt:{SHA_A}", "--starting-client", "CAM_1"], "--warm-start-digest"),
        ([f"--warm-start", "source:/scratch/a.pt:abc", "--starting-client", "CAM_1"], "not a sha256"),
        ([f"--warm-start", f"source:relative.pt:{SHA_A}", "--starting-client", "CAM_1"], "absolute path"),
        ([f"--warm-start", f"source:/scratch/m.pt:{SHA_A}", "--starting-client", "CAM_1", "--warm-start-digest",
          SHA_B, "--mirror-path", "/scratch/m.pt"], "overwrite its own source"),
        (["--warm-start", "continue"], "--mirror-path"),
        (["--warm-start-digest", SHA_B], "makes no sense"),
        (["--warm-start", "sideways"], "expected fresh, continue"),
        (["--starting-client", "VHIO_1"], "not in --sites"),
        (["--mirror-path", "/scratch/m.ckpt"], "expected a .pt"),
    ],
)
def test_warm_start_requests_that_cannot_work_are_refused(argv, message):
    with pytest.raises(bcj.BuildError, match=message):
        plan("--name", "x", "--model", "MST", "--rounds", "5", *argv)


def test_continue_reads_and_writes_the_given_mirror():
    p = plan("--name", "x", "--model", "MST", "--rounds", "5", "--warm-start", "continue",
             "--mirror-path", "/scratch/mediswarm_latest_global_chain.pt", "--warm-start-digest", SHA_B)
    assert p.persistor_mode == "require"
    assert p.source_ckpt == p.mirror_path == "/scratch/mediswarm_latest_global_chain.pt"
    assert p.launcher_env["ODELIA_WARM_START_SHA256"] == SHA_B


# ── MS6, active learning, --env ───────────────────────────────────────────


def test_archive_globals_enables_both_components(tmp_path):
    job = build(tmp_path, "--name", "arc", "--model", "MST", "--rounds", "2", "--archive-globals",
                "--archive-root", "/startupkit/archive")
    _, server, client = confs(job)
    assert component(client, "ms6_global_streamer")["args"]["enabled"] is True
    archive = component(server, "ms6_global_archive")["args"]
    assert archive["enabled"] is True and archive["archive_root"] == "/startupkit/archive"


def test_archive_globals_needs_the_components_in_the_template(tmp_path, capsys):
    tree = make_tree(tmp_path, ms6_components=False)
    build(tmp_path, "--name", "arc", "--model", "MST", "--rounds", "2", "--archive-globals", tree=tree, expect=2)
    assert "ms6_global_archive" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--ms6-capture-sites", "VHIO_1"], "not in --sites"),
        (["--ms6-capture-sites", "CAM_1", "--ms6-capture-rounds", "5"], "do not exist"),
        (["--ms6-capture-rounds", "1"], "need --ms6-capture-sites"),
        (["--archive-root", "/x"], "needs --archive-globals"),
        (["--al-strategy", "entropy"], "go together"),
        (["--al-strategy", "random", "--al-budget", "0"], "positive"),
        (["--al-strategy", "random", "--al-budget", "5", "--al-select-round", "5"], "training rounds are"),
        (["--al-strategy", "random", "--al-budget", "5", "--train-include-list", "/scratch/uids.txt"], "choose one"),
        (["--al-seed", "3"], "need --al-strategy"),
        (["--env", "MODEL_NAME=MST"], "set by --model"),
        (["--env", "ODELIA_DP_SIGMA=2"], "set by --dp-sigma"),
        (["--env", "KEY_METRIC=val/x"], "job template"),
        (["--env", "FOO=a b"], "must be non-empty"),
        (["--env", "FOO={x}"], "must be non-empty"),
        (["--env", "FOO=1", "--env", "FOO=2"], "given twice"),
        (["--env", "lower-case=1"], "expected KEY=VALUE"),
        (["--sites", "CAM_1,CAM_1"], "duplicate"),
        (["--name", "bad name"], "--name"),
    ],
)
def test_other_requests_that_cannot_work_are_refused(argv, message):
    base = ["--name", "x", "--model", "MST", "--rounds", "5"]
    if "--name" in argv:
        base = ["--model", "MST", "--rounds", "5"]
    with pytest.raises(bcj.BuildError, match=message):
        plan(*base, *argv)


def test_capture_all_and_include_list():
    p = plan("--name", "x", "--model", "MST", "--rounds", "5", "--ms6-capture-sites", "all",
             "--train-include-list", "/scratch/d32/uids.txt")
    assert p.launcher_env["ODELIA_MS6_CAPTURE_SITES"] == "all"
    assert p.launcher_env["ODELIA_MS6_CAPTURE_ROUNDS"] == "all"
    assert p.launcher_env["ODELIA_TRAIN_INCLUDE_LIST"] == "/scratch/d32/uids.txt"
    assert p.hooks == ["al_subset", "ms6_capture"]


def test_unknown_odelia_env_is_flagged_not_fatal(tmp_path):
    job = build(tmp_path, "--name", "envx", "--model", "MST", "--rounds", "2", "--env", "ODELIA_TYPO=1")
    manifest = json.loads((job / "job_manifest.json").read_text())
    assert any("ODELIA_TYPO" in w for w in manifest["warnings"])


# ── The job's own code must support what the job asks for ────────────────


def test_old_job_code_is_refused_with_reasons(tmp_path, capsys):
    tree = make_tree(tmp_path, persistor=OLD_PERSISTOR, hooks=("round_hooks",), data="DATA = {'unilateral': 1}\n")
    build(tmp_path, "--name", "old", "--model", "MST", "--rounds", "2", "--config", "unilateral_BD",
          "--dp-sigma", "1", "--dp-clip-norm", "1", tree=tree, expect=2)
    err = capsys.readouterr().err
    assert "does not accept ['data_config', 'model_name']" in err
    assert "CONFIG 'unilateral_BD' is not known" in err
    assert "round hook module 'dp_noise'" in err


def test_unknown_model_is_refused(tmp_path, capsys):
    build(tmp_path, "--name", "m", "--model", "NotAModel", "--rounds", "2", expect=2)
    err = capsys.readouterr().err
    assert "'NotAModel' is not known" in err
    # The refusal lists what the job's code would accept (#636 get_supported_model_names).
    assert "MST, MST_SAMMed2D, MedicalNet, ResNet18, ResNet101, challenge_1DivideAndConquer" in err


@pytest.mark.parametrize("model", ["Swin3D", "MST_sammed2d", "challenge_NoSuchTeam"])
def test_names_create_model_rejects_are_refused(tmp_path, capsys, model):
    # Swin3D was removed in #636; a quoted mention elsewhere in the file must not count.
    build(tmp_path, "--name", "m", "--model", model, "--rounds", "2", expect=2)
    assert f"{model!r} is not known" in capsys.readouterr().err


@pytest.mark.parametrize(
    "alias, unified", [("2BCN_AIM", "challenge_2BCN_AIM"), ("challenge", "challenge_1DivideAndConquer")]
)
def test_unified_aliases_build_with_a_warning(tmp_path, alias, unified):
    # create_model() unifies these spellings itself, so they work; the warning asks for the full name.
    job = build(tmp_path, "--name", "alias", "--model", alias, "--rounds", "2")
    manifest = json.loads((job / "job_manifest.json").read_text())
    assert any(f"reads this as {unified}" in w for w in manifest["warnings"])


def test_model_list_of_this_checkout_is_read_statically():
    names, aliases = bcj.supported_model_names(JOBS_DIR / "_shared" / "custom" / "models" / "models_config.py")
    assert {"MST", "MST_SAMMed2D", "MedicalNet", "ResNet101", "challenge_1DivideAndConquer"} <= set(names)
    assert "Swin3D" not in names and len(names) == len(set(names))
    assert aliases["1DivideAndConquer"] == "challenge_1DivideAndConquer"
    assert aliases["challenge"] == "challenge_1DivideAndConquer"


def test_model_check_falls_back_to_text_for_code_before_636(tmp_path, capsys):
    tree = make_tree(tmp_path, models=OLD_MODELS)
    assert bcj.supported_model_names(tree / "application/jobs/_shared/custom/models/models_config.py") is None
    build(tmp_path, "--name", "old_ok", "--model", "MST_SAMMed2D", "--rounds", "2", tree=tree)
    build(tmp_path, "--name", "old_bad", "--model", "NotAModel", "--rounds", "2", tree=tree, expect=2)
    assert "'NotAModel' is not known" in capsys.readouterr().err


def test_no_code_checks_is_recorded(tmp_path):
    tree = make_tree(tmp_path, persistor=OLD_PERSISTOR)
    job = build(tmp_path, "--name", "nc", "--model", "MST", "--rounds", "2", "--no-code-checks", tree=tree)
    manifest = json.loads((job / "job_manifest.json").read_text())
    assert manifest["code_checks"].startswith("skipped")
    assert any("code-support checks skipped" in w for w in manifest["warnings"])


def test_persistor_args_taken_from_kwargs_count_as_accepted(tmp_path):
    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "warm_continue.py").write_text(
        "class WarmStartablePTFileModelPersistor:\n"
        "    def __init__(self, warm_start_mode='auto', **kw):\n"
        "        self.data_config = kw.pop('data_config', None)\n"
        "        self.model_name = kw.get('model_name')\n"
    )
    accepted = bcj.persistor_accepted_args(custom)
    assert {"data_config", "model_name", "source_ckpt_file_full_name"} <= accepted
    assert "source_sha256" not in accepted


# ── Roster safety and output handling ─────────────────────────────────────


def test_vhio_must_be_listed_to_appear(tmp_path, capsys):
    tree = make_tree(tmp_path)
    server = tree / "application/jobs/ODELIA_ternary_classification/app/config/config_fed_server.conf"
    text = server.read_text().replace("num_rounds = 20", 'num_rounds = 20\n      aggr_clients = ["VHIO_1"]', 1)
    server.write_text(text)
    build(tmp_path, "--name", "v", "--model", "MST", "--rounds", "2", tree=tree, expect=2)
    assert "no VHIO_1 unless listed" in capsys.readouterr().err


def test_vhio_is_fine_when_listed(tmp_path):
    job = build(tmp_path, "--name", "v8", "--model", "MST", "--rounds", "2", "--sites", "CAM_1,VHIO_1")
    meta, _, _ = confs(job)
    assert meta["deploy_map"]["app"] == ["server", "CAM_1", "VHIO_1"]


def test_existing_job_needs_force(tmp_path, capsys):
    tree = make_tree(tmp_path)
    job = build(tmp_path, "--name", "dup", "--model", "MST", "--rounds", "2", tree=tree)
    build(tmp_path, "--name", "dup", "--model", "MST", "--rounds", "2", tree=tree, expect=2)
    assert "exists" in capsys.readouterr().err
    build(tmp_path, "--name", "dup", "--model", "MST_SAMMed2D", "--rounds", "2", "--force", tree=tree)
    assert launcher_env(confs(job)[2])["MODEL_NAME"] == "MST_SAMMed2D"
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["dup"]


def test_dry_run_writes_nothing(tmp_path, capsys):
    tree = make_tree(tmp_path)
    assert bcj.main(["--name", "dry", "--model", "MST", "--rounds", "2", "--source", f"dir:{tree}", "--dry-run",
                     "--json"]) == 0
    captured = capsys.readouterr()
    manifest = json.loads(captured.out)
    assert manifest["job_name"] == "dry" and "dry run" in captured.err


def test_repo_source_builds_from_this_checkout(capsys):
    # Current job code may predate the 1.9.0 features; the checks are covered above.
    assert bcj.main(["--name", "repo_dry", "--model", "MST", "--rounds", "2", "--dry-run", "--no-code-checks",
                     "--json"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["source"]["kind"] == "repo"
    assert manifest["custom_files_sha256"]


def _git_usable():
    try:
        subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], check=True, capture_output=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


@pytest.mark.skipif(not _git_usable(), reason="no usable git checkout")
def test_git_source_builds_from_a_committed_ref(capsys):
    assert bcj.main(["--name", "git_dry", "--model", "MST", "--rounds", "2", "--dry-run", "--no-code-checks",
                     "--source", "git:HEAD", "--json"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["source"]["kind"] == "git" and manifest["source"]["dirty"] is False
    assert len(manifest["source"]["git_commit"]) == 40


def test_launcher_rewrite_keeps_template_tokens_in_place():
    script = "KEY_METRIC=val/AUC_ROC MODEL_NAME=old python3 custom/{app_script}  {app_config} "
    out = bcj.render_launcher(script, {"MODEL_NAME": "MST", "CONFIG": "unilateral"})
    assert out == "KEY_METRIC=val/AUC_ROC MODEL_NAME=MST CONFIG=unilateral python3 custom/{app_script}  {app_config} "
    with pytest.raises(bcj.BuildError, match="quoted"):
        bcj.split_launcher("A='x y' python3 main.py")
