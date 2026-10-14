"""Pinned warm starts in WarmStartablePTFileModelPersistor (release 1.9.0, contract 3.6; D2.5/D3.2).

Covers the pinned source, spelled 'source' or 'auto' + source_sha256 (the job builder's form): only the
starting client needs the file, it must match source_sha256, other clients never read it. Also
source_sha256 in require mode and its refusal with fresh, refusal of a mirror path equal to the source,
provenance from the persistor's model_name/data_config args instead of the container env, and one run
through NVFlare's real PTFileModelPersistor showing that the round-0 global it builds from the source file
has the state digest the job builder records (ODELIA_WARM_START_SHA256).

The existing modes are covered, unchanged, by test_warm_continue.py and test_warm_start_provenance.py.
"""

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import SHARED_CUSTOM_DIR, import_module_from_path

START_CLIENT_KEY = "cwf.start_client"
CONFIG_KEY = "cwf.config"
TASK_DATA_KEY = "__task_data__"


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))


def _install_mocks(monkeypatch):
    """NVFlare stand-ins: the base persistor 'loads' whatever source path it is left with."""

    class DummyPTFileModelPersistor:
        def __init__(self, **kwargs):
            self.source_ckpt_file_full_name = kwargs.get("source_ckpt_file_full_name")
            self.logger = _Logger()
            self.panics = []
            self.loaded_from = []

        def handle_event(self, event, fl_ctx):
            return None

        def load_model(self, fl_ctx):
            self.loaded_from.append(self.source_ckpt_file_full_name)
            return "loaded"

        def save_model(self, ml, fl_ctx):
            Path(self._ckpt_save_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self._ckpt_save_path).write_bytes(b"latest global")

        def log_info(self, fl_ctx, message):
            self.logger.info(message)

        def log_warning(self, fl_ctx, message):
            self.logger.warning(message)

        def system_panic(self, reason, fl_ctx):
            self.panics.append(reason)

    names = [
        "nvflare", "nvflare.apis", "nvflare.apis.event_type", "nvflare.apis.fl_constant", "nvflare.apis.fl_context",
        "nvflare.apis.workspace", "nvflare.app_common", "nvflare.app_common.app_event_type", "nvflare.app_opt",
        "nvflare.app_opt.pt", "nvflare.app_opt.pt.file_model_persistor",
    ]
    modules = {name: types.ModuleType(name) for name in names}
    modules["nvflare.apis.event_type"].EventType = object
    modules["nvflare.apis.fl_constant"].FLContextKey = SimpleNamespace(APP_ROOT="__app_root__", TASK_DATA=TASK_DATA_KEY)
    modules["nvflare.apis.fl_context"].FLContext = object
    modules["nvflare.apis.workspace"].WorkspaceConstants = SimpleNamespace(CUSTOM_FOLDER_NAME="custom")
    modules["nvflare.app_common.app_event_type"].AppEventType = SimpleNamespace(
        GLOBAL_BEST_MODEL_AVAILABLE="GLOBAL_BEST_MODEL_AVAILABLE"
    )
    modules["nvflare.app_opt.pt.file_model_persistor"].PTFileModelPersistor = DummyPTFileModelPersistor
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def wc(monkeypatch):
    _install_mocks(monkeypatch)
    module = import_module_from_path("warm_continue_source_under_test", SHARED_CUSTOM_DIR / "warm_continue.py")
    yield module
    sys.modules.pop("warm_continue_source_under_test", None)


class FakeContext:
    """FLContext of the CCWF configure task: the task data names the starting client."""

    def __init__(self, me, start_client=None, job_id="job-1"):
        self.me = me
        self.props = {}
        if start_client is not None:
            self.props[TASK_DATA_KEY] = {CONFIG_KEY: {START_CLIENT_KEY: start_client}}
        self.job_id = job_id

    def get_prop(self, key, default=None):
        return self.props.get(key, default)

    def get_identity_name(self):
        return self.me

    def get_job_id(self):
        return self.job_id


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "odelia_warmstart_87c5bbee_best.pt"
    path.write_bytes(b"pretend 722 MB of 1DivideAndConquer weights")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _persistor(wc, src, digest, tmp_path, **extra):
    args = dict(
        warm_start_mode="source",
        source_ckpt_file_full_name=str(src),
        source_sha256=digest,
        latest_global_path=str(tmp_path / "mediswarm_latest_global_d32_entropy_b20.pt"),
    )
    args.update(extra)
    return wc.WarmStartablePTFileModelPersistor(**args)


@pytest.fixture(params=["source", "auto"])
def pinned_mode(request):
    """Both spellings of a pinned warm start: explicit 'source', and 'auto' + source_sha256, which is what
    scripts/admin/build_consortium_job.py writes for --warm-start source:<path>:<sha256>."""
    return request.param


# --------------------------------------------------------------------------------------------------
# Pinned source: only the starting client needs the file
# --------------------------------------------------------------------------------------------------


def test_starting_client_verifies_and_loads_the_source(wc, source, tmp_path, pinned_mode):
    src, digest = source
    # sha256sum output in either case
    persistor = _persistor(wc, src, digest.upper(), tmp_path, warm_start_mode=pinned_mode)
    assert persistor.load_model(FakeContext("RSH_1", start_client="RSH_1")) == "loaded"
    assert persistor.loaded_from == [str(src)]
    assert persistor.panics == []
    assert any("verified" in m for _, m in persistor.logger.messages)


def test_starting_client_without_the_file_fails_configure(wc, source, tmp_path, pinned_mode):
    src, digest = source
    src.unlink()
    persistor = _persistor(wc, src, digest, tmp_path, warm_start_mode=pinned_mode)
    with pytest.raises(wc.WarmStartRefused, match="WARM_START_SOURCE_MISSING"):
        persistor.load_model(FakeContext("RSH_1", start_client="RSH_1"))
    assert len(persistor.panics) == 1 and "WARM_START_SOURCE_MISSING" in persistor.panics[0]
    assert persistor.loaded_from == []  # never falls back to an untrained model


def test_starting_client_with_another_file_is_refused(wc, source, tmp_path, pinned_mode):
    src, digest = source
    src.write_bytes(b"the October MST global, same path")
    persistor = _persistor(wc, src, digest, tmp_path, warm_start_mode=pinned_mode)
    with pytest.raises(wc.WarmStartRefused, match="WARM_START_SOURCE_DIGEST_MISMATCH"):
        persistor.load_model(FakeContext("UKA_1", start_client="UKA_1"))
    assert digest in persistor.panics[0]
    assert persistor.loaded_from == []


def test_other_clients_do_not_need_or_read_the_file(wc, source, tmp_path, pinned_mode):
    src, digest = source
    src.unlink()  # not present at this site, as at every partner except the starting client
    persistor = _persistor(wc, src, digest, tmp_path, warm_start_mode=pinned_mode)
    assert persistor.load_model(FakeContext("UMCU_1", start_client="RSH_1")) == "loaded"
    assert persistor.loaded_from == [None]  # base persistor initialises from its own model
    assert persistor.panics == []
    info = [m for level, m in persistor.logger.messages if level == "info"]
    assert any("not the starting client (RSH_1 is)" in m for m in info)


def test_other_client_holding_a_different_file_is_not_disturbed(wc, source, tmp_path, pinned_mode):
    src, digest = source
    src.write_bytes(b"something else entirely")
    persistor = _persistor(wc, src, digest, tmp_path, warm_start_mode=pinned_mode)
    assert persistor.load_model(FakeContext("MHA_1", start_client="RSH_1")) == "loaded"
    assert persistor.panics == []


def test_unknown_role_is_treated_as_the_starting_client(wc, source, tmp_path, pinned_mode):
    src, digest = source
    src.unlink()
    persistor = _persistor(wc, src, digest, tmp_path, warm_start_mode=pinned_mode)
    with pytest.raises(wc.WarmStartRefused):
        persistor.load_model(FakeContext("RSH_1", start_client=None))  # no CCWF configure task in sight


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"source_sha256": None}, "needs source_sha256"),
        ({"source_ckpt_file_full_name": None}, "needs source_ckpt_file_full_name"),
        ({"source_sha256": "abc123"}, "64-character hex"),
        ({"warm_start_mode": "fresh"}, "fresh never reads one"),
    ],
)
def test_source_mode_configuration_errors(wc, source, tmp_path, overrides, message):
    src, digest = source
    with pytest.raises(ValueError, match=message):
        _persistor(wc, src, digest, tmp_path, **overrides)


def test_mirror_path_equal_to_the_source_is_refused(wc, source, tmp_path, pinned_mode):
    src, digest = source
    with pytest.raises(ValueError, match="overwrite the checkpoint it is pinned to"):
        _persistor(wc, src, digest, tmp_path, warm_start_mode=pinned_mode, latest_global_path=str(src))


def test_source_mode_mirrors_to_the_experiment_path_with_lineage(wc, source, tmp_path, monkeypatch):
    src, digest = source
    monkeypatch.setenv("MODEL_NAME", "MST")  # the container default; this job trains 1DC
    monkeypatch.setenv("CONFIG", "unilateral")
    persistor = _persistor(wc, src, digest, tmp_path, model_name="1DivideAndConquer", data_config="unilateral")
    persistor._ckpt_save_path = str(tmp_path / "run" / "FL_global_model.pt")
    persistor.save_model(ml=object(), fl_ctx=FakeContext("UMCU_1", job_id="job-d32"))

    mirror = Path(persistor.latest_global_path)
    assert mirror.read_bytes() == b"latest global"
    assert src.read_bytes() != b"latest global"  # the pinned source is never touched
    sidecar = json.loads(Path(wc.sidecar_path(str(mirror))).read_text())
    assert sidecar["model_name"] == "1DivideAndConquer" and sidecar["model_name_source"] == "arg"
    assert sidecar["data_config"] == "unilateral" and sidecar["data_config_source"] == "arg"
    assert sidecar["warm_start_mode"] == "source"
    assert sidecar["warm_start_source_sha256"] == digest
    assert sidecar["job_id"] == "job-d32"
    assert sidecar["sha256"] == hashlib.sha256(b"latest global").hexdigest()


# --------------------------------------------------------------------------------------------------
# source_sha256 in the existing modes
# --------------------------------------------------------------------------------------------------


def test_require_mode_verifies_a_pinned_digest(wc, source, tmp_path):
    src, digest = source
    persistor = wc.WarmStartablePTFileModelPersistor(
        warm_start_mode="require", source_ckpt_file_full_name=str(src), source_sha256=digest
    )
    assert persistor.load_model(FakeContext("A")) == "loaded"
    src.write_bytes(b"changed")
    persistor = wc.WarmStartablePTFileModelPersistor(
        warm_start_mode="require", source_ckpt_file_full_name=str(src), source_sha256=digest
    )
    with pytest.raises(wc.WarmStartRefused, match="DIGEST_MISMATCH"):
        persistor.load_model(FakeContext("A"))


def test_auto_mode_with_a_pinned_digest_never_starts_fresh_silently(wc, source, tmp_path):
    src, digest = source
    src.unlink()
    persistor = wc.WarmStartablePTFileModelPersistor(
        warm_start_mode="auto", source_ckpt_file_full_name=str(src), source_sha256=digest
    )
    assert persistor.source_ckpt_file_full_name == str(src)  # not dropped, as auto does without a digest
    with pytest.raises(wc.WarmStartRefused, match="WARM_START_SOURCE_MISSING"):
        persistor.load_model(FakeContext("RSH_1", start_client="RSH_1"))
    assert persistor.loaded_from == []


def test_auto_mode_without_a_digest_is_unchanged(wc, tmp_path):
    persistor = wc.WarmStartablePTFileModelPersistor(
        warm_start_mode="auto", source_ckpt_file_full_name=str(tmp_path / "missing.pt")
    )
    assert persistor.source_ckpt_file_full_name is None
    assert persistor.load_model(FakeContext("A")) == "loaded"


# --------------------------------------------------------------------------------------------------
# Provenance from the job's args, not the container env
# --------------------------------------------------------------------------------------------------


def _mirror_with_sidecar(wc, tmp_path, **sidecar):
    mirror = tmp_path / "mediswarm_latest_global.pt"
    mirror.write_bytes(b"a mirrored global")
    Path(wc.sidecar_path(str(mirror))).write_text(json.dumps(sidecar))
    return mirror


def test_model_name_arg_overrides_the_container_env(wc, tmp_path, monkeypatch):
    """Parent env MST (kit default), job pinned to 1DivideAndConquer: the MST mirror is refused."""
    monkeypatch.setenv("MODEL_NAME", "MST")
    mirror = _mirror_with_sidecar(wc, tmp_path, model_name="MST", job_id="7a678d9a", sha256="")
    persistor = wc.WarmStartablePTFileModelPersistor(
        warm_start_mode="require", source_ckpt_file_full_name=str(mirror), model_name="1DivideAndConquer"
    )
    assert persistor.load_model(FakeContext("A")) is None
    assert len(persistor.panics) == 1 and "WARM_START_MODEL_MISMATCH" in persistor.panics[0]
    assert "'1DivideAndConquer'" in persistor.panics[0]


def test_without_the_arg_the_env_is_still_used(wc, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "MST")
    mirror = _mirror_with_sidecar(wc, tmp_path, model_name="MST", job_id="7a678d9a", sha256="")
    persistor = wc.WarmStartablePTFileModelPersistor(warm_start_mode="require", source_ckpt_file_full_name=str(mirror))
    assert persistor.load_model(FakeContext("A")) == "loaded"


def test_data_config_mismatch_is_refused_when_the_job_states_its_config(wc, tmp_path, monkeypatch):
    monkeypatch.setenv("CONFIG", "unilateral")
    mirror = _mirror_with_sidecar(wc, tmp_path, model_name="1DivideAndConquer", data_config="unilateral")
    persistor = wc.WarmStartablePTFileModelPersistor(
        warm_start_mode="require", source_ckpt_file_full_name=str(mirror),
        model_name="1DivideAndConquer", data_config="unilateral_BD",
    )
    assert persistor.load_model(FakeContext("A")) is None
    assert "WARM_START_DATA_CONFIG_MISMATCH" in persistor.panics[0]


def test_a_legacy_sidecar_counts_as_unilateral(wc, tmp_path):
    mirror = _mirror_with_sidecar(wc, tmp_path, model_name="1DivideAndConquer", job_id="old", sha256="")
    bd_job = wc.WarmStartablePTFileModelPersistor(
        warm_start_mode="require", source_ckpt_file_full_name=str(mirror),
        model_name="1DivideAndConquer", data_config="unilateral_BD",
    )
    assert bd_job.load_model(FakeContext("A")) is None
    assert "predates data_config" in bd_job.panics[0]
    unilateral_job = wc.WarmStartablePTFileModelPersistor(
        warm_start_mode="require", source_ckpt_file_full_name=str(mirror),
        model_name="1DivideAndConquer", data_config="unilateral",
    )
    assert unilateral_job.load_model(FakeContext("A")) == "loaded"


def test_data_config_from_the_env_only_warns(wc, tmp_path, monkeypatch):
    """Without the arg the container's CONFIG is a guess: existing jobs keep loading, with a warning."""
    monkeypatch.setenv("CONFIG", "unilateral")
    mirror = _mirror_with_sidecar(wc, tmp_path, model_name="MST", data_config="unilateral_BD")
    monkeypatch.setenv("MODEL_NAME", "MST")
    persistor = wc.WarmStartablePTFileModelPersistor(warm_start_mode="require", source_ckpt_file_full_name=str(mirror))
    assert persistor.load_model(FakeContext("A")) == "loaded"
    assert persistor.panics == []
    assert any("WARM_START_DATA_CONFIG_MISMATCH" in m for level, m in persistor.logger.messages if level == "warning")


# --------------------------------------------------------------------------------------------------
# The real persistor: the round-0 global built from the source has the digest the builder records
# --------------------------------------------------------------------------------------------------


def test_real_persistor_round0_global_matches_checkpoint_state_digest(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("nvflare.app_opt.pt.file_model_persistor")
    from nvflare.apis.fl_constant import FLContextKey, ReservedKey
    from nvflare.apis.fl_context import FLContext
    from nvflare.apis.shareable import Shareable

    digest_module = import_module_from_path("state_digest_for_persistor", SHARED_CUSTOM_DIR / "state_digest.py")
    for name in ("warm_continue_real_under_test",):
        sys.modules.pop(name, None)
    wc_real = import_module_from_path("warm_continue_real_under_test", SHARED_CUSTOM_DIR / "warm_continue.py")

    torch.manual_seed(0)
    trained = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    src = tmp_path / "odelia_warmstart.pt"
    torch.save({"model": trained.state_dict(), "train_conf": {"train": {"model": "Sequential"}}}, src)
    file_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    expected_state_digest = digest_module.checkpoint_state_digest(str(src))

    def context(me, start_client):
        ctx = FLContext()
        task = Shareable()
        task[CONFIG_KEY] = {START_CLIENT_KEY: start_client}
        ctx.set_prop(FLContextKey.TASK_DATA, task, private=True, sticky=False)
        ctx.set_prop(ReservedKey.IDENTITY_NAME, me, private=False, sticky=False)
        return ctx

    def persistor():
        p = wc_real.WarmStartablePTFileModelPersistor(
            # What the job builder writes for --warm-start source:<path>:<sha256> (contract 3.6).
            warm_start_mode="auto", source_ckpt_file_full_name=str(src), source_sha256=file_sha,
            latest_global_path=str(tmp_path / "mirror.pt"), model_name="Sequential",
        )
        torch.manual_seed(1)
        p.model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))  # untrained
        panics = []
        monkeypatch.setattr(p, "system_panic", lambda reason, fl_ctx: panics.append(reason))
        return p, panics

    starting, panics = persistor()
    learnable = starting.load_model(context("RSH_1", "RSH_1"))
    assert panics == []
    weights = learnable["weights"]
    assert digest_module.state_digest(weights) == expected_state_digest  # what ODELIA_WARM_START_SHA256 holds

    other, panics = persistor()
    learnable = other.load_model(context("UMCU_1", "RSH_1"))
    assert panics == []
    # A non-starting client builds its (discarded) model from its own initialisation, not from the file.
    assert digest_module.state_digest(learnable["weights"]) != expected_state_digest
