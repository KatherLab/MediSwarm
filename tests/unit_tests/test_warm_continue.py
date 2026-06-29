import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_CUSTOM_DIR = REPO_ROOT / "application" / "jobs" / "_shared" / "custom"


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))


def _install_warm_continue_nvflare_mocks(monkeypatch):
    class DummyPTFileModelPersistor:
        def __init__(self, **kwargs):
            self.source_ckpt_file_full_name = kwargs.get("source_ckpt_file_full_name")
            self.logger = _Logger()
            self.panics = []

        def handle_event(self, event, fl_ctx):
            return None

        def load_model(self, fl_ctx):
            return "loaded"

        def log_info(self, fl_ctx, message):
            self.logger.info(message)

        def log_warning(self, fl_ctx, message):
            self.logger.messages.append(("warning", message))

        def system_panic(self, reason, fl_ctx):
            self.panics.append(reason)

    modules = {
        "nvflare": types.ModuleType("nvflare"),
        "nvflare.apis": types.ModuleType("nvflare.apis"),
        "nvflare.apis.event_type": types.ModuleType("nvflare.apis.event_type"),
        "nvflare.apis.fl_constant": types.ModuleType("nvflare.apis.fl_constant"),
        "nvflare.apis.fl_context": types.ModuleType("nvflare.apis.fl_context"),
        "nvflare.apis.workspace": types.ModuleType("nvflare.apis.workspace"),
        "nvflare.app_common": types.ModuleType("nvflare.app_common"),
        "nvflare.app_common.app_event_type": types.ModuleType("nvflare.app_common.app_event_type"),
        "nvflare.app_opt": types.ModuleType("nvflare.app_opt"),
        "nvflare.app_opt.pt": types.ModuleType("nvflare.app_opt.pt"),
        "nvflare.app_opt.pt.file_model_persistor": types.ModuleType("nvflare.app_opt.pt.file_model_persistor"),
    }
    modules["nvflare.apis.event_type"].EventType = object
    modules["nvflare.apis.fl_constant"].FLContextKey = SimpleNamespace(APP_ROOT="__app_root__")
    modules["nvflare.apis.fl_context"].FLContext = object
    modules["nvflare.apis.workspace"].WorkspaceConstants = SimpleNamespace(CUSTOM_FOLDER_NAME="custom")
    modules["nvflare.app_common.app_event_type"].AppEventType = SimpleNamespace(
        GLOBAL_BEST_MODEL_AVAILABLE="GLOBAL_BEST_MODEL_AVAILABLE"
    )
    modules["nvflare.app_opt.pt.file_model_persistor"].PTFileModelPersistor = DummyPTFileModelPersistor

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _install_controller_nvflare_mocks(monkeypatch):
    class FakeFLContext:
        def __init__(self, identity=None, props=None, peer_context=None):
            self.identity = identity
            self.props = props or {}
            self.peer_context = peer_context

        def get_peer_context(self):
            return self.peer_context

        def get_identity_name(self):
            return self.identity

        def get_prop(self, key):
            return self.props.get(key)

        def set_prop(self, *args, **kwargs):
            return None

    class FakeClientStatus:
        def __init__(self):
            self.last_report_time = None
            self.num_reports = 0
            self.status = None
            self.last_progress_time = None

    class FakeGatherer:
        pass

    class FakeSwarmClientController:
        pass

    class FakeSwarmServerController:
        pass

    def status_report_from_dict(report):
        return SimpleNamespace(
            error=report.get("error"),
            timestamp=report.get("timestamp"),
            last_round=report.get("last_round", 0),
            action=report.get("action", "train"),
            all_done=report.get("all_done", False),
        )

    modules = {
        "nvflare": types.ModuleType("nvflare"),
        "nvflare.apis": types.ModuleType("nvflare.apis"),
        "nvflare.apis.fl_context": types.ModuleType("nvflare.apis.fl_context"),
        "nvflare.apis.shareable": types.ModuleType("nvflare.apis.shareable"),
        "nvflare.app_common": types.ModuleType("nvflare.app_common"),
        "nvflare.app_common.app_constant": types.ModuleType("nvflare.app_common.app_constant"),
        "nvflare.app_common.app_event_type": types.ModuleType("nvflare.app_common.app_event_type"),
        "nvflare.app_common.ccwf": types.ModuleType("nvflare.app_common.ccwf"),
        "nvflare.app_common.ccwf.common": types.ModuleType("nvflare.app_common.ccwf.common"),
        "nvflare.app_common.ccwf.server_ctl": types.ModuleType("nvflare.app_common.ccwf.server_ctl"),
        "nvflare.app_common.ccwf.swarm_client_ctl": types.ModuleType("nvflare.app_common.ccwf.swarm_client_ctl"),
        "nvflare.app_common.ccwf.swarm_server_ctl": types.ModuleType("nvflare.app_common.ccwf.swarm_server_ctl"),
    }
    modules["nvflare.apis.fl_context"].FLContext = FakeFLContext
    modules["nvflare.apis.shareable"].ReturnCode = SimpleNamespace(OK="OK", EXECUTION_EXCEPTION="EXECUTION_EXCEPTION")
    modules["nvflare.apis.shareable"].make_reply = lambda rc: {"return_code": rc}
    modules["nvflare.app_common.app_constant"].AppConstants = SimpleNamespace(
        CURRENT_ROUND="current_round",
        TRAINING_RESULT="training_result",
        AGGREGATION_ACCEPTED="aggregation_accepted",
    )
    modules["nvflare.app_common.app_event_type"].AppEventType = SimpleNamespace(
        BEFORE_CONTRIBUTION_ACCEPT="before_contribution_accept",
        AFTER_CONTRIBUTION_ACCEPT="after_contribution_accept",
    )
    modules["nvflare.app_common.ccwf.common"].Constant = SimpleNamespace(STATUS_REPORTS="status_reports")
    modules["nvflare.app_common.ccwf.common"].status_report_from_dict = status_report_from_dict
    modules["nvflare.app_common.ccwf.server_ctl"].ClientStatus = FakeClientStatus
    modules["nvflare.app_common.ccwf.swarm_client_ctl"].Gatherer = FakeGatherer
    modules["nvflare.app_common.ccwf.swarm_client_ctl"].SwarmClientController = FakeSwarmClientController
    modules["nvflare.app_common.ccwf.swarm_server_ctl"].SwarmServerController = FakeSwarmServerController

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return FakeFLContext


def _import_module(module_name, path):
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def warm_continue(monkeypatch):
    _install_warm_continue_nvflare_mocks(monkeypatch)
    return _import_module("warm_continue_under_test", SHARED_CUSTOM_DIR / "warm_continue.py")


@pytest.fixture
def fault_tolerant_ccwf(monkeypatch):
    fl_context_cls = _install_controller_nvflare_mocks(monkeypatch)
    module = _import_module("fault_tolerant_ccwf_under_test", SHARED_CUSTOM_DIR / "fault_tolerant_ccwf.py")
    return module, fl_context_cls


def test_auto_missing_absolute_checkpoint_starts_fresh(warm_continue, tmp_path):
    missing = tmp_path / "missing.pt"
    persistor = warm_continue.WarmStartablePTFileModelPersistor(
        warm_start_mode="auto",
        source_ckpt_file_full_name=str(missing),
    )

    assert persistor.source_ckpt_file_full_name is None


def test_auto_present_checkpoint_warm_starts(warm_continue, tmp_path):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"checkpoint")

    persistor = warm_continue.WarmStartablePTFileModelPersistor(
        warm_start_mode="auto",
        source_ckpt_file_full_name=str(checkpoint),
    )

    assert persistor.source_ckpt_file_full_name == str(checkpoint)


def test_fresh_ignores_present_checkpoint(warm_continue, tmp_path):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"checkpoint")

    persistor = warm_continue.WarmStartablePTFileModelPersistor(
        warm_start_mode="fresh",
        source_ckpt_file_full_name=str(checkpoint),
    )

    assert persistor.source_ckpt_file_full_name is None


def test_require_present_checkpoint_warm_starts(warm_continue, tmp_path):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"checkpoint")

    persistor = warm_continue.WarmStartablePTFileModelPersistor(
        warm_start_mode="require",
        source_ckpt_file_full_name=str(checkpoint),
    )

    assert persistor.source_ckpt_file_full_name == str(checkpoint)


def test_require_missing_checkpoint_panics_during_load_model(warm_continue, tmp_path):
    missing = tmp_path / "missing.pt"

    persistor = warm_continue.WarmStartablePTFileModelPersistor(
        warm_start_mode="require",
        source_ckpt_file_full_name=str(missing),
    )

    assert persistor.source_ckpt_file_full_name == str(missing)
    assert persistor.load_model(SimpleNamespace(get_prop=lambda key: None)) is None
    assert len(persistor.panics) == 1
    assert "WARM_START_REQUIRED_MISSING" in persistor.panics[0]
    assert str(missing) in persistor.panics[0]


def test_continue_negative_path_uses_expected_require_error_string(warm_continue, tmp_path):
    missing = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError) as exc:
        warm_continue.resolve_source_checkpoint(str(missing), "require")

    assert "WARM_START_REQUIRED_MISSING" in str(exc.value)
    assert str(missing) in str(exc.value)


def test_invalid_warm_start_mode_raises(warm_continue):
    with pytest.raises(ValueError, match="Invalid warm_start_mode"):
        warm_continue.WarmStartablePTFileModelPersistor(
            warm_start_mode="resume",
            source_ckpt_file_full_name=None,
        )


def _make_controller_with_report(module, fl_context_cls, error):
    reports_key = module.Constant.STATUS_REPORTS
    peer_ctx = fl_context_cls(
        identity="site1",
        props={
            reports_key: {
                "wf": {
                    "error": error,
                    "timestamp": 1,
                    "last_round": 2,
                    "action": "train",
                    "all_done": False,
                }
            }
        },
    )
    fl_ctx = fl_context_cls(peer_context=peer_ctx)

    controller = module.FaultTolerantSwarmServerController.__new__(module.FaultTolerantSwarmServerController)
    controller.workflow_id = "wf"
    controller.min_clients = 2
    controller.asked_to_stop = False
    controller.client_statuses = {
        "site1": module.ClientStatus(),
        "site2": module.ClientStatus(),
        "site3": module.ClientStatus(),
    }
    controller.log_debug = lambda *args, **kwargs: None
    controller.log_info = lambda *args, **kwargs: None
    controller.log_warning = lambda *args, **kwargs: None
    panics = []
    controller.system_panic = lambda message, ctx: panics.append(message)
    return controller, fl_ctx, panics


def test_warm_start_required_missing_aborts_instead_of_pruning(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, panics = _make_controller_with_report(
        module,
        fl_context_cls,
        "WARM_START_REQUIRED_MISSING: required warm-start checkpoint missing: /scratch/mediswarm_latest_global.pt",
    )

    controller._update_client_status(fl_ctx)

    assert controller.asked_to_stop is True
    assert "site1" in controller.client_statuses
    assert len(panics) == 1
    assert "non-tolerable warm-start failure" in panics[0]


def test_transient_error_is_still_pruned_when_min_clients_remain(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, panics = _make_controller_with_report(module, fl_context_cls, "MODEL_UNRECOGNIZED")

    controller._update_client_status(fl_ctx)

    assert controller.asked_to_stop is False
    assert "site1" not in controller.client_statuses
    assert panics == []
