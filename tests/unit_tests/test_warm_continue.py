import importlib.util
import sys
import threading
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
            if event == "GLOBAL_BEST_MODEL_AVAILABLE" and getattr(self, "_best_ckpt_save_path", None):
                Path(self._best_ckpt_save_path).parent.mkdir(parents=True, exist_ok=True)
                Path(self._best_ckpt_save_path).write_bytes(b"best global")
            return None

        def load_model(self, fl_ctx):
            return "loaded"

        def save_model(self, ml, fl_ctx):
            Path(self._ckpt_save_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self._ckpt_save_path).write_bytes(b"latest global")

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
    engine_key = "__engine__"

    class FakeShareable(dict):
        def __init__(self):
            super().__init__()
            self.headers = {}

        def set_header(self, key, value):
            self.headers[key] = value

        def get_header(self, key, default=None):
            return self.headers.get(key, default)

        def get_return_code(self, default=None):
            return self.get("return_code", default)

    class FakeFLContext:
        def __init__(self, identity=None, props=None, peer_context=None, engine=None, run_abort_signal=None):
            self.identity = identity
            self.props = props or {}
            self.peer_context = peer_context
            self.engine = engine
            self.run_abort_signal = run_abort_signal

        def clone(self):
            return FakeFLContext(
                identity=self.identity,
                props=dict(self.props),
                peer_context=self.peer_context,
                engine=self.engine,
                run_abort_signal=self.run_abort_signal,
            )

        def put(self, key, value, private, sticky):
            if key == engine_key:
                self.engine = value
            else:
                self.props[key] = value

        def get_engine(self):
            return self.engine

        def get_run_abort_signal(self):
            return self.run_abort_signal

        def get_peer_context(self):
            return self.peer_context

        def set_peer_context(self, ctx):
            self.peer_context = ctx

        def get_identity_name(self):
            return self.identity

        def get_prop(self, key):
            return self.props.get(key)

        def set_prop(self, key, value, private, sticky):
            self.props[key] = value

    class FakeClientStatus:
        def __init__(self):
            self.last_report_time = None
            self.num_reports = 0
            self.status = None
            self.last_progress_time = None

    class FakeTrainerStatus:
        def __init__(self, name):
            self.name = name
            self.reply_time = None
            self.busy = False

    class FakeGatherer:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.trainers = list(kwargs.get("trainers", []))
            self.trainer_statuses = {t: FakeTrainerStatus(t) for t in self.trainers}
            self.min_responses_required = kwargs.get("min_responses_required", len(self.trainers))
            self.for_round = kwargs.get("for_round")
            self.lock = threading.Lock()

        def is_done(self):
            return "base-is-done"

    class FakeSwarmClientController:
        def __init__(self, *args, learn_task_ack_timeout=30, learn_task_timeout=None, **kwargs):
            self.learn_task_ack_timeout = learn_task_ack_timeout
            self.learn_task_timeout = learn_task_timeout
            self.do_learn_task_name = "swarm_learn"
            self.learn_task = None
            self.allow_busy_task = True
            self.asked_to_stop = False

        def do_learn_task(self, name, task_data, fl_ctx, abort_signal):
            self.base_seen_fl_ctx = fl_ctx
            return fl_ctx.get_engine()

        def start_run(self, fl_ctx):
            self.base_start_run_fl_ctx = fl_ctx

        def _scatter(self, task_data, for_round, fl_ctx):
            self.base_seen_fl_ctx = fl_ctx
            return fl_ctx.get_engine()

        def process_config(self, fl_ctx):
            self.base_process_config_called = True
            return None

        def broadcast_and_wait(self, task, fl_ctx, targets=None, min_responses=1,
                               wait_time_after_min_received=0, abort_signal=None):
            self.base_sends = getattr(self, "base_sends", []) + [(task, list(targets or []))]
            reply = FakeShareable()
            reply["return_code"] = getattr(self, "base_send_rc", "OK")
            return {t: reply for t in (targets or [])}

        def is_task_secure(self, fl_ctx):
            return False

        def get_config_prop(self, name, default=None):
            if not getattr(self, "config", None):
                return default
            return self.config.get(name, default)

    class FakeSwarmServerController:
        def __init__(self, *args, **kwargs):
            self.init_kwargs = kwargs

        def handle_event(self, event_type, fl_ctx):
            self.base_events = getattr(self, "base_events", []) + [event_type]

        def _configure_clients(self, learn_config, fl_ctx, abort_signal):
            return getattr(self, "base_configure_result", True)

        def _is_configured(self, client_name):
            return client_name in getattr(self, "configured", set(self.client_statuses))

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
        "nvflare.apis.event_type": types.ModuleType("nvflare.apis.event_type"),
        "nvflare.apis.fl_constant": types.ModuleType("nvflare.apis.fl_constant"),
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
        "nvflare.security": types.ModuleType("nvflare.security"),
        "nvflare.security.logging": types.ModuleType("nvflare.security.logging"),
        "nvflare.apis.controller_spec": types.ModuleType("nvflare.apis.controller_spec"),
    }

    class FakeTask:
        def __init__(self, name, data, timeout=0, secure=False, **kwargs):
            self.name = name
            self.data = data
            self.timeout = timeout
            self.secure = secure

    modules["nvflare.apis.controller_spec"].Task = FakeTask
    modules["nvflare.apis.fl_constant"].ReservedKey = SimpleNamespace(
        ENGINE=engine_key,
        RC="__rc__",
        TASK_NAME="__task_name__",
    )
    modules["nvflare.apis.fl_constant"].ReservedTopic = SimpleNamespace(DO_TASK="__do_task__")
    modules["nvflare.apis.fl_constant"].FLContextKey = SimpleNamespace(
        DISCONNECTED_CLIENT_NAME="__disconnected_client__",
        RECONNECTED_CLIENT_NAME="__reconnected_client__",
        WORKFLOW="__workflow__",
    )
    modules["nvflare.apis.event_type"].EventType = SimpleNamespace(
        CLIENT_DISCONNECTED="_client_disconnected",
        CLIENT_RECONNECTED="_client_reconnected",
    )
    modules["nvflare.apis.fl_context"].FLContext = FakeFLContext
    modules["nvflare.apis.shareable"].Shareable = FakeShareable
    modules["nvflare.apis.shareable"].ReturnCode = SimpleNamespace(
        OK="OK",
        ERROR="ERROR",
        EXECUTION_EXCEPTION="EXECUTION_EXCEPTION",
        MODEL_UNRECOGNIZED="MODEL_UNRECOGNIZED",
        SERVICE_UNAVAILABLE="SERVICE_UNAVAILABLE",
        TIMEOUT="TIMEOUT",
    )
    def make_reply(rc):
        reply = FakeShareable()
        reply["return_code"] = rc
        return reply

    modules["nvflare.apis.shareable"].make_reply = make_reply
    modules["nvflare.app_common.app_constant"].AppConstants = SimpleNamespace(
        CURRENT_ROUND="current_round",
        TRAINING_RESULT="training_result",
        AGGREGATION_ACCEPTED="aggregation_accepted",
        NUM_ROUNDS="num_rounds",
    )
    modules["nvflare.app_common.app_event_type"].AppEventType = SimpleNamespace(
        BEFORE_CONTRIBUTION_ACCEPT="before_contribution_accept",
        AFTER_CONTRIBUTION_ACCEPT="after_contribution_accept",
    )
    modules["nvflare.app_common.ccwf.common"].Constant = SimpleNamespace(
        STATUS_REPORTS="status_reports",
        CLIENTS="clients",
        TRAIN_CLIENTS="train_clients",
        AGGR_CLIENTS="aggr_clients",
        RESULT_CLIENTS="result_clients",
        AGGREGATOR="aggregator",
    )
    modules["nvflare.app_common.ccwf.common"].ResultType = SimpleNamespace(BEST="best", LAST="last")
    modules["nvflare.app_common.ccwf.common"].status_report_from_dict = status_report_from_dict
    modules["nvflare.security.logging"].secure_format_traceback = lambda *args, **kwargs: ""
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


def test_latest_global_save_is_mirrored_for_future_continue(warm_continue, tmp_path):
    mirror = tmp_path / "scratch" / "mediswarm_latest_global.pt"
    run_ckpt = tmp_path / "run" / "FL_global_model.pt"
    persistor = warm_continue.WarmStartablePTFileModelPersistor(latest_global_path=str(mirror))
    persistor._ckpt_save_path = str(run_ckpt)

    persistor.save_model(ml=object(), fl_ctx=SimpleNamespace())

    assert mirror.read_bytes() == b"latest global"
    assert ("info", f"WarmStart: mirrored latest global -> {mirror}") in persistor.logger.messages


def test_best_global_event_is_still_mirrored(warm_continue, tmp_path):
    mirror = tmp_path / "scratch" / "mediswarm_latest_global.pt"
    best_ckpt = tmp_path / "run" / "best_FL_global_model.pt"
    persistor = warm_continue.WarmStartablePTFileModelPersistor(latest_global_path=str(mirror))
    persistor._best_ckpt_save_path = str(best_ckpt)

    persistor.handle_event("GLOBAL_BEST_MODEL_AVAILABLE", SimpleNamespace())

    assert mirror.read_bytes() == b"best global"
    assert ("info", f"WarmStart: mirrored best global -> {mirror}") in persistor.logger.messages


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
    controller.log_error = lambda *args, **kwargs: None
    controller.prune_notify_timeout = 5.0
    controller.pruned_clients = []
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


class _SequenceEngine:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.forwarded_attribute = "forwarded"

    def send_aux_request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.responses.pop(0)


def _make_permission_retry_adapter(module, responses):
    warnings = []
    controller = SimpleNamespace(
        request_to_submit_learn_result_task_name="request_submit",
        log_warning=lambda fl_ctx, message: warnings.append((fl_ctx, message)),
        aggregator_replacement=lambda name: None,
        note_permission_granted_by=lambda aggr: None,
        me="me",
    )
    engine = _SequenceEngine(responses)
    return module._PermissionReplyRetryEngine(engine, controller), engine, warnings


def test_missing_permission_reply_becomes_retryable_service_unavailable(fault_tolerant_ccwf):
    module, _ = fault_tolerant_ccwf
    adapter, engine, warnings = _make_permission_retry_adapter(
        module,
        responses=[
            {},
            {"MHA_1": module.make_reply(module.ReturnCode.OK)},
        ],
    )

    first = adapter.send_aux_request(
        targets=["MHA_1"],
        topic="request_submit",
        request={},
        timeout=60,
        fl_ctx="ctx",
        secure=False,
    )
    second = adapter.send_aux_request(
        targets=["MHA_1"],
        topic="request_submit",
        request={},
        timeout=60,
        fl_ctx="ctx",
        secure=False,
    )

    assert first["MHA_1"]["return_code"] == module.ReturnCode.SERVICE_UNAVAILABLE
    assert second["MHA_1"]["return_code"] == module.ReturnCode.OK
    assert len(engine.calls) == 2
    assert len(warnings) == 1
    assert "treating it as transient and retrying" in warnings[0][1]
    assert adapter.forwarded_attribute == "forwarded"


def test_explicit_permission_rejection_is_not_rewritten(fault_tolerant_ccwf):
    module, _ = fault_tolerant_ccwf
    rejected = {"MHA_1": module.make_reply(module.ReturnCode.MODEL_UNRECOGNIZED)}
    adapter, _, warnings = _make_permission_retry_adapter(module, responses=[rejected])

    actual = adapter.send_aux_request(
        targets=["MHA_1"],
        topic="request_submit",
        request={},
        timeout=60,
        fl_ctx="ctx",
        secure=False,
    )

    assert actual is rejected
    assert warnings == []


def test_explicit_falsey_permission_rejection_is_not_rewritten(fault_tolerant_ccwf):
    module, _ = fault_tolerant_ccwf

    class FalseyReply:
        def __bool__(self):
            return False

        def get_return_code(self, default=None):
            return module.ReturnCode.MODEL_UNRECOGNIZED

    explicit_reply = FalseyReply()
    rejected = {"MHA_1": explicit_reply}
    adapter, _, warnings = _make_permission_retry_adapter(module, responses=[rejected])

    actual = adapter.send_aux_request(
        targets=["MHA_1"],
        topic="request_submit",
        request={},
        timeout=60,
        fl_ctx="ctx",
        secure=False,
    )

    assert actual is rejected
    assert actual["MHA_1"] is explicit_reply
    assert warnings == []


def test_retry_adapter_is_context_local_for_learning_task(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    delegate = _SequenceEngine([])
    original_fl_ctx = fl_context_cls(engine=delegate)
    controller = module.FaultTolerantSwarmClientController.__new__(
        module.FaultTolerantSwarmClientController
    )
    controller.request_to_submit_learn_result_task_name = "request_submit"
    controller.log_warning = lambda *args, **kwargs: None
    controller._round_lock = threading.Lock()
    controller._round = {}
    controller._aggr_replacement = {}
    task_data = module.Shareable()
    task_data.set_header(module.AppConstants.CURRENT_ROUND, 0)
    task_data.set_header(module.Constant.AGGREGATOR, "site9")

    seen_engine = controller.do_learn_task(
        name="train",
        task_data=task_data,
        fl_ctx=original_fl_ctx,
        abort_signal=SimpleNamespace(triggered=False),
    )

    assert isinstance(seen_engine, module._PermissionReplyRetryEngine)
    assert seen_engine._engine is delegate
    assert controller.base_seen_fl_ctx is not original_fl_ctx
    assert original_fl_ctx.get_engine() is delegate
    assert controller._round["num"] == 0 and controller._round["aggr"] == "site9"
    assert controller._round["task_data"] is task_data and controller._round["done"] is False


def test_start_run_installs_fault_tolerant_gatherer_used_by_inherited_controller(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller = module.FaultTolerantSwarmClientController()
    controller.log_info = lambda *args: None
    fl_ctx = fl_context_cls()

    controller.start_run(fl_ctx)

    swarm_module = sys.modules["nvflare.app_common.ccwf.swarm_client_ctl"]
    assert swarm_module.Gatherer is module.FaultTolerantGatherer
    assert controller.base_start_run_fl_ctx is fl_ctx


def test_learn_scatter_retries_only_missing_none_and_timeout(fault_tolerant_ccwf):
    module, _ = fault_tolerant_ccwf
    ok_a = module.make_reply(module.ReturnCode.OK)
    timeout_b = module.make_reply(module.ReturnCode.TIMEOUT)
    ok_b = module.make_reply(module.ReturnCode.OK)
    error_c = module.make_reply(module.ReturnCode.ERROR)
    engine = _SequenceEngine(
        [
            {"A": ok_a, "B": timeout_b, "C": None},
            {"B": ok_b, "C": error_c},
        ]
    )
    warnings = []
    controller = SimpleNamespace(
        do_learn_task_name="swarm_learn",
        asked_to_stop=False,
        log_warning=lambda fl_ctx, message: warnings.append(message),
    )
    request = module.make_reply(module.ReturnCode.OK)
    request.set_header(module.ReservedKey.TASK_NAME, "swarm_learn")
    request.set_header(module.AppConstants.CURRENT_ROUND, 3)
    adapter = module._LearnScatterRetryEngine(
        engine=engine,
        controller=controller,
        deadline=module.time.time() + 1,
        attempt_timeout=0.1,
        retry_interval=0.001,
    )

    actual = adapter.send_aux_request(
        targets=["A", "B", "C"],
        topic=module.ReservedTopic.DO_TASK,
        request=request,
        timeout=30,
        fl_ctx="ctx",
        secure=False,
    )

    assert engine.calls[0][1]["targets"] == ["A", "B", "C"]
    assert engine.calls[1][1]["targets"] == ["B", "C"]
    assert actual == {"A": ok_a, "B": ok_b, "C": error_c}
    assert len(warnings) == 1
    assert "retrying only those clients" in warnings[0]


def test_learn_scatter_never_retries_explicit_terminal_errors(fault_tolerant_ccwf):
    module, _ = fault_tolerant_ccwf
    execution_error = module.make_reply(module.ReturnCode.EXECUTION_EXCEPTION)
    engine = _SequenceEngine([{"UKA_1": execution_error}])
    controller = SimpleNamespace(
        do_learn_task_name="swarm_learn",
        asked_to_stop=False,
        log_warning=lambda *args: None,
    )
    request = module.make_reply(module.ReturnCode.OK)
    request.set_header(module.ReservedKey.TASK_NAME, "swarm_learn")
    adapter = module._LearnScatterRetryEngine(
        engine=engine,
        controller=controller,
        deadline=module.time.time() + 1,
        attempt_timeout=0.1,
        retry_interval=0.001,
    )

    actual = adapter.send_aux_request(
        targets=["UKA_1"],
        topic=module.ReservedTopic.DO_TASK,
        request=request,
        timeout=30,
        fl_ctx="ctx",
        secure=False,
    )

    assert actual == {"UKA_1": execution_error}
    assert len(engine.calls) == 1


def test_learn_scatter_stops_retrying_when_run_is_aborted(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    abort_signal = SimpleNamespace(triggered=False)

    class AbortAfterFirstEngine(_SequenceEngine):
        def send_aux_request(self, *args, **kwargs):
            response = super().send_aux_request(*args, **kwargs)
            abort_signal.triggered = True
            return response

    engine = AbortAfterFirstEngine([{}])
    controller = SimpleNamespace(
        do_learn_task_name="swarm_learn",
        asked_to_stop=False,
        log_warning=lambda *args: None,
    )
    request = module.make_reply(module.ReturnCode.OK)
    request.set_header(module.ReservedKey.TASK_NAME, "swarm_learn")
    adapter = module._LearnScatterRetryEngine(
        engine=engine,
        controller=controller,
        deadline=module.time.time() + 86400,
        attempt_timeout=3600,
        retry_interval=5,
    )

    actual = adapter.send_aux_request(
        targets=["UKA_1"],
        topic=module.ReservedTopic.DO_TASK,
        request=request,
        timeout=86400,
        fl_ctx=fl_context_cls(run_abort_signal=abort_signal),
        secure=False,
    )

    assert actual == {}
    assert len(engine.calls) == 1


def test_scatter_timeout_defaults_to_stock_ack_timeout(fault_tolerant_ccwf):
    module, _ = fault_tolerant_ccwf

    controller = module.FaultTolerantSwarmClientController(
        learn_task_ack_timeout=37,
        learn_task_timeout=90,
    )

    assert controller.learn_task_scatter_attempt_timeout == 37
    assert controller.learn_task_scatter_retry_interval == 5.0


def _return_code(reply, default=None):
    return reply.get_return_code(default)


def test_duplicate_learn_request_is_acknowledged_without_requeue(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller = module.FaultTolerantSwarmClientController()
    controller.log_info = lambda *args: None
    controller.log_error = lambda *args: None
    controller.update_status = lambda **kwargs: None
    queued = []
    controller.set_learn_task = lambda task_data, fl_ctx: queued.append(task_data) or True
    fl_ctx = fl_context_cls(peer_context=fl_context_cls(identity="USZ_1"))
    request = module.make_reply(module.ReturnCode.OK)
    request.set_header(module.AppConstants.CURRENT_ROUND, 4)

    first = controller._try_process_learn_request(request, fl_ctx)
    duplicate = controller._try_process_learn_request(request, fl_ctx)

    assert _return_code(first) == module.ReturnCode.OK
    assert _return_code(duplicate) == module.ReturnCode.OK
    assert queued == [request]
    assert controller._last_accepted_learn_round == 4


def test_older_learn_request_is_rejected_without_requeue(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller = module.FaultTolerantSwarmClientController()
    controller.log_info = lambda *args: None
    controller.log_error = lambda *args: None
    controller.update_status = lambda **kwargs: None
    queued = []
    controller.set_learn_task = lambda task_data, fl_ctx: queued.append(task_data) or True
    fl_ctx = fl_context_cls(peer_context=fl_context_cls(identity="USZ_1"))
    current = module.make_reply(module.ReturnCode.OK)
    current.set_header(module.AppConstants.CURRENT_ROUND, 5)
    stale = module.make_reply(module.ReturnCode.OK)
    stale.set_header(module.AppConstants.CURRENT_ROUND, 4)

    controller._try_process_learn_request(current, fl_ctx)
    rejected = controller._try_process_learn_request(stale, fl_ctx)

    assert _return_code(rejected) == module.ReturnCode.MODEL_UNRECOGNIZED
    assert queued == [current]


def _make_strict_gatherer(module, fl_ctx, accepted=True):
    updates = []
    gatherer = module.FaultTolerantGatherer.__new__(module.FaultTolerantGatherer)
    gatherer.for_round = 2
    gatherer.min_responses_required = 2
    gatherer.trainer_statuses = {
        "A": SimpleNamespace(reply_time=1),
        "B": SimpleNamespace(reply_time=None),
    }
    gatherer.min_resps_received_time = None
    gatherer.timeout = None
    gatherer.start_time = module.time.time()
    gatherer.executor = SimpleNamespace(update_status=lambda **kwargs: updates.append(kwargs))
    gatherer.aggregator = SimpleNamespace(accept=lambda result, ctx: accepted)
    gatherer.log_error = lambda *args: None
    gatherer.log_warning = lambda *args: None
    gatherer.log_info = lambda *args: None
    gatherer.fire_event = lambda *args: None
    gatherer.fl_ctx = fl_ctx
    return gatherer, updates


def test_strict_gather_bad_result_reports_same_error_and_does_not_count(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    gatherer, updates = _make_strict_gatherer(module, fl_context_cls())
    result = module.make_reply(module.ReturnCode.ERROR)
    result.set_header(module.AppConstants.CURRENT_ROUND, 2)

    reply = gatherer._do_gather("B", result, gatherer.fl_ctx)

    assert _return_code(reply) == module.ReturnCode.ERROR
    assert gatherer.trainer_statuses["B"].reply_time is None
    assert updates == [{"action": "gather", "error": module.ReturnCode.ERROR}]


def test_strict_gather_rejected_contribution_is_uncounted(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    gatherer, updates = _make_strict_gatherer(module, fl_context_cls(), accepted=False)
    events = []
    gatherer.fire_event = lambda event_type, fl_ctx: events.append(event_type)
    result = module.make_reply(module.ReturnCode.OK)
    result.set_header(module.AppConstants.CURRENT_ROUND, 2)

    reply = gatherer._do_gather("B", result, gatherer.fl_ctx)

    assert _return_code(reply) == module.ReturnCode.EXECUTION_EXCEPTION
    assert gatherer.trainer_statuses["B"].reply_time is None
    assert gatherer.min_resps_received_time is None
    assert updates == [{"action": "gather", "error": module.ReturnCode.EXECUTION_EXCEPTION}]
    assert events == [
        module.AppEventType.BEFORE_CONTRIBUTION_ACCEPT,
        module.AppEventType.AFTER_CONTRIBUTION_ACCEPT,
    ]
    assert gatherer.fl_ctx.get_prop(module.AppConstants.AGGREGATION_ACCEPTED) is False


def test_strict_gather_counts_final_response_only_after_accept_finishes(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    gatherer, updates = _make_strict_gatherer(module, fl_context_cls())
    accept_started = module.threading.Event()
    allow_accept_to_finish = module.threading.Event()
    replies = []

    def accept(result, fl_ctx):
        accept_started.set()
        assert allow_accept_to_finish.wait(timeout=1)
        return True

    gatherer.aggregator = SimpleNamespace(accept=accept)
    result = module.make_reply(module.ReturnCode.OK)
    result.set_header(module.AppConstants.CURRENT_ROUND, 2)

    gather_thread = module.threading.Thread(
        target=lambda: replies.append(gatherer._do_gather("B", result, gatherer.fl_ctx))
    )
    gather_thread.start()
    assert accept_started.wait(timeout=1)

    assert gatherer.is_done() is False
    assert gatherer.trainer_statuses["B"].reply_time is None

    allow_accept_to_finish.set()
    gather_thread.join(timeout=1)
    assert not gather_thread.is_alive()
    assert [_return_code(reply) for reply in replies] == [module.ReturnCode.OK]
    assert gatherer.trainer_statuses["B"].reply_time is not None
    assert gatherer.is_done() is True
    assert updates == []


def test_strict_gather_accept_exception_does_not_count_response(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    gatherer, _ = _make_strict_gatherer(module, fl_context_cls())
    gatherer.aggregator = SimpleNamespace(accept=lambda result, fl_ctx: (_ for _ in ()).throw(RuntimeError("boom")))
    result = module.make_reply(module.ReturnCode.OK)
    result.set_header(module.AppConstants.CURRENT_ROUND, 2)

    with pytest.raises(RuntimeError, match="boom"):
        gatherer._do_gather("B", result, gatherer.fl_ctx)

    assert gatherer.trainer_statuses["B"].reply_time is None
    assert gatherer.is_done() is False


def test_strict_gather_after_accept_event_exception_does_not_count_response(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    gatherer, _ = _make_strict_gatherer(module, fl_context_cls())

    def fire_event(event_type, fl_ctx):
        if event_type == module.AppEventType.AFTER_CONTRIBUTION_ACCEPT:
            raise RuntimeError("after hook failed")

    gatherer.fire_event = fire_event
    result = module.make_reply(module.ReturnCode.OK)
    result.set_header(module.AppConstants.CURRENT_ROUND, 2)

    with pytest.raises(RuntimeError, match="after hook failed"):
        gatherer._do_gather("B", result, gatherer.fl_ctx)

    assert gatherer.trainer_statuses["B"].reply_time is None
    assert gatherer.is_done() is False


def test_strict_gather_timeout_never_finishes_partial_aggregation(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    gatherer, updates = _make_strict_gatherer(module, fl_context_cls())
    gatherer.start_time = module.time.time() - 10
    gatherer.timeout = 1

    assert gatherer.is_done() is False
    assert gatherer.is_done() is False
    assert updates == [{"action": "gather_timeout", "error": module.ReturnCode.TIMEOUT}]


@pytest.mark.parametrize(
    "job_name",
    [
        "challenge_1DivideAndConquer",
        "challenge_2BCN_AIM",
        "challenge_3agaldran",
        "challenge_4abmil",
        "challenge_5pimed",
        "ODELIA_ternary_classification",
    ],
)
def test_production_swarm_client_configs_keep_result_refs_and_control_retries_alive(job_name):
    client_config = (
        REPO_ROOT / "application" / "jobs" / job_name / "app" / "config" / "config_fed_client.conf"
    ).read_text()

    assert 'path = "fault_tolerant_ccwf.FaultTolerantSwarmClientController"' in client_config
    assert "last_result_transfer_timeout = 86400" in client_config
    assert "download_complete_timeout = 86400" in client_config
    assert "learn_task_scatter_attempt_timeout = 3600" in client_config
    assert "learn_task_scatter_retry_interval = 5" in client_config
    assert "request_to_submit_result_msg_timeout = 60" in client_config
    assert "request_to_submit_result_interval = 5" in client_config
    assert "max_concurrent_submissions = 1" in client_config
    # Reusable templates must not bake in the current ODELIA deployment's site count.
    assert "min_responses_required = 5" in client_config


# --- structural guard (#545 follow-up): parameter names, independent of the sidecar ---

def _two_disjoint_models():
    import torch
    a = torch.nn.Module(); a.enc = torch.nn.Linear(3, 2)
    b = torch.nn.Module(); b.head = torch.nn.Linear(3, 2)
    return a, b


def test_unlabelled_checkpoint_of_another_architecture_is_refused(warm_continue, tmp_path, monkeypatch):
    """The 2026-09-11 case: no sidecar, disjoint parameter names -> refuse."""
    import torch
    written_by, running = _two_disjoint_models()
    ckpt = tmp_path / "mediswarm_latest_global.pt"
    torch.save({"model": written_by.state_dict(),
                "train_conf": {"train": {"model": "ResidualEncoderClsLightning"}}}, ckpt)
    monkeypatch.setenv("MODEL_NAME", "MST")

    persistor = warm_continue.WarmStartablePTFileModelPersistor(
        warm_start_mode="auto", source_ckpt_file_full_name=str(ckpt))
    persistor.model = running

    assert persistor.load_model(SimpleNamespace(get_prop=lambda key: None)) is None
    assert len(persistor.panics) == 1
    msg = persistor.panics[0]
    assert "WARM_START_MODEL_MISMATCH" in msg
    assert "shares no parameter names" in msg
    assert "ResidualEncoderClsLightning" in msg      # the label read out of the file (E2)
    assert "enc.weight" in msg and "head.weight" in msg


def test_unlabelled_checkpoint_of_the_same_architecture_still_loads(warm_continue, tmp_path, monkeypatch):
    """Pre-provenance mirrors of the right model must keep working -- with a warning."""
    import torch
    written_by, _ = _two_disjoint_models()
    running = torch.nn.Module(); running.enc = torch.nn.Linear(3, 2)   # same names
    ckpt = tmp_path / "latest.pt"
    torch.save({"model": written_by.state_dict()}, ckpt)
    monkeypatch.setenv("MODEL_NAME", "MST")

    persistor = warm_continue.WarmStartablePTFileModelPersistor(
        warm_start_mode="auto", source_ckpt_file_full_name=str(ckpt))
    persistor.model = running

    assert persistor.load_model(SimpleNamespace(get_prop=lambda key: None)) == "loaded"
    assert persistor.panics == []
    kinds = [k for k, _ in persistor.logger.messages]
    assert "warning" in kinds                                     # "carries no provenance"
    assert any("structure OK" in m for _, m in persistor.logger.messages)


def test_structural_check_catches_a_sidecar_that_lies(warm_continue, tmp_path, monkeypatch):
    """A sidecar naming the right model does not excuse disjoint parameters."""
    import torch
    written_by, running = _two_disjoint_models()
    ckpt = tmp_path / "latest.pt"
    torch.save({"model": written_by.state_dict()}, ckpt)
    warm_continue.write_provenance(str(ckpt), "MST", "job-1", "deadbeef")
    monkeypatch.setenv("MODEL_NAME", "MST")

    persistor = warm_continue.WarmStartablePTFileModelPersistor(
        warm_start_mode="auto", source_ckpt_file_full_name=str(ckpt))
    persistor.model = running

    assert persistor.load_model(SimpleNamespace(get_prop=lambda key: None)) is None
    assert len(persistor.panics) == 1 and "shares no parameter names" in persistor.panics[0]


def test_structural_check_is_skipped_when_no_model_is_built(warm_continue, tmp_path):
    """Not a torch file and no model attribute: old behaviour, nothing crashes."""
    ckpt = tmp_path / "latest.pt"
    ckpt.write_bytes(b"not a checkpoint")
    persistor = warm_continue.WarmStartablePTFileModelPersistor(
        warm_start_mode="auto", source_ckpt_file_full_name=str(ckpt))
    assert persistor.load_model(SimpleNamespace(get_prop=lambda key: None)) == "loaded"
    assert persistor.panics == []


def test_checkpoint_keys_reads_a_string_train_conf(warm_continue, tmp_path):
    import torch
    m = torch.nn.Module(); m.x = torch.nn.Linear(2, 1)
    ckpt = tmp_path / "c.pt"
    torch.save({"model": m.state_dict(), "train_conf": "{'train': {'model': 'MST'}}"}, ckpt)
    keys, label = warm_continue.checkpoint_keys(str(ckpt))
    assert keys == {"x.weight", "x.bias"} and label == "MST"


# ---------------------------------------------------------------------------
# #595: pruning is announced to the surviving clients, who act on it
# ---------------------------------------------------------------------------


class _RecordingEngine:
    """Records send_aux_request calls; answers OK for every target unless told otherwise."""

    def __init__(self, silent=()):
        self.calls = []
        self.silent = set(silent)

    def send_aux_request(self, targets, topic, request, timeout, fl_ctx, secure=False):
        self.calls.append({"targets": list(targets), "topic": topic, "request": dict(request), "timeout": timeout})
        replies = {}
        for t in targets:
            if t in self.silent:
                continue
            reply = type(request)()
            reply["return_code"] = "OK"
            replies[t] = reply
        return replies


def _make_server(module, fl_context_cls, active=("site1", "site2", "site3", "site4"), min_clients=3, silent=()):
    engine = _RecordingEngine(silent=silent)
    fl_ctx = fl_context_cls(engine=engine)
    controller = module.FaultTolerantSwarmServerController.__new__(module.FaultTolerantSwarmServerController)
    controller.workflow_id = "wf"
    controller.min_clients = min_clients
    controller.asked_to_stop = False
    controller.prune_notify_timeout = 5.0
    controller.pruned_clients = []
    controller.client_statuses = {name: module.ClientStatus() for name in active}
    logs = []
    controller.log_debug = lambda ctx, msg: None
    controller.log_info = lambda ctx, msg: logs.append(("info", msg))
    controller.log_warning = lambda ctx, msg: logs.append(("warning", msg))
    controller.log_error = lambda ctx, msg: logs.append(("error", msg))
    panics = []
    controller.system_panic = lambda message, ctx: panics.append(message)
    return controller, fl_ctx, engine, logs, panics


def test_error_report_prune_notifies_survivors(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, panics = _make_controller_with_report(module, fl_context_cls, "MODEL_UNRECOGNIZED")
    engine = _RecordingEngine()
    fl_ctx.engine = engine

    controller._update_client_status(fl_ctx)

    assert panics == []
    assert "site1" not in controller.client_statuses
    assert controller.pruned_clients == ["site1"]
    assert len(engine.calls) == 1
    call = engine.calls[0]
    assert call["topic"] == module.prune_topic("wf")
    assert sorted(call["targets"]) == ["site2", "site3"]
    assert call["request"][module.PRUNE_KEY_PRUNED] == ["site1"]
    assert sorted(call["request"][module.PRUNE_KEY_ACTIVE]) == ["site2", "site3"]
    assert "MODEL_UNRECOGNIZED" in call["request"][module.PRUNE_KEY_REASON]


def test_disconnect_prunes_and_notifies_when_min_clients_remain(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, engine, logs, panics = _make_server(module, fl_context_cls)
    fl_ctx.set_prop(module.FLContextKey.DISCONNECTED_CLIENT_NAME, "site4", private=True, sticky=False)

    controller.handle_event(module.EventType.CLIENT_DISCONNECTED, fl_ctx)

    assert controller.base_events == [module.EventType.CLIENT_DISCONNECTED]  # stock handling still ran
    assert sorted(controller.client_statuses) == ["site1", "site2", "site3"]
    assert panics == []
    assert engine.calls[0]["request"][module.PRUNE_KEY_PRUNED] == ["site4"]
    assert sorted(engine.calls[0]["targets"]) == ["site1", "site2", "site3"]
    assert any("acknowledged by all active clients" in msg for level, msg in logs if level == "info")


def test_disconnect_in_strict_mode_is_logged_not_pruned(fault_tolerant_ccwf):
    """A strict run waits for a site that lost its VPN: it keeps training and submits later."""
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, engine, logs, panics = _make_server(module, fl_context_cls, min_clients=4)
    fl_ctx.set_prop(module.FLContextKey.DISCONNECTED_CLIENT_NAME, "site4", private=True, sticky=False)

    controller.handle_event(module.EventType.CLIENT_DISCONNECTED, fl_ctx)

    assert sorted(controller.client_statuses) == ["site1", "site2", "site3", "site4"]
    assert engine.calls == []
    assert panics == []
    assert any("not pruned" in msg for level, msg in logs if level == "warning")


def test_disconnect_of_already_pruned_or_unknown_client_is_ignored(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, engine, logs, panics = _make_server(module, fl_context_cls)
    fl_ctx.set_prop(module.FLContextKey.DISCONNECTED_CLIENT_NAME, "nobody", private=True, sticky=False)

    controller.handle_event(module.EventType.CLIENT_DISCONNECTED, fl_ctx)

    assert len(controller.client_statuses) == 4
    assert engine.calls == []


def test_unconfigured_clients_are_pruned_before_the_start_task(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, engine, logs, panics = _make_server(module, fl_context_cls)
    controller.configured = {"site1", "site2", "site4"}  # site3's worker died at launch

    assert controller._configure_clients({}, fl_ctx, None) is True

    assert sorted(controller.client_statuses) == ["site1", "site2", "site4"]
    assert controller.pruned_clients == ["site3"]
    assert engine.calls[0]["request"][module.PRUNE_KEY_PRUNED] == ["site3"]
    assert "not configured" in engine.calls[0]["request"][module.PRUNE_KEY_REASON]
    assert sorted(engine.calls[0]["targets"]) == ["site1", "site2", "site4"]


def test_configure_failure_is_passed_through(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, engine, logs, panics = _make_server(module, fl_context_cls)
    controller.base_configure_result = False

    assert controller._configure_clients({}, fl_ctx, None) is False
    assert engine.calls == []


def test_prune_notice_names_clients_that_did_not_acknowledge(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, engine, logs, panics = _make_server(module, fl_context_cls, silent=("site2",))

    controller._prune(["site4"], "test", fl_ctx)

    warnings = [msg for level, msg in logs if level == "warning"]
    assert any("no acknowledgement from ['site2']" in msg for msg in warnings)


def test_reconnected_pruned_client_stays_pruned(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, engine, logs, panics = _make_server(module, fl_context_cls)
    controller._prune(["site4"], "test", fl_ctx)
    fl_ctx.set_prop(module.FLContextKey.RECONNECTED_CLIENT_NAME, "site4", private=True, sticky=False)

    controller.handle_event(module.EventType.CLIENT_RECONNECTED, fl_ctx)

    assert "site4" not in controller.client_statuses
    assert any("stays pruned" in msg for level, msg in logs if level == "warning")


class _FakeTrainerStatus:
    def __init__(self):
        self.reply_time = None


def _make_gatherer(module, trainers, min_responses_required):
    g = module.FaultTolerantGatherer.__new__(module.FaultTolerantGatherer)
    g.trainers = list(trainers)
    g.trainer_statuses = {t: _FakeTrainerStatus() for t in trainers}
    g.min_responses_required = min_responses_required
    g.for_round = 2
    g.lock = threading.Lock()
    g.logs = []
    g.log_warning = lambda ctx, msg: g.logs.append(msg)
    return g


def test_gatherer_drops_pruned_trainers_that_have_not_replied(fault_tolerant_ccwf):
    module, _ = fault_tolerant_ccwf
    g = _make_gatherer(module, ["a", "b", "c", "d"], min_responses_required=3)
    g.trainer_statuses["a"].reply_time = 1.0

    dropped = g.drop_trainers({"d", "zzz"}, None)

    assert dropped == ["d"]
    assert sorted(g.trainer_statuses) == ["a", "b", "c"]
    assert g.trainers == ["a", "b", "c"]
    assert g.min_responses_required == 3
    assert g._all_responses_required() is True  # 3 of 3: strict among the survivors


def test_gatherer_keeps_a_pruned_trainer_whose_result_arrived(fault_tolerant_ccwf):
    module, _ = fault_tolerant_ccwf
    g = _make_gatherer(module, ["a", "b", "c"], min_responses_required=2)
    g.trainer_statuses["c"].reply_time = 1.0

    assert g.drop_trainers({"c"}, None) == []
    assert sorted(g.trainer_statuses) == ["a", "b", "c"]


def test_gatherer_lowers_min_responses_when_trainers_shrink_below_it(fault_tolerant_ccwf):
    module, _ = fault_tolerant_ccwf
    g = _make_gatherer(module, ["a", "b", "c", "d"], min_responses_required=4)

    g.drop_trainers({"d"}, None)

    assert g.min_responses_required == 3


def _make_client(module, fl_context_cls, me="site1"):
    controller = module.FaultTolerantSwarmClientController.__new__(module.FaultTolerantSwarmClientController)
    controller.me = me
    controller.workflow_id = "wf"
    controller.config = {
        module.Constant.CLIENTS: ["site1", "site2", "site3", "site4"],
        module.Constant.TRAIN_CLIENTS: ["site1", "site2", "site3", "site4"],
        module.Constant.AGGR_CLIENTS: ["site1", "site4"],
        module.Constant.RESULT_CLIENTS: ["site1", "site2", "site3", "site4"],
    }
    controller.trainers = ["site1", "site2", "site3", "site4"]
    controller.aggrs = ["site1", "site4"]
    controller.gatherer = None
    controller._round_lock = threading.Lock()
    controller._round = {}
    controller._aggr_replacement = {}
    controller.aggregator_death_grace = 0.3
    controller.aggregator_death_poll = 0.02
    controller.request_to_submit_learn_result_task_name = "request_submit"
    controller.report_learn_result_task_name = "report_result"
    controller.request_to_submit_result_max_wait = 0
    controller.request_to_submit_result_msg_timeout = 1
    controller.request_to_submit_result_interval = 0.01
    controller.learn_task_ack_timeout = 30
    controller.learn_task_timeout = 60
    controller.min_responses_required = 3
    controller.wait_time_after_min_resps_received = 5
    controller.max_concurrent_submissions = 1
    controller.metric_comparator = object()
    controller.aggregator = object()
    waiter_calls = []
    controller.gatherer_waiter = SimpleNamespace(set=lambda: waiter_calls.append(True))
    controller.waiter_calls = waiter_calls
    logs = []
    controller.log_info = lambda ctx, msg: logs.append(("info", msg))
    controller.log_warning = lambda ctx, msg: logs.append(("warning", msg))
    controller.log_error = lambda ctx, msg: logs.append(("error", msg))
    return controller, fl_context_cls(identity=me), logs


def test_client_applies_prune_to_every_list_and_the_live_gatherer(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, logs = _make_client(module, fl_context_cls)
    gatherer = _make_gatherer(module, ["site1", "site2", "site3", "site4"], min_responses_required=3)
    controller.gatherer = gatherer
    request = module.Shareable()
    request[module.PRUNE_KEY_PRUNED] = ["site4"]
    request[module.PRUNE_KEY_ACTIVE] = ["site1", "site2", "site3"]
    request[module.PRUNE_KEY_REASON] = "deemed disconnected by the server"

    reply = controller._process_prune_notice(module.prune_topic("wf"), request, fl_ctx)

    assert reply["return_code"] == "OK"
    assert controller.get_config_prop(module.Constant.TRAIN_CLIENTS) == ["site1", "site2", "site3"]
    assert controller.get_config_prop(module.Constant.AGGR_CLIENTS) == ["site1"]
    assert controller.get_config_prop(module.Constant.RESULT_CLIENTS) == ["site1", "site2", "site3"]
    assert controller.get_config_prop(module.Constant.CLIENTS) == ["site1", "site2", "site3"]
    assert controller.trainers == ["site1", "site2", "site3"]
    assert controller.aggrs == ["site1"]
    assert sorted(gatherer.trainer_statuses) == ["site1", "site2", "site3"]
    assert any("server pruned ['site4']" in msg for level, msg in logs if level == "warning")


def test_client_ignores_prune_before_configuration_but_still_acks(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, logs = _make_client(module, fl_context_cls)
    controller.config = None
    controller.trainers = None
    request = module.Shareable()
    request[module.PRUNE_KEY_PRUNED] = ["site4"]

    reply = controller._process_prune_notice("t", request, fl_ctx)

    assert reply["return_code"] == "OK"
    assert any("before configuration" in msg for level, msg in logs if level == "warning")


def test_client_pruned_itself_only_logs(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, logs = _make_client(module, fl_context_cls, me="site4")
    request = module.Shareable()
    request[module.PRUNE_KEY_PRUNED] = ["site4"]
    request[module.PRUNE_KEY_ACTIVE] = ["site1", "site2", "site3"]

    controller._process_prune_notice("t", request, fl_ctx)

    assert controller.trainers == ["site1", "site2", "site3", "site4"]  # untouched
    assert any("pruned THIS client" in msg for level, msg in logs if level == "error")


def test_client_registers_prune_handler_after_configuration(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, logs = _make_client(module, fl_context_cls)
    registered = []
    controller.engine = SimpleNamespace(
        register_aux_message_handler=lambda topic, message_handle_func: registered.append((topic, message_handle_func))
    )

    controller.process_config(fl_ctx)

    assert controller.base_process_config_called is True
    assert registered == [(module.prune_topic("wf"), controller._process_prune_notice)]


# ---------------------------------------------------------------------------
# #595: the pruned client was the round's aggregator
# ---------------------------------------------------------------------------


def _round_state(num=2, aggr="site4", **extra):
    rd = {"num": num, "aggr": aggr, "task_data": "task-data", "fl_ctx": None, "result": None,
          "submitted_to": None, "granted_by": None, "done": False}
    rd.update(extra)
    return rd


def _perm_request(module, round_num=2):
    req = module.Shareable()
    req.set_header(module.AppConstants.CURRENT_ROUND, round_num)
    return req


def test_permission_adapter_redirects_to_the_replacement_and_rekeys(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, logs = _make_client(module, fl_context_cls, me="site1")
    controller._round = _round_state()
    controller._aggr_replacement = {"site4": "site2"}
    engine = _RecordingEngine()
    adapter = module._PermissionReplyRetryEngine(engine, controller)

    resp = adapter.send_aux_request(targets=["site4"], topic="request_submit", request=_perm_request(module),
                                    timeout=1, fl_ctx=fl_ctx, secure=False)

    assert engine.calls[0]["targets"] == ["site2"]
    assert "site4" in resp and resp["site4"]["return_code"] == "OK"
    assert "site2" not in resp
    assert controller._round["granted_by"] == "site2"


def test_permission_adapter_answers_locally_when_this_client_took_over(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, logs = _make_client(module, fl_context_cls, me="site1")
    controller._round = _round_state()
    controller._aggr_replacement = {"site4": "site1"}
    engine = _RecordingEngine()
    fl_ctx.engine = SimpleNamespace(new_context=lambda: fl_context_cls(identity="site1"))
    seen = []

    def fake_submission_request(topic, request, ctx):
        seen.append((topic, ctx.get_peer_context().get_identity_name()))
        reply = module.Shareable()
        reply["return_code"] = "OK"
        return reply

    controller._process_submission_request = fake_submission_request
    adapter = module._PermissionReplyRetryEngine(engine, controller)

    resp = adapter.send_aux_request(targets=["site4"], topic="request_submit", request=_perm_request(module),
                                    timeout=1, fl_ctx=fl_ctx, secure=False)

    assert engine.calls == []
    assert seen == [("request_submit", "site1")]
    assert resp["site4"]["return_code"] == "OK"
    assert controller._round["granted_by"] == "site1"


def test_permission_adapter_records_who_granted_without_redirect(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, logs = _make_client(module, fl_context_cls, me="site1")
    controller._round = _round_state()
    adapter = module._PermissionReplyRetryEngine(_RecordingEngine(), controller)

    adapter.send_aux_request(targets=["site4"], topic="request_submit", request=_perm_request(module),
                             timeout=1, fl_ctx=fl_ctx, secure=False)

    assert controller._round["granted_by"] == "site4"


def test_pruned_aggregator_is_replaced_by_the_first_remaining_candidate(fault_tolerant_ccwf):
    """Every survivor computes the same replacement; only the elected one sets up a gatherer."""
    module, fl_context_cls = fault_tolerant_ccwf
    elected, fl_ctx, logs = _make_client(module, fl_context_cls, me="site1")
    elected._round = _round_state()
    other, fl_ctx2, logs2 = _make_client(module, fl_context_cls, me="site2")
    other._round = _round_state()
    request = module.Shareable()
    request[module.PRUNE_KEY_PRUNED] = ["site4"]
    request[module.PRUNE_KEY_ACTIVE] = ["site1", "site2", "site3"]
    request[module.PRUNE_KEY_REASON] = "deemed disconnected by the server"

    elected._process_prune_notice("t", request, fl_ctx)
    other._process_prune_notice("t", request, fl_ctx2)

    assert elected.aggregator_replacement("site4") == "site1"
    assert other.aggregator_replacement("site4") == "site1"
    g = elected.gatherer
    assert isinstance(g, module.FaultTolerantGatherer)
    assert g.for_round == 2 and g.trainers == ["site1", "site2", "site3"] and g.task_data == "task-data"
    assert elected.waiter_calls == [True]
    assert other.gatherer is None and other.waiter_calls == []
    assert any("takes over the round" in msg for level, msg in logs if level == "warning")


def test_prune_of_a_non_aggregator_does_not_touch_the_round(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, logs = _make_client(module, fl_context_cls, me="site1")
    controller._round = _round_state(aggr="site2")
    request = module.Shareable()
    request[module.PRUNE_KEY_PRUNED] = ["site4"]

    controller._process_prune_notice("t", request, fl_ctx)

    assert controller._aggr_replacement == {}
    assert controller.gatherer is None


def test_accepted_result_is_resubmitted_when_its_aggregator_dies(fault_tolerant_ccwf, monkeypatch):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, logs = _make_client(module, fl_context_cls, me="site2")
    controller._round = _round_state(done=True, submitted_to="site4", result="my-result")
    calls = []
    controller._submit_result_to = lambda aggr, rnd, result, ctx, need_permission: (
        calls.append((aggr, rnd, result, need_permission)) or {"return_code": "OK"})

    class SyncThread:
        def __init__(self, target, args=(), name=None, daemon=None):
            self._t, self._a = target, args

        def start(self):
            self._t(*self._a)

    monkeypatch.setattr(module.threading, "Thread", SyncThread)
    request = module.Shareable()
    request[module.PRUNE_KEY_PRUNED] = ["site4"]

    controller._process_prune_notice("t", request, fl_ctx)

    assert calls == [("site1", 2, "my-result", True)]
    assert controller._round["submitted_to"] == "site1"


def test_result_send_is_redirected_when_the_aggregator_was_replaced(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, logs = _make_client(module, fl_context_cls, me="site2")
    controller._round = _round_state(granted_by="site1")
    controller._aggr_replacement = {"site4": "site1"}
    calls = []
    ok = module.Shareable(); ok["return_code"] = "OK"
    controller._submit_result_to = lambda aggr, rnd, result, ctx, need_permission: (
        calls.append((aggr, rnd, result, need_permission)) or ok)
    task = module.Task(name="report_result", data="my-result", timeout=30)

    resp = controller.broadcast_and_wait(task, fl_ctx, ["site4"], 1)

    assert calls == [("site1", 2, "my-result", False)]  # permission already came from site1 via the adapter
    assert resp == {"site4": ok}
    assert controller._round["done"] is True and controller._round["submitted_to"] == "site1"


def test_failed_result_send_waits_for_the_prune_then_redirects(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, logs = _make_client(module, fl_context_cls, me="site2")
    controller._round = _round_state()
    controller.base_send_rc = "ERROR"  # the stock send to the dying aggregator fails
    ok = module.Shareable(); ok["return_code"] = "OK"
    calls = []
    controller._submit_result_to = lambda aggr, rnd, result, ctx, need_permission: (
        calls.append((aggr, need_permission)) or ok)
    # the server prunes the aggregator shortly after the failed send
    threading.Timer(0.05, lambda: controller._aggr_replacement.update({"site4": "site3"})).start()
    task = module.Task(name="report_result", data="my-result", timeout=30)

    resp = controller.broadcast_and_wait(task, fl_ctx, ["site4"], 1)

    assert controller.base_sends[0][1] == ["site4"]
    assert calls == [("site3", True)]
    assert resp == {"site4": ok}


def test_failed_result_send_without_a_prune_keeps_the_stock_error(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, logs = _make_client(module, fl_context_cls, me="site2")
    controller._round = _round_state()
    controller.base_send_rc = "ERROR"
    task = module.Task(name="report_result", data="my-result", timeout=30)

    resp = controller.broadcast_and_wait(task, fl_ctx, ["site4"], 1)

    assert resp["site4"]["return_code"] == "ERROR"
    assert controller._round["done"] is False


def test_other_tasks_pass_through_broadcast_and_wait(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, logs = _make_client(module, fl_context_cls, me="site2")
    task = module.Task(name="something_else", data="x")

    resp = controller.broadcast_and_wait(task, fl_ctx, ["site4"], 1)

    assert controller.base_sends[0] == (task, ["site4"])
    assert resp["site4"]["return_code"] == "OK"


def test_submit_result_to_self_asks_and_gathers_locally(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, logs = _make_client(module, fl_context_cls, me="site1")
    fl_ctx.engine = SimpleNamespace(new_context=lambda: fl_context_cls(identity="site1"))
    ok = module.Shareable(); ok["return_code"] = "OK"
    perms, gathers = [], []
    controller._process_submission_request = lambda topic, req, ctx: perms.append(topic) or ok
    controller._resolve_lazy_refs = lambda result, ctx: result
    controller._process_learn_result = lambda result, ctx, abort: gathers.append(result) or ok

    reply = controller._submit_result_to("site1", 2, "my-result", fl_ctx, need_permission=True)

    assert perms == ["request_submit"] and gathers == ["my-result"]
    assert reply is ok
    assert controller.base_sends == [] if hasattr(controller, "base_sends") else True


def test_submit_result_to_remote_asks_then_sends(fault_tolerant_ccwf):
    module, fl_context_cls = fault_tolerant_ccwf
    controller, fl_ctx, logs = _make_client(module, fl_context_cls, me="site2")
    engine = _RecordingEngine()
    fl_ctx.engine = engine

    reply = controller._submit_result_to("site3", 2, "my-result", fl_ctx, need_permission=True)

    assert engine.calls[0]["targets"] == ["site3"] and engine.calls[0]["topic"] == "request_submit"
    assert controller.base_sends[0][1] == ["site3"] and controller.base_sends[0][0].data == "my-result"
    assert reply["return_code"] == "OK"
