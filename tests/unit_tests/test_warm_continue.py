import importlib.util
import re
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

    class FakeGatherer:
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
    }
    modules["nvflare.apis.fl_constant"].ReservedKey = SimpleNamespace(
        ENGINE=engine_key,
        RC="__rc__",
        TASK_NAME="__task_name__",
    )
    modules["nvflare.apis.fl_constant"].ReservedTopic = SimpleNamespace(DO_TASK="__do_task__")
    modules["nvflare.apis.fl_context"].FLContext = FakeFLContext
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
    modules["nvflare.app_common.ccwf.common"].Constant = SimpleNamespace(STATUS_REPORTS="status_reports")
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

    seen_engine = controller.do_learn_task(
        name="train",
        task_data={},
        fl_ctx=original_fl_ctx,
        abort_signal=SimpleNamespace(triggered=False),
    )

    assert isinstance(seen_engine, module._PermissionReplyRetryEngine)
    assert seen_engine._engine is delegate
    assert controller.base_seen_fl_ctx is not original_fl_ctx
    assert original_fl_ctx.get_engine() is delegate


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
    if job_name == "challenge_1DivideAndConquer":
        # The live production job is pinned to the current 8-site deployment; the exact
        # site list is asserted by test_1dc_config_pins_exact_strict_eight_sites_and_retry_settings.
        assert "min_responses_required = 8" in client_config
    else:
        # Reusable templates must not bake in the current ODELIA deployment's site count.
        assert "min_responses_required = 5" in client_config
def test_1dc_config_pins_exact_strict_eight_sites_and_retry_settings():
    expected_sites = {"CAM_1", "VHIO_1", "USZ_1", "RUMC_1", "MHA_1", "RSH_1", "UMCU_1", "UKA_1"}
    job_dir = REPO_ROOT / "application" / "jobs" / "challenge_1DivideAndConquer"
    server_config = (job_dir / "app" / "config" / "config_fed_server.conf").read_text()
    client_config = (job_dir / "app" / "config" / "config_fed_client.conf").read_text()
    metadata = (job_dir / "meta.conf").read_text()

    participating = re.search(r"participating_clients\s*=\s*\[(.*?)\]", server_config, re.DOTALL)
    mandatory = re.search(r"mandatory_clients\s*=\s*\[(.*?)\]", metadata, re.DOTALL)

    assert participating is not None
    assert mandatory is not None
    assert set(re.findall(r'"([^"]+)"', participating.group(1))) == expected_sites
    assert set(re.findall(r'"([^"]+)"', mandatory.group(1))) == expected_sites
    assert "min_clients = 8" in server_config
    assert "configure_min_clients = 8" in server_config
    assert "min_responses_required = 8" in client_config
    assert "learn_task_scatter_attempt_timeout = 3600" in client_config
    assert "learn_task_scatter_retry_interval = 5" in client_config


def test_prepare_script_default_roster_matches_the_pinned_1dc_config():
    """The eight-site roster is declared twice on purpose and must not drift.

    The committed 1DC job config pins the roster so a job submitted directly --
    bypassing prepare_odelia_job.sh -- cannot fall back to a partial quorum
    (failure mode F8). prepare_odelia_job.sh carries the same roster as its
    default for the normal admin path. Neither is redundant, but they have to
    agree, so assert it rather than trusting a comment.
    """
    prepare_script = (REPO_ROOT / "kit_admin_tools" / "prepare_odelia_job.sh").read_text()
    server_config = (
        REPO_ROOT / "application" / "jobs" / "challenge_1DivideAndConquer"
        / "app" / "config" / "config_fed_server.conf"
    ).read_text()

    default_clients = re.search(
        r'^DEFAULT_STRICT_CLIENTS="([^"]*)"', prepare_script, re.MULTILINE
    )
    participating = re.search(
        r"participating_clients\s*=\s*\[(.*?)\]", server_config, re.DOTALL
    )

    assert default_clients is not None, "DEFAULT_STRICT_CLIENTS not found in prepare_odelia_job.sh"
    assert participating is not None, "participating_clients not found in the 1DC server config"

    script_roster = {c.strip() for c in default_clients.group(1).split(",") if c.strip()}
    config_roster = set(re.findall(r'"([^"]+)"', participating.group(1)))

    assert script_roster == config_roster, (
        "prepare_odelia_job.sh and the 1DC job config disagree on the site roster: "
        f"only in script {sorted(script_roster - config_roster)}, "
        f"only in config {sorted(config_roster - script_roster)}"
    )
