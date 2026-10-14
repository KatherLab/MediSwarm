"""Unit tests for application/jobs/_shared/custom/round_hooks.py (CPU only).

Covers the loading rules (list order, hard import errors, the feature-variable safety net, swarm-only
activation), the call order, and the placement of hook callbacks in front of NVFlare's FLCallback,
including one run through a real Lightning Trainer with the real FLCallback so the claim "a callback
inserted before FLCallback runs before the model is sent" is checked against the installed Lightning.
"""

import importlib
import sys
import textwrap
from pathlib import Path

import pytest

from conftest import import_module_from_path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROUND_HOOKS_PATH = REPO_ROOT / "application" / "jobs" / "_shared" / "custom" / "round_hooks.py"

# A hook module that records every call it receives into ctx.state["calls"] as (module name, point).
RECORDING_HOOK = """
from round_hooks import RoundHook


class _Hook(RoundHook):
    def __init__(self, name):
        self.name = name

    def _record(self, ctx, point):
        ctx.state.setdefault("calls", []).append((self.name, point))

    def on_start(self, ctx):
        self._record(ctx, "on_start")

    def after_receive(self, ctx):
        self._record(ctx, "after_receive")

    def before_fit(self, ctx):
        self._record(ctx, "before_fit")

    def lightning_callbacks(self, ctx):
        self._record(ctx, "lightning_callbacks")
        return []

    def after_round(self, ctx):
        self._record(ctx, "after_round")

    def on_end(self, ctx):
        self._record(ctx, "on_end")


def create_hook(env, logger):
    return _Hook(__name__)
"""

FEATURE_CASES = [
    ("ODELIA_DP_SIGMA", "2.4", "dp_noise"),
    ("ODELIA_DP_MEASURE_ONLY", "0", "dp_noise"),
    ("ODELIA_MS6_CAPTURE_SITES", "all", "ms6_capture"),
    ("ODELIA_AL_STRATEGY", "entropy", "al_subset"),
    ("ODELIA_TRAIN_INCLUDE_LIST", "/data/uids.txt", "al_subset"),
    # An empty value still counts: it usually means a template value got lost on the way.
    ("ODELIA_DP_CLIP_NORM", "", "dp_noise"),
]


@pytest.fixture
def rh(monkeypatch):
    """A fresh round_hooks module, registered as 'round_hooks' so hook modules import this copy."""
    monkeypatch.delitem(sys.modules, "round_hooks", raising=False)
    module = import_module_from_path("round_hooks", ROUND_HOOKS_PATH)
    yield module
    sys.modules.pop("round_hooks", None)


@pytest.fixture
def hook_modules(tmp_path, monkeypatch):
    """Write throwaway hook modules into a directory on sys.path; forget them afterwards."""
    hook_dir = tmp_path / "hooks"
    hook_dir.mkdir()
    monkeypatch.syspath_prepend(str(hook_dir))
    written = []

    def write(name, body=RECORDING_HOOK):
        (hook_dir / f"{name}.py").write_text(textwrap.dedent(body))
        sys.modules.pop(name, None)
        importlib.invalidate_caches()
        written.append(name)

    yield write
    for name in written:
        sys.modules.pop(name, None)


@pytest.fixture
def swarm_env():
    return {"TRAINING_MODE": "swarm"}


# --------------------------------------------------------------------------------------------------
# load_hooks: list order, activation, hard errors
# --------------------------------------------------------------------------------------------------


def test_no_hooks_without_variable(rh, swarm_env):
    assert rh.load_hooks(swarm_env) == []


def test_hooks_load_and_run_in_list_order(rh, hook_modules, swarm_env):
    hook_modules("hook_b")
    hook_modules("hook_a")
    hooks = rh.load_hooks({**swarm_env, "ODELIA_ROUND_HOOKS": " hook_b, hook_a ,"})
    assert [hook.name for hook in hooks] == ["hook_b", "hook_a"]

    ctx = rh.RoundContext()
    rh.start_hooks(hooks, ctx, fl_callback_cls=_NeverMatches)
    for point in ("after_receive", "before_fit", "after_round", "on_end"):
        rh.call_hooks(hooks, point, ctx)

    assert ctx.state["calls"] == [
        # every on_start first, then the callbacks are collected (also in list order)
        ("hook_b", "on_start"), ("hook_a", "on_start"),
        ("hook_b", "lightning_callbacks"), ("hook_a", "lightning_callbacks"),
        ("hook_b", "after_receive"), ("hook_a", "after_receive"),
        ("hook_b", "before_fit"), ("hook_a", "before_fit"),
        ("hook_b", "after_round"), ("hook_a", "after_round"),
        ("hook_b", "on_end"), ("hook_a", "on_end"),
    ]


def test_create_hook_returning_none_means_inactive(rh, hook_modules, swarm_env):
    hook_modules("inactive_hook", "def create_hook(env, logger):\n    return None\n")
    assert rh.load_hooks({**swarm_env, "ODELIA_ROUND_HOOKS": "inactive_hook"}) == []


def test_create_hook_receives_env_and_a_logger(rh, hook_modules, swarm_env):
    hook_modules(
        "env_hook",
        """
        from round_hooks import RoundHook
        SEEN = {}

        def create_hook(env, logger):
            SEEN["value"] = env.get("ENV_HOOK_VALUE")
            SEEN["logger"] = logger
            return RoundHook()
        """,
    )
    rh.load_hooks({**swarm_env, "ODELIA_ROUND_HOOKS": "env_hook", "ENV_HOOK_VALUE": "42"})
    seen = sys.modules["env_hook"].SEEN
    assert seen["value"] == "42"
    assert seen["logger"] is not None


def test_unimportable_module_is_a_hard_error(rh, swarm_env):
    with pytest.raises(rh.RoundHookError, match="no_such_hook_module_xyz"):
        rh.load_hooks({**swarm_env, "ODELIA_ROUND_HOOKS": "no_such_hook_module_xyz"})


def test_module_failing_at_import_is_a_hard_error(rh, hook_modules, swarm_env):
    hook_modules("broken_hook", "raise RuntimeError('broken at import')\n")
    with pytest.raises(rh.RoundHookError, match="broken_hook") as excinfo:
        rh.load_hooks({**swarm_env, "ODELIA_ROUND_HOOKS": "broken_hook"})
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_module_without_create_hook_is_a_hard_error(rh, hook_modules, swarm_env):
    hook_modules("no_factory_hook", "VALUE = 1\n")
    with pytest.raises(rh.RoundHookError, match="create_hook"):
        rh.load_hooks({**swarm_env, "ODELIA_ROUND_HOOKS": "no_factory_hook"})


def test_create_hook_must_return_a_round_hook(rh, hook_modules, swarm_env):
    hook_modules("wrong_type_hook", "def create_hook(env, logger):\n    return object()\n")
    with pytest.raises(rh.RoundHookError, match="RoundHook"):
        rh.load_hooks({**swarm_env, "ODELIA_ROUND_HOOKS": "wrong_type_hook"})


@pytest.mark.parametrize("value", ["dp-noise", "custom.dp_noise", "../dp_noise", "dp_noise,dp_noise"])
def test_malformed_or_duplicate_names_are_rejected(rh, swarm_env, value):
    with pytest.raises(rh.RoundHookError):
        rh.load_hooks({**swarm_env, "ODELIA_ROUND_HOOKS": value})


# --------------------------------------------------------------------------------------------------
# Safety net and swarm-only activation
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("variable,value,module", FEATURE_CASES)
def test_feature_variable_without_its_module_raises(rh, swarm_env, variable, value, module):
    with pytest.raises(rh.RoundHookError, match=module) as excinfo:
        rh.load_hooks({**swarm_env, variable: value})
    assert variable in str(excinfo.value)


def test_listing_another_module_does_not_satisfy_the_safety_net(rh, hook_modules, swarm_env):
    hook_modules("ms6_capture")
    with pytest.raises(rh.RoundHookError, match="dp_noise"):
        rh.load_hooks({**swarm_env, "ODELIA_ROUND_HOOKS": "ms6_capture", "ODELIA_DP_SIGMA": "1.0"})


@pytest.mark.parametrize("variable,value,module", FEATURE_CASES)
def test_feature_variable_with_its_module_listed_loads(rh, hook_modules, swarm_env, variable, value, module):
    hook_modules(module)
    hooks = rh.load_hooks({**swarm_env, "ODELIA_ROUND_HOOKS": module, variable: value})
    assert [hook.name for hook in hooks] == [module]


def test_safety_net_runs_before_any_import(rh, swarm_env):
    # The unlisted DP variable is reported even though the listed module would also fail to import:
    # nothing is imported (no module code runs) for a job that has to stop anyway.
    with pytest.raises(rh.RoundHookError, match="dp_noise"):
        rh.load_hooks({**swarm_env, "ODELIA_ROUND_HOOKS": "no_such_hook_module_xyz", "ODELIA_DP_SIGMA": "1"})


@pytest.mark.parametrize("mode", ["local_training", "preflight_check", ""])
def test_hooks_are_inactive_outside_swarm(rh, mode):
    env = {"ODELIA_ROUND_HOOKS": "no_such_hook_module_xyz", "ODELIA_DP_SIGMA": "1"}
    if mode:
        env["TRAINING_MODE"] = mode
    assert rh.load_hooks(env) == []


def test_load_hooks_defaults_to_os_environ(rh, monkeypatch):
    monkeypatch.setenv("TRAINING_MODE", "swarm")
    monkeypatch.setenv("ODELIA_AL_BUDGET", "40")
    monkeypatch.delenv("ODELIA_ROUND_HOOKS", raising=False)
    with pytest.raises(rh.RoundHookError, match="al_subset"):
        rh.load_hooks()


# --------------------------------------------------------------------------------------------------
# call_hooks: no swallowing
# --------------------------------------------------------------------------------------------------


def test_hook_exception_propagates_and_stops_later_hooks(rh):
    calls = []

    class Failing(rh.RoundHook):
        name = "failing"

        def before_fit(self, ctx):
            raise ValueError("budget exhausted")

    class Later(rh.RoundHook):
        name = "later"

        def before_fit(self, ctx):
            calls.append("later")

    with pytest.raises(ValueError, match="budget exhausted"):
        rh.call_hooks([Failing(), Later()], "before_fit", rh.RoundContext())
    assert calls == []


def test_unknown_call_point_is_rejected(rh):
    with pytest.raises(ValueError):
        rh.call_hooks([], "before_validate", rh.RoundContext())


# --------------------------------------------------------------------------------------------------
# Callback placement before FLCallback
# --------------------------------------------------------------------------------------------------


class _NeverMatches:
    """FLCallback stand-in for tests that insert no callbacks."""


def _fake_fl_classes():
    from pytorch_lightning.callbacks import Callback

    class FakeFLCallback(Callback):
        pass

    class FakeRestoreState(Callback):
        pass

    class Plain(Callback):
        pass

    return FakeFLCallback, FakeRestoreState, Plain


class _Trainer:
    def __init__(self, callbacks):
        self.callbacks = callbacks


def test_callbacks_are_inserted_immediately_before_fl_callback(rh):
    from pytorch_lightning.callbacks import ModelCheckpoint

    FakeFLCallback, FakeRestoreState, Plain = _fake_fl_classes()
    first, checkpoint, fl, restore = Plain(), ModelCheckpoint(), FakeFLCallback(), FakeRestoreState()
    trainer = _Trainer([first, checkpoint, fl, restore])
    a, b = Plain(), Plain()

    rh.insert_before_fl_callback(trainer, [a, b], fl_callback_cls=FakeFLCallback)

    assert trainer.callbacks == [first, checkpoint, a, b, fl, restore]


def test_start_hooks_inserts_callbacks_of_all_hooks_in_hook_order(rh):
    FakeFLCallback, FakeRestoreState, Plain = _fake_fl_classes()
    fl = FakeFLCallback()
    a1, a2, b1 = Plain(), Plain(), Plain()

    class HookA(rh.RoundHook):
        name = "a"

        def lightning_callbacks(self, ctx):
            return [a1, a2]

    class HookB(rh.RoundHook):
        name = "b"

        def lightning_callbacks(self, ctx):
            return [b1]

    ctx = rh.RoundContext(trainer=_Trainer([fl, FakeRestoreState()]))
    assert rh.start_hooks([HookA(), HookB()], ctx, fl_callback_cls=FakeFLCallback) == [a1, a2, b1]
    assert ctx.trainer.callbacks[:4] == [a1, a2, b1, fl]


def test_start_hooks_uses_a_trainer_replaced_in_on_start(rh):
    FakeFLCallback, _, Plain = _fake_fl_classes()
    old_trainer, new_trainer = _Trainer([FakeFLCallback()]), _Trainer([FakeFLCallback()])
    cb = Plain()

    class Replacing(rh.RoundHook):
        name = "replacing"

        def on_start(self, ctx):
            ctx.trainer = new_trainer

        def lightning_callbacks(self, ctx):
            return [cb]

    ctx = rh.RoundContext(trainer=old_trainer)
    rh.start_hooks([Replacing()], ctx, fl_callback_cls=FakeFLCallback)
    assert cb in new_trainer.callbacks
    assert cb not in old_trainer.callbacks


def test_insert_refuses_callbacks_lightning_would_move_or_that_send(rh):
    from pytorch_lightning.callbacks import ModelCheckpoint

    FakeFLCallback, _, _ = _fake_fl_classes()
    trainer = _Trainer([FakeFLCallback()])
    with pytest.raises(ValueError, match="reordering"):
        rh.insert_before_fl_callback(trainer, [ModelCheckpoint()], fl_callback_cls=FakeFLCallback)
    with pytest.raises(ValueError, match="FLCallback"):
        rh.insert_before_fl_callback(trainer, [FakeFLCallback()], fl_callback_cls=FakeFLCallback)
    with pytest.raises(TypeError):
        rh.insert_before_fl_callback(trainer, [object()], fl_callback_cls=FakeFLCallback)


def test_insert_requires_exactly_one_fl_callback(rh):
    FakeFLCallback, _, Plain = _fake_fl_classes()
    with pytest.raises(RuntimeError, match="flare.patch"):
        rh.insert_before_fl_callback(_Trainer([Plain()]), [Plain()], fl_callback_cls=FakeFLCallback)


def test_hook_callback_runs_before_the_real_fl_callback_sends(rh, monkeypatch, tmp_path):
    """Real Lightning Trainer + real NVFlare FLCallback; only the client transport is faked.

    Checks with the installed Lightning that (1) the Trainer's reordering at construction keeps the
    inserted callback right before FLCallback, (2) its on_train_end runs before FLCallback sends, and
    (3) what it changes in on_train_end is what gets sent (the property a DP-noise callback relies on).
    """
    torch = pytest.importorskip("torch")
    pl = pytest.importorskip("pytorch_lightning")
    api = pytest.importorskip("nvflare.app_opt.lightning.api")
    from nvflare.app_common.abstract.fl_model import FLModel
    from nvflare.client.config import ConfigKey
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader, TensorDataset

    events = []
    task = {"model": None}

    class Tiny(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 2)

        def training_step(self, batch, batch_idx):
            x, y = batch
            return torch.nn.functional.cross_entropy(self.layer(x), y)

        def validation_step(self, batch, batch_idx):
            x, y = batch
            self.log("val/loss", torch.nn.functional.cross_entropy(self.layer(x), y))

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    class Perturb(pl.Callback):
        def on_train_end(self, trainer, pl_module):
            events.append("hook_callback.on_train_end")
            with torch.no_grad():
                pl_module.layer.bias.fill_(42.0)

    def send(model, clear_cache=True):
        events.append("send")
        task["sent"] = model

    monkeypatch.setattr(api, "init", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "get_config", lambda: {ConfigKey.TASK_EXCHANGE: {ConfigKey.TRAIN_WITH_EVAL: True}})
    monkeypatch.setattr(api, "receive", lambda *args, **kwargs: task["model"])
    monkeypatch.setattr(api, "send", send)
    monkeypatch.setattr(api, "clear", lambda: None)
    monkeypatch.setattr(api, "is_train", lambda: True)
    monkeypatch.setattr(api, "is_evaluate", lambda: False)
    monkeypatch.setattr(api, "is_submit_model", lambda: False)

    torch.manual_seed(0)
    model = Tiny()
    global_params = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    task["model"] = FLModel(params=global_params, current_round=0, total_rounds=1)
    data = TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,)))
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, logger=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0, default_root_dir=str(tmp_path),
        callbacks=[ModelCheckpoint(dirpath=str(tmp_path / "ckpt"), save_last=True)],
    )
    api.patch(trainer, load_state_dict_strict=False)
    hook_callback = Perturb()
    rh.insert_before_fl_callback(trainer, [hook_callback], fl_callback_cls=api.FLCallback)

    names = [type(cb).__name__ for cb in trainer.callbacks]
    assert names.index("Perturb") == names.index("FLCallback") - 1
    assert names.index("ModelCheckpoint") < names.index("Perturb")

    trainer.validate(model, dataloaders=DataLoader(data, batch_size=4))
    # FLCallback loaded the (all-zero) received global on this first validate of the round.
    assert torch.count_nonzero(model.layer.weight) == 0
    trainer.fit(model, train_dataloaders=DataLoader(data, batch_size=4), val_dataloaders=DataLoader(data, batch_size=4))

    assert events == ["hook_callback.on_train_end", "send"]
    sent = task["sent"]
    assert torch.all(sent.params["layer.bias"] == 42.0)
    assert "val/loss" in sent.metrics  # metrics of the global's validation travel with the update
