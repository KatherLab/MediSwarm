"""Unit tests for application/jobs/_shared/custom/dp_noise.py, the DP noise round hook (T3.3, #593). CPU only.

Covers the configuration rules, the clipping and noise math, which state-dict entries are noised,
restored or left alone, and the hook inside a real Lightning Trainer with NVFlare's real FLCallback (only
the client transport is faked): the update is rewritten before FLCallback sends it, measure-only and
equal-weight modes, per-round epsilon, the budget refusal at the right round, the evaluation-only round,
and that the "16-mixed" training precision keeps float32 parameters.

Seeded generators appear only here, to predict the noise; the hook itself always seeds from os.urandom.
"""

import importlib
import json
import logging
import math
import sys
import types

import pytest

from conftest import SHARED_CUSTOM_DIR

torch = pytest.importorskip("torch")
pl = pytest.importorskip("pytorch_lightning")

from pytorch_lightning.loggers.logger import Logger  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

SIGMA, CLIP, DELTA = 2.4, 0.5, 1e-5
NOISE_ENV = {"ODELIA_DP_SIGMA": str(SIGMA), "ODELIA_DP_CLIP_NORM": str(CLIP)}
# compute_epsilon(2.4, r, 1e-5), r = 1..20 (verified in test_privacy_accounting.py)
EPSILONS_SIGMA_2_4 = [1.77, 2.60, 3.26, 3.84, 4.36, 4.84, 5.29, 5.72, 6.12, 6.51,
                      6.89, 7.25, 7.60, 7.94, 8.28, 8.60, 8.92, 9.23, 9.54, 9.84]
TRAINABLE = ["layer.weight", "layer.bias"]


@pytest.fixture
def dp(monkeypatch, tmp_path):
    """A fresh dp_noise imported from custom/, together with the round_hooks/privacy_accounting it uses.

    Runs in an empty working directory: create_hook keeps the release count of noise mode in the working
    directory (the job's app dir in production), which must never be the repository checkout.
    """
    names = ("dp_noise", "round_hooks", "privacy_accounting")
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(SHARED_CUSTOM_DIR))
    module = importlib.import_module("dp_noise")
    yield module
    for name in names:
        sys.modules.pop(name, None)


# --------------------------------------------------------------------------------------------------
# Tiny model: trainable layer, a frozen parameter, a float and an integer buffer that change in training
# (like BatchNorm's running statistics and counter)
# --------------------------------------------------------------------------------------------------


class TinyNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 2)
        self.frozen = torch.nn.Parameter(torch.ones(3), requires_grad=False)
        self.register_buffer("float_buffer", torch.zeros(2))
        self.register_buffer("int_buffer", torch.zeros((), dtype=torch.long))

    def training_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            self.float_buffer.add_(x.mean(0)[:2])
            self.int_buffer.add_(1)
        return torch.nn.functional.cross_entropy(self.layer(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val/loss", torch.nn.functional.cross_entropy(self.layer(x), y))

    def configure_optimizers(self):
        return torch.optim.SGD([p for p in self.parameters() if p.requires_grad], lr=0.5)


def _loader(n=8, seed=7):
    generator = torch.Generator().manual_seed(seed)
    data = TensorDataset(torch.randn(n, 4, generator=generator), torch.randint(0, 2, (n,), generator=generator))
    return DataLoader(data, batch_size=4)


class RecordingLogger(Logger):
    """Lightning logger that keeps every log_metrics call (ClientLogger/TensorBoardLogger stand-in)."""

    def __init__(self):
        super().__init__()
        self.records = []

    @property
    def name(self):
        return "recording"

    @property
    def version(self):
        return 0

    def log_metrics(self, metrics, step=None):
        self.records.append((dict(metrics), step))

    def log_hyperparams(self, params, *args, **kwargs):
        pass

    def dp_records(self):
        return [({k: v for k, v in metrics.items() if k.startswith("dp/")}, step)
                for metrics, step in self.records if any(k.startswith("dp/") for k in metrics)]


def _clone(state):
    return {k: v.detach().clone() for k, v in state.items()}


def _perturbed(state, scale=0.3, seed=11):
    """A 'received global' that differs from the model in every float entry (and the integer buffer)."""
    generator = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in state.items():
        if v.is_floating_point():
            out[k] = v.detach().clone() + scale * torch.randn(v.shape, generator=generator, dtype=v.dtype)
        else:
            out[k] = v.detach().clone() + 5
    return out


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------


def test_noise_mode_defaults(dp):
    config = dp.parse_config(NOISE_ENV)
    assert config.mode == dp.MODE_NOISE
    assert (config.sigma, config.clip_norm, config.delta) == (SIGMA, CLIP, DELTA)
    assert config.target_epsilon is None and config.equal_weights is False
    assert config.noise_std == pytest.approx(SIGMA * CLIP)


def test_noise_mode_with_every_option(dp):
    config = dp.parse_config({**NOISE_ENV, "ODELIA_DP_DELTA": "1e-6", "ODELIA_DP_TARGET_EPSILON": "10",
                              "ODELIA_DP_EQUAL_WEIGHTS": "1", "ODELIA_DP_MEASURE_ONLY": "0"})
    assert (config.delta, config.target_epsilon, config.equal_weights) == (1e-6, 10.0, True)


def test_measure_and_equal_weight_modes(dp):
    assert dp.parse_config({"ODELIA_DP_MEASURE_ONLY": "1"}).mode == dp.MODE_MEASURE
    measure = dp.parse_config({"ODELIA_DP_MEASURE_ONLY": "1", "ODELIA_DP_CLIP_NORM": "0.5", "ODELIA_DP_EQUAL_WEIGHTS": "1"})
    assert (measure.clip_norm, measure.equal_weights, measure.noise_std) == (0.5, True, 0.0)
    weights_only = dp.parse_config({"ODELIA_DP_EQUAL_WEIGHTS": "1"})
    assert (weights_only.mode, weights_only.equal_weights) == (dp.MODE_EQUAL_WEIGHTS, True)


@pytest.mark.parametrize("env,message", [
    ({"ODELIA_DP_SIGMA": "2.4"}, "ODELIA_DP_CLIP_NORM"),
    ({"ODELIA_DP_SIGMA": "0", "ODELIA_DP_CLIP_NORM": "1"}, "> 0"),
    ({"ODELIA_DP_SIGMA": "-1", "ODELIA_DP_CLIP_NORM": "1"}, "> 0"),
    ({"ODELIA_DP_SIGMA": "nan", "ODELIA_DP_CLIP_NORM": "1"}, "finite"),
    ({"ODELIA_DP_SIGMA": "", "ODELIA_DP_CLIP_NORM": "1"}, "empty"),
    ({"ODELIA_DP_SIGMA": "two", "ODELIA_DP_CLIP_NORM": "1"}, "not a number"),
    ({"ODELIA_DP_SIGMA": "2.4", "ODELIA_DP_CLIP_NORM": "0"}, "> 0"),
    ({**NOISE_ENV, "ODELIA_DP_DELTA": "1"}, "< 1"),
    ({**NOISE_ENV, "ODELIA_DP_TARGET_EPSILON": "0"}, "> 0"),
    ({**NOISE_ENV, "ODELIA_DP_MEASURE_ONLY": "1"}, "both"),
    ({"ODELIA_DP_MEASURE_ONLY": "1", "ODELIA_DP_TARGET_EPSILON": "3"}, "only apply"),
    ({"ODELIA_DP_MEASURE_ONLY": "1", "ODELIA_DP_DELTA": "1e-5"}, "only apply"),
    ({"ODELIA_DP_CLIP_NORM": "1"}, "without ODELIA_DP_SIGMA"),
    ({"ODELIA_DP_TARGET_EPSILON": "3", "ODELIA_DP_EQUAL_WEIGHTS": "1"}, "without ODELIA_DP_SIGMA"),
    ({}, "no mode"),
    ({"ODELIA_DP_MEASURE_ONLY": "0"}, "no mode"),
    ({"ODELIA_DP_MEASURE_ONLY": "yes"}, "0 or 1"),
    ({"ODELIA_DP_EQUAL_WEIGHTS": ""}, "0 or 1"),
    ({**NOISE_ENV, "ODELIA_DP_SIGAM": "2.4"}, "Unknown"),
])
def test_ambiguous_or_invalid_configuration_stops_the_job(dp, env, message):
    with pytest.raises(dp.DPConfigError, match=message):
        dp.parse_config(env)


def test_noise_refuses_per_case_return_and_the_legacy_budget_name(dp):
    with pytest.raises(dp.DPConfigError, match="ODELIA_RETURN_PER_CASE=0"):
        dp.create_hook({**NOISE_ENV, "ODELIA_RETURN_PER_CASE": "1"}, None)
    assert dp.create_hook({**NOISE_ENV, "ODELIA_RETURN_PER_CASE": "0"}, None).config.mode == dp.MODE_NOISE
    # A calibration run carries no guarantee, so it does not need the per-case return switched off.
    assert dp.create_hook({"ODELIA_DP_MEASURE_ONLY": "1", "ODELIA_RETURN_PER_CASE": "1"}, None) is not None
    with pytest.raises(dp.DPConfigError, match="ODELIA_DP_TARGET_EPSILON"):
        dp.create_hook({**NOISE_ENV, "PRIVACY_EPSILON_TARGET": "8"}, None)


def test_create_hook_reads_the_final_eval_round_flag(dp):
    assert dp.create_hook(NOISE_ENV, None).final_eval_round is False
    assert dp.create_hook({**NOISE_ENV, "ODELIA_FINAL_EVAL_ROUND": "1"}, None).final_eval_round is True


def test_load_hooks_builds_the_dp_hook(dp):
    import round_hooks

    hooks = round_hooks.load_hooks({"TRAINING_MODE": "swarm", "ODELIA_ROUND_HOOKS": "dp_noise", **NOISE_ENV})
    assert len(hooks) == 1 and isinstance(hooks[0], dp.DPNoiseHook) and hooks[0].name == "dp_noise"
    # Listed without any DP variable it is an error, never a silently inactive hook.
    with pytest.raises(dp.DPConfigError, match="no mode"):
        round_hooks.load_hooks({"TRAINING_MODE": "swarm", "ODELIA_ROUND_HOOKS": "dp_noise"})


# --------------------------------------------------------------------------------------------------
# Clipping and noise math
# --------------------------------------------------------------------------------------------------


def _update(seed=0, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    return {"a": torch.randn(30, 20, generator=generator, dtype=dtype),
            "b": torch.randn(50, generator=generator, dtype=dtype)}


def test_clipping_scales_the_whole_update_to_c_and_keeps_its_direction(dp):
    update = _update()
    before = _clone(update)
    norm = dp.l2_norm(update.values())
    assert norm > 1.0
    released, pre_norm, released_norm, clipped = dp.clip_and_noise(update, clip_norm=1.0, noise_multiplier=0.0)

    assert clipped and pre_norm == pytest.approx(norm, rel=1e-12)
    assert released_norm == pytest.approx(1.0, rel=1e-6)
    assert dp.l2_norm(released.values()) == pytest.approx(1.0, rel=1e-6)
    flat_in = torch.cat([update[k].flatten() for k in ("a", "b")])
    flat_out = torch.cat([released[k].flatten() for k in ("a", "b")])
    assert torch.nn.functional.cosine_similarity(flat_in, flat_out, dim=0).item() == pytest.approx(1.0, abs=1e-12)
    for k in update:  # the caller's update is not modified
        assert torch.equal(update[k], before[k])


def test_update_within_c_is_released_unchanged_without_noise(dp):
    update = _update()
    norm = dp.l2_norm(update.values())
    released, pre_norm, released_norm, clipped = dp.clip_and_noise(update, clip_norm=2 * norm, noise_multiplier=0.0)
    assert not clipped and released_norm == pytest.approx(norm, rel=1e-12)
    for k in update:
        assert torch.equal(released[k], update[k])


def test_noise_is_gaussian_with_std_sigma_times_c(dp):
    # 1.2 million float32 coordinates, the dtype the parameters have in training.
    update = {"w": torch.zeros(1000, 1000), "b": torch.zeros(200_000)}
    released, _, released_norm, _ = dp.clip_and_noise(update, CLIP, SIGMA, torch.Generator().manual_seed(2024))
    noise = torch.cat([v.flatten() for v in released.values()]).double()
    std = SIGMA * CLIP
    n = noise.numel()
    assert noise.std().item() == pytest.approx(std, rel=0.01)
    assert abs(noise.mean().item()) < 5 * std / math.sqrt(n)
    assert released_norm == pytest.approx(std * math.sqrt(n), rel=0.01)


def test_noise_is_added_on_top_of_the_clipped_update(dp):
    update = {"w": torch.full((600, 1000), 0.01)}  # norm 7.75 > C: clipped to C first
    released, pre_norm, _, clipped = dp.clip_and_noise(update, CLIP, SIGMA, torch.Generator().manual_seed(5))
    assert clipped
    residual = released["w"] - update["w"] * (CLIP / pre_norm)
    assert residual.std().item() == pytest.approx(SIGMA * CLIP, rel=0.01)


def test_default_generator_draws_fresh_noise_every_time(dp):
    update = {"w": torch.zeros(1000)}
    first, *_ = dp.clip_and_noise(update, CLIP, SIGMA)
    second, *_ = dp.clip_and_noise(update, CLIP, SIGMA)
    assert not torch.equal(first["w"], second["w"])
    a, b = dp.new_generator(), dp.new_generator()
    assert not torch.equal(torch.randn(16, generator=a), torch.randn(16, generator=b))


def test_a_seeded_generator_reproduces_the_noise(dp):
    """Only what lets these tests predict the noise; production never passes a seeded generator."""
    update = _update(dtype=torch.float32)
    a, *_ = dp.clip_and_noise(update, CLIP, SIGMA, torch.Generator().manual_seed(9))
    b, *_ = dp.clip_and_noise(update, CLIP, SIGMA, torch.Generator().manual_seed(9))
    assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("clip_norm,sigma", [(0.0, 1.0), (-1.0, 1.0), (float("inf"), 1.0), (1.0, -0.5), (1.0, float("nan"))])
def test_clip_and_noise_rejects_invalid_parameters(dp, clip_norm, sigma):
    with pytest.raises(ValueError):
        dp.clip_and_noise({"w": torch.ones(3)}, clip_norm, sigma)


# --------------------------------------------------------------------------------------------------
# Which entries are noised, restored or left alone
# --------------------------------------------------------------------------------------------------


def test_partition_noises_trainable_parameters_and_restores_everything_else(dp):
    torch.manual_seed(0)
    model = TinyNet()
    model.register_buffer("_class_weight", torch.tensor([0.5, 2.0]))  # local-only, like the class weights
    received = _perturbed(model.state_dict())
    del received["_class_weight"]

    partition = dp.partition_state(model, received)
    assert [name for name, _ in partition.trainable] == TRAINABLE
    assert [name for name, _ in partition.restored] == ["frozen", "float_buffer", "int_buffer"]
    assert partition.local_only == ["_class_weight"] and partition.trainable_not_in_global == []
    assert partition.coordinates == 4 * 2 + 2

    local = _clone(model.state_dict())
    released = {name: torch.full_like(local[name], 0.25) for name in TRAINABLE}
    dp.write_release(partition, received, released)
    state = model.state_dict()
    for name in TRAINABLE:
        assert torch.allclose(state[name], received[name] + 0.25)
    for name in ("frozen", "float_buffer", "int_buffer"):
        assert torch.equal(state[name], received[name])
    assert state["int_buffer"].dtype == torch.long  # integer buffers are copied, never noised or cast
    assert torch.equal(state["_class_weight"], local["_class_weight"])


def test_tied_weights_are_noised_once(dp):
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(3, 3, bias=False)
            self.decoder = torch.nn.Linear(3, 3, bias=False)
            self.decoder.weight = self.encoder.weight

    model = Tied()
    received = _perturbed(model.state_dict())
    partition = dp.partition_state(model, received)
    assert [name for name, _ in partition.trainable] == ["encoder.weight"]
    assert partition.restored == []


def test_compute_update_accepts_numpy_globals_and_half_parameters(dp):
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.float16))
    update = dp.compute_update([("p", param)], {"p": torch.zeros(4, dtype=torch.float64).numpy()})
    assert update["p"].dtype == torch.float32 and torch.equal(update["p"], torch.ones(4))
    with pytest.raises(RuntimeError, match="shape"):
        dp.compute_update([("p", param)], {"p": torch.zeros(5)})


# --------------------------------------------------------------------------------------------------
# The hook with a real Trainer and NVFlare's real FLCallback (client transport faked)
# --------------------------------------------------------------------------------------------------


class FakeRuntime:
    """Client transport: one task per round; what FLCallback sends (DIFF semantics: only keys the global
    has) becomes the next global, as for a single site."""

    def __init__(self, fl_model_cls, initial_params, total_rounds):
        self.fl_model_cls = fl_model_cls
        self.global_params = _clone(initial_params)
        self.total_rounds = total_rounds
        self.round = -1
        self.current = None
        self.handed_out = []
        self.sent = []  # (round, params cloned at send time, meta)
        self.events = []

    def next_task(self):
        self.round += 1
        self.current = self.fl_model_cls(params=_clone(self.global_params), current_round=self.round,
                                         total_rounds=self.total_rounds)
        self.handed_out.append(self.current)
        return self.current

    def receive(self, *args, **kwargs):
        return self.current

    def send(self, model, clear_cache=True):
        self.events.append("send")
        params = _clone(model.params)  # FLCallback sends views of the live module: copy them now
        self.sent.append((self.round, params, dict(model.meta or {})))
        self.global_params = {k: params[k] for k in self.global_params if k in params}

    def clear(self):
        self.current = None


class Snapshot(pl.Callback):
    """Placed right before the DP callback: records the local model W as training left it."""

    def __init__(self, events):
        self.events = events
        self.states = []

    def on_train_end(self, trainer, pl_module):
        self.events.append("snapshot")
        self.states.append(_clone(pl_module.state_dict()))


def _stack(dp, monkeypatch, tmp_path, env, total_rounds=3, seed=1234, transfer_type="DIFF", model=None,
           final_eval_round=False, precision=None):
    api = pytest.importorskip("nvflare.app_opt.lightning.api")
    from nvflare.app_common.abstract.fl_model import FLModel
    from nvflare.client.config import ConfigKey
    import round_hooks

    if model is None:
        torch.manual_seed(0)
        model = TinyNet()
    runtime = FakeRuntime(FLModel, _perturbed(model.state_dict()), total_rounds)
    for name, value in {
        "init": lambda *args, **kwargs: None,
        "get_config": lambda: {ConfigKey.TASK_EXCHANGE: {ConfigKey.TRAIN_WITH_EVAL: True}},
        "receive": runtime.receive,
        "send": runtime.send,
        "clear": runtime.clear,
        "is_train": lambda: True,
        "is_evaluate": lambda: False,
        "is_submit_model": lambda: False,
    }.items():
        monkeypatch.setattr(api, name, value)
    monkeypatch.setattr(dp, "_read_transfer_type", lambda: transfer_type)

    recorder = RecordingLogger()
    kwargs = dict(accelerator="cpu", devices=1, max_epochs=1, logger=recorder, enable_progress_bar=False,
                  enable_model_summary=False, num_sanity_val_steps=0, enable_checkpointing=False,
                  default_root_dir=str(tmp_path))
    if precision is not None:
        kwargs["precision"] = precision
    trainer = pl.Trainer(**kwargs)
    api.patch(trainer, load_state_dict_strict=False)

    hook = dp.DPNoiseHook(dp.parse_config(env), logger=logging.getLogger("test_dp_noise"),
                          final_eval_round=final_eval_round, generator=torch.Generator().manual_seed(seed))
    ctx = round_hooks.RoundContext(logger=logging.getLogger("test_dp_noise"), site_name="SITE_A", model=model,
                                   trainer=trainer)
    round_hooks.start_hooks([hook], ctx, fl_callback_cls=api.FLCallback)
    snapshot = Snapshot(runtime.events)
    position = trainer.callbacks.index(hook.callback)
    trainer.callbacks = trainer.callbacks[:position] + [snapshot] + trainer.callbacks[position:]

    def run_round(eval_only=False, skip_after_receive=False):
        """The call sequence of main.py's swarm loop for one round."""
        runtime.next_task()
        ctx.input_model = runtime.receive()
        ctx.current_round = ctx.input_model.current_round
        ctx.total_rounds = ctx.input_model.total_rounds
        ctx.is_final_round = total_rounds is not None and ctx.current_round >= total_rounds - 1
        ctx.is_eval_only_round = eval_only
        if not skip_after_receive:
            round_hooks.call_hooks([hook], "after_receive", ctx)
        trainer.validate(model, dataloaders=_loader())
        round_hooks.call_hooks([hook], "before_fit", ctx)
        trainer.fit(model, train_dataloaders=_loader(), val_dataloaders=_loader())
        round_hooks.call_hooks([hook], "after_round", ctx)

    return types.SimpleNamespace(api=api, model=model, runtime=runtime, trainer=trainer, hook=hook, ctx=ctx,
                                 recorder=recorder, snapshot=snapshot, run_round=run_round, rh=round_hooks)


def test_update_is_clipped_and_noised_before_fl_callback_sends(dp, monkeypatch, tmp_path):
    seed = 31
    s = _stack(dp, monkeypatch, tmp_path, NOISE_ENV, seed=seed)
    s.run_round()

    assert s.runtime.events == ["snapshot", "send"]  # the DP rewrite happened between the two
    received = s.runtime.handed_out[0].params
    local = s.snapshot.states[0]
    _, sent, meta = s.runtime.sent[0]

    # Recompute the expected release with the same generator stream (same key order).
    update = {k: local[k] - received[k] for k in TRAINABLE}
    expected, pre_norm, _, clipped = dp.clip_and_noise(update, CLIP, SIGMA, torch.Generator().manual_seed(seed))
    for k in TRAINABLE:
        assert torch.allclose(sent[k], received[k] + expected[k], atol=1e-6)
        assert not torch.allclose(sent[k], local[k], atol=1e-3)
    # Buffers and the frozen parameter go out at the received global's value: no local change leaves.
    for k in ("frozen", "float_buffer", "int_buffer"):
        assert torch.equal(sent[k], received[k])
    assert not torch.equal(local["int_buffer"], received["int_buffer"])  # (training did change them)

    stats = s.hook.history[0]
    assert stats.update_norm == pytest.approx(pre_norm) and stats.clipped == clipped
    assert stats.epsilon == pytest.approx(EPSILONS_SIGMA_2_4[0], abs=0.005) and stats.releases == 1
    # Weighting unchanged without ODELIA_DP_EQUAL_WEIGHTS: FLCallback's own value, 2 batches x 1 epoch.
    assert meta["NUM_STEPS_CURRENT_ROUND"] == 2
    names = [type(cb).__name__ for cb in s.trainer.callbacks]
    assert names.index("DPNoiseCallback") == names.index("FLCallback") - 1


def test_update_within_c_gets_noise_but_no_clipping(dp, monkeypatch, tmp_path):
    s = _stack(dp, monkeypatch, tmp_path, {"ODELIA_DP_SIGMA": "1e-3", "ODELIA_DP_CLIP_NORM": "1e6"})
    s.run_round()
    stats = s.hook.history[0]
    assert stats.clipped is False and s.recorder.dp_records()[0][0]["dp/clipped"] == 0.0
    received, local, (_, sent, _) = s.runtime.handed_out[0].params, s.snapshot.states[0], s.runtime.sent[0]
    for k in TRAINABLE:  # noise std 1e3: the sent update is noise-dominated, far from the local model
        assert (sent[k] - local[k]).abs().max() > 1.0


def test_dp_callback_runs_before_fl_callback_with_fake_callbacks(dp):
    """Same order check without a Trainer: Lightning calls on_train_end along trainer.callbacks."""
    import round_hooks

    class FLCallback(pl.Callback):  # fake, matched by class name like the real one
        _is_training = True

        def __init__(self):
            self.sent = None

        def on_train_end(self, trainer, pl_module):
            self.sent = _clone(pl_module.state_dict())

    torch.manual_seed(0)
    model = TinyNet()
    received = _perturbed(model.state_dict())
    fl = FLCallback()
    trainer = types.SimpleNamespace(callbacks=[fl], loggers=[], world_size=1)
    hook = dp.DPNoiseHook(dp.parse_config(NOISE_ENV), generator=torch.Generator().manual_seed(3))
    hook.transfer_type = "DIFF"
    ctx = round_hooks.RoundContext(model=model, trainer=trainer)
    ctx.input_model = types.SimpleNamespace(params=received)
    ctx.current_round = 0
    round_hooks.insert_before_fl_callback(trainer, hook.lightning_callbacks(ctx), fl_callback_cls=FLCallback)

    local = _clone(model.state_dict())
    for callback in trainer.callbacks:
        callback.on_train_end(trainer, model)

    update = {k: local[k] - received[k] for k in TRAINABLE}
    expected, *_ = dp.clip_and_noise(update, CLIP, SIGMA, torch.Generator().manual_seed(3))
    for k in TRAINABLE:
        assert torch.allclose(fl.sent[k], received[k] + expected[k], atol=1e-6)


def test_per_round_epsilon_is_streamed_with_the_other_dp_metrics(dp, monkeypatch, tmp_path):
    s = _stack(dp, monkeypatch, tmp_path, NOISE_ENV, total_rounds=3)
    for _ in range(3):
        s.run_round()
    records = s.recorder.dp_records()
    assert [step for _, step in records] == [0, 1, 2]
    assert [m["dp/epsilon"] for m, _ in records] == pytest.approx(EPSILONS_SIGMA_2_4[:3], abs=0.005)
    for metrics, _ in records:
        assert set(metrics) == {"dp/epsilon", "dp/update_norm", "dp/clipped", "dp/noise_std", "dp/released_norm"}
        assert metrics["dp/noise_std"] == pytest.approx(SIGMA * CLIP)
        assert metrics["dp/clipped"] in (0.0, 1.0) and metrics["dp/update_norm"] > 0
    assert len(s.runtime.sent) == 3


def test_epsilon_after_twenty_releases(dp):
    """20 releases at sigma 2.4, delta 1e-5: epsilon 1.77 after the first, 9.84 after the last."""
    torch.manual_seed(0)
    model = TinyNet()
    hook = dp.DPNoiseHook(dp.parse_config(NOISE_ENV), generator=torch.Generator().manual_seed(0))
    hook.transfer_type = "DIFF"
    recorder = RecordingLogger()
    trainer = types.SimpleNamespace(callbacks=[], loggers=[recorder], world_size=1)
    epsilons = []
    for r in range(20):
        ctx = types.SimpleNamespace(current_round=r, is_eval_only_round=False,
                                    input_model=types.SimpleNamespace(params=_perturbed(model.state_dict(), seed=r)))
        epsilons.append(hook.release(trainer, model, ctx).epsilon)
    assert epsilons == pytest.approx(EPSILONS_SIGMA_2_4, abs=0.005)
    assert epsilons[0] == pytest.approx(1.7716, abs=1e-4) and epsilons[-1] == pytest.approx(9.8416, abs=1e-4)
    assert [m["dp/epsilon"] for m, _ in recorder.dp_records()] == epsilons


def test_a_restarted_subprocess_does_not_under_count(dp):
    torch.manual_seed(0)
    model = TinyNet()
    hook = dp.DPNoiseHook(dp.parse_config(NOISE_ENV), generator=torch.Generator().manual_seed(0))
    hook.transfer_type = "DIFF"
    ctx = types.SimpleNamespace(current_round=5, is_eval_only_round=False,
                                input_model=types.SimpleNamespace(params=_perturbed(model.state_dict())))
    stats = hook.release(types.SimpleNamespace(callbacks=[], loggers=[], world_size=1), model, ctx)
    assert stats.releases == 6 and stats.epsilon == pytest.approx(EPSILONS_SIGMA_2_4[5], abs=0.005)


def _release_round(hook, model, r):
    hook.transfer_type = "DIFF"
    ctx = types.SimpleNamespace(current_round=r, is_eval_only_round=False,
                                input_model=types.SimpleNamespace(params=_perturbed(model.state_dict(), seed=r)))
    return hook.release(types.SimpleNamespace(callbacks=[], loggers=[], world_size=1), model, ctx)


def test_the_release_count_survives_a_subprocess_restart(dp, tmp_path):
    """A round repeated by a restarted subprocess after its update was sent counts as a new release."""
    torch.manual_seed(0)
    model = TinyNet()
    path = str(tmp_path / dp.STATE_FILE_NAME)
    first = dp.DPNoiseHook(dp.parse_config(NOISE_ENV), generator=torch.Generator().manual_seed(0), state_path=path)
    _release_round(first, model, 0)
    _release_round(first, model, 1)
    saved = json.loads((tmp_path / dp.STATE_FILE_NAME).read_text())
    assert saved["releases"] == 2 and saved["last_round"] == 1
    assert saved["epsilon"] == pytest.approx(EPSILONS_SIGMA_2_4[1], abs=0.005)

    restarted = dp.DPNoiseHook(dp.parse_config(NOISE_ENV), generator=torch.Generator().manual_seed(1), state_path=path)
    assert restarted.accountant.steps == 2
    stats = _release_round(restarted, model, 1)  # round 1 again: by round index alone it would be release 2
    assert stats.releases == 3 and stats.epsilon == pytest.approx(EPSILONS_SIGMA_2_4[2], abs=0.005)
    assert json.loads((tmp_path / dp.STATE_FILE_NAME).read_text())["releases"] == 3


def test_a_refused_release_does_not_change_the_saved_count(dp, tmp_path):
    torch.manual_seed(0)
    model = TinyNet()
    path = str(tmp_path / dp.STATE_FILE_NAME)
    env = {**NOISE_ENV, "ODELIA_DP_TARGET_EPSILON": "1.8"}  # one release (1.77) fits
    hook = dp.DPNoiseHook(dp.parse_config(env), generator=torch.Generator().manual_seed(0), state_path=path)
    _release_round(hook, model, 0)
    with pytest.raises(dp.PrivacyBudgetExceeded):
        _release_round(hook, model, 1)
    assert json.loads((tmp_path / dp.STATE_FILE_NAME).read_text())["releases"] == 1


@pytest.mark.parametrize("content,message", [
    ("{not json", "cannot be read"),
    (json.dumps({"releases": -1, "sigma": SIGMA, "clip_norm": CLIP, "delta": DELTA}), "no valid release count"),
    (json.dumps({"releases": True, "sigma": SIGMA, "clip_norm": CLIP, "delta": DELTA}), "no valid release count"),
    (json.dumps({"releases": 3, "sigma": 1.0, "clip_norm": CLIP, "delta": DELTA}), "other parameters"),
])
def test_an_unusable_count_file_stops_the_job(dp, tmp_path, content, message):
    path = tmp_path / dp.STATE_FILE_NAME
    path.write_text(content)
    with pytest.raises(RuntimeError, match=message):
        dp.DPNoiseHook(dp.parse_config(NOISE_ENV), state_path=str(path))


def test_saving_the_count_is_best_effort(dp, tmp_path, caplog):
    torch.manual_seed(0)
    model = TinyNet()
    hook = dp.DPNoiseHook(dp.parse_config(NOISE_ENV), generator=torch.Generator().manual_seed(0),
                          state_path=str(tmp_path / "missing_dir" / dp.STATE_FILE_NAME))
    with caplog.at_level(logging.WARNING):
        stats = _release_round(hook, model, 0)
    assert stats.releases == 1 and "could not save the release count" in caplog.text


def test_create_hook_keeps_the_count_in_the_working_directory(dp, tmp_path):
    """The working directory of the training subprocess is the job's app dir (the fixture chdirs to tmp_path)."""
    assert dp.create_hook(NOISE_ENV, None).state_path == str(tmp_path / dp.STATE_FILE_NAME)
    assert dp.create_hook({"ODELIA_DP_MEASURE_ONLY": "1"}, None).state_path is None
    assert dp.create_hook({"ODELIA_DP_EQUAL_WEIGHTS": "1"}, None).state_path is None


def test_budget_refusal_happens_at_the_right_round(dp):
    """Target 8 at sigma 2.4 allows 14 releases (epsilon 7.94); the 15th (round 14, epsilon 8.28) is refused
    before training and again at the release, with nothing sent and the model untouched."""
    torch.manual_seed(0)
    model = TinyNet()
    env = {**NOISE_ENV, "ODELIA_DP_TARGET_EPSILON": "8"}
    hook = dp.DPNoiseHook(dp.parse_config(env), generator=torch.Generator().manual_seed(0))
    hook.transfer_type = "DIFF"
    trainer = types.SimpleNamespace(callbacks=[], loggers=[], world_size=1)

    def ctx_for(r):
        return types.SimpleNamespace(current_round=r, total_rounds=None, is_eval_only_round=False,
                                     input_model=types.SimpleNamespace(params=_perturbed(model.state_dict(), seed=r)))

    for r in range(14):
        hook.after_receive(ctx_for(r))
        hook.release(trainer, model, ctx_for(r))
    assert hook.accountant.steps == 14
    assert hook.accountant.epsilon() == pytest.approx(7.94, abs=0.005)

    ctx = ctx_for(14)
    with pytest.raises(dp.PrivacyBudgetExceeded, match="round 14 would be release 15"):
        hook.after_receive(ctx)
    before = _clone(model.state_dict())
    with pytest.raises(dp.PrivacyBudgetExceeded):
        hook.release(trainer, model, ctx)
    assert all(torch.equal(model.state_dict()[k], before[k]) for k in before)
    assert hook.accountant.steps == 14 and len(hook.history) == 14


@pytest.mark.parametrize("total_rounds,final_eval_round,refused", [
    (20, False, True), (15, False, True), (14, False, False), (15, True, False), (16, True, True),
])
def test_a_job_configured_past_the_budget_is_refused_before_training(dp, total_rounds, final_eval_round, refused):
    hook = dp.DPNoiseHook(dp.parse_config({**NOISE_ENV, "ODELIA_DP_TARGET_EPSILON": "8"}),
                          final_eval_round=final_eval_round)
    ctx = types.SimpleNamespace(current_round=0, total_rounds=total_rounds, is_eval_only_round=False)
    if refused:
        with pytest.raises(dp.PrivacyBudgetExceeded, match="allows 14"):
            hook.after_receive(ctx)
    else:
        hook.after_receive(ctx)


def test_refused_round_sends_nothing_in_the_real_training_loop(dp, monkeypatch, tmp_path):
    """Target 3.0 allows two releases (1.77, 2.60). Round 2 is refused in after_receive, before training;
    a loop that skipped that check is still stopped in on_train_end, before FLCallback sends."""
    s = _stack(dp, monkeypatch, tmp_path, {**NOISE_ENV, "ODELIA_DP_TARGET_EPSILON": "3"}, total_rounds=None)
    s.run_round()
    s.run_round()
    assert len(s.runtime.sent) == 2
    with pytest.raises(dp.PrivacyBudgetExceeded):
        s.run_round()
    with pytest.raises(dp.PrivacyBudgetExceeded):
        s.run_round(skip_after_receive=True)
    assert len(s.runtime.sent) == 2 and s.hook.accountant.steps == 2


def test_evaluation_only_round_is_left_untouched(dp, monkeypatch, tmp_path):
    s = _stack(dp, monkeypatch, tmp_path, {**NOISE_ENV, "ODELIA_DP_TARGET_EPSILON": "1.8"}, total_rounds=2,
               final_eval_round=True)
    s.run_round()  # release 1: epsilon 1.77, the whole budget
    s.run_round(eval_only=True)  # would be over budget if it counted; it must not even check

    _, sent, _ = s.runtime.sent[1]
    local = s.snapshot.states[1]
    assert all(torch.equal(sent[k], local[k]) for k in local)  # sent exactly as the model was
    assert s.hook.accountant.steps == 1 and len(s.hook.history) == 1
    assert len(s.recorder.dp_records()) == 1


def test_evaluation_only_round_without_fit(dp, monkeypatch, tmp_path, caplog):
    """The call sequence of main.py's evaluation-only round (release contract 3.3): after_receive, then
    on_end; no before_fit, no trainer.fit, no after_round. The hook neither checks the budget nor counts."""
    s = _stack(dp, monkeypatch, tmp_path, {**NOISE_ENV, "ODELIA_DP_TARGET_EPSILON": "1.8"}, total_rounds=2,
               final_eval_round=True)
    s.run_round()  # training round 0: release 1, epsilon 1.77, the whole budget
    s.runtime.next_task()
    s.ctx.input_model = s.runtime.receive()
    s.ctx.current_round, s.ctx.is_final_round, s.ctx.is_eval_only_round = 1, True, True
    with caplog.at_level(logging.INFO):
        s.rh.call_hooks([s.hook], "after_receive", s.ctx)
        s.rh.call_hooks([s.hook], "on_end", s.ctx)
    assert s.hook.accountant.steps == 1 and len(s.runtime.sent) == 1
    assert "DP summary: 1 release(s), epsilon 1.7716" in caplog.text


def test_no_release_when_fl_callback_sends_nothing(dp, monkeypatch, tmp_path):
    s = _stack(dp, monkeypatch, tmp_path, NOISE_ENV)
    monkeypatch.setattr(s.api, "is_train", lambda: False)  # not a training task: FLCallback sends no model
    s.run_round()
    assert s.runtime.sent == [] and s.hook.accountant.steps == 0 and s.hook.history == []


def test_measure_only_sends_the_update_unchanged_and_streams_its_norm(dp, monkeypatch, tmp_path):
    env = {"ODELIA_DP_MEASURE_ONLY": "1", "ODELIA_DP_CLIP_NORM": "1e-6"}
    s = _stack(dp, monkeypatch, tmp_path, env, total_rounds=2)
    s.run_round()
    s.run_round()
    for r in range(2):
        received, local, (_, sent, _) = s.runtime.handed_out[r].params, s.snapshot.states[r], s.runtime.sent[r]
        assert all(torch.equal(sent[k], local[k]) for k in local)
        norm = math.sqrt(sum(float(((local[k] - received[k]).double() ** 2).sum()) for k in TRAINABLE))
        metrics, step = s.recorder.dp_records()[r]
        assert step == r and set(metrics) == {"dp/update_norm", "dp/clipped"}
        assert metrics["dp/update_norm"] == pytest.approx(norm, rel=1e-6)
        assert metrics["dp/clipped"] == 1.0  # C=1e-6 would have clipped it
    assert s.hook.accountant is None


@pytest.mark.parametrize("env", [{**NOISE_ENV, "ODELIA_DP_EQUAL_WEIGHTS": "1"}, {"ODELIA_DP_EQUAL_WEIGHTS": "1"}])
def test_equal_weights_report_one_step_per_round(dp, monkeypatch, tmp_path, env):
    s = _stack(dp, monkeypatch, tmp_path, env, total_rounds=2)
    s.run_round()
    s.run_round()
    assert [meta["NUM_STEPS_CURRENT_ROUND"] for _, _, meta in s.runtime.sent] == [1, 1]
    assert s.trainer.estimated_stepping_batches > 1
    if "ODELIA_DP_SIGMA" not in env:  # equal weights alone adds no noise
        assert all(torch.equal(s.runtime.sent[0][1][k], s.snapshot.states[0][k]) for k in TRAINABLE)
        assert s.recorder.dp_records() == []


def test_full_transfer_with_local_only_entries_is_refused(dp):
    torch.manual_seed(0)
    model = TinyNet()
    model.register_buffer("_class_weight", torch.tensor([0.5, 2.0]))
    received = _perturbed(model.state_dict())
    del received["_class_weight"]
    ctx = types.SimpleNamespace(current_round=0, is_eval_only_round=False,
                                input_model=types.SimpleNamespace(params=received))
    trainer = types.SimpleNamespace(callbacks=[], loggers=[], world_size=1)

    for transfer_type in ("FULL", None):
        hook = dp.DPNoiseHook(dp.parse_config(NOISE_ENV))
        hook.transfer_type = transfer_type
        with pytest.raises(RuntimeError, match="DIFF"):
            hook.release(trainer, model, ctx)
    hook = dp.DPNoiseHook(dp.parse_config(NOISE_ENV))
    hook.transfer_type = "DIFF"
    weight = model.state_dict()["_class_weight"].clone()
    assert hook.release(trainer, model, ctx).releases == 1
    assert torch.equal(model.state_dict()["_class_weight"], weight)  # not sent under DIFF, left alone


def test_read_transfer_type_from_the_client_api_config(dp, monkeypatch):
    client_api = pytest.importorskip("nvflare.client.api")
    from nvflare.client.config import ConfigKey

    monkeypatch.setattr(client_api, "get_config", lambda: {ConfigKey.TASK_EXCHANGE: {ConfigKey.TRANSFER_TYPE: "DIFF"}})
    assert dp._read_transfer_type() == "DIFF"

    def not_initialised():
        raise RuntimeError("APIContext is None")

    monkeypatch.setattr(client_api, "get_config", not_initialised)
    assert dp._read_transfer_type() is None


def test_callback_after_fl_callback_is_refused_before_training(dp, monkeypatch, tmp_path):
    s = _stack(dp, monkeypatch, tmp_path, NOISE_ENV)
    callbacks = [cb for cb in s.trainer.callbacks if cb is not s.hook.callback]
    s.trainer.callbacks = callbacks + [s.hook.callback]  # now after FLCallback
    s.runtime.next_task()
    s.ctx.input_model, s.ctx.current_round = s.runtime.current, 0
    with pytest.raises(RuntimeError, match="after FLCallback"):
        s.trainer.fit(s.model, train_dataloaders=_loader())
    assert s.runtime.sent == []


def test_missing_callback_is_caught_before_and_after_training(dp, monkeypatch, tmp_path):
    s = _stack(dp, monkeypatch, tmp_path, NOISE_ENV)
    s.ctx.current_round, s.ctx.is_eval_only_round = 0, False
    s.trainer.callbacks = [cb for cb in s.trainer.callbacks if cb is not s.hook.callback]
    with pytest.raises(RuntimeError, match="not in trainer.callbacks"):
        s.hook.before_fit(s.ctx)
    with pytest.raises(RuntimeError, match="did not run"):
        s.hook.after_round(s.ctx)


def test_banner_and_summary_state_the_guarantee(dp, caplog):
    caplog.set_level(logging.INFO)
    torch.manual_seed(0)
    model = TinyNet()
    model.bn = torch.nn.BatchNorm1d(2)
    hook = dp.DPNoiseHook(dp.parse_config({**NOISE_ENV, "ODELIA_DP_TARGET_EPSILON": "10"}),
                          logger=logging.getLogger("dp_banner"))
    hook.on_start(types.SimpleNamespace(model=model))
    hook.on_end(types.SimpleNamespace())
    text = caplog.text
    assert "DP ACTIVE" in text and "client-level" in text and "NOT record-level" in text
    assert "allows 20 training rounds" in text
    assert "BatchNorm" in text and "round-0 statistics" in text
    assert "DP summary: 0 release(s)" in text

    caplog.clear()
    measure = dp.DPNoiseHook(dp.parse_config({"ODELIA_DP_MEASURE_ONLY": "1"}), logger=logging.getLogger("dp_banner"))
    measure.on_start(types.SimpleNamespace(model=None))
    assert "NO privacy guarantee" in caplog.text


# --------------------------------------------------------------------------------------------------
# Training precision: "16-mixed" (threedcnn_ptl.prepare_training) keeps float32 parameters
# --------------------------------------------------------------------------------------------------


def test_16_mixed_precision_plugin_does_not_convert_parameters():
    """The GPU configuration: MixedPrecision only autocasts; convert_module leaves the parameters alone."""
    import warnings

    from pytorch_lightning.plugins.precision import MixedPrecision

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a CUDA GradScaler on a CPU-only host warns and disables itself
        plugin = MixedPrecision(precision="16-mixed", device="cuda")
    module = TinyNet()
    assert plugin.convert_module(module) is module
    assert {p.dtype for p in module.parameters()} == {torch.float32}


def test_mixed_precision_training_keeps_float32_parameters_through_a_dp_round(dp, monkeypatch, tmp_path):
    from pytorch_lightning.plugins.precision import MixedPrecision

    try:  # on CPU Lightning maps "16-mixed" to bf16 autocast (same MixedPrecision plugin) or refuses it
        s = _stack(dp, monkeypatch, tmp_path, NOISE_ENV, precision="16-mixed")
    except Exception:
        s = _stack(dp, monkeypatch, tmp_path, NOISE_ENV, precision="bf16-mixed")
    assert isinstance(s.trainer.precision_plugin, MixedPrecision)
    s.run_round()
    _, sent, _ = s.runtime.sent[0]
    assert {sent[k].dtype for k in TRAINABLE} == {torch.float32}
    assert {p.dtype for p in s.model.parameters()} == {torch.float32}
    assert s.hook.history[0].releases == 1
