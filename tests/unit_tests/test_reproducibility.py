"""Tests of the reproducibility round hook: ODELIA_SEED and the round-0 warm-start digest check.

The hook is implicit: load_hooks adds it whenever one of its variables is present, because the job builder
(scripts/admin/build_consortium_job.py) sets both variables without listing a module. The contract test at
the end pins that agreement, so neither side can drift into a silently ignored seed or digest.
The whole-run behaviour (a mismatch stops a site before training, a seed reproduces a run) is covered end to
end through main.py in test_al_subset.py.
"""

import importlib
import importlib.util
import logging
import sys
import types

import pytest

from conftest import REPO_ROOT, SHARED_CUSTOM_DIR

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("pytorch_lightning")

MODULES = ("reproducibility", "al_subset", "al_selection", "state_digest", "loader_workers", "round_hooks")
SWARM = {"TRAINING_MODE": "swarm"}
DIGEST = "ab" * 32


@pytest.fixture
def rp(monkeypatch):
    """reproducibility imported the way main.py's load_hooks imports it (custom/ on sys.path)."""
    monkeypatch.syspath_prepend(str(SHARED_CUSTOM_DIR))
    for name in MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module("reproducibility")
    yield module
    for name in MODULES:
        sys.modules.pop(name, None)


def _ctx(round_num, params, data_module=None):
    import round_hooks

    ctx = round_hooks.RoundContext(logger=logging.getLogger("t"), site_name="SITE_A", job_id="job-1",
                                   data_module=data_module)
    ctx.current_round, ctx.total_rounds = round_num, 5
    ctx.input_model = types.SimpleNamespace(params=params)
    return ctx


def _params():
    generator = torch.Generator().manual_seed(3)
    return {"layer.weight": torch.randn(3, 4, generator=generator), "layer.bias": torch.randn(3, generator=generator)}


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------


def test_nothing_set_means_no_hook(rp):
    assert rp.create_hook({}, logging.getLogger("t")) is None


def test_values_are_parsed_and_normalised(rp):
    hook = rp.create_hook({"ODELIA_SEED": "11", "ODELIA_WARM_START_SHA256": DIGEST.upper()}, None)
    assert (hook.train_seed, hook.warm_start_sha256) == (11, DIGEST)


@pytest.mark.parametrize(
    "env,message",
    [
        ({"ODELIA_SEED": "-1"}, "must be in"),
        ({"ODELIA_SEED": str(2**32)}, "must be in"),
        ({"ODELIA_SEED": "seven"}, "must be an integer"),
        ({"ODELIA_SEED": " "}, "set but empty"),
        ({"ODELIA_WARM_START_SHA256": "1234"}, "64-character hex"),
        ({"ODELIA_WARM_START_SHA256": "g" * 64}, "64-character hex"),
        ({"ODELIA_WARM_START_SHA256": ""}, "set but empty"),
    ],
)
def test_malformed_values_fail_at_load(rp, env, message):
    with pytest.raises(rp.ReproducibilityConfigError, match=message):
        rp.create_hook(env, None)


# --------------------------------------------------------------------------------------------------
# load_hooks adds the hook by itself
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("variable,value", [("ODELIA_SEED", "7"), ("ODELIA_WARM_START_SHA256", DIGEST)])
def test_implicit_without_listing(rp, variable, value):
    import round_hooks

    hooks = round_hooks.load_hooks({**SWARM, variable: value}, logging.getLogger("t"))
    assert [h.name for h in hooks] == ["reproducibility"]


def test_implicit_hook_runs_before_the_listed_ones(rp):
    import round_hooks

    env = {**SWARM, "ODELIA_ROUND_HOOKS": "al_subset", "ODELIA_AL_STRATEGY": "random", "ODELIA_AL_BUDGET": "3",
           "ODELIA_WARM_START_SHA256": DIGEST}
    assert [h.name for h in round_hooks.load_hooks(env)] == ["reproducibility", "al_subset"]


def test_an_explicitly_listed_position_is_kept_and_never_duplicated(rp):
    import round_hooks

    env = {**SWARM, "ODELIA_ROUND_HOOKS": "al_subset,reproducibility", "ODELIA_AL_STRATEGY": "random",
           "ODELIA_AL_BUDGET": "3", "ODELIA_SEED": "1"}
    assert [h.name for h in round_hooks.load_hooks(env)] == ["al_subset", "reproducibility"]


def test_not_added_without_its_variables_and_ignored_outside_swarm_mode(rp, caplog):
    import round_hooks

    assert round_hooks.load_hooks(dict(SWARM)) == []
    with caplog.at_level(logging.WARNING):
        assert round_hooks.load_hooks({"TRAINING_MODE": "local_training", "ODELIA_SEED": "7"}) == []
    assert "ODELIA_SEED" in caplog.text


def test_builder_contract_both_variables_need_no_listed_hook(rp):
    """build_consortium_job.py writes --seed/--warm-start-digest without ODELIA_ROUND_HOOKS (CODE_READ_ENV)."""
    path = REPO_ROOT / "scripts" / "admin" / "build_consortium_job.py"
    if not path.is_file():
        pytest.skip("builder not in this tree")
    name = "_test_admin_build_consortium_job"
    builder = sys.modules.get(name)
    if builder is None:
        spec = importlib.util.spec_from_file_location(name, path)
        builder = importlib.util.module_from_spec(spec)
        sys.modules[name] = builder
        spec.loader.exec_module(builder)
    env = {"ODELIA_SEED": "4", "ODELIA_WARM_START_SHA256": DIGEST}
    assert builder.derive_hooks(env) == []
    assert set(env) <= set(builder.CODE_READ_ENV)

    import round_hooks

    assert [h.name for h in round_hooks.load_hooks({**SWARM, **env})] == ["reproducibility"]


# --------------------------------------------------------------------------------------------------
# Round-0 warm-start digest check
# --------------------------------------------------------------------------------------------------


def test_round0_digest_match_passes_and_is_recorded(rp):
    import state_digest

    params = _params()
    hook = rp.ReproducibilityHook(warm_start_sha256=state_digest.state_digest(params))
    ctx = _ctx(0, params)
    hook.after_receive(ctx)
    assert ctx.state["reproducibility"]["round0_global_digest"] == state_digest.state_digest(params)
    hook.after_receive(_ctx(1, {"anything": torch.zeros(1)}))  # later rounds are not checked


def test_round0_digest_mismatch_stops_the_job(rp):
    hook = rp.ReproducibilityHook(warm_start_sha256=DIGEST)
    with pytest.raises(rp.WarmStartDigestMismatch, match="did not seed round 0 from the intended") as excinfo:
        hook.after_receive(_ctx(0, _params()))
    assert DIGEST in str(excinfo.value) and "SITE_A" in str(excinfo.value)


def test_a_round0_task_without_parameters_stops_the_job(rp):
    with pytest.raises(rp.WarmStartDigestMismatch, match="carries no model parameters"):
        rp.ReproducibilityHook(warm_start_sha256=DIGEST).after_receive(_ctx(0, {}))


def test_digest_ignores_the_numpy_vs_torch_transport_and_the_class_weight_buffer(rp):
    import state_digest

    params = _params()
    hook = rp.ReproducibilityHook(warm_start_sha256=state_digest.state_digest(params))
    received = {k: v.numpy() for k, v in params.items()}
    received["_class_weight"] = np.ones(3, dtype=np.float32)  # a site-local buffer is not part of the global
    hook.after_receive(_ctx(0, received))


def test_a_process_that_starts_late_cannot_check_round0(rp, caplog):
    hook = rp.ReproducibilityHook(warm_start_sha256=DIGEST)
    with caplog.at_level(logging.WARNING):
        hook.after_receive(_ctx(3, _params()))
    assert "cannot be checked" in caplog.text


# --------------------------------------------------------------------------------------------------
# ODELIA_SEED
# --------------------------------------------------------------------------------------------------


class _Loader:
    def __init__(self):
        self.shut = False
        self._iterator = types.SimpleNamespace(_shutdown_workers=self._shutdown)

    def _shutdown(self):
        self.shut = True


def test_seed_sets_the_rngs_and_the_sampler_seed_and_drops_a_cached_loader(rp):
    loader = _Loader()
    data_module = types.SimpleNamespace(seed=0, _train_loader=loader)
    ctx = _ctx(None, None, data_module=data_module)
    rp.ReproducibilityHook(train_seed=1234).on_start(ctx)
    assert torch.initial_seed() == 1234
    first = torch.rand(3)
    torch.manual_seed(1234)
    assert torch.equal(first, torch.rand(3))
    assert data_module.seed == 1234 and data_module._train_loader is None and loader.shut
    assert ctx.state["reproducibility"]["train_seed"] == 1234


def test_without_a_seed_on_start_changes_nothing(rp):
    data_module = types.SimpleNamespace(seed=0, _train_loader="cached")
    rp.ReproducibilityHook(warm_start_sha256=DIGEST).on_start(_ctx(None, None, data_module=data_module))
    assert data_module.seed == 0 and data_module._train_loader == "cached"
