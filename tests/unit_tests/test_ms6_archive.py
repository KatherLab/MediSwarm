"""Tests of the MS6 global archive (application/jobs/_shared/custom/ms6_archive.py), CPU only.

The client streamer and the server archive talk through NVFlare's real ObjectStreamer and FileStreamer;
only the aux transport between them is replaced by a loopback that FOBS-encodes every message, so the chunk
protocol, the stream-done callback and the reply carrying the archive status run as in a job. The swarm
controller is a stand-in that records globals the way ClientSideController.record_last_result does
(persistor.save writes FL_global_model.pt).
"""

import collections
import importlib
import json
import logging
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nvflare.app_common.streamers.file_streamer")

from nvflare.apis.event_type import EventType  # noqa: E402
from nvflare.apis.fl_constant import ReservedKey  # noqa: E402
from nvflare.apis.fl_context import FLContext  # noqa: E402
from nvflare.apis.streaming import StreamableEngine  # noqa: E402
from nvflare.app_common.app_constant import AppConstants  # noqa: E402
from nvflare.app_common.app_event_type import AppEventType  # noqa: E402
from nvflare.app_common.ccwf.common import Constant  # noqa: E402
from nvflare.fuel.utils import fobs  # noqa: E402
from nvflare.private.stream_runner import ObjectStreamer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_CUSTOM_DIR = REPO_ROOT / "application" / "jobs" / "_shared" / "custom"
JOB_ID = "job-ms6"


@pytest.fixture
def ms6(monkeypatch):
    monkeypatch.syspath_prepend(str(SHARED_CUSTOM_DIR))
    for name in ("round_hooks", "ms6_capture", "ms6_archive"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    archive = importlib.import_module("ms6_archive")
    # Disk-space reserves are about real disks; keep them out of the way of a test's tmp dir.
    monkeypatch.setattr(archive, "MIN_FREE_RESERVE_BYTES", 0)
    yield types.SimpleNamespace(archive=archive, capture=importlib.import_module("ms6_capture"))
    for name in ("round_hooks", "ms6_capture", "ms6_archive"):
        sys.modules.pop(name, None)


# --------------------------------------------------------------------------------------------------
# Loopback transport and stand-ins
# --------------------------------------------------------------------------------------------------


class LoopbackAux:
    """One side's AuxRunner: a request goes to the peer side's handler, FOBS-encoded like on the wire."""

    def __init__(self, make_ctx):
        self.handlers = {}
        self.peer = None
        self.make_ctx = make_ctx
        self.requests = 0

    def register_aux_message_handler(self, topic, message_handle_func):
        self.handlers[topic] = message_handle_func

    def send_aux_request(self, targets, topic, request, timeout, fl_ctx, bulk_send=False, optional=False,
                         secure=False):
        self.requests += 1
        handler = self.peer.handlers.get(topic) if self.peer else None
        if handler is None:
            return {}
        receiver_ctx = self.peer.make_ctx()
        peer_ctx = FLContext()
        peer_ctx.set_public_props(fl_ctx.get_all_public_props())
        receiver_ctx.set_peer_context(peer_ctx)
        reply = handler(topic, fobs.loads(fobs.dumps(request)), receiver_ctx)
        return {"server": fobs.loads(fobs.dumps(reply))}


class FakeEngine(StreamableEngine):
    def __init__(self, identity, job_id, run_dir=None):
        self.identity = identity
        self.job_id = job_id
        self.run_dir = run_dir
        self.aux = LoopbackAux(self.new_context)
        self.streamer = ObjectStreamer(self.aux)

    def new_context(self):
        ctx = FLContext()
        ctx.put(key=ReservedKey.ENGINE, value=self, private=True, sticky=False)
        ctx.put(key=ReservedKey.RUN_NUM, value=self.job_id, private=False, sticky=True)
        ctx.put(key=ReservedKey.IDENTITY_NAME, value=self.identity, private=False, sticky=False)
        return ctx

    def stream_objects(self, channel, topic, stream_ctx, targets, producer, fl_ctx, optional=False, secure=False):
        return self.streamer.stream(channel=channel, topic=topic, stream_ctx=stream_ctx, targets=targets,
                                    producer=producer, fl_ctx=fl_ctx, secure=secure, optional=optional)

    def register_stream_processing(self, channel, topic, factory, stream_done_cb=None, consumed_cb=None,
                                   **cb_kwargs):
        self.streamer.register_stream_processing(channel, topic, factory, stream_done_cb, consumed_cb, **cb_kwargs)

    def shutdown_streamer(self):
        self.streamer.shutdown()

    def get_workspace(self):
        return types.SimpleNamespace(get_run_dir=lambda job_id: str(self.run_dir))

    def fire_event(self, event_type, fl_ctx):
        pass


class FakePersistor:
    def __init__(self, path):
        self._ckpt_save_path = str(path)

    def save(self, state):
        torch.save({"model": state, "train_conf": {"train": {"model": "Tiny"}}}, self._ckpt_save_path)


class FakeController:
    """Records globals like ClientSideController.record_last_result: persistor.save writes FL_global_model.pt."""

    def __init__(self, app_dir, num_rounds=3, starting=True):
        self.persistor = FakePersistor(Path(app_dir) / "FL_global_model.pt")
        self.config = {AppConstants.NUM_ROUNDS: num_rounds, Constant.START_ROUND: 0}
        self.is_starting_client = starting
        self.recorded = []

    def get_config_prop(self, name, default=None):
        return self.config.get(name, default)

    def record_last_result(self, fl_ctx, round_num, result):
        self.recorded.append(round_num)
        self.persistor.save(result)


def make_state(seed):
    generator = torch.Generator().manual_seed(seed)
    return collections.OrderedDict(
        [("layer.weight", torch.randn(64, 32, generator=generator)),
         ("layer.bias", torch.randn(64, generator=generator)),
         ("_class_weight", torch.ones(3))]
    )


@pytest.fixture
def swarm(ms6, tmp_path):
    """One client (streamer + controller) and the server (archive), joined by the loopback."""
    client_engine = FakeEngine("SITE_A", JOB_ID)
    server_engine = FakeEngine("server", JOB_ID, run_dir=tmp_path / "server_run")
    server_engine.run_dir.mkdir()
    client_engine.aux.peer, server_engine.aux.peer = server_engine.aux, client_engine.aux
    app_dir = tmp_path / "app_SITE_A"
    app_dir.mkdir()

    def start_archive(enabled=True, archive_root=None):
        archive = ms6.archive.MS6GlobalArchive(enabled=enabled, archive_root=str(archive_root or tmp_path / "archive"))
        archive.handle_event(EventType.START_RUN, server_engine.new_context())
        return archive

    def start_client(enabled=True, **streamer_args):
        streamer_args.setdefault("retry_interval", 0.0)
        streamer = ms6.archive.MS6GlobalStreamer(enabled=enabled, **streamer_args)
        controller = FakeController(app_dir)
        ctx = client_engine.new_context()
        ctx.set_prop(Constant.EXECUTOR, controller, private=True, sticky=False)
        streamer.handle_event(Constant.EXECUTOR_INITIALIZED, ctx)
        return types.SimpleNamespace(streamer=streamer, controller=controller)

    def build(enabled=True, archive_enabled=True, archive_root=None, **streamer_args):
        archive = start_archive(archive_enabled, archive_root)
        client = start_client(enabled, **streamer_args)
        return types.SimpleNamespace(archive=archive, streamer=client.streamer, controller=client.controller)

    def start_round(parts, round_number, state):
        ctx = client_engine.new_context()
        ctx.set_prop(AppConstants.CURRENT_ROUND, round_number, private=True, sticky=True)
        ctx.set_prop(AppConstants.GLOBAL_MODEL, {"weights": {k: v.numpy() for k, v in state.items()}},
                     private=True, sticky=True)
        parts.streamer.handle_event(AppEventType.ROUND_STARTED, ctx)

    def aggregate(parts, round_number, state):
        parts.controller.record_last_result(client_engine.new_context(), round_number, state)

    yield types.SimpleNamespace(build=build, start_archive=start_archive, start_client=start_client,
                                start_round=start_round, aggregate=aggregate, app_dir=app_dir,
                                archive_dir=tmp_path / "archive" / JOB_ID, client_engine=client_engine,
                                server_engine=server_engine, tmp_path=tmp_path)


def _manifest(directory):
    return json.loads((Path(directory) / "manifest.json").read_text())


# --------------------------------------------------------------------------------------------------
# End to end: client streamer -> server archive
# --------------------------------------------------------------------------------------------------


def test_every_global_of_the_run_is_archived_with_matching_digests(swarm, ms6):
    parts = swarm.build(chunk_size=4096)  # several chunks per file
    states = [make_state(seed) for seed in range(4)]
    swarm.start_round(parts, 0, states[0])  # starting client: the initial global G_0
    for r in range(3):
        swarm.aggregate(parts, r, states[r + 1])  # round r's aggregate is G_{r+1}
    # The final round's record_last_result returns only once everything reached the archive.
    assert parts.streamer._pending == 0
    parts.archive.handle_event(EventType.END_RUN, swarm.server_engine.new_context())
    parts.streamer.handle_event(EventType.END_RUN, swarm.client_engine.new_context())

    archive_dir = swarm.archive_dir
    manifest = _manifest(archive_dir)
    assert manifest["job_id"] == JOB_ID and manifest["fallback_dir"] is False
    assert sorted(manifest["rounds"], key=int) == ["0", "1", "2", "3"]
    assert manifest["missing_rounds"] == [] and manifest["conflicts"] == [] and manifest["rejected"] == []
    assert manifest["expected"] == {"start_round": 0, "num_rounds": 3}
    sums = (archive_dir / "SHA256SUMS").read_text().splitlines()
    assert len(sums) == 4
    for r in range(4):
        path = archive_dir / f"round_{r:02d}_global.pt"
        entry = manifest["rounds"][str(r)]
        assert entry["file"] == path.name and entry["sender"] == "SITE_A"
        assert entry["kind"] == ("initial" if r == 0 else "aggregated")
        assert entry["sha256"] == ms6.capture.file_sha256(path) and f"{entry['sha256']}  {path.name}" in sums
        model = torch.load(path, map_location="cpu", weights_only=False)["model"]
        for key, value in states[r].items():
            assert torch.equal(model[key], value)
        # The digest a site's own capture of G_r would carry.
        assert entry["tensor_sha256"] == ms6.capture.tensor_digest(states[r])[0]
        assert entry["n_tensors"] == 2  # _class_weight left out
    # Delivered copies leave the outbox; the pointer for download_job lands in the server run dir.
    assert not [p for p in (swarm.app_dir / "ms6_outbox").iterdir() if not p.name.startswith(".")]
    assert json.loads((swarm.server_engine.run_dir / "ms6_global_archive.json").read_text())["rounds"].keys() == \
        manifest["rounds"].keys()


def test_a_disabled_streamer_leaves_the_controller_untouched(swarm):
    parts = swarm.build(enabled=False)
    assert "record_last_result" not in vars(parts.controller)  # still the class method: nothing wrapped
    swarm.aggregate(parts, 0, make_state(1))
    assert not (swarm.app_dir / "ms6_outbox").exists()
    assert _manifest(swarm.archive_dir)["rounds"] == {}


def test_a_disabled_archive_registers_nothing(swarm, ms6):
    parts = swarm.build(archive_enabled=False, enabled=False)
    assert parts.archive.archive_dir is None
    assert not (swarm.tmp_path / "archive").exists()
    assert swarm.server_engine.streamer.registry.find(ms6.archive.CHANNEL, ms6.archive.TOPIC) is None


def test_without_a_receiver_training_goes_on_and_the_copy_stays_in_the_outbox(swarm, caplog):
    parts = swarm.build(archive_enabled=False, max_attempts=2)
    state = make_state(1)
    with caplog.at_level(logging.WARNING):
        swarm.aggregate(parts, 2, state)  # the final round: waits for the (failing) delivery, then returns
    assert parts.controller.recorded == [2]  # the controller's own work happened
    assert (swarm.app_dir / "ms6_outbox" / "round_03_global.pt").exists()
    assert "NOT archived" in caplog.text and "attempt 2/2" in caplog.text
    parts.streamer.handle_event(EventType.END_RUN, swarm.client_engine.new_context())
    assert "did not reach the coordinator" in caplog.text


def test_the_final_round_waits_for_delivery_but_not_forever(swarm, ms6, monkeypatch, caplog):
    parts = swarm.build(final_stream_timeout=0.3)
    release = threading.Event()
    real_stream = ms6.archive.FileStreamer.stream_file

    def slow_stream(**kwargs):
        release.wait(5)
        return real_stream(**kwargs)

    monkeypatch.setattr(ms6.archive.FileStreamer, "stream_file", staticmethod(slow_stream))
    started = time.monotonic()
    with caplog.at_level(logging.ERROR):
        swarm.aggregate(parts, 2, make_state(3))
    assert 0.25 < time.monotonic() - started < 3
    assert "still not delivered" in caplog.text
    release.set()
    assert parts.streamer.wait_until_idle(10)
    assert "3" in _manifest(swarm.archive_dir)["rounds"]


def test_only_the_starting_client_archives_the_initial_global_and_only_once(swarm, ms6):
    parts = swarm.build()
    parts.controller.is_starting_client = False
    swarm.start_round(parts, 0, make_state(0))
    assert parts.streamer._pending == 0 and not (swarm.app_dir / "ms6_outbox").exists()
    parts.controller.is_starting_client = True
    swarm.start_round(parts, 1, make_state(1))  # not the start round
    swarm.start_round(parts, 0, make_state(0))
    swarm.start_round(parts, 0, make_state(5))  # a repeated ROUND_STARTED for round 0
    assert parts.streamer.wait_until_idle(10)
    rounds = _manifest(swarm.archive_dir)["rounds"]
    assert list(rounds) == ["0"]
    assert rounds["0"]["tensor_sha256"] == ms6.capture.tensor_digest(make_state(0))[0]


def test_resending_a_round_is_a_duplicate_and_a_different_global_is_kept_as_conflict(swarm, ms6):
    archive = swarm.start_archive()
    first = swarm.start_client()
    swarm.aggregate(first, 0, make_state(1))
    assert first.streamer.wait_until_idle(10)
    # Another client (e.g. the #595 take-over aggregator) sends round 1 again: same bytes, then other bytes.
    second = swarm.start_client()
    swarm.aggregate(second, 0, make_state(1))
    assert second.streamer.wait_until_idle(10)
    third = swarm.start_client()
    swarm.aggregate(third, 0, make_state(2))
    assert third.streamer.wait_until_idle(10)
    archive.handle_event(EventType.END_RUN, swarm.server_engine.new_context())
    manifest = _manifest(swarm.archive_dir)
    assert list(manifest["rounds"]) == ["1"]
    (conflict,) = manifest["conflicts"]
    assert conflict["file"].startswith("round_01_global.conflict-SITE_A-")
    assert (swarm.archive_dir / conflict["file"]).exists()
    assert not [p for p in (swarm.app_dir / "ms6_outbox").iterdir() if not p.name.startswith(".")]


def test_a_corrupted_transfer_is_refused_and_retried(swarm, ms6, monkeypatch, caplog):
    parts = swarm.build(max_attempts=2)
    real_describe = parts.streamer._describe

    def wrong_digest(path):
        info = real_describe(path)
        info[ms6.archive.CTX_SHA256] = "0" * 64
        return info

    monkeypatch.setattr(parts.streamer, "_describe", wrong_digest)
    with caplog.at_level(logging.WARNING):
        swarm.aggregate(parts, 2, make_state(1))
    manifest = _manifest(swarm.archive_dir)
    assert manifest["rounds"] == {} and len(manifest["rejected"]) == 2
    assert "sender declared" in manifest["rejected"][0]["reason"]
    assert not list((swarm.archive_dir / ".incoming").iterdir())  # no partial file left behind
    assert (swarm.app_dir / "ms6_outbox" / "round_03_global.pt").exists()


def test_a_stream_from_another_job_is_refused_without_retry(swarm, ms6, caplog):
    parts = swarm.build(max_attempts=3)
    swarm.client_engine.job_id = "some-other-job"
    with caplog.at_level(logging.WARNING):
        swarm.aggregate(parts, 2, make_state(1))
    manifest = _manifest(swarm.archive_dir)
    assert manifest["rounds"] == {}
    assert len(manifest["rejected"]) == 1 and "some-other-job" in manifest["rejected"][0]["reason"]
    assert "attempt 2/3" not in caplog.text


def test_files_above_the_size_cap_are_refused(swarm, ms6):
    parts = swarm.build(max_attempts=1)
    parts.archive.max_file_bytes = 1000
    swarm.aggregate(parts, 2, make_state(1))
    assert _manifest(swarm.archive_dir)["rounds"] == {}
    assert (swarm.app_dir / "ms6_outbox" / "round_03_global.pt").exists()


def test_an_unwritable_archive_root_falls_back_to_the_run_directory(swarm, caplog):
    blocker = swarm.tmp_path / "file_not_dir"
    blocker.write_text("")
    with caplog.at_level(logging.WARNING):
        parts = swarm.build(archive_root=blocker / "archive")
    fallback = swarm.server_engine.run_dir / "ms6_global_archive"
    assert parts.archive.archive_dir == fallback and "not usable" in caplog.text
    swarm.aggregate(parts, 2, make_state(1))
    manifest = _manifest(fallback)
    assert manifest["fallback_dir"] is True and list(manifest["rounds"]) == ["3"]


def test_missing_rounds_are_reported_at_the_end(swarm, caplog):
    parts = swarm.build()
    swarm.aggregate(parts, 0, make_state(1))
    swarm.aggregate(parts, 2, make_state(3))
    with caplog.at_level(logging.ERROR):
        parts.archive.handle_event(EventType.END_RUN, swarm.server_engine.new_context())
    assert _manifest(swarm.archive_dir)["missing_rounds"] == [0, 2]
    assert "MISSING" in caplog.text


def test_the_server_never_unpickles_what_it_receives(swarm, ms6, monkeypatch):
    parts = swarm.build()

    def no_load(*args, **kwargs):
        raise AssertionError("the archive must not deserialize client files")

    calls = []
    real_load = torch.load

    def guarded_load(path, *args, **kwargs):
        # The sender reads its own outbox file for the tensor digest; the server must not read anything.
        calls.append(str(path))
        if str(swarm.archive_dir) in str(path):
            no_load()
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded_load)
    swarm.aggregate(parts, 2, make_state(1))
    assert "3" in _manifest(swarm.archive_dir)["rounds"]
    assert all(str(swarm.app_dir) in call for call in calls)


def test_the_archive_labels_globals_with_the_model_the_persistor_pins(swarm, monkeypatch):
    """The client job does not see the launcher's MODEL_NAME; the job builder pins model_name on the persistor."""
    monkeypatch.delenv("MODEL_NAME", raising=False)
    parts = swarm.build()
    parts.controller.persistor.model_name = "MST"
    swarm.aggregate(parts, 0, make_state(1))
    assert parts.streamer.wait_until_idle(30)
    assert _manifest(swarm.archive_dir)["rounds"]["1"]["model_name"] == "MST"
    parts.streamer.model_name = "explicit"  # the component arg wins
    assert parts.streamer._model_name() == "explicit"
    parts.streamer.model_name = None
    del parts.controller.persistor.model_name
    monkeypatch.setenv("MODEL_NAME", "FromEnv")  # last resort: the client job's own environment
    assert parts.streamer._model_name() == "FromEnv"
    parts.streamer.handle_event(EventType.END_RUN, swarm.client_engine.new_context())


def test_component_arguments_are_validated(ms6):
    with pytest.raises(ValueError):
        ms6.archive.MS6GlobalStreamer(chunk_size=0)
    with pytest.raises(ValueError):
        ms6.archive.MS6GlobalStreamer(max_attempts=0)
    with pytest.raises(ValueError):
        ms6.archive.MS6GlobalStreamer(final_stream_timeout=0)
    with pytest.raises(ValueError):
        ms6.archive.MS6GlobalArchive(max_file_bytes=0)
    # Defaults: off.
    assert ms6.archive.MS6GlobalStreamer().enabled is False
    assert ms6.archive.MS6GlobalArchive().enabled is False and ms6.archive.MS6GlobalArchive().archive_root == "/archive"


@pytest.mark.parametrize("template", ["ODELIA_ternary_classification", "challenge_1DivideAndConquer"])
def test_job_templates_carry_both_components_disabled(template):
    config_dir = REPO_ROOT / "application" / "jobs" / template / "app" / "config"
    client = (config_dir / "config_fed_client.conf").read_text()
    server = (config_dir / "config_fed_server.conf").read_text()
    for text, component_id, path in (
        (client, "ms6_global_streamer", "ms6_archive.MS6GlobalStreamer"),
        (server, "ms6_global_archive", "ms6_archive.MS6GlobalArchive"),
    ):
        block = text[text.index(f'id = "{component_id}"'):]
        block = block[: block.index("}") + 1]
        assert f'path = "{path}"' in block
        assert "enabled = false" in block
    assert 'archive_root = "/archive"' in server[server.index('id = "ms6_global_archive"'):]
