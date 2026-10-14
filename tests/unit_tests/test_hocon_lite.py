"""scripts/admin/hocon_lite.py: the job builder's dependency-free HOCON reader and patcher.

The builder runs on admin hosts without pyhocon, so it reads the job configs itself. The
cross-check below proves that reading equals pyhocon's (NVFlare's parser) on every job
config in the repository; the edit tests prove that patching keeps comments and layout.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
JOB_CONFIGS = sorted(
    list((REPO_ROOT / "application" / "jobs").glob("*/meta.conf"))
    + list((REPO_ROOT / "application" / "jobs").glob("*/app/config/*.conf"))
)


def _load():
    name = "_test_admin_hocon_lite"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "admin" / "hocon_lite.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hocon = _load()


def test_repo_has_job_configs():
    assert len(JOB_CONFIGS) >= 20


@pytest.mark.parametrize("path", JOB_CONFIGS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_every_job_config_reads_like_pyhocon(path):
    text = path.read_text()
    ours = hocon.loads(text)
    assert hocon.duplicate_keys(hocon.parse(text)) == []
    pyhocon = pytest.importorskip("pyhocon")
    reference = hocon.parse_with_pyhocon(text)
    assert reference is not None and pyhocon
    assert ours == reference


def test_insert_into_an_object_that_only_holds_a_comment_keeps_the_comment():
    text = (
        "{\n"
        "  model {\n"
        "    path = \"x.create\"\n"
        "    args {\n"
        "      # chosen by MODEL_NAME\n"
        "    }\n"
        "  }\n"
        "}\n"
    )
    conf = hocon.ConfText(text)
    conf.set_value(hocon.nested("model", "args"), "model_name", "MST", comment="pinned")
    assert "      # chosen by MODEL_NAME\n      model_name = \"MST\"  # pinned\n    }" in conf.text
    assert hocon.loads(conf.text) == {"model": {"path": "x.create", "args": {"model_name": "MST"}}}


def test_inline_objects_are_opened_up():
    conf = hocon.ConfText('components = [\n  {\n    id = "g"\n    args {}\n  }\n]\n')
    conf.set_value(hocon.component_args("g"), "enabled", True)
    assert hocon.loads(conf.text)["components"][0]["args"] == {"enabled": True}
    conf = hocon.ConfText("a { b = 1 }\n")
    conf.set_value(hocon.nested("a"), "c", "x")
    assert hocon.loads(conf.text) == {"a": {"b": 1, "c": "x"}}


def test_replacing_a_value_keeps_its_trailing_comment():
    conf = hocon.ConfText("workflows = [\n  {\n    id = \"w\"\n    args {\n      n = 5  # keep me\n    }\n  }\n]\n")
    wf = lambda root: hocon.get_value(hocon.find_in_list(root, "workflows", "w"), "args")  # noqa: E731
    conf.set_value(wf, "n", 7)
    conf.set_value(wf, "clients", ["A", "B"])
    assert "n = 7  # keep me" in conf.text
    assert hocon.loads(conf.text)["workflows"][0]["args"] == {"n": 7, "clients": ["A", "B"]}


def test_braceless_root_append_and_list_replace():
    conf = hocon.ConfText('name = "x"\ndeploy_map {\n  app = [\n    "@ALL"\n  ]\n}\nmin_clients = 2\n')
    conf.set_value(hocon.root_object, "name", "job")
    conf.set_value(hocon.nested("deploy_map"), "app", ["server", "A"])
    conf.set_value(hocon.root_object, "mandatory_clients", ["A"])
    assert hocon.loads(conf.text) == {
        "name": "job",
        "deploy_map": {"app": ["server", "A"]},
        "min_clients": 2,
        "mandatory_clients": ["A"],
    }


def test_punctuation_inside_strings_and_comments_is_not_syntax():
    text = 'script = "A=1 python3 custom/{app_script}  {app_config} "  # x = { [\nb = "#not a comment"\n'
    assert hocon.loads(text) == {"script": "A=1 python3 custom/{app_script}  {app_config} ", "b": "#not a comment"}


def test_duplicate_keys_are_reported():
    tree = hocon.parse("args {\n  model_name = \"A\"\n  model_name = \"B\"\n}\n")
    assert hocon.duplicate_keys(tree) == ["args.model_name"]
    with pytest.raises(hocon.ConfError):
        hocon.get_field(tree.fields[0].value, "model_name")


@pytest.mark.parametrize(
    "text",
    ["a = ${b}\n", 'include "other.conf"\n', "a += 1\n", 'a = """x"""\n', "a = \"open\n", "a {\n", "a = [1, 2\n"],
)
def test_unsupported_or_broken_text_is_rejected(text):
    with pytest.raises(hocon.ConfError):
        hocon.loads(text)


def test_values():
    assert hocon.loads("a = 1e-5\nb = 0.01\nc = true\nd = null\ne = 86400\nf = foo bar\n") == {
        "a": 1e-5,
        "b": 0.01,
        "c": True,
        "d": None,
        "e": 86400,
        "f": "foo bar",
    }
    assert hocon.render(["server", "A"]) == '["server", "A"]'
    assert hocon.render(False) == "false"
