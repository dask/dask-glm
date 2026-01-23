import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --slow option to run")
    for item in items:
        # Check for @pytest.mark.slow on the function
        if item.get_closest_marker("slow"):
            item.add_marker(skip_slow)
            continue
        # Check for pytest.param(..., marks=pytest.mark.slow) in parametrize
        if not hasattr(item, "callspec"):
            continue
        for marker in item.iter_markers("parametrize"):
            param_names = [n.strip() for n in marker.args[0].split(",")]
            for param in marker.args[1]:
                if not hasattr(param, "marks") or not param.marks:
                    continue
                has_slow = any(
                    getattr(m, "name", None) == "slow"
                    or (hasattr(m, "mark") and m.mark.name == "slow")
                    for m in param.marks
                )
                if not has_slow:
                    continue
                # Check if all values from this param match the callspec
                match = True
                for i, name in enumerate(param_names):
                    if i < len(param.values):
                        if item.callspec.params.get(name) != param.values[i]:
                            match = False
                            break
                if match:
                    item.add_marker(skip_slow)
                    break
