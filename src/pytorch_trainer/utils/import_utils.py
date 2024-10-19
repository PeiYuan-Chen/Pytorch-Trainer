import importlib
import importlib.util


def is_package_avaiable(pkg_name: str) -> bool:
    return importlib.util.find_spec(pkg_name) is not None
