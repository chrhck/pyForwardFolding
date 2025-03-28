import packaging.version
import pytest

import pyForwardFolding


@pytest.mark.parametrize(
    "version,expected_version",
    ((pyForwardFolding.__version__, "0.1.0"),),
)
def test_version_matches_expected(version: str, expected_version: str) -> None:
    assert version == expected_version


def test_version_is_valid() -> None:
    packaging.version.parse(pyForwardFolding.__version__)
