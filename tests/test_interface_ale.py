import pytest

from xfsi_solver.tools.interface_ale import ROWS, check_interface_ale_rows, interface_ale_description


@pytest.mark.parametrize("rows", [("momentum",), ("mass",), ("momentum", "mass"), ("mass", "momentum"), ["momentum"]])
def test_rows_accepted(rows):
    check_interface_ale_rows(rows)


@pytest.mark.parametrize("rows", [(), "momentum", ("momentum", "momentum"), ("pressure",), ("momentum", "continuity")])
def test_rows_rejected(rows):
    with pytest.raises(ValueError, match="interface_ale_rows"):
        check_interface_ale_rows(rows)


def test_description():
    assert interface_ale_description(False, None) == "off"
    assert interface_ale_description(True, None) == (
        "on, momentum and mass rows, integrated with the quadrature of the residual"
    )
    # The rows are named in the order of ROWS, whatever the order given.
    assert interface_ale_description(True, 2, ("mass", "momentum")) == interface_ale_description(True, 2, ROWS)
    assert interface_ale_description(True, 2, ("momentum",)) == "on, momentum rows, integrated with quadrature degree 2"
