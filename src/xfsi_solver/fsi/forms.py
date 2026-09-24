# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Helpers for selecting and simplifying UFL forms."""

import ufl
import ufl.algorithms


def _subdomain_ids(integral):
    sid = integral.subdomain_id()
    return tuple(sid) if isinstance(sid, tuple) else (sid,)


def restrict_to_cells(form: ufl.Form, marker: int) -> ufl.Form:
    """The cell integrals of ``form`` over the cells tagged ``marker``.

    Every integral in ``form`` must be a cell integral over a single tagged
    subdomain, so that no contribution is silently dropped or duplicated.
    """
    selected = []
    for integral in form.integrals():
        ids = _subdomain_ids(integral)
        if integral.integral_type() != "cell" or len(ids) != 1 or not isinstance(ids[0], int):
            raise ValueError(
                f"Cannot split {integral.integral_type()} integral over subdomain(s) {ids} by cell marker"
            )
        if ids[0] == marker:
            selected.append(integral)
    return ufl.Form(selected)


def nonzero(form: ufl.Form):
    """``form`` with derivatives expanded, or ``None`` if it vanishes identically."""
    form = ufl.algorithms.expand_derivatives(form)
    return None if form.empty() else form
