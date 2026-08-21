import pytest
from jax import numpy as jnp

import fdx
from fdx.stencils import Stencil


def test_stencil_solves_expected_central_difference_system():
    stencil = Stencil(offsets=[-1, 0, 1], partials={(1,): 1})

    assert stencil.ndims == 1
    assert stencil.spacings == [1]
    assert stencil.values[(-1,)] == pytest.approx(-0.5)
    assert stencil.values[(1,)] == pytest.approx(0.5)
    assert stencil.accuracy == 2

    system_matrix, terms = stencil._system_matrix()
    assert terms == [(0,), (1,), (2,)]
    assert jnp.array_equal(
        system_matrix,
        jnp.array([[1, 1, 1], [-1, 0, 1], [1, 0, 1]]),
    )
    assert stencil._system_matrix_row((2,)) == [1, 0, 1]
    assert stencil._multinomial_powers(2) == [(2,)]
    assert stencil._rows_are_linearly_independent([[1, 0, 1], [-1, 0, 1]])
    assert not stencil._rows_are_linearly_independent([[1, 0, 1], [1, 0, 1]])
    assert str(stencil) == repr(stencil) == str(stencil.values)

    stencil.max_order = 1
    with pytest.raises(Exception, match="Not enough terms"):
        stencil._system_matrix()


def test_stencil_applies_consistently_on_points_masks_and_slices():
    stencil = Stencil(offsets=[-1, 0, 1], partials={(1,): 1})
    field = jnp.arange(6.0)
    mask = jnp.array([False, True, True, True, False, False])

    assert stencil(field, at=2) == pytest.approx(1.0)
    assert jnp.allclose(
        stencil(field, on=mask),
        jnp.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
    )
    assert jnp.allclose(
        stencil(field, on=(slice(1, 4),)),
        jnp.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
    )
    assert jnp.allclose(
        stencil(field, on=(slice(-4, -1),)),
        jnp.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0]),
    )
    assert stencil._canonic_slice(slice(-4, -1, 2), len(field)) == slice(2, 5, 2)

    with pytest.raises(Exception, match="Cannot evaluate outside of grid"):
        stencil(field, at=0)

    with pytest.raises(Exception, match="Cannot specify both"):
        stencil(field, at=2, on=mask)


def test_stencil_broadcasts_spacings_and_shifts_masks_in_2d():
    stencil = Stencil(offsets=[(-1, 0), (1, 0)], partials={(1, 0): 1}, spacings=2.0)
    mask = jnp.array(
        [
            [False, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )
    x = jnp.arange(4.0) * 2.0
    field = jnp.broadcast_to(x[:, None], mask.shape)

    assert stencil.ndims == 2
    assert stencil.spacings == [2.0, 2.0]
    assert stencil.values[(-1, 0)] == pytest.approx(-0.25)
    assert stencil.values[(1, 0)] == pytest.approx(0.25)
    assert stencil._multinomial_powers(2) == [(0, 2), (1, 1), (2, 0)]
    assert jnp.array_equal(
        stencil._make_offset_mask(mask, (1, 0)),
        jnp.array(
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, True, False, False],
                [False, False, True, False],
            ],
            dtype=bool,
        ),
    )
    assert jnp.array_equal(
        stencil._make_offset_mask(mask, (-1, 0)),
        jnp.array(
            [
                [False, True, False, False],
                [False, False, True, False],
                [False, False, False, False],
                [False, False, False, False],
            ],
            dtype=bool,
        ),
    )
    assert jnp.allclose(
        stencil(field, on=mask),
        jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_stencil_set_matches_operator_application():
    field = jnp.arange(7.0) ** 2
    op = fdx.FinDiff(0, 1.0, 1, acc=2)
    stencil_set = op.stencil(field.shape)
    expected = op(field)

    assert stencil_set.char_pts == (("L",), ("C",), ("H",))
    assert stencil_set._typical_index_tuple_for_char_point(("L",)) == (0,)
    assert stencil_set._typical_index_tuple_for_char_point(("C",)) == (3,)
    assert stencil_set._typical_index_tuple_for_char_point(("H",)) == (6,)
    assert str(stencil_set) == repr(stencil_set)

    assert stencil_set.apply(field, 3) == pytest.approx(expected[3])
    assert stencil_set.apply(field, (0,)) == pytest.approx(expected[0])
    assert stencil_set.apply(field, (6,)) == pytest.approx(expected[6])
    assert jnp.allclose(stencil_set.apply_all(field), expected)
