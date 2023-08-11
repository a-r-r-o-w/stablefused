import pytest
import numpy as np
import torch

from stablefused.utils import lerp, slerp


test_cases_lerp = [
    # Test cases for lerp with t as a float
    (np.array([1.0, 2.0]), np.array([4.0, 6.0]), 0.0, np.array([1.0, 2.0])),
    (np.array([1.0, 2.0]), np.array([4.0, 6.0]), 1.0, np.array([4.0, 6.0])),
    (np.array([1.0, 2.0]), np.array([4.0, 6.0]), 0.5, np.array([2.5, 4.0])),
    # Test cases for lerp with t as an np.ndarray
    (
        np.array([1.0, 2.0]),
        np.array([4.0, 6.0]),
        np.array([0.0, 1.0]),
        np.array([[1.0, 2.0], [4.0, 6.0]]),
    ),
    (
        np.array([1.0, 2.0]),
        np.array([4.0, 6.0]),
        np.array([0.5, 0.25]),
        np.array([[2.5, 4.0], [1.75, 3.0]]),
    ),
    # Test cases for lerp with t as a torch.Tensor
    (
        np.array([1.0, 2.0]),
        np.array([4.0, 6.0]),
        torch.Tensor([0.0, 1.0]),
        np.array([[1.0, 2.0], [4.0, 6.0]]),
    ),
    (
        np.array([1.0, 2.0]),
        np.array([4.0, 6.0]),
        torch.Tensor([0.5, 0.25]),
        np.array([[2.5, 4.0], [1.75, 3.0]]),
    ),
]

test_cases_slerp = [
    # Test cases for slerp with t as a float
    (np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0, np.array([1.0, 0.0])),
    (np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1.0, np.array([0.0, 1.0])),
    (np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.5, np.array([0.707107, 0.707107])),
    # Test cases for slerp with t as an array
    (
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ),
    (
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.5, 0.25]),
        np.array([[0.707107, 0.707107], [0.923880, 0.382683]]),
    ),
    # Test cases for slerp with t as a torch.Tensor
    (
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        torch.Tensor([0.0, 1.0]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ),
    (
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        torch.Tensor([0.5, 0.25]),
        np.array([[0.707107, 0.707107], [0.923880, 0.382683]]),
    ),
]


@pytest.mark.parametrize("v0, v1, t, expected", test_cases_lerp)
def test_lerp(v0, v1, t, expected):
    result = lerp(v0, v1, t)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("v0, v1, t, expected", test_cases_slerp)
def test_slerp(v0, v1, t, expected):
    v0_torch = torch.from_numpy(v0)
    v1_torch = torch.from_numpy(v1)

    result = slerp(v0, v1, t)
    result_torch = slerp(v0_torch, v1_torch, t).numpy()

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(result_torch, expected, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    pytest.main()
