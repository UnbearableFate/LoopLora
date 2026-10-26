import torch


def test_cuda_available_and_matmul():
    assert torch.cuda.is_available()
    a = torch.randn(64, 64, device="cuda", dtype=torch.float64)
    torch.testing.assert_close(a @ torch.eye(64, device="cuda", dtype=torch.float64), a)
