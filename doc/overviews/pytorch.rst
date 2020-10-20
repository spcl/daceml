PyTorch Integration
===================
A PyTorch ``nn.Module`` can be imported using the :class:`~daceml.pytorch.DaceModule` wrapper or :class:`~daceml.pytorch.dace_module` decorator.

.. testcode::

    import torch
    import torch.nn.functional as F
    from daceml.pytorch import DaceModule, dace_module
    
    # Input and size definition
    B, H, P, SM, SN = 2, 16, 64, 512, 512
    N = P * H
    Q, K, V = [torch.randn([SN, B, N]), torch.randn([SM, B, N]), torch.randn([SM, B, N])]
    
    # DaCe module used as a wrapper
    ptmodel = torch.nn.MultiheadAttention(N, H, bias=False)
    dace_model = DaceModule(ptmodel)
    outputs_wrapped = dace_model(Q, K, V)
    
    # DaCe module used as a decorator
    @dace_module
    class Model(nn.Module):
        def __init__(self, kernel_size):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 4, kernel_size)
            self.conv2 = nn.Conv2d(4, 4, kernel_size)
    
        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))
    
    dace_model = Model(3)
    outputs_dec = dace_model(torch.rand(1, 1, 8, 8))

.. testoutput::
    :hide:
    :options: +ELLIPSIS

    ...
