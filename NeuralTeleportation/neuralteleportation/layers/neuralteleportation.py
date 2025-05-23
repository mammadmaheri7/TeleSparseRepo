import torch
from torch.nn.modules import Flatten


class NeuralTeleportationLayerMixin(object):
    cob: torch.Tensor = None
    prev_cob: torch.Tensor = None
    next_cob: torch.Tensor = None

    @classmethod
    def base_layer(cls):
        return cls.__bases__[-1]

    def teleport(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        raise NotImplemented

    def apply_cob(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        raise NotImplemented


class COBForwardMixin(object):
    cob_field: str
    reshape_cob: bool
    last_cob: bool = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if getattr(self, self.cob_field) is None:
            if self.last_cob:
                setattr(self, self.cob_field, torch.ones(input.shape[-1]))
            else:
                setattr(self, self.cob_field, torch.ones(input.shape[1]))

        if self.reshape_cob:
            if self.last_cob:
                # print("====== state 1 happed ====== , input.shape:",input.shape)
                cob_shape = (input.shape[-1],) + (1,) * (input.dim() - 2)
            else:
                # print("====== state 2 happed ====== , input.shape:",input.shape)
                cob_shape = (input.shape[1],) + tuple([1 for _ in range(input.dim() - 2)])
            
            # cob_view = getattr(self, self.cob_field).view(cob_shape).float().type_as(input).detach()
            cob_view = getattr(self, self.cob_field).view(cob_shape).float().type_as(input)
            setattr(self, self.cob_field, cob_view)

        return self._forward(input)

    def _forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FlattenCOB(NeuralTeleportationLayerMixin, Flatten):

    def apply_cob(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        pass

    def teleport(self, prev_cob: torch.Tensor, next_cob: torch.Tensor):
        pass


