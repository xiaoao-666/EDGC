# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from abc import abstractmethod

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from megatron import get_args
from megatron import mpu
from .module import MegatronModule
from megatron.mpu.reducers import PowerSGDReducer
from megatron.schedules_utils import Utils


class MemoryBuffer:

    def __init__(self, numel, dtype):
        self.numel = numel
        self.dtype = dtype
        self.data = torch.zeros(self.numel,
                                dtype=self.dtype,
                                device=torch.cuda.current_device(),
                                requires_grad=False)

    def zero(self):
        """Reset the buffer to zero."""
        self.data.zero_()

    def get(self, shape, start_index):
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, \
            'requested tensor is out of the buffer range.'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor


def calculate_entropy(grad, bins=1000, sample_ratio=0.25):
    stride = max(1, int(1 / sample_ratio))
    flat_grad = grad.flatten()
    mask = torch.isfinite(flat_grad)
    if not mask.any():
        return torch.tensor(0.0, device=grad.device)
    valid_grad = flat_grad[mask]
    sample_grad = valid_grad[::stride]
    min_val, max_val = torch.min(sample_grad), torch.max(sample_grad)
    hist = torch.histc(sample_grad, bins=bins, min=min_val.item(), max=max_val.item())
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -torch.sum(hist * torch.log(hist))


class DistributedDataParallelBase(MegatronModule, ABC):
    """Abstract class for DDP."""

    def __init__(self, module):
        super(DistributedDataParallelBase, self).__init__()
        # Keep a pointer to the model.
        self.module = module

    @abstractmethod
    def allreduce_gradients(self):
        pass

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(destination, prefix,
                                                          keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)


class DistributedDataParallel(DistributedDataParallelBase):
    """DDP with contiguous buffers options to storre and accumulate gradients.
    This class:
        - has the potential to reduce memory fragmentation.
        - provides the option to do the gradient accumulation
          in a type other than the params type (for example fp32)

    Arguments:
        module: input model.
        accumulate_allreduce_grads_in_fp32: if true do the gradient accumulation
            and the gradient all-reduce all in in float32. If this option is
            true, we require `use_contiguous_buffers` to be true too.
        use_contiguous_buffers: if true, use a contiguous buffer to store the
            gradients.
    """

    def __init__(self, module,
                 accumulate_allreduce_grads_in_fp32,
                 use_contiguous_buffers):

        super(DistributedDataParallel, self).__init__(module)

        self.gradient_sample_ratio = 0.25
        self.iteration_sample_ratio = 0.1
        self.initial_error = False
        self.max_rank_error = 0.0
        self.min_rank_error = 0.0
        self.reducer_buffer = None
        self.reducer = None
        self.args = get_args()
        self.args.grad_comp = False
        self.accumulate_allreduce_grads_in_fp32 \
            = accumulate_allreduce_grads_in_fp32
        self.use_contiguous_buffers = use_contiguous_buffers
        # If we are using fp32-accumulate-allreduce explicitly
        # this means we need main grads in a continous buffer.
        if self.accumulate_allreduce_grads_in_fp32:
            assert self.use_contiguous_buffers
        # ===================================
        # Rest of this part applies only to
        # the case we use continuous buffers.
        # ===================================
        self._grad_buffers = None
        if self.use_contiguous_buffers:
            self._grad_buffers = {}

            # Simple function to define buffer type.
            def _get_buffer_type(param):
                return torch.float if \
                    self.accumulate_allreduce_grads_in_fp32 else param.dtype

            # First calculate total number of elements per type.
            type_num_elements = {}
            for name, param in self.module.named_parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] = type_num_elements.get(dtype, 0) \
                                               + param.data.nelement()

            # Allocate the buffer.
            for dtype, num_elements in type_num_elements.items():
                self._grad_buffers[dtype] = MemoryBuffer(num_elements, dtype)

            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.
            for name, param in self.module.named_parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] -= param.data.nelement()
                    param.main_grad = self._grad_buffers[dtype].get(
                        param.data.shape, type_num_elements[dtype])

            # Backward hook.
            # Accumalation function for the gradients. We need
            # to store them so they don't go out of scope.
            self.grad_accs = []
            # Loop over all the parameters in the model.
            for name, param in self.module.named_parameters():
                if param.requires_grad:
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator functtion.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_param_hook(param))
                    self.grad_accs.append(grad_acc)

    def _record_entropy(self, tensors, total_entropy):
        for tensor in tensors:
            entropy = calculate_entropy(tensor, sample_ratio=self.gradient_sample_ratio)
            total_entropy += entropy.item()
        return total_entropy

    def update(self):
        if not self.args.grad_comp:
            self.reducer = None
            self.reducer_buffer = {}
            return

        self.reducer_buffer = {}

        # Allocate contiguous buffers if needed
        if self.use_contiguous_buffers:
            def _get_buffer_type(param):
                return torch.float if self.accumulate_allreduce_grads_in_fp32 else param.dtype

            type_num_elements = {}
            for _, param in self.module.named_parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] = type_num_elements.get(dtype, 0) + param.data.nelement()

            for dtype, num_elements in type_num_elements.items():
                self.reducer_buffer[dtype] = MemoryBuffer(num_elements, dtype)

        # Configure PowerSGDReducer
        group = mpu.get_data_parallel_group()
        group_num = mpu.get_data_parallel_world_size()
        device = torch.cuda.current_device()
        seed = self.args.comp_seed
        fp16 = self.args.fp16

        if self.args.find_rank_upper_limit:
            rank = self._get_find_rank()
            self.reducer = PowerSGDReducer(
                random_seed=seed,
                device=device,
                group=group,
                group_num=group_num,
                rank=rank,
                start_iter=0,
                use_error_feedback=self.args.use_error_feedback,
                fp16=fp16
            )
        else:
            rank = self._get_adaptive_rank()
            self.reducer = PowerSGDReducer(
                random_seed=seed,
                device=device,
                group=group,
                group_num=group_num,
                rank=rank,
                start_iter=0,
                use_error_feedback=self.args.use_error_feedback,
                fp16=fp16
            )

    def _get_find_rank(self):
        """Helper to determine rank when finding rank upper limit."""
        if self.args.mapped_rank is not None:
            return int(self.args.mapped_rank)
        if self.args.is_loading_checkpoint:
            return int(Utils.mapped_rank[-1] if Utils.mapped_rank else self.args.max_rank)
        return int(self.args.max_rank)

    def _get_adaptive_rank(self):
        """Helper to determine rank during adaptive compression."""
        if self.args.is_loading_checkpoint:
            delta_iter = self.args.curr_iteration - self.args.latest_iteration
        else:
            delta_iter = self.args.curr_iteration
        return 2 ** int((delta_iter - 9) / 3)
 
    def _make_param_hook(self, param):

        # Hook used for back-prop.
        def param_hook(*unused):
            # Add the gradient to the buffer.
            if param.grad.data is not None:
                # The gradient function of linear layers is fused with GEMMs
                param.main_grad.add_(param.grad.data)
                # Now we can deallocate grad memory.
                param.grad = None

        return param_hook

    def zero_grad_buffer(self):
        assert self._grad_buffers is not None, 'buffers are not initialized.'
        for _, buffer_ in self._grad_buffers.items():
            buffer_.zero()

    def broadcast_params(self):
        for param in self.module.parameters():
            torch.distributed.broadcast(param.data,
                                        src=mpu.get_data_parallel_src_rank(),
                                        group=mpu.get_data_parallel_group())

    def _eval_error(self, tensor, rank):
        matrix = tensor.view(tensor.shape[0], -1)
        n, m = matrix.shape
        r = min(n, m, rank)
        q = torch.randn(m, r, device=tensor.device, dtype=torch.float)
        q, _ = torch.linalg.qr(q, mode='reduced')
        p = matrix @ q
        p, _ = torch.linalg.qr(p, mode='reduced')
        q = matrix.t() @ p
        recon = (p @ q.t()).view_as(tensor)
        error = torch.sum(torch.abs(recon - tensor)).item()
        return error

    def _handle_compression_error(self, tensor, should_sample, is_rank_0):
        if self.args.find_rank_upper_limit and not self.initial_error:
            if self.args.is_loading_checkpoint:
                self.max_rank_error = float(Utils.read_error_from_csv(self.args.max_error_path))
            else:
                self.max_rank_error = self._eval_error(tensor, self.args.max_rank)
                Utils.append_init_error_to_csv(self.args.max_error_path, self.max_rank_error)
            self.initial_error = True

        if self.current_iter % self.args.window_size == 0 and self.current_iter != 0 and is_rank_0 and self.args.compute_end_warm_up is None:
            self.min_rank_error = self._eval_error(tensor, int(self.args.max_rank / 4))
            if self.min_rank_error < self.max_rank_error:
                self.args.compute_end_warm_up = self.args.curr_iteration

        if should_sample:
            self._record_entropy([tensor], 0.0)

    def allreduce_gradients(self):

        total_entropy = 0.0
        iter_sample_interval = int(1 / self.iteration_sample_ratio)
        should_sample = (self.args.curr_iteration % iter_sample_interval == 0)
        is_rank_0 = (torch.distributed.get_rank() == 0)
        if self._grad_buffers is not None:
            if self.args.grad_comp:
                need_callback = self.reducer.reduce(self.module, self._grad_buffers, self.reducer_buffer)
                if need_callback:
                    for (_, buffer_), (_, reduced_) in zip(self._grad_buffers.items(), self.reducer_buffer.items()):
                        buffer_.data.copy_(reduced_.data)
                        if should_sample:
                            total_entropy = self._record_entropy([buffer_.data], total_entropy)
            else:
                for _, buffer_ in self._grad_buffers.items():
                    buffer_.data /= mpu.get_data_parallel_world_size()
                    torch.distributed.all_reduce(buffer_.data, group=mpu.get_data_parallel_group())
                    self._handle_compression_error(buffer_.data, should_sample, is_rank_0)
        else:
            # Bucketize and reduce
            buckets = {}
            for param in self.module.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = param.data.type()
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)
                    param.main_grad = param.grad

            for tp, bucket in buckets.items():
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                coalesced /= mpu.get_data_parallel_world_size()
                torch.distributed.all_reduce(coalesced, group=mpu.get_data_parallel_group())
                self._handle_compression_error(coalesced, should_sample, is_rank_0)
                unflattened = _unflatten_dense_tensors(coalesced, grads)
                for buf, synced in zip(grads, unflattened):
                    buf.copy_(synced)

        if should_sample:
            Utils.entropy.append(total_entropy)

