"""Implement tensor computations with pytorch."""
from enum import Enum, auto
from functools import partial
from itertools import count
import os
import queue
import shutil
import time
import threading
from typing import Optional, Union, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist

from flexgen.utils import (GB, T, cpu_mem_stats, vector_gather,
    np_dtype_to_torch_dtype, torch_dtype_to_np_dtype,
    torch_dtype_to_num_bytes)

general_copy_compressed = TorchCompressedDevice = None
global_cpu_device = None
global_disk_device = None


def fix_recursive_import():
    global general_copy_compressed, TorchCompressedDevice, global_cpu_device
    from flexgen import compression
    general_copy_compressed = compression.general_copy_compressed
    TorchCompressedDevice = compression.TorchCompressedDevice


class DeviceType(Enum):
    CPU = auto()
    CUDA = auto()
    DISK = auto()
    MIXED = auto()
    COMPRESSED = auto()

    @staticmethod
    def convert(name):
        if name == "cpu":
            return DeviceType.CPU
        elif name == "cuda":
            return DeviceType.CUDA
        elif name == "disk":
            return DeviceType.DISK
        elif name == "mixed":
            return DeviceType.MIXED
        elif name == "compressed":
            return DeviceType.COMPRESSED
        else:
            raise ValueError(f"Invalid name: {name}")


class TorchTensor:
    """
    Wrap pytorch tensors to support
      - Unified representation for normal and compressed tensors on
        GPUs, CPUs, disks and mixed devices.
      - Asynchronous copy between tensors on any formats and any devices.

    This is achieved by implementing the data movement APIs for primitive cases
    and using recursive structures to handle other combinations.

    Note:
    For a tensor on a TorchDevice, self.data is a primitive tensor.
      type: torch.Tensor.
    For a tensor on a TorchDisk, self.data is a filename.
      type: str
    For a tensor on a TorchMixedDevice, self.data is (tensors, segment_points)
      type: Tuple[Tuple[TorchTensor], Tuple[int]]
    For a tensor on a TorchCompressedDevice, self.data is (data, scale, compression_config)
      type: Tuple[TorchTensor, TorchTensor, CompressionConfig]
    """
    name_count = count()

    def __init__(self, shape, dtype, data, device, name=None, seg_dim=None):
        if isinstance(data, torch.Tensor):
            assert data.device == device.dev

        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.device = device

        # Whether delete the file when the tensor is deleted
        self.delete_file = True

        self.name = name or TorchTensor.next_name()
        self.seg_dim=seg_dim

    @property
    def bytes(self):
        return np.prod(self.shape) * torch_dtype_to_num_bytes[self.dtype]

    @classmethod
    def next_name(cls):
        return f"t_{next(cls.name_count)}"

    @classmethod
    def create_from_torch(cls, data, device, name=None):
        return cls(data.shape, data.dtype, data, device, name=name)

    def delete(self):
        assert self.device is not None, "already deleted"
        if self.device.device_type == DeviceType.DISK:
            self.device.delete(self)
        self.device = self.data = None
    def load_from_np(self, np_array):
        if self.device.device_type == DeviceType.DISK:
            with open(self.data, "wb") as fout:
                np.save(fout, np_array)
        else:
            if self.device.device_type == DeviceType.COMPRESSED:
                tmp = torch.from_numpy(np_array)
                tmp = global_cpu_device.compressed_device.compress(tmp, self.data[2])
                general_copy(self, None, tmp, None)
            else:
                self.data.copy_(torch.from_numpy(np_array))

    def load_from_np_file(self, filename):
        if self.device.device_type == DeviceType.DISK:
            shutil.copy(filename, self.data)
        else:
            self.load_from_np(np.load(filename))

    def copy(self, dst, src_indices=None):
        if src_indices:
            assert all(x.step is None for x in src_indices)
            shape = tuple(x.stop - x.start for x in src_indices
                ) + self.shape[len(src_indices):]
        else:
            shape = self.shape

        if dst.device_type == DeviceType.COMPRESSED:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype], self.data[2])
        else:
            ret = dst.allocate(shape, torch_dtype_to_np_dtype[self.dtype])
        general_copy(ret, None, self, src_indices, seg_dim=1)
        return ret
    def balanced_copy(self, dst, dst_seg_lengths, seg_dim):
        # src_seg_lengths = [self.data[1][i] - self.data[1][i-1] for i in range(1, len(self.data[1]))]
        
        if self.data[1] == dst_seg_lengths:
            return self, False
        else:
            if seg_dim == None:
                ret = dst.allocate_all(self.shape, torch_dtype_to_np_dtype[self.dtype])
                general_copy(ret, None, self, None, seg_dim)
                return ret, True
            else:
                ret = dst.allocate(self.shape, torch_dtype_to_np_dtype[self.dtype], dst_seg_lengths, seg_dim=seg_dim)
                general_copy(ret, None, self, None, seg_dim)
                return ret, True
    def smart_copy(self, dst, src_indices=None):
        if self.device == dst:
            return self, False
        return self.copy(dst, src_indices=src_indices), True
    def move(self, dst):
        if self.device == dst:
            return self
        ret = self.copy(dst)
        self.delete()
        return ret

    def __str__(self):
        return (f"TorchTensor(shape={self.shape}, dtype={str(self.dtype)}, "
                f"device={self.device.name if self.device else None})")

class TorchDevice:
    """Wrap tensor and computation APIs of a single CPU or GPU."""

    def __init__(self, name, mem_capacity=None, flops=None):
        self.name = name
        self.mem_capacity = mem_capacity
        self.flops = flops

        self.dev = torch.device(name)
        self.device_type = DeviceType.convert(self.dev.type)
        self.compressed_device = TorchCompressedDevice(self)

        self.links = {}

        self.attention_compute_workspace = None
        self.workspace_pt = 0

        if self.device_type == DeviceType.CPU:
            global global_cpu_device
            global_cpu_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        if self.device_type == DeviceType.CPU:
            pin_memory = True if pin_memory is None else pin_memory
        else:
            pin_memory = False
        dtype = np_dtype_to_torch_dtype[dtype]
        data = torch.empty(shape, dtype=dtype, pin_memory=pin_memory, device=self.dev)
        return TorchTensor.create_from_torch(data, self, name=name)

    def delete(self, tensor):
        pass

    def init_attention_compute_workspace(self, config, task, policy):
        if self.device_type != DeviceType.CPU:
            return  # Only CPU requires this fp32 workspace

        if not policy.compress_cache:
            b = policy.gpu_batch_size
            n_head = config.n_head
            head_dim = config.input_dim // n_head
            max_seq_len = task.prompt_len + task.gen_len - 1
            self.attention_compute_workspace = []
            self.workspace_pt = 0

            # We currently separate SelfAttention and MLP as two layers,
            # so we only need one workspace instead of two.
            for i in range(1 if policy.sep_layer else 2):
                shape = (max_seq_len, b * n_head, head_dim)
                k_cache = self.allocate(shape, np.float32, pin_memory=False)
                v_cache = self.allocate(shape, np.float32, pin_memory=False)
                self.attention_compute_workspace.append((k_cache, v_cache))
        else:
            self.compressed_device.init_attention_compute_workspace(
                config, task, policy)

    def next_attention_compute_workspace(self):
        self.workspace_pt = (self.workspace_pt + 1) % len(
            self.attention_compute_workspace)
        return self.attention_compute_workspace[self.workspace_pt]

    def del_attention_compute_workspace(self):
        self.attention_compute_workspace = None

    def gen_attention_mask(self, token_ids, pad_token_id, donate):
        data = token_ids.data.ne(pad_token_id)
        if donate[0]: token_ids.delete()
        return TorchTensor.create_from_torch(data, self)

    def extend_attention_mask(self, attention_mask, donate):
        bs = attention_mask.shape[0]
        data = torch.concat((attention_mask.data,
             torch.ones((bs, 1), dtype=attention_mask.dtype, device=self.dev)), dim=1)
        if donate[0]: attention_mask.delete()
        return TorchTensor.create_from_torch(data, self)

    def opt_input_embed_dist(self, inputs, attention_mask, w_token, w_pos, pad_token_id, donate, world_size):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
            w_pos = w_pos.device.decompress(w_pos)

        token_ids = inputs.data
        mask = attention_mask.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # token embedding

        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)
        # pos embedding
        positions = torch.cumsum(mask, dim=1).int() * mask + 1

        # cut positions if `past_key_values_length` is > 0
        past_key_values_length = mask.shape[1] - token_ids.shape[1]
        positions = positions[:, past_key_values_length:]

        pos_embed = F.embedding(positions, w_pos.data)
        data = token_embed + pos_embed
        b, s, dist_h = data.shape
        output = torch.empty((world_size, b, s, dist_h) , device=self.dev, dtype=data.dtype)
        dist.all_gather_into_tensor(output, data)
        output = output.permute(1, 2, 0, 3).reshape(b, s, -1)

        return TorchTensor.create_from_torch(output, self)
    def opt_input_embed(self, inputs, attention_mask, w_token, w_pos, pad_token_id, donate):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
            w_pos = w_pos.device.decompress(w_pos)

        token_ids = inputs.data
        mask = attention_mask.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # token embedding
        # shape: (b, prompt_len or 1, h)
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)
        # mask shape: (b, prompt_len + generated_len)
        # pos embedding
        positions = torch.cumsum(mask, dim=1).int() * mask + 1

        # cut positions if `past_key_values_length` is > 0
        past_key_values_length = mask.shape[1] - token_ids.shape[1]
        # positions shape: (b, prompt_len or 1)
        positions = positions[:, past_key_values_length:]
        #pos_embed shape: (b, prompt_len or 1, h)
        pos_embed = F.embedding(positions, w_pos.data)

        data = token_embed + pos_embed
        return TorchTensor.create_from_torch(data, self)
    def opt_output_embed(self, inputs, w_ln, b_ln, w_token, donate,
                         do_sample, temperature):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        b, s, h = inputs.shape
        
        # last_hidden shape: (b, 1, h)
        last_hidden = F.layer_norm(inputs.data[:, -1, :], (h,), weight=w_ln.data, bias=b_ln.data)
        if donate[0]: inputs.delete()

        # output embedding
        # last_token_logits shape: (b, 1, v)
        last_token_logits = F.linear(last_hidden, w_token.data)
        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        return TorchTensor.create_from_torch(ids, self)
    def opt_output_embed_dist(self, inputs, w_ln, b_ln, w_token, donate,
                         do_sample, temperature, world_size):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)

        b, s, h = inputs.shape
        
        # last_hidden shape: (b, 1, h)
        last_hidden = F.layer_norm(inputs.data[:, -1, :], (h,), weight=w_ln.data, bias=b_ln.data)
        if donate[0]: inputs.delete()

        # output embedding
        
        assert w_token.data.shape[0] % world_size == 0
        # last_token_logits shape: (b, v // world_size)
        # w_token.data shape: (v // world_size, h)
        last_token_logits = F.linear(last_hidden, w_token.data)
        # gathered_shape shape: (b, v)
        b, per_gpu_v = last_token_logits.shape
        gathered = torch.empty((world_size, b, per_gpu_v), device=self.dev, dtype=last_token_logits.dtype)
        dist.all_gather_into_tensor(gathered, last_token_logits)
        gathered = gathered.permute(1, 0, 2).reshape(b, -1)
        if do_sample and not temperature < 1e-5:
            probs = torch.softmax(gathered / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = gathered.argmax(dim=1, keepdim=True)
        return TorchTensor.create_from_torch(ids, self)

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        return k_cache, v_cache
    def init_cache_one_gpu_batch_dist(self, config, task, policy, world_size):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        assert (gpu_batch_size * num_head) % world_size == 0
        shape = (prompt_len + gen_len - 1, (gpu_batch_size * num_head) // world_size, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16, pin_memory=pin_memory)
        return k_cache, v_cache
    def mha_dist(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config, rank, world_size):
        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)
        
        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        assert n_head % world_size == 0
        # w_ shape: (h // world_size, h)
        # b_ shape: (h // world_size,)
        # q, k, v shape: (b, s, h // world_size)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        
        # shape: (b, s, n_head // world_size, head_dim)
        q = q.view(b, s, -1, head_dim)
        k = k.view(b, s, -1, head_dim)
        v = v.view(b, s, -1, head_dim)

        # shape: (b * n_head // world_size, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(-1, s, head_dim)
        # shape: (b * n_head // world_size, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(-1, head_dim, s)
        # shape: (b * n_head // world_size, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(-1, s, head_dim)

        # shape: (b * n_head // world_size, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head // world_size, s, s)
        attn_weights = attn_weights.view(b, -1, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        # shape: (b * n_head // world_size, s, s)
        attn_weights = attn_weights.view(-1, s, s)
        attn_weights = F.softmax(attn_weights, dim=2)
        # shape: (b, n_head // world_size, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, -1, s, head_dim)
        # shape: (b, s, h // world_size)
        value = value.transpose(1, 2).reshape(b, s, -1)
        # w_out shape: (h, h // world_size)
        # b_out shape: (h,)
        
        if rank == 0:
            value = F.linear(value, w_out.data, bias=b_out.data)
            value.add_(inputs.data)
        else:
            value = F.linear(value, w_out.data)
        dist.all_reduce(value)
        # value shape: (b, s, h)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # shape: (s, b * n_head // world_size, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)
        # shape: (b, s, h), (s, b * n_head // world_size, head_dim), (s, b * n_head // world_size, head_dim)
        return TorchTensor.create_from_torch(value, self), k, v

    def mha(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
            w_out, b_out, w_ln, b_ln, n_head, donate, compress_cache, comp_config):
        """Multi-head attention (prefill phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)

        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_head, head_dim)
        v = v.view(b, s, n_head, head_dim)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)
        
        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=self.dev)
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)
        
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        if compress_cache:
            k = self.compressed_device.compress(k, comp_config)
            v = self.compressed_device.compress(v, comp_config)
        else:
            k = TorchTensor.create_from_torch(k, self)
            v = TorchTensor.create_from_torch(v, self)

        return TorchTensor.create_from_torch(value, self), k, v
    def mha_gen_dist(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate,
                attn_sparsity, compress_cache, comp_config, rank, world_size):
        """Multi-head attention (decoding phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        assert n_head % world_size == 0
        # hidden shape: (b, 1, h)
        # w_ shape: (h // world_size, h)
        # b_ shape: (h // world_size,)
        # q, k, v shape: (b, 1, h // world_size)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, 1, n_head // world_size, head_dim)
        q = q.view(b, tgt_s, -1, head_dim)
        k = k.view(b, tgt_s, -1, head_dim)
        v = v.view(b, tgt_s, -1, head_dim)

        # shape: (b * n_head // world_size, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(-1, tgt_s, head_dim)
        # shape: (1, b * n_head // world_size, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, -1, head_dim)
        # shape: (1, b * n_head // world_size, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, -1, head_dim)

        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:  # Dense attention
                if compress_cache:
                    # shape: (s, b * n_head // world_size, head_dim)
                    k = k_cache.device.decompress(k_cache)[:src_s]
                    v = v_cache.device.decompress(v_cache)[:src_s]
                else:
                    # shape: (s, b * n_head // world_size, head_dim)
                    k = k_cache.data[:src_s]
                    v = v_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                v[src_s - 1:src_s] = v_new

                # shape: (b * n_head // world_size, head_dim, s)
                k = k.permute(1, 2, 0).reshape(-1, head_dim, src_s)
                # shape: (b * n_head // world_size, s, head_dim)
                v = v.permute(1, 0, 2).reshape(-1, src_s, head_dim)

                if k.is_cuda:
                    # shape: (b, n_head // world_size, 1, head_dim)
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim)
                else:
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim).cuda().half()
            else:  # Sparse attention
                # shape: (s, b * n_head // world_size, head_dim)
                k = k_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                # shape: (b * n_head // world_size, head_dim, s)
                k = k.permute(1, 2, 0).reshape(-1, head_dim, src_s)

                if k.is_cuda:
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity)
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity).cuda().half()
        else:  # Mixed device attention
            assert attn_sparsity >= 1.0
            # shape: (b, n_head // world_size, 1, head_dim)
            value = self._mixed_device_attention(q, k_cache, v_cache,
                k_new, v_new, attention_mask.data, b, src_s, tgt_s,
                n_head, head_dim)
        # value shape: (b, 1, h // world_size)
        # w_out shape: (h, h // world_size)
        # b_out shape: (h,)
        value = value.transpose(1, 2).view(b, tgt_s, -1)
        if rank == 0:
            value = F.linear(value, w_out.data, bias=b_out.data)
        else:
            value = F.linear(value, w_out.data)
        
        dist.all_reduce(value)
        value.add_(inputs.data)
        # value shape: (b, 1, h)
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k_new = k[:, :, s_:].permute(2, 0, 1)
                v_new = v[:, s_:, :].permute(1, 0, 2)
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)
        # shape: (b, 1, h), (1, b * n_head // world_size, head_dim), (1, b * n_head // world_size, head_dim)
        return TorchTensor.create_from_torch(value, self), k_new, v_new


    def mha_gen(self, inputs, attention_mask, w_q, b_q, w_k, b_k, w_v, b_v,
                w_out, b_out, w_ln, b_ln, n_head, k_cache, v_cache, donate,
                attn_sparsity, compress_cache, comp_config):
        """Multi-head attention (decoding phase)."""
        # decompress weights
        if w_q.device.device_type == DeviceType.COMPRESSED:
            w_q = w_q.device.decompress(w_q)
            w_k = w_k.device.decompress(w_k)
            w_v = w_v.device.decompress(w_v)
            w_out = w_out.device.decompress(w_out)

        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        # shape: (b, 1, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, 1, n_head, head_dim)
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_head, head_dim)
        v = v.view(b, tgt_s, n_head, head_dim)

        # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)

        if isinstance(k_cache, TorchTensor):
            if attn_sparsity >= 1.0:  # Dense attention
                if compress_cache:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.device.decompress(k_cache)[:src_s]
                    v = v_cache.device.decompress(v_cache)[:src_s]
                else:
                    # shape: (s, b * n_head, head_dim)
                    k = k_cache.data[:src_s]
                    v = v_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                v[src_s - 1:src_s] = v_new

                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
                # shape: (b * n_head, s, head_dim)
                v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)

                if k.is_cuda:
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim)
                else:
                    q = q.float().cpu()
                    k, v = k.float(), v.float()
                    value = self._attention_value(q, k, v, attention_mask.data,
                        b, src_s, tgt_s, n_head, head_dim).cuda().half()
            else:  # Sparse attention
                # shape: (s, b * n_head, head_dim)
                k = k_cache.data[:src_s]
                k[src_s - 1:src_s] = k_new
                # shape: (b * n_head, head_dim, s)
                k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)

                if k.is_cuda:
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity)
                else:
                    q = q.float().cpu()
                    value = self._sparse_attention_value(q, k, v_new, v_cache,
                        attention_mask.data, b, src_s, tgt_s, n_head, head_dim,
                        attn_sparsity).cuda().half()
        else:  # Mixed device attention
            assert attn_sparsity >= 1.0
            value = self._mixed_device_attention(q, k_cache, v_cache,
                k_new, v_new, attention_mask.data, b, src_s, tgt_s,
                n_head, head_dim)

        # shape: (b, 1, h)
        value = value.transpose(1, 2).view(b, tgt_s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        if compress_cache:
            if comp_config.group_dim == 0:
                s_ = src_s // comp_config.group_size * comp_config.group_size
                k_new = k[:, :, s_:].permute(2, 0, 1)
                v_new = v[:, s_:, :].permute(1, 0, 2)
            k_new = self.compressed_device.compress(k_new, comp_config)
            v_new = self.compressed_device.compress(v_new, comp_config)
        else:
            k_new = TorchTensor.create_from_torch(k_new, self)
            v_new = TorchTensor.create_from_torch(v_new, self)

        return TorchTensor.create_from_torch(value, self), k_new, v_new

    def _attention_weights(self, q, k, mask, b, src_s, n_head):
        # shape: (b * n_head, 1, s)
        attn_weights = torch.bmm(q, k)
        # shape: (b, 1, 1, s)
        mask = mask.view(b, 1, 1, src_s)
        # shape: (b * n_head, 1, s)
        attn_weights = attn_weights.view(b, -1, 1, src_s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(-1, 1, src_s)
        attn_weights = F.softmax(attn_weights, dim=2)
        return attn_weights

    def _attention_value(self, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        # shape: (b, n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, -1, tgt_s, head_dim)
    def _sparse_attention_value(self, q, k, v_new, v_cache, mask, b,
                                src_s, tgt_s, n_head, head_dim, attn_sparsity):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        topk = int(attn_sparsity * (attn_weights.shape[2] - 1))
        topk_weights, topk_indices = attn_weights[:, :, :-1].topk(
            topk, dim=2, sorted=False)
        topk_indices = topk_indices.view(-1, topk).transpose(0, 1)
        # shape: (b * n_head, 1, topk+1)
        attn_weights = torch.cat([topk_weights,
            attn_weights[:, :, -1].unsqueeze(-1)], dim=-1)

        if k.is_cuda:
            v_home = v_cache
            v_buf = self.allocate((topk+1, -1, head_dim), np.float16)
            topk_indices = topk_indices.cpu()
        else:
            (v_home, v_buf) = v_cache

        # shape: (s, b * n_head, head_dim)
        indices_src = topk_indices
        indices_tgt = (slice(0, indices_src.shape[0]), slice(0, v_home.shape[1]))
        general_copy(v_buf, indices_tgt, v_home, indices_src)
        v_home.device.synchronize()

        # shape: (topk+1, b * n_head, head_dim)
        v = v_buf.data[:topk+1]
        v[topk:topk+1] = v_new
        # shape: (b * n_head, topk+1, head_dim)
        v = v.permute(1, 0, 2).reshape(-1, topk+1, head_dim)

        # shape: (b * n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, -1, tgt_s, head_dim)

    def _mixed_device_attention(self, q, k_cache, v_cache, k_new, v_new,
            mask, b, src_s, tgt_s, n_head, head_dim):
        # The caches are stored on both gpu and cpu.
        # Compute attention on gpu for caches stored on gpu.
        # Compute attention on cpu for caches stored on cpu.
        k_gpu, k_cpu = k_cache[0].data, k_cache[1].data
        v_gpu, v_cpu = v_cache[0].data, v_cache[1].data
        seg = k_gpu.shape[1]

        # Compute GPU part
        b_gpu = seg // n_head
        q_gpu = q[:seg]
        # shape: (s, b * n_head, head_dim)
        k_gpu = k_gpu[:src_s, :seg, :]
        v_gpu = v_gpu[:src_s, :seg, :]
        k_gpu[src_s-1:src_s, :, :] = k_new[:, :seg, :]
        v_gpu[src_s-1:src_s, :, :] = v_new[:, :seg, :]
        # shape: (b * n_head, head_dim, s)
        k_gpu = k_gpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_gpu = v_gpu.permute(1, 0, 2)

        mask_gpu = mask[:b_gpu].cuda()
        value_gpu = self._attention_value(q_gpu, k_gpu, v_gpu, mask_gpu,
            b_gpu, src_s, tgt_s, n_head, head_dim)

        # Compute CPU Part
        b_cpu = b - b_gpu
        q_cpu = q[seg:].float().cpu()
        # shape: (s, b * n_head, head_dim)
        k_cpu = k_cpu[:src_s, seg:, :]
        v_cpu = v_cpu[:src_s, seg:, :]
        k_cpu[src_s-1:src_s, :, :] = k_new[:, seg:, :]
        v_cpu[src_s-1:src_s, :, :] = v_new[:, seg:, :]
        # shape: (b * n_head, head_dim, s)
        k_cpu = k_cpu.permute(1, 2, 0)
        # shape: (b * n_head, s, head_dim)
        v_cpu = v_cpu.permute(1, 0, 2)

        mask_cpu = mask[b_gpu:]
        value_cpu = self._attention_value(q_cpu, k_cpu, v_cpu, mask_cpu,
            b_cpu, src_s, tgt_s, n_head, head_dim)

        value = torch.cat([value_gpu, value_cpu.cuda().half()], dim=0)
        return value

    def mlp(self, inputs, wi, bi, wo, bo, w_ln, b_ln, donate):
        # decompress weights
        if wi.device.device_type == DeviceType.COMPRESSED:
            wi = wi.device.decompress(wi)
            wo = wo.device.decompress(wo)

        b, s, h = inputs.shape
        out = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        
        out = F.linear(out, wi.data, bias=bi.data)
        
        F.relu(out, inplace=True)
        out = F.linear(out, wo.data, bias=bo.data)
        
        out.add_(inputs.data)
        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, self)
    def mlp_dist(self, inputs, wi, bi, wo, bo, w_ln, b_ln, donate, rank, world_size):
        # decompress weights
        assert wi.shape[0] % world_size == 0
        if wi.device.device_type == DeviceType.COMPRESSED:
            wi = wi.device.decompress(wi)
            wo = wo.device.decompress(wo)

        b, s, h = inputs.shape
        # wi shape: (4*h // world_size, h)
        # bi shape: (4*h // world_Size,)

        out = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        
        out = F.linear(out, wi.data, bias=bi.data)
        
        # out shape: (b, s, 4*h // world_size)
        F.relu(out, inplace=True)
        # wo shape: (h, 4*h // world_size)
        # bo shape: (h, )
        if rank == 0:
            out = F.linear(out, wo.data, bias=bo.data)
            out.add_(inputs.data)
        else:
            out = F.linear(out, wo.data)
        
        dist.all_reduce(out)
        # out shape: (b, s, h)
        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, self)

    def synchronize(self):
        torch.cuda.synchronize()

    def mem_stats(self):
        if self.device_type == DeviceType.CUDA:
            cur_mem = torch.cuda.memory_allocated(self.dev)
            peak_mem = torch.cuda.max_memory_allocated(self.dev)
        elif self.device_type == DeviceType.CPU:
            cur_mem = cpu_mem_stats()
            peak_mem = 0
        else:
            raise NotImplementedError()

        return cur_mem, peak_mem

    def print_stats(self, output_file=None):
        torch.cuda.synchronize()
        cur_mem, peak_mem = self.mem_stats()

        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(f"TorchDevice: {self.name}\n")
                f.write(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                        f" peak_mem: {peak_mem/GB:.4f} GB\n")
        else:
            print(f"TorchDevice: {self.name}")
            print(f"  cur_mem: {cur_mem/GB:.4f} GB, "
                  f" peak_mem: {peak_mem/GB:.4f} GB")

        return cur_mem, peak_mem

    def __str__(self):
        return f"TorchDevice(name={self.name})"


class TorchDisk:
    """Manage tensors stored on a disk."""

    def __init__(self, path, mem_capacity=None, cuda_id=0, num_copy_threads=4):
        self.name = path
        self.path = os.path.abspath(os.path.expanduser(path))
        self.mem_capacity = mem_capacity

        self.device_type = DeviceType.DISK
        self.compressed_device = TorchCompressedDevice(self)

        if os.path.exists(self.path):
            assert os.path.isdir(self.path)
        else:
            os.makedirs(self.path)

        self.links = {}

        # Copy threads
        self.copy_queue = queue.Queue()
        self.copy_threads = [
            threading.Thread(
                target=copy_worker_func, args=(self.copy_queue, cuda_id)
            ) for _ in range(num_copy_threads)
        ]
        for t in self.copy_threads:
            t.start()

        global global_disk_device
        global_disk_device = self

    def add_link(self, link):
        dst = link.b if link.a == self else link.a
        self.links[dst] = link

    def allocate(self, shape, dtype, pin_memory=None, name=None):
        name = name or TorchTensor.next_name()
        path = os.path.join(self.path, name)
        np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=dtype)
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           path, self, name=name)

    def delete(self, tensor):
        if os.path.exists(tensor.data) and tensor.delete_file:
            os.remove(tensor.data)

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        k_cache = self.allocate(shape, np.float16)
        v_cache = self.allocate(shape, np.float16)
        return k_cache, v_cache
    def init_cache_one_gpu_batch_dist(self, config, task, policy, world_size):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        assert (gpu_batch_size * num_head) % world_size == 0
        shape = (prompt_len + gen_len - 1, (gpu_batch_size * num_head) // world_size, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        k_cache = self.allocate(shape, np.float16)
        v_cache = self.allocate(shape, np.float16)
        return k_cache, v_cache
    def submit_copy(self, *args):
        self.copy_queue.put_nowait(args)

    def synchronize(self):
        self.copy_queue.join()

    def close_copy_threads(self):
        for _ in range(len(self.copy_threads)):
            self.copy_queue.put_nowait(None)
        for t in self.copy_threads:
            t.join()
        self.copy_queue.join()
        self.copy_queue = None

    def mem_stats(self):
        raise NotImplementedError()

    def print_stats(self):
        raise NotImplementedError()

    def __del__(self):
        if self.copy_queue:
            self.close_copy_threads()


# Segment dimension for tensors stored on TorchMixedDevice

class TorchMixedDevice:
    """Manage tensors stored on multiple physical devices."""

    def __init__(self, base_devices):
        self.name = "mixed"
        self.device_type = DeviceType.MIXED
        self.base_devices = base_devices

    def allocate(self, shape, dtype, seg_lengths, pin_memory=None, name=None, seg_dim=1):
        assert sum(seg_lengths) == shape[seg_dim]
        assert len(seg_lengths) == len(self.base_devices)
        seg_points = [0]
        for l in seg_lengths:
            seg_points.append(seg_points[-1] + l)

        devices = self.base_devices
        tensors = []
        for i in range(len(devices)):
            seg_len = seg_points[i+1] - seg_points[i]
            if seg_len == 0:
                tensors.append(None)
            else:
                seg_shape = shape[:seg_dim] + (seg_len,) + shape[seg_dim+1:]
                tensors.append(devices[i].allocate(seg_shape, dtype,
                    pin_memory=pin_memory))

        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                        (tensors, seg_points), self, name=name)
    ## allocate tensor with same size at both gpu and cpu 
    # def allocate_balanced(self, shape, dtype, seg_lengths, pin_memory=None, name=None, seg_dim=1):
    #     assert sum(seg_lengths) == shape[seg_dim]
    #     assert len(seg_lengths) == len(self.base_devices)
    #     seg_points = [0]
    #     for l in seg_lengths:
    #         seg_points.append(seg_points[-1] + l)
    #     devices = self.base_devices
    #     tensors = []
    #     for i in range(len(devices)):
    #         if seg_lengths[i] == 0:
    #             tensors.append(None)
    #         else:
    #             seg_shape = shape[:seg_dim] + (seg_lengths[i],) + shape[seg_dim+1:]
    #             tensors.append(devices[i].allocate(seg_shape, dtype, pin_memory=pin_memory))
    #     return TorchTensor(shape, np_dtype_to_torch_dtype[dtype], tensors, self, name=name, seg_dim=seg_dim)

    ## allocate tensor with same size at both gpu and cpu 
    def allocate_all(self, shape, dtype, pin_memory=None, name=None):
        devices = self.base_devices
        tensors = []
        for i in range(2):
            tensors.append(devices[i].allocate(shape, dtype, pin_memory=pin_memory))
        tensors.append(None)
        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype], (tensors,None), self, name=name)
    def delete(self, tensor):
        for x in self.tensor.data[0]:
            if x:
                x.delete()
    def init_cache_one_gpu_batch(self, config, task, policy, seg_dim=1):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)

        # We have to round to a multiple of `num_head`
        if policy.cache_disk_percent == 0:
            len_gpu = int(shape[seg_dim] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = shape[seg_dim]  - len_gpu
            len_disk = 0
        else:
            len_gpu = int(shape[seg_dim] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = int(shape[seg_dim] * policy.cache_cpu_percent / 100) // num_head * num_head
            len_disk = shape[seg_dim] - len_gpu - len_cpu
        lens = [len_gpu, len_cpu, len_disk]

        pin_memory = False
        k_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        return k_cache, v_cache
    def gather(self, shape, gpu_tensor, cpu_tensor, seg_dim):
        gpu_len = gpu_tensor.shape[seg_dim]
        cpu_len = gpu_tensor.shape[seg_dim]
        input = TorchTensor(shape, dtype=gpu_tensor.dtype, data=[[gpu_tensor, cpu_tensor, None], [0, gpu_len, gpu_len + cpu_len, gpu_len + cpu_len]], device=self, seg_dim=seg_dim)
        output = self.allocate_all(shape, dtype=gpu_tensor.dtype)
        input.balanced_copy(output.data[0][0], None, seg_dim=seg_dim)
        input.balanced_copy(output.data[0][1], None, seg_dim=seg_dim)
        return output

    # def reduce():

    def init_weight_balanced(self, weight_specs, dev_percents, n_head=None):
        weights = []
        
        for i in range(len(weight_specs)):
            shape, dtype, filename, seg_dim = weight_specs[i]
            
            if len(shape) < 2:
                pin_memory = True
            else:
                pin_memory = False
            w_data = self.base_devices[1].allocate(shape, dtype, pin_memory=pin_memory)
            if "_DUMMY_" not in filename:
                w_data.load_from_np_file(weight_specs[i][2])
            else:
                w_data.load_from_np(np.ones(shape, dtype))
            if seg_dim is not None:
                ## all weight in gpu and cpu
                if dev_percents[2] == 0:
                    if n_head:
                        len_gpu = int(n_head * dev_percents[0] / 100) * shape[seg_dim] // n_head
                        len_cpu = shape[seg_dim] - len_gpu
                        len_disk = 0
                    else:
                        len_gpu = int(shape[seg_dim] * dev_percents[0] / 100)
                        len_cpu = shape[seg_dim] - len_gpu
                        len_disk = 0
                ## all weight in gpu, cpu and disk
                else:
                    if n_head:
                        len_gpu = int(n_head * dev_percents[0] / 100) * shape[seg_dim] // n_head
                        len_cpu = int(n_head * dev_percents[1] / 100) * shape[seg_dim] // n_head
                        len_disk = shape[seg_dim] - len_gpu - len_cpu
                    else:
                        len_gpu = int(shape[seg_dim] * dev_percents[0] / 100)
                        len_cpu = int(shape[seg_dim] * dev_percents[1] / 100)
                        len_disk = shape[seg_dim] - len_gpu - len_cpu
                lens = [len_gpu, len_cpu, len_disk]


                weight = self.allocate(shape, dtype, seg_lengths=lens, pin_memory=pin_memory, seg_dim=seg_dim)
                general_copy(weight, None, w_data, None, seg_dim=seg_dim)
            else: ##split dim is None, distribute whole weight to both gpu and cpu
                weight = self.allocate_all(shape, dtype, pin_memory=pin_memory)
                general_copy(weight, None, w_data, None, seg_dim=seg_dim)
            weights.append(weight)
            w_data.delete()
        return weights
    def init_cache_one_gpu_batch_balanced(self, config, task, policy, seg_dim=1):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        if policy.cache_disk_percent == 0:
            len_gpu = gpu_batch_size * int(num_head * policy.cache_gpu_percent / 100)
            len_cpu = shape[seg_dim] - len_gpu
            len_disk = 0
        else:
            len_gpu = gpu_batch_size * int(num_head * policy.cache_gpu_percent / 100)
            len_cpu = gpu_batch_size * int(num_head * policy.cache_gpu_percent / 100)
            len_disk = shape[seg_dim] - len_gpu - len_cpu
        lens = [len_gpu, len_cpu, len_disk]

        pin_memory = False
        k_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        return k_cache, v_cache
    def init_cache_one_gpu_batch_dist(self, config, task, policy, world_size, seg_dim=1):
        raise NotImplementedError
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        assert (gpu_batch_size * num_head) % world_size == 0
        shape = (prompt_len + gen_len - 1, (gpu_batch_size * num_head) // world_size, hidden_size // num_head)
        
        # We have to round to a multiple of `num_head`
        if policy.cache_disk_percent == 0:
            len_gpu = int(shape[seg_dim] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = shape[seg_dim]  - len_gpu
            len_disk = 0
        else:
            len_gpu = int(shape[seg_dim] * policy.cache_gpu_percent / 100) // num_head * num_head
            len_cpu = int(shape[seg_dim] * policy.cache_cpu_percent / 100) // num_head * num_head
            len_disk = shape[seg_dim] - len_gpu - len_cpu
        lens = [len_gpu, len_cpu, len_disk]

        pin_memory = False
        k_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            seg_lengths=lens, pin_memory=pin_memory)
        return k_cache, v_cache
    def opt_input_embed_balanced(self, inputs, attention_mask, w_token, w_pos, pad_token_id, donate):
        # decompress weights
        if w_token.device.device_type == DeviceType.COMPRESSED:
            w_token = w_token.device.decompress(w_token)
            w_pos = w_pos.device.decompress(w_pos)

        token_ids = inputs.data
        mask = attention_mask.data
        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # token embedding
        
        token_embed_gpu = F.embedding(token_ids, w_token.data[0][0], pad_token_id)
        token_embed_cpu = F.embedding(token_ids, w_token.data[0][1], pad_token_id)
        # pos embedding
        positions = torch.cumsum(mask, dim=1).int() * mask + 1

        # cut positions if `past_key_values_length` is > 0
        past_key_values_length = mask.shape[1] - token_ids.shape[1]
        positions = positions[:, past_key_values_length:]

        pos_embed_gpu = F.embedding(positions, w_pos.data[0][0])
        pos_embed_cpu = F.embedding(positions, w_pos.data[0][1])
        data_gpu = token_embed_gpu + pos_embed_gpu
        data_cpu = token_embed_cpu + pos_embed_cpu
        b, s, gpu_h = data_gpu.shape
        b, s, cpu_h = data_cpu.shape
        output = self.gather((b, s, gpu_h + cpu_h), data_gpu, data_cpu, seg_dim=2)
        return output
        # output = torch.empty((b, s, gpu_h + cpu_h), )
        # output = torch.empty((world_size, b, s, dist_h) , device=self.dev, dtype=data.dtype)
        # dist.all_gather_into_tensor(output, data)
        # output = output.permute(1, 2, 0, 3).reshape(b, s, -1)

        # return TorchTensor.create_from_torch(output, self)
class TorchLink:
    """An I/O link between two devices."""

    def __init__(self, a, b, a_to_b_bandwidth, b_to_a_bandwidth):
        self.a = a
        self.b = b
        self.a_to_b_bandwidth = a_to_b_bandwidth
        self.b_to_a_bandwidth = b_to_a_bandwidth

        a.add_link(self)
        b.add_link(self)

    def io_time(self, src, dst, size):
        if src == self.a:
            assert dst == self.b
            bandwidth = self.a_to_b_bandwidth
        elif src == self.b:
            assert dst == self.a
            bandwidth = self.b_to_a_bandwidth
        else:
            raise ValueError(f"Invalid source {src}")

        if force_io_time is not None:
            return force_io_time

        return size / bandwidth

def get_seg_points(seg_lengths):
    lst = [0]
    for i in range(seg_lengths):
        lst.append(i + lst[-1])
    return lst

    

def general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice], seg_dim):
    """Launch a general asynchronous copy between two tensors.
    It is equivalent to `dst[dst_indices] = src[src_indices]` in numpy syntax.
    The copy is asynchronous. To wait for the copy to complete, you need to call
    >>> env.disk.synchronize()
    >>> torch.cuda.synchronize()
    """
    if src.device.device_type == DeviceType.MIXED and dst.device.device_type == DeviceType.MIXED:
        # if seg_dim == None: ## allocate_all
        #     dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
        #     src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
        #     for i in range(len(dst.device.base_devices))
        # else:  
        dst_seg_points = dst.data[1]
        src_seg_points = src.data[1]
        dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
        src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
        for i in range(len(src.device.base_devices)):
            if src_seg_points[i] == src_seg_points[i+1]:
                continue
            for j in range(len(dst.device.base_devices)):
                if dst_seg_points[j] == dst_seg_points[j+1]:
                    continue
                start = max(src_seg_points[i], dst_seg_points[j])
                end = min(src_seg_points[i+1], dst_seg_points[j+1])
                if start >= end:
                    continue
                src_indices = src_indices[:seg_dim] + (slice(start - src_seg_points[i], end - src_seg_points[i]), ) + src_indices[seg_dim + 1:]
                dst_indices = dst_indices[:seg_dim] + (slice(start - dst_seg_points[j], end - dst_seg_points[j]), ) + dst_indices[seg_dim + 1:]
                general_copy(dst.data[0][j], dst_indices, src.data[0][i], src_indices, seg_dim=seg_dim)

    elif dst.device.device_type == DeviceType.MIXED:
        assert src.device.device_type != DeviceType.MIXED
        seg_points = dst.data[1]
        if seg_dim == None:
            for i in range(len(dst.device.base_devices)):
                if dst.data[0][i] is not None:
                    general_copy(dst.data[0][i], None, src, None, seg_dim=seg_dim)
        else:
            for i in range(len(dst.device.base_devices)):
                if seg_points[i] == seg_points[i+1]:
                    continue
                src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
                dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
                tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1], seg_dim=seg_dim)
                tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1], base=seg_points[i], seg_dim=seg_dim)
                general_copy(dst.data[0][i], tmp_dst_indices, src, tmp_src_indices, seg_dim=seg_dim)
    elif src.device.device_type == DeviceType.MIXED:
        # The tensor is on mixed devices, do recursive calls
        assert dst.device.device_type != DeviceType.MIXED
        seg_points = src.data[1]

        for i in range(len(src.device.base_devices)):
            if seg_points[i] == seg_points[i+1]:
                continue
            src_indices = src_indices or tuple(slice(0, x) for x in src.shape)
            dst_indices = dst_indices or tuple(slice(0, x) for x in dst.shape)
            tmp_src_indices = cut_indices(src_indices, seg_points[i], seg_points[i+1], base=seg_points[i], seg_dim=seg_dim)
            tmp_dst_indices = cut_indices(dst_indices, seg_points[i], seg_points[i+1], seg_dim=seg_dim)
            general_copy(dst, tmp_dst_indices, src.data[0][i], tmp_src_indices, seg_dim=seg_dim)
    elif (src.device.device_type == DeviceType.COMPRESSED or
          dst.device.device_type == DeviceType.COMPRESSED):
        # The tensor is compressed, do recursive calls
        general_copy_compressed(dst, dst_indices, src, src_indices)
    elif src.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        src.device.submit_copy(dst, dst_indices, src, src_indices)
    elif dst.device.device_type == DeviceType.DISK:
        # The tensor is on the disk, dispatch to copy threads for asynchronous copy
        dst.device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CUDA and
          dst.device.device_type == DeviceType.CPU and
          not dst.data.is_pinned() and src.shape[0] > 1):
        # The cpu tensor is not pinned, dispatch to copy threads and use pin_memory
        # as a relay
        global_disk_device.submit_copy(dst, dst_indices, src, src_indices)
    elif (src.device.device_type == DeviceType.CPU and
          dst.device.device_type == DeviceType.CUDA and
          not src.data.is_pinned()):
        # The cpu tensor is not pinned, use pin_memory as a relay
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        src = src.pin_memory()
        dst.copy_(src, non_blocking=True)
    else:
        # The normal path
        src = src.data[src_indices] if src_indices else src.data
        dst = dst.data[dst_indices] if dst_indices else dst.data
        dst.copy_(src, non_blocking=True)


def cut_indices(indices, start, stop, base=0, seg_dim=None):
    assert all(x.step is None for x in indices)
    if seg_dim == None:
        return indices
    seg = indices[seg_dim]
    return (indices[:seg_dim] +
            (slice(max(seg.start, start) - base, min(seg.stop, stop) - base),) +
            indices[seg_dim + 1:])


def map_to_torch_tensor(tensor, indices):
    if tensor.device.device_type == DeviceType.DISK:
        data = torch.from_numpy(np.lib.format.open_memmap(tensor.data))
    else:
        data = tensor.data

    # BC: this is supposed to only handle the sparse v_cache case
    if torch.is_tensor(indices):
        return vector_gather(data, indices)
    return data[indices] if indices else data


def copy_worker_func(queue, cuda_id):
    """The copy worker thread."""
    torch.cuda.set_device(cuda_id)

    cpu_buf = torch.empty((1 * GB,), dtype=torch.float16, pin_memory=True)
    copy_stream = torch.cuda.Stream()

    with torch.cuda.stream(copy_stream):
        while True:
            item = queue.get()
            if item is None:
                queue.task_done()
                return

            dst, dst_indices, src, src_indices = item
            src_data = map_to_torch_tensor(src, src_indices)
            dst_data = map_to_torch_tensor(dst, dst_indices)

            if (src.device.device_type == DeviceType.CUDA or
                dst.device.device_type == DeviceType.CUDA):
                # Use a pinned cpu buffer as a relay
                size = np.prod(src_data.shape)
                tmp_cpu_buf = cpu_buf[:size].view(src_data.shape)
                tmp_cpu_buf.copy_(src_data)
                dst_data.copy_(tmp_cpu_buf)
            else:
                dst_data.copy_(src_data)

            queue.task_done()
