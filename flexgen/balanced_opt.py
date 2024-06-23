"""
Usage:
python3 -m flexgen.flex_opt --model facebook/opt-1.3b --gpu-batch-size 32 --percent 100 0 100 0 100 0
"""

import argparse
import dataclasses
import os
import pickle
import time
from typing import Union, List, Optional

import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from flexgen.compression import CompressionConfig
from flexgen.opt_config import OptConfig, get_opt_config, download_opt_weights
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, DeviceType, general_copy, fix_recursive_import)
from flexgen.timer import timers
from flexgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)

fix_recursive_import()

DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes


@dataclasses.dataclass(frozen=True)
class Policy:
    gpu_batch_size: int

    # percent = a means a%
    InputEmbed_w_gpu_percent: float
    InputEmbed_w_cpu_percent: float
    OutputEmbed_w_gpu_percent: float
    OutputEmbed_w_cpu_percent: float
    SelfAttention_w_gpu_percent: float
    SelfAttention_w_cpu_percent: float
    MLP_w_gpu_percent: float
    MLP_w_cpu_percent: float

    InputEmbed_comp_gpu_percent: float
    OutputEmbed_comp_gpu_percent: float
    SelfAttention_comp_gpu_percent: float
    MLP_comp_gpu_percent: float

    cache_gpu_percent: float
    cache_cpu_percent: float
    # w_gpu_percent: float
    # w_cpu_percent: float
    

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to separate attention and mlp as two layers
    sep_layer: bool

    # Whether to use pinned memory for weights on CPU
    pin_weight: bool
    compress_cache = False
    # Whether to compute attention on CPU
    # cpu_cache_compute: bool

    # Sparsity of attention weights
    # attn_sparsity: float

    # Compress weights with group-wise quantization
    # compress_weight: bool
    # comp_weight_config: CompressionConfig

    # Compress KV cache with group-wise quantization
    # compress_cache: bool
    # comp_cache_config: CompressionConfig
    @property
    def InputEmbed_w_disk_percent(self):
        return 100 - self.InputEmbed_w_gpu_percent - self.InputEmbed_w_cpu_percent
    @property
    def OutputEmbed_w_disk_percent(self):
        return 100 - self.OutputEmbed_w_gpu_percent - self.OutputEmbed_w_cpu_percent
    @property
    def SelfAttention_w_disk_percent(self):
        return 100 - self.SelfAttention_w_gpu_percent - self.SelfAttention_w_cpu_percent
    @property
    def MLP_w_disk_percent(self):
        return 100 - self.MLP_w_gpu_percent - self.MLP_w_cpu_percent
    # @property
    # def w_disk_percent(self):
    #     return 100 - self.w_gpu_percent - self.w_cpu_percent
    @property
    def InputEmbed_comp_cpu_percent(self):
        return 100 - self.InputEmbed_comp_gpu_percent
    @property
    def OutputEmbed_comp_cpu_percent(self):
        return 100 - self.OutputEmbed_comp_gpu_percent
    @property
    def SelfAttention_comp_cpu_percent(self):
        return 100 - self.SelfAttention_comp_gpu_percent
    @property
    def MLP_comp_cpu_percent(self):
        return 100 - self.MLP_comp_gpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent

class InputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        # self.compute = self.env.gpu
        # self.weight_load_dst = self.compute

        self.task = None

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, s, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.max_seq_len, self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_token
            ((v, h), dtype, path + "decoder.embed_tokens.weight", 1),
            # w_pos
            ((s + 2, h), dtype, path + "decoder.embed_positions.weight", 1),
        ]
        dev_percents = [self.policy.InputEmbed_w_gpu_percent, self.policy.InputEmbed_w_cpu_percent, self.policy.InputEmbed_w_disk_percent]
        weights = self.env.mixed.init_weight_balanced(weight_specs, dev_percents)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, check_time):
        if check_time:
            timers("InputEmbed_load_weight").start(self.sync)
        w_token, w_pos = weight_home.val
        comp_gpu_percent = self.policy.InputEmbed_comp_gpu_percent
        v, h = w_token.shape
        len_gpu = int((h * comp_gpu_percent) / 100)
        len_cpu = h - len_gpu
        ##compute on gpu or cpu
        seg_lengths = [len_gpu, len_cpu, 0]
        
        ## w_token stored in gpu to comp_w_token(gpu)
        weight_read_buf.store((w_token.balanced_copy(self.env.mixed, seg_lengths, 1), w_pos.balanced_copy(self.env.mixed, seg_lengths, 1)))
        if check_time:
            timers("InputEmbed_load_weight").stop(self.sync)
    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i, check_time):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i, check_time):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len), np.int64

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, check_time):
        # Compute input embedding
        if check_time:
            timers("InputEmbed_comp").start(self.sync)
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.balanced_copy(self.env.mixed, None, None)
        # mask, donate[1] = attention_mask.val.smart_copy(self.compute)
        # Clear the weight_read_buf if it is the last gpu batch
        (w_token, donate[2]), (w_pos, donate[3]) = weight_read_buf.pop()

        h = self.compute.opt_input_embed_balanced(h, mask,
            w_token, w_pos, self.config.pad_token_id, donate)
        hidden.val = h
        if check_time:
            timers("InputEmbed_comp").stop(self.sync)

class OutputEmbed:
    def __init__(self, config, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = self.compute

        self.task = None

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_ln
            ((h,), dtype, path + "decoder.layer_norm.weight", None),
            # b_ln
            ((h,), dtype, path + "decoder.layer_norm.bias", None),
            # w_token
            ((v, h), dtype, path + "decoder.embed_tokens.weight", 0),
        ]
        # weights = init_weight_list_balanced(weight_specs, self.policy, self.env, 'OutputEmbed')
        dev_percents = [self.policy.OutputEmbed_w_gpu_percent, self.policy.OutputEmbed_w_cpu_percent, self.policy.OutputEmbed_w_disk_percent]
        weights = self.env.mixed.init_weight_balanced(weight_specs, dev_percents)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, check_time):
        if check_time:
            timers("OutputEmbed_load_weight").start(self.sync)
        w_ln, b_ln, w_token = weight_home.val
        comp_gpu_percent = self.policy.OutputEmbed_comp_gpu_percent
        v, h = w_token.shape
        len_gpu = int((v * comp_gpu_percent) / 100)
        len_cpu = v - len_gpu
        seg_lengths = [len_gpu, len_cpu, 0]
        dst1 = self.weight_load_dst
        dst2 = self.compute
        weight_read_buf.store((w_ln.balanced_copy(self.env.mixed, None, None), b_ln.balanced_copy(self.env.mixed, None, None), w_token.balanced_copy(self.env.mixed, seg_lengths, 0)))
        if check_time:
            timers("OutputEmbed_load_weight").stop(self.sync)
    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i, check_time):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i, check_time):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, check_time):
        if check_time:
            timers("OutputEmbed_comp").start(self.sync)
        donate = [False] * 4
        h, donate[0] = hidden.val, True

        # Clear the weight_read_buf if it is the last gpu batch
        (w_ln, donate[1]), (b_ln, donate[2]), (w_token, donate[3]) = weight_read_buf.pop()

        h = self.compute.opt_output_embed(h, w_ln, b_ln, w_token, donate,
            self.task.do_sample, self.task.temperature)
        hidden.val = h
        if check_time:
            timers("OutputEmbed_comp").stop(self.sync)
class SelfAttention:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = self.compute
        self.attention_compute = self.env.cpu

        self.task = None

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}.self_attn"))
        weight_specs = [
                # w_q
                ((h, h), dtype, path + ".q_proj.weight", 0),
                # b_q
                ((h,), dtype, path + ".q_proj.bias", 0),
                # w_k
                ((h, h), dtype, path + ".k_proj.weight", 0),
                # b_k
                ((h,), dtype, path + ".k_proj.bias", 0),
                # w_v
                ((h, h), dtype, path + ".v_proj.weight", 0),
                # b_v
                ((h,), dtype, path + ".v_proj.bias", 0),
                # w_out
                ((h, h), dtype, path + ".out_proj.weight", 1),
                # b_out
                ((h,), dtype, path + ".out_proj.bias", None),
                # w_ln
                ((h,), dtype, path + "_layer_norm.weight", None),
                # b_ln
                ((h,), dtype, path + "_layer_norm.bias", None),
        ]
        dev_percents = [self.policy.SelfAttention_w_gpu_percent, self.policy.SelfAttention_w_cpu_percent, self.policy.SelfAttention_w_disk_percent]
        weights = self.env.mixed.init_weight_balanced(weight_specs, dev_percents, n_head = self.config.n_head)
        weight_home.store(weights)
    def load_weight(self, weight_home, weight_read_buf, check_time):
        if check_time:
            timers("SelfAttention_load_weight").start(self.sync)
        w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln = weight_home.val
        dst1 = self.weight_load_dst
        dst2 = self.compute
        comp_gpu_percent = self.policy.SelfAttention_comp_gpu_percent
        h, = b_q.shape
        len_gpu = int((h * comp_gpu_percent) / 100)
        len_cpu = h - len_gpu
        seg_lengths = [len_gpu, len_cpu, 0]
        weight_read_buf.store((
            w_q.balanced_copy(self.env.mixed, seg_lengths, 0), b_q.balanced_copy(self.env.mixed, seg_lengths, 0), 
            w_k.balanced_copy(self.env.mixed, seg_lengths, 0), b_k.balanced_copy(self.env.mixed, seg_lengths, 0),
            w_v.balanced_copy(self.env.mixed, seg_lengths, 0), b_v.balanced_copy(self.env.mixed, seg_lengths, 0),
            w_out.balanced_copy(self.env.mixed, seg_lengths, 1), b_out.balanced_copy(self.env.mixed, None, None),
            w_ln.balanced_copy(self.env.mixed, None, None), b_ln.balanced_copy(self.env.mixed, None, None)))
        if check_time:
            timers("SelfAttention_load_weight").stop(self.sync)

    def init_cache_one_gpu_batch(self, cache_home):
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy)
        cache_home.store(cache)
    def load_cache(self, cache_home, cache_read_buf, i, check_time):
        if i == 0:  # prefill, no cache
            return
        if check_time:
            timers("SelfAttention_load_cache").start(self.sync)
        # shape: (s, b * n_head, head_dim) ?
        k_home, v_home = cache_home.val

        # Pick code path

        if (k_home.device.device_type == DeviceType.MIXED and
            k_home.data[0][0] is not None):
            path = 1
        else:
            path = 0
        dst = self.attention_compute
        if path == 0:  # Copy to CPU temporary workspace
            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(0, k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)
            general_copy(v_buf, indices, v_home, indices)
            cache_read_buf.store(((k_buf, False), (v_buf, False)))
        elif path == 1:  # Copy to both GPU and CPU
            # The caches are stored on both GPU and other devices.
            # Compute attention on gpu for caches stored on gpu.
            # Compute attention on cpu for caches stored on cpu/disk.
            gpu_k_buf = k_home.data[0][0]
            gpu_v_buf = v_home.data[0][0]

            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(gpu_k_buf.shape[1], k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)
            general_copy(v_buf, indices, v_home, indices)
            cache_read_buf.store((((gpu_k_buf, k_buf,), False),
                                  ((gpu_v_buf, v_buf,), False)))
            assert self.policy.attn_sparsity >= 1.0
        else:
            raise ValueError(f"Invalid path: {path}")
        if check_time:
            timers("SelfAttention_load_cache").stop(self.sync)
    def store_cache(self, cache_home, cache_write_buf, i, check_time):
        # shape: (s, b * n_head, head_dim)
        if check_time:
            timers("SelfAttention_store_cache").start(self.sync)
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()

        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return

        if i == 0:  # prefill
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1]))
        else:  # decoding
            pos = self.task.prompt_len + i
            indices = (slice(pos - k_new.shape[0], pos),
                       slice(0, k_new.shape[1]))

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)
        if check_time:
            timers("SelfAttention_store_cache").stop(self.sync)
    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, check_time):
        if check_time:
            timers("SelfAttention_comp").start(self.sync)
        n_head = self.config.n_head

        donate = [False] * 14
        h, donate[0] = hidden.val, True

        # Clear the weight_read_buf if it is the last gpu batch
        ((w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]),
            (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (b_out, donate[9]),
            (w_ln, donate[10]), (b_ln, donate[11])) = weight_read_buf.pop()

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            h, new_k_cache, new_v_cache = self.compute.mha_balanced(h, mask, w_q, b_q,
                w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate)
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:  # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()
            h, new_k_cache, new_v_cache = self.compute.mha_gen_balanced(h, mask, w_q,
                b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head,
                k_cache, v_cache, donate)
            cache_write_buf.store((new_k_cache, new_v_cache))
        hidden.val = h
        # print(h.data)
        if check_time:
            timers("SelfAttention_comp").stop(self.sync)
class MLP:
    def __init__(self, config, env, policy, layer_id):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = self.compute

        self.task = None

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}."))
        weight_specs = [
            # wi
            ((4 * h, h), dtype, path + "fc1.weight", 0),
            # bi
            ((4 * h,), dtype, path + "fc1.bias", 0),
            # wo
            ((h, 4 * h), dtype, path + "fc2.weight", 1),
            # bo
            ((h,), dtype, path + "fc2.bias", None),
            # w_ln
            ((h,), dtype, path + "final_layer_norm.weight", None),
            # b_ln
            ((h,), dtype, path + "final_layer_norm.bias", None),
        ]
        dev_percents = [self.policy.MLP_w_gpu_percent, self.policy.MLP_w_cpu_percent, self.policy.MLP_w_disk_percent]
        weights = self.env.mixed.init_weight_balanced(weight_specs, dev_percents)
        weight_home.store(weights)
        # print(f'mlp first data {weights[0].data[0][0]}')
    def load_weight(self, weight_home, weight_read_buf, check_time):
        if check_time:
            timers("MLP_load_weight").start(self.sync)
        wi, bi, wo, bo, w_ln, b_ln = weight_home.val
        comp_gpu_percent = self.policy.MLP_comp_gpu_percent
        h,  = bo.shape
        len_gpu = int((4 * h *  comp_gpu_percent) / 100)
        len_cpu = 4 * h - len_gpu
        seg_lengths = [len_gpu, len_cpu, 0]
        weight_read_buf.store((
            wi.balanced_copy(self.env.mixed, seg_lengths, 0), bi.balanced_copy(self.env.mixed, seg_lengths, 0),
            wo.balanced_copy(self.env.mixed, seg_lengths, 1), bo.balanced_copy(self.env.mixed, None, None),
            w_ln.balanced_copy(self.env.mixed, None, None), b_ln.balanced_copy(self.env.mixed, None, None)
        ))
        # dst1 = self.weight_load_dst
        # dst2 = self.compute
        # weight_read_buf.store((
        #     wi.smart_copy(dst1), bi.smart_copy(dst2),
        #     wo.smart_copy(dst1), bo.smart_copy(dst2),
        #     w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))
        if check_time:
            timers("MLP_load_weight").stop(self.sync)

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i, check_time):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i, check_time):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, check_time):
        if check_time:
            timers("MLP_comp").start(self.sync)
        donate = [False] * 7
        h, donate[0] = hidden.val, True

        # Clear the weight_read_buf if it is the last gpu batch
        ((wi, donate[1]), (bi, donate[2]), (wo, donate[3]), (bo, donate[4]),
            (w_ln, donate[5]), (b_ln, donate[6])) = weight_read_buf.pop()

        h = self.compute.mlp(h, wi, bi, wo, bo, w_ln, b_ln, donate)
        hidden.val = h
        if check_time:
            timers("MLP_comp").stop(self.sync)

class TransformerLayer:
    def __init__(self, config, env, policy, i):
        self.attention = SelfAttention(config, env, policy, i)
        self.mlp = MLP(config, env, policy, i)
        self.policy = policy
        self.compute = self.attention.compute

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def set_task(self, task):
        self.attention.set_task(task)
        self.mlp.set_task(task)

    def init_weight(self, weight_home, path):
        home1, home2 = ValueHolder(), ValueHolder()
        self.attention.init_weight(home1, path)
        self.mlp.init_weight(home2, path)
        weight_home.store((home1, home2))

    def load_weight(self, weight_home, weight_read_buf, check_time):
        read_buf1, read_buf2 = ValueHolder(), ValueHolder()
        home1, home2 = weight_home.val
        self.attention.load_weight(home1, read_buf1, check_time)
        self.mlp.load_weight(home2, read_buf2, check_time)

        weight_read_buf.store((read_buf1, read_buf2))

    def load_cache(self, cache_home, cache_read_buf, i, check_time):
        self.attention.load_cache(cache_home, cache_read_buf, i, check_time)

    def store_cache(self, cache_home, cache_write_buf, i, check_time):
        self.attention.store_cache(cache_home, cache_write_buf, i, check_time)

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, check_time):
        read_buf1, read_buf2 = weight_read_buf.pop()

        self.attention.forward(hidden, cache_read_buf, read_buf1, attention_mask,
                               cache_write_buf, i, check_time)
        self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i, check_time)

class OptLM:
    def __init__(self,
                 config: Union[str, OptConfig],
                 env: ExecutionEnv,
                 path: str,
                 policy: Policy):
        if isinstance(config, str):
            config = get_opt_config(config)
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy

        layers = []
        layers.append(InputEmbed(self.config, self.env, self.policy))
        for i in range(self.config.num_hidden_layers):
            if policy.sep_layer:
                layers.append(SelfAttention(self.config, self.env, self.policy, i))
                layers.append(MLP(self.config, self.env, self.policy, i))
            else:
                layers.append(TransformerLayer(self.config, self.env, self.policy, i))
        layers.append(OutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)


        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers = self.num_layers

        # cache[j]
        self.cache_home = array_1d(num_layers, ValueHolder)
        self.cache_read_buf = array_1d(num_layers, ValueHolder)
        self.cache_write_buf = array_1d(num_layers, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        # attention_mask
        self.attention_mask = ValueHolder()

        self.task = None
        self.init_all_weights()

    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)

    def init_weight(self, j):
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))
        check_path = os.path.join(expanded_path, "decoder.embed_positions.weight")
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            download_opt_weights(self.config.name, self.path)
        self.layers[j].init_weight(self.weight_home[j], expanded_path)
    def load_weight(self, i, j, check_time, overlap=True):
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from weight_home to weight_read_buf
        if overlap:
            with torch.cuda.stream(self.load_weight_stream):
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], check_time)
        else:
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], check_time)

    def delete_weight(self, j):
        for x in self.weight_home[j].pop():
            if isinstance(x, ValueHolder):
                for y in x.pop():
                    y.delete()
            else:
                x.delete()

    def init_cache(self, j):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j])

    def load_cache(self, i, j, check_time, overlap=True):
        # Handle corner cases
        if i == 0:  # prefill, no cache
            return
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from cache_home to cache_read_buf
        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j], self.cache_read_buf[j], i, check_time)
        else:
            self.layers[j].load_cache(self.cache_home[j], self.cache_read_buf[j], i, check_time)

    def store_cache(self, i, j, check_time, overlap=True):
        # Handle corner cases
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return
        if i == self.task.gen_len - 1:  # last token, no need to store cache
            self.cache_write_buf[j].pop()
            return

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j], self.cache_write_buf[j], i, check_time)
        else:
            self.layers[j].store_cache(self.cache_home[j], self.cache_write_buf[j], i, check_time)

    def delete_cache(self, j):
        v = self.cache_home[j].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, i):
        # Load to hidden states buffers
        dst = self.env.gpu
        # dst = self.env.mixed
        gpu_batch_size = self.policy.gpu_batch_size
        if i == 0:  # load from the input ids
            val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
            val.load_from_np(self.output_ids[:gpu_batch_size, :self.task.prompt_len])
            # val = dst.allocate_all((gpu_batch_size, self.task.prompt_len), np.int32)
            # val.data[0].load_from_np(self.output_ids[:gpu_batch_size, :self.task.prompt_len])
            # val.data[1].load_from_np(self.output_ids[:gpu_batch_size, :self.task.prompt_len])
        else:  # load from the last generated token
            pos = self.task.prompt_len + i
            val = dst.allocate((gpu_batch_size, 1), np.int32)
            val.load_from_np(self.output_ids[:gpu_batch_size, pos-1:pos])
            # val = dst.allocate_all((gpu_batch_size, 1), np.int32)
            # val.data[0].load_from_np(self.output_ids[:gpu_batch_size, pos-1:pos])
            # val.data[1].load_from_np(self.output_ids[:gpu_batch_size, pos-1:pos])
        self.hidden.store(val)

    def store_hidden(self, i):
        # Handle corner cases
        # Store to hidden states buffers
        gpu_batch_size = self.policy.gpu_batch_size
        ids = self.hidden.pop().data.detach().cpu().numpy()
        pos = self.task.prompt_len + i
        if self.task.stop:
            stopped = self.stopped[:gpu_batch_size]
            self.output_ids[:gpu_batch_size, pos:pos+1] = np.where(
                stopped, self.config.pad_token_id, ids)
            stopped[:] = np.logical_or(stopped, ids == self.task.stop)
        else:
            self.output_ids[:gpu_batch_size, pos:pos+1] = ids

    def compute_layer(self, i, j, check_time):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        self.layers[j].forward(self.hidden, self.cache_read_buf[j],
            self.weight_read_buf[j], self.attention_mask,
            self.cache_write_buf[j], i, check_time)

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def init_all_weights(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for j in range(self.num_layers):
            self.init_weight(j)
            # print(f'{j} th layer, weight len {len(self.weight_home[j].val)}')

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j)

    def update_attention_mask(self, i):
        if i > 0:
            mask = self.attention_mask
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        gpu_batch_size = self.policy.gpu_batch_size
        input_ids = self.output_ids[:gpu_batch_size, :self.task.prompt_len]

        attention_compute = self.env.cpu
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask.store(val)

    def generate(self,
                 inputs: Union[np.array, List[List[int]]],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: Optional[int] = None,
                 debug_mode: Optional[str] = None,
                 cut_gen_len: Optional[int] = None,
                 verbose: int = 0,
                 check_time: bool = False):
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
            check_time=check_time
        )
        num_layers = self.num_layers
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

        # Output token ids
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        assert gpu_batch_size == len(task.inputs)

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers = self.num_layers
        for j in range(num_layers):
            self.cache_home[j].clear()
            self.cache_read_buf[j].clear()
            self.cache_write_buf[j].clear()
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
        self.attention_mask.clear()
        self.hidden = ValueHolder()

        # Init cache
        self.set_task(task)
        for j in range(num_layers):
            self.init_cache(j)
        self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)

        self.generation_loop_overlap_single_batch()

        # Delete cache
        for j in range(num_layers):
            self.delete_cache(j)
        self.env.cpu.del_attention_compute_workspace()

        return self.output_ids
    
    def generation_loop_overlap_single_batch(self):
        # Prologue
        self.load_weight(0, 0, True)
        self.sync()
        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            self.update_attention_mask(i)
            self.load_hidden(i)
            for j in range(self.num_layers):
                self.load_weight(i, j+1, self.task.check_time)
                self.load_cache(i, j+1, self.task.check_time)
                self.compute_layer(i, j, self.task.check_time)
                self.store_cache(i, j-1, self.task.check_time)
                self.sync()
            self.store_hidden(i)
            timers("generate").stop()

            if self.task.stop and np.all(self.stopped):
                break

def get_filename(args):
    model_size = args.model.split('-')[-1]
    per_layer_weight_percent = ""
    per_layer_computation_percent = ""
    for i in range(len(args.per_layer_weight_percent)):
        per_layer_weight_percent += str(args.per_layer_weight_percent[i]) + "-"
    for i in range(len(args.per_layer_computation_percent)):
        per_layer_computation_percent += str(args.per_layer_computation_percent[i]) + "-"
    cache_percent = ""
    for i in range(len(args.cache_percent)):
        cache_percent += str(args.cache_percent[i]) + "-"
    filename = f"fo-{model_size}-gbs{args.gpu_batch_size}-" \
               f"prompt{args.prompt_len}-" \
               f"gen{args.gen_len}-per_layer_weight_percent-{per_layer_weight_percent}-per_layer_computation_percent-{per_layer_computation_percent}-cache_percent-{cache_percent}"
    if args.cpu_cache_compute:
        filename += "cpu-cache"
    else:
        filename += "gpu-cache"
    if args.compress_weight:
        filename += "-compw"
    if args.compress_cache:
        filename += "-compc"
    return filename


def get_test_inputs(prompt_len, num_prompts, tokenizer):
    prompts = ["Paris is the capital city of"]
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len).input_ids
    return (input_ids[0],) * num_prompts


def run_flexgen(args):
    print(f"<run_flexgen>: args.model: {args.model}")
    if args.model == "facebook/galactica-30b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b", padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    num_prompts = args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # Task and policy
    warmup_inputs = get_test_inputs(512, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    policy = Policy(args.gpu_batch_size, 
                    args.per_layer_weight_percent[0], args.per_layer_weight_percent[1],
                    args.per_layer_weight_percent[2], args.per_layer_weight_percent[3],
                    args.per_layer_weight_percent[4], args.per_layer_weight_percent[5],
                    args.per_layer_weight_percent[6], args.per_layer_weight_percent[7],
                    args.per_layer_computation_percent[0], args.per_layer_computation_percent[1],
                    args.per_layer_computation_percent[2], args.per_layer_computation_percent[3],
                    args.cache_percent[0], args.cache_percent[1],
                    args.overlap, args.sep_layer, args.pin_weight,
                    # args.cpu_cache_compute, args.attn_sparsity,
                    # args.compress_weight,
                    # CompressionConfig(num_bits=4, group_size=64,
                    #                   group_dim=0, symmetric=False),
                    # args.compress_cache,
                    # CompressionConfig(num_bits=4, group_size=64,
                    #                   group_dim=2, symmetric=False))
    )
    # assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    opt_config = get_opt_config(args.model)
    cache_size = opt_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = opt_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {opt_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    print("init weight...")
    model = OptLM(opt_config, env, args.path, policy)

    try:
        print("warmup - generate")
        
        output_ids = model.generate(
            warmup_inputs, max_new_tokens=2, verbose=args.verbose, check_time=True)
        InputEmbed_load_weight = timers("InputEmbed_load_weight").costs
        InputEmbed_comp = timers("InputEmbed_comp").costs
        OutputEmbed_load_weight = timers("OutputEmbed_load_weight").costs
        OutputEmbed_comp = timers("OutputEmbed_comp").costs
        SelfAttention_load_weight = timers("SelfAttention_load_weight").costs
        SelfAttention_load_cache = timers("SelfAttention_load_cache").costs
        SelfAttention_store_cache = timers("SelfAttention_store_cache").costs
        SelfAttention_comp = timers("SelfAttention_comp").costs
        MLP_load_weight = timers("MLP_load_weight").costs
        MLP_comp = timers("MLP_comp").costs
        print('InputEmbed')
        print('selfattention load_weight', str(np.mean(SelfAttention_load_weight)))
        print('inputembed comp', str(np.mean(InputEmbed_comp[1:])))
        print('selfattention load cache', str(np.mean(SelfAttention_load_cache)))

        print('SelfAttention')
        print('MLP load weight', str(np.mean(MLP_load_weight)))
        print('selfattention comp', str(np.mean(SelfAttention_comp[1:])))

        print('MLP')
        print('selfattention load_weight', str(np.mean(SelfAttention_load_weight)))
        print('MLP comp', str(np.mean(MLP_comp[1:])))
        print('selfattention load cache', str(np.mean(SelfAttention_load_cache)))
        print('selfattention store cache', str(np.mean(SelfAttention_store_cache)))

        print('OutputEmbed')
        print('inputembed load weight', str(np.mean(InputEmbed_load_weight)))
        print('outputembed comp', str(np.mean(OutputEmbed_comp[1:])))

        print("benchmark - generate")
        timers("generate").reset()
        output_ids = model.generate(
            inputs, max_new_tokens=args.gen_len,
            debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
        costs = timers("generate").costs
        
              
    finally:
        env.close_copy_threads()

    # Log output
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in [0, len(outputs)-1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"
        if args.verbose >= 2:
            print(show_str)

    gpu.print_stats()
    cpu.print_stats()
    projected = bool(args.debug_mode or cut_gen_len)

    if args.log_file == "auto":
        filename = get_filename(args) + ".log"
    else:
        filename = args.log_file

    log_str = write_benchmark_log(filename,
        opt_config.model_bytes(), cache_size, hidden_size,
        gpu_peak_mem, projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput)
    if args.verbose >= 1:
        print(log_str)


def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b",
        help="The model name.")
    parser.add_argument("--path", type=str, default="~/opt_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=10)
    parser.add_argument("--cache-percent", nargs="+", type=int,
        default=[80, 20],
        help="two numbers. They are "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, ")
    parser.add_argument("--per-layer-weight-percent", nargs="+", type=int,
        default=[20, 80, 20, 80, 20, 80, 20, 80],
        help="Eight numbers. They are "
         "the percentage of InputEmbed weight on GPU, "
         "the percentage of InputEmbed weight on CPU, "
         "the percentage of OutputEmbed weight on GPU, "
         "the percentage of OutputEmbed weight on CPU, "
         "the percentage of SelfAttention weight on GPU, "
         "the percentage of SelfAttention weight on CPU, "
         "the percentage of MLP weight on GPU, "
         "the percentage of MLP weight on CPU, ")
    parser.add_argument("--per-layer-computation-percent", nargs="+", type=int,
        default=[80, 20, 80, 20],
        help="Four numbers. They are "
         "the percentage of InputEmbed computation on GPU, "
         "the percentage of OutputEmbed computation on GPU, "
         "the percentage of SelfAttention computation on GPU, "
         "the percentage of MLP computation on GPU, ")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    # parser.add_argument("--cpu-cache-compute", action="store_true")
    # parser.add_argument("--attn-sparsity", type=float, default=1.0)
    # parser.add_argument("--compress-weight", action="store_true",
    #     help="Whether to compress weight.")
    # parser.add_argument("--compress-cache", action="store_true",
    #     help="Whether to compress cache.")


    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()
    print(args)
    # args = argparse.Namespace(model='facebook/opt-66b', path='~/opt_weights', offload_dir='~/flexgen_offload_dir', prompt_len=512, gen_len=32, cut_gen_len=None, debug_mode=None, gpu_batch_size=1, percent=[0, 100, 0, 100], sep_layer=True, pin_weight=True, cpu_cache_compute=False, attn_sparsity=1.0, compress_weight=False, compress_cache=False, log_file='auto', no_log=False, verbose=2, overlap=True)
    assert len(args.per_layer_weight_percent) == 8
    assert len(args.per_layer_computation_percent) == 4
    assert len(args.cache_percent) == 2
    assert args.sep_layer == True
    assert args.cache_percent[0] <= args.per_layer_computation_percent[2]
    for i in range(4):
        assert args.per_layer_weight_percent[2*i] <= args.per_layer_computation_percent[i]
    run_flexgen(args)

