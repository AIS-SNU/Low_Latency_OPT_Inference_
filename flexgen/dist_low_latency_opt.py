import argparse
from itertools import count
import os
import pickle
import traceback
from typing import Union, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from flexgen.compression import CompressionConfig
from flexgen.dist_utils import initialize_distributed
from flexgen.low_latency_opt import (Policy, InputEmbed, OutputEmbed, SelfAttention,
                              MLP, TransformerLayer, OptLM, get_filename,
                              add_parser_arguments, get_test_inputs,
                              DUMMY_WEIGHT, get_choice)
from flexgen.opt_config import get_opt_config, download_opt_weights
from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
    TorchMixedDevice, TorchTensor, DeviceType, general_copy)
from flexgen.timer import timers
from flexgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, array_4d, str2bool, project_decode_latency, torch_dtype_to_np_dtype)


def init_weight_list_dist(weight_specs, policy, env, rank, world_size):
    dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
    dev_choices = [env.disk, env.cpu, env.gpu]
    sizes = [np.prod(spec[0]) for spec in weight_specs]
    sizes_cumsum = np.cumsum(sizes)
    ret = []
    for i in range(len(weight_specs)):
        mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
        home = get_choice(mid_percent * 100, dev_percents, dev_choices)
        shape, dtype, filename, split_dim = weight_specs[i]

        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight
        if split_dim is not None:
            assert shape[split_dim] % world_size == 0
            split = shape[split_dim] // world_size
            shape = list(shape)
            shape[split_dim] = split
            shape = tuple(shape)
            if not compress:
                weight = home.allocate(shape, dtype, pin_memory=pin_memory)
                if DUMMY_WEIGHT not in filename:
                    w_data = np.split(np.load(weight_specs[i][2]), [rank*split, (rank+1)*split], axis=split_dim)[1]
                    weight.load_from_np(w_data)
                else:
                    weight.load_frome_np(np.ones(shape, dtype))
            else:
                weight = home.compressed_device.allocate(
                    shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

                if DUMMY_WEIGHT not in filename:
                    w_data = np.split(np.load(weight_specs[i][2]), [rank*split, (rank+1)*split], axis=split_dim)[1]
                    weight.load_from_np(w_data)
                else:
                    for i in range(2):
                        x = weight.data[i]
                        x.load_from_np(np.ones(x.shape, torch.dtype_to_np_dtype[x.dtype]))
        else:
            if not compress:
                weight = home.allocate(shape, dtype, pin_memory=pin_memory)

                if DUMMY_WEIGHT not in filename:
                    weight.load_from_np_file(weight_specs[i][2])
                else:
                    weight.load_from_np(np.ones(shape, dtype))
                    #weight.load_from_np(np.random.rand(*shape).astype(dtype))
            else:
                weight = home.compressed_device.allocate(
                    shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

                if DUMMY_WEIGHT not in filename:
                    weight.load_from_np_file(weight_specs[i][2])
                else:
                    for i in range(2):
                        x = weight.data[i]
                        x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))
        ret.append(weight)
    return ret

class DistInputEmbed(InputEmbed):
    def __init__(self, config, env, policy, rank, world_size):
        super().__init__(config, env, policy)
        self.rank = rank
        self.world_size = world_size
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
        weights = init_weight_list_dist(weight_specs, self.policy, self.env, self.rank, self.world_size)

        weight_home.store(weights)
        # print(f'inputembed rank {self.rank} first data {weights[0].data[0][0]}')

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i):
        # Compute input embedding
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)
        # Clear the weight_read_buf if it is the last gpu batch
        (w_token, donate[2]), (w_pos, donate[3]) = weight_read_buf.pop()

        h = self.compute.opt_input_embed_dist(h, mask,
            w_token, w_pos, self.config.pad_token_id, donate, self.world_size)
        hidden.val = h
        

class DistOutputEmbed(OutputEmbed):
    def __init__(self, config, env, policy, rank, world_size):
        super().__init__(config, env, policy)
        self.rank = rank
        self.world_size = world_size
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
        weights = init_weight_list_dist(weight_specs, self.policy, self.env, self.rank, self.world_size)
        weight_home.store(weights)
        # print(f'outputembed rank {self.rank} first data {weights[2].data[0]}')
    def load_weight(self, weight_home, weight_read_buf):
        w_ln, b_ln, w_token = weight_home.val
        dst1 = self.weight_load_dst
        dst2 = self.compute
        weight_read_buf.store((w_ln.smart_copy(dst2), b_ln.smart_copy(dst2),
                w_token.smart_copy(dst1)))
    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask, cache_write_buf, i):
        donate = [False] * 4
        h, donate[0] = hidden.val, True

        # Clear the weight_read_buf if it is the last gpu batch
        (w_ln, donate[1]), (b_ln, donate[2]), (w_token, donate[3]) = weight_read_buf.pop()

        h = self.compute.opt_output_embed_dist(h, w_ln, b_ln, w_token, donate,
            self.task.do_sample, self.task.temperature, self.world_size)
        hidden.val = h


class DistSelfAttention(SelfAttention):
    def __init__(self, config, env, policy, layer_id, rank, world_size):
        super().__init__(config, env, policy, layer_id)
        self.rank = rank
        self.world_size = world_size
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
        weights = init_weight_list_dist(weight_specs, self.policy, self.env, self.rank, self.world_size)
        weight_home.store(weights)
        # print(f'selfattention rank {self.rank} first data {weights[0].data[0][0]}')
    def init_cache_one_gpu_batch(self, cache_home):
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device
        
        cache = device.init_cache_one_gpu_batch_dist(self.config, self.task, self.policy, self.world_size)
        cache_home.store(cache)
    
    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i):
        n_head = self.config.n_head

        donate = [False] * 14
        h, donate[0] = hidden.val, True

        # Clear the weight_read_buf if it is the last gpu batch
        ((w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]),
            (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (b_out, donate[9]),
            (w_ln, donate[10]), (b_ln, donate[11])) = weight_read_buf.pop()

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            h, new_k_cache, new_v_cache = self.compute.mha_dist(h, mask, w_q, b_q,
                w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                self.policy.compress_cache, self.policy.comp_cache_config, self.rank, self.world_size)
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:  # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()
            h, new_k_cache, new_v_cache = self.compute.mha_gen_dist(h, mask, w_q,
                b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head,
                k_cache, v_cache, donate, self.policy.attn_sparsity,
                self.policy.compress_cache, self.policy.comp_cache_config, self.rank, self.world_size)
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h
        # print(f'rank {self.rank} hidden {h.data}' )

class DistMLP(MLP):
    def __init__(self, config, env, policy, layer_id, rank, world_size):
        super().__init__(config, env, policy, layer_id)
        self.rank = rank
        self.world_size = world_size

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
        weights = init_weight_list_dist(weight_specs, self.policy, self.env, self.rank, self.world_size)
        weight_home.store(weights)
        # print(f'mlp rank {self.rank} first data {weights[0].data[0][0]}')

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i):
        donate = [False] * 7
        h, donate[0] = hidden.val, True

        # Clear the weight_read_buf if it is the last gpu batch
        ((wi, donate[1]), (bi, donate[2]), (wo, donate[3]), (bo, donate[4]),
            (w_ln, donate[5]), (b_ln, donate[6])) = weight_read_buf.pop()

        h = self.compute.mlp_dist(h, wi, bi, wo, bo, w_ln, b_ln, donate, self.rank, self.world_size)
        hidden.val = h

class DistTransformerLayer(TransformerLayer):
    def __init__(self, config, env, policy, i, rank, world_size):
        self.attention = DistSelfAttention(config, env, policy, i, rank, world_size)
        self.mlp = DistMLP(config, env, policy, i, rank, world_size)
        self.policy = policy
        self.compute = self.attention.compute
        self.rank = rank
        self.world_size = world_size

    def load_weight(self, weight_home, weight_read_buf):
        read_buf1, read_buf2 = ValueHolder(), ValueHolder()
        home1, home2 = weight_home.val
        self.attention.load_weight(home1, read_buf1)
        self.mlp.load_weight(home2, read_buf2)
        weight_read_buf.store((read_buf1, read_buf2))

    def load_cache(self, cache_home, cache_read_buf, i):
        self.attention.load_cache(cache_home, cache_read_buf, i)

    def store_cache(self, cache_home, cache_write_buf, i):
        self.attention.store_cache(cache_home, cache_write_buf, i)

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i):
        read_buf1, read_buf2 = weight_read_buf.pop()
        self.attention.forward(hidden, cache_read_buf, read_buf1, attention_mask,
                               cache_write_buf, i)
        self.mlp.forward(hidden, None, read_buf2, attention_mask, None, i)


#os.environ["NCCL_DEBUG"] = "TRACE"
class DistTP(OptLM):
    def __init__(self, config, env, path, policy, rank,
                 world_size, comm_device, async_comm=False):
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.rank = rank
        self.world_size = world_size
        self.async_comm = async_comm
        if comm_device == "cpu":
            self.comm_device = self.env.cpu
        elif comm_device == "gpu":
            self.comm_device = self.env.gpu
        else:
            raise ValueError(f"Invalid comm_device: {comm_device}")

        layers = []
        layers.append(DistInputEmbed(self.config, self.env, self.policy, self.rank, self.world_size))
        for i in range(self.config.num_hidden_layers):
            if policy.sep_layer:
                layers.append(DistSelfAttention(self.config, self.env, self.policy, i, self.rank, self.world_size))
                layers.append(DistMLP(self.config, self.env, self.policy, i, self.rank, self.world_size))
            else:
                layers.append(DistTransformerLayer(self.config, self.env, self.policy, i, self.world_size))
        layers.append(DistOutputEmbed(self.config, self.env, self.policy, self.rank, self.world_size))
        self.layers = layers
        self.num_layers = len(layers)
        
        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        self.task = None
        self.init_all_weights()
    def init_weight(self, j):
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))
        check_path = os.path.join(expanded_path, "decoder.embed_positions.weight")
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            download_opt_weights(self.config.name, self.path)
        self.layers[j].init_weight(self.weight_home[j], expanded_path)
        
    def load_weight(self, i, j):
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            i += 1
        if i == self.execute_gen_len:
            return

        # Load from weight_home to weight_read_buf
        with torch.cuda.stream(self.load_weight_stream):
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j])

    def init_cache(self, j):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j])

    def load_cache(self, i, j):
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            i += 1
        if i == self.execute_gen_len:
            return

        # Load from cache_home to cache_read_buf
        with torch.cuda.stream(self.load_cache_stream):
            self.layers[j].load_cache(self.cache_home[j], self.cache_read_buf[j], i)

    def store_cache(self, i, j):
        # Handle corner cases
        if j == -1:
            j = self.num_layers - 1
            i -= 1
        if i == -1:
            return

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        with torch.cuda.stream(self.store_cache_stream):
            self.layers[j].store_cache(self.cache_home[j], self.cache_write_buf[j], i)

    def delete_cache(self, j):
        v = self.cache_home[j].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, i, j):
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            i += 1
        if i == self.execute_gen_len:
            return

        # Load to hidden states buffers
        dst = self.layers[j].compute
        if j == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            if i == 0:  # load from the input ids
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[:gpu_batch_size, :self.task.prompt_len])
            else:  # load from the last generated token
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[:gpu_batch_size, pos-1:pos])
            self.hidden.store(val)
        else:
            return
    def store_hidden(self, i, j):
        # Handle corner cases
        if j == -1:
            j = self.num_layers - 1
            i -= 1
        if i == -1:
            return

        if j == self.num_layers - 1:
            # store to output
            gpu_batch_size = self.policy.gpu_batch_size
            ids = self.hidden.pop().data.detach().cpu().numpy()
            pos = self.task.prompt_len + i
            self.output_ids[:gpu_batch_size, pos:pos+1] = ids

    def compute_layer(self, i, j):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        self.layers[j].forward(self.hidden, self.cache_read_buf[j],
            self.weight_read_buf[j], self.attention_mask,
            self.cache_write_buf[j], i)

    def update_attention_mask(self, i):
        if i > 0:
            mask = self.attention_mask
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        gpu_batch_size = self.policy.gpu_batch_size
        input_ids = self.output_ids[:gpu_batch_size, :self.task.prompt_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
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
                 verbose: int = 0):
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
        )
        assert stop is None, "Not implemented."
        world_size = self.world_size
        num_layers = self.num_layers
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

        # Output token ids
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)

        # Intermediate tensors
        # The following buffers store values used
        # for the w-th rank, i-th token, j-th layer
        self.cache_home = array_1d(num_layers, ValueHolder)
        self.cache_read_buf = array_1d(num_layers, ValueHolder)
        self.cache_write_buf = array_1d(num_layers, ValueHolder)
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        self.hidden = ValueHolder()
        self.attention_mask = ValueHolder()

        # Init cache
        self.set_task(task)
        for j in range(num_layers):
            self.init_cache(j)
        if self.policy.cpu_cache_compute:
            raise NotImplementedError()
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)

        dist.barrier()

        # Generate
        if not overlap:
            # No overlap, easy to understand, suitable for debugging
            self.generation_loop_normal()
        else:
            # Overlap I/O and compute
            self.generation_loop_overlap_one_batch()

        # Delete cache
        for j in range(num_layers):
            self.delete_cache(j)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()

        return self.output_ids

    def generation_loop_normal(self):
        self.sending_tag = 0
        self.receiving_tag = 0
        last_sending_job = None
        for i in range(self.execute_gen_len):
            timer_name = "generate-prompt" if i == 0 else "generate"
            timers(timer_name).start()
            self.update_attention_mask(i)

            for j in range(self.num_layers):
                self.load_weight(i, j)
                self.sync()

                self.load_cache(i, j)
                self.sync()
                self.compute_layer(i, j)
                self.sync()
                self.store_cache(i, j)
                self.sync()
            timers(timer_name).stop()
        
        if self.world_size > 1:
            dist.barrier()


    def generation_loop_overlap_one_batch(self):
        # Prologue
        self.load_weight(0, 0)
        self.sync()
        self.sending_tag = 0
        self.receiving_tag = 0

        # Generate
        for i in range(self.execute_gen_len):
            timer_name = "generate-prompt" if i == 0 else "generate"
            timers(timer_name).start()
            self.update_attention_mask(i)
            for j in range(self.num_layers):
                self.load_weight(i, j+1)
                self.load_cache(i, j+1)
                self.load_hidden(i, j)
                self.compute_layer(i, j)
                self.store_cache(i, j-1)
                self.store_hidden(i, j)
                self.sync()
            timers(timer_name).stop()
        if self.world_size > 1:
            dist.barrier()


def comm_test(comm_device):
    # A small all_reduce for warmup.
    a = torch.ones(1).to(comm_device)
    dist.all_reduce(a)
    assert a.item() == args.world_size


def run_flexgen_dist(args):
    t_name = args.model.replace("175b", "66b")
    tokenizer = AutoTokenizer.from_pretrained(t_name, padding_side="left")
    num_inner_iterations = args.num_inner_iterations if args.num_inner_iterations is not None else args.world_size
    num_prompts = args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # Task and policy
    warmup_inputs = get_test_inputs(512, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)

    gpu = TorchDevice(f"cuda:{args.local_rank}")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir, None, args.local_rank)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))
    TorchTensor.name_count = count(start=args.rank, step=args.world_size)

    comm_test(gpu.dev if args.comm_device == "gpu" else cpu.dev)

    policy = Policy(args.gpu_batch_size,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    opt_config = get_opt_config(args.model)
    model = DistTP(opt_config, env, args.path, policy, args.rank,
                      args.world_size, args.comm_device, async_comm=args.async_comm)
    cache_size = opt_config.cache_bytes_dist(num_prompts, prompt_len + gen_len, args.world_size)
    hidden_size = opt_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {opt_config.model_bytes_dist(args.world_size)/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    try:
        print("warmup - generate")
        output_ids = model.generate(
            warmup_inputs, max_new_tokens=2, verbose=args.verbose)

        print("benchmark - generate")
        for timer_name in ["generate-prompt", "generate"]:
            timers(timer_name).reset()
        output_ids = model.generate(
            inputs, max_new_tokens=args.gen_len,
            debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
        prompt_costs = timers("generate-prompt").costs
        generate_costs = timers("generate").costs
    finally:
        env.close_copy_threads()

    if args.rank != args.world_size - 1:
        return

    # Log output
    prefill_latency = sum(prompt_costs)
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        costs = np.array(generate_costs).reshape(-1, cut_gen_len-1).sum(axis=0).tolist()
        decode_latency = project_decode_latency([None] + costs, prompt_len, gen_len)
    else:
        decode_latency = sum(generate_costs)
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
        print(show_str)

    gpu.print_stats()
    cpu.print_stats()
    projected = args.debug_mode or cut_gen_len

    log_str = (f"model size: {opt_config.model_bytes()/GB:.3f} GB\t"
               f"cache size: {cache_size/GB:.3f} GB\t"
               f"hidden size (prefill): {hidden_size/GB:.3f} GB\n"
               f"peak gpu mem: {gpu_peak_mem / GB:.3f} GB\n"
               f"prefill latency: {prefill_latency:.2f} s\t"
               f"prefill throughput: {prefill_throughput:.2f} token/s\n"
               f"decode latency: {decode_latency:.2f} s\t"
               f"decode throughput: {decode_throughput:.2f} token/s\n"
               f"total latency: {total_latency:.2f} s\t"
               f"total throughput: {total_throughput:.2f} token/s")
    print(log_str)

    if not args.no_log:
        if args.log_file == "auto":
            basename = f"rank-{args.rank}-{get_filename(args)}"
            log_filename = basename + ".log"
        else:
            log_filename = args.log_file
        with open(log_filename, "a") as fout:
            fout.write(log_str + "\n")


def add_distributed_parser_arguments(parser):
    parser.add_argument('--head-ip', type=str, default=None, help='the IP address of the head node')
    parser.add_argument('--port', type=int, default=None, help='the port of the head node')
    parser.add_argument('--rank', metavar='I', type=int, default=None)
    parser.add_argument('--local-rank', metavar='I', type=int, default=None)
    parser.add_argument('--world-size', metavar='N', type=int, default=None)
    parser.add_argument('--use-mpi', action='store_true', default=False,
                        help="Get distributed info from MPI")
    parser.add_argument('--comm-device', type=str, default='gpu',
                        choices=['gpu', 'cpu'],
                        help='communication through gpu nvlink or cpu memory '
                             'and socket')
    parser.add_argument('--num-inner-iterations', metavar='I', type=int, default=None)
    parser.add_argument('--async-comm', action='store_true', default=False,
                        help="Use asynchronous communication")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    add_distributed_parser_arguments(parser)
    args = parser.parse_args()

    if args.head_ip is not None and args.port is not None:
        if args.use_mpi:
            args.world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
            args.rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
            args.local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
        initialize_distributed(args.head_ip, args.port, args.world_size,
                               args.rank, args.local_rank, args.comm_device)
    else:
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    assert len(args.percent) == 4

    try:
        run_flexgen_dist(args)
    except Exception as e:
        print(str(e))
        traceback.print_exc()
        raise e
