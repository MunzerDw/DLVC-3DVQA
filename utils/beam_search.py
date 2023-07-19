from contextlib import contextmanager
from typing import Sequence, Union

import torch
import torch.nn as nn

TensorOrSequence = Union[Sequence[torch.Tensor], torch.Tensor]
TensorOrNone = Union[torch.Tensor, None]

def get_batch_size(x: TensorOrSequence) -> int:
    if isinstance(x, torch.Tensor):
        b_s = x.size(0)
    else:
        b_s = x[0].size(0)
    return b_s


def get_device(x: TensorOrSequence) -> int:
    if isinstance(x, torch.Tensor):
        b_s = x.device
    else:
        b_s = x[0].device
    return b_s

# https://github.com/aimagelab/meshed-memory-transformer
class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_stateful = False
        self._state_names = []
        self._state_defaults = dict()

    def register_state(self, name: str, default: TensorOrNone):
        self._state_names.append(name)
        if default is None:
            self._state_defaults[name] = None
        else:
            self._state_defaults[name] = default.clone().detach()
        self.register_buffer(name, default)

    def states(self):
        for name in self._state_names:
            yield self._buffers[name]
        for m in self.children():
            if isinstance(m, Module):
                yield from m.states()

    def apply_to_states(self, fn):
        for name in self._state_names:
            self._buffers[name] = fn(self._buffers[name])
        for m in self.children():
            if isinstance(m, Module):
                m.apply_to_states(fn)

    def _init_states(self, batch_size: int):
        for name in self._state_names:
            if self._state_defaults[name] is None:
                self._buffers[name] = None
            else:
                self._buffers[name] = (
                    self._state_defaults[name]
                    .clone()
                    .detach()
                    .to(self._buffers[name].device)
                )
                self._buffers[name] = self._buffers[name].unsqueeze(0)
                self._buffers[name] = self._buffers[name].expand(
                    [
                        batch_size,
                    ]
                    + list(self._buffers[name].shape[1:])
                )
                self._buffers[name] = self._buffers[name].contiguous()

    def _reset_states(self):
        for name in self._state_names:
            if self._state_defaults[name] is None:
                self._buffers[name] = None
            else:
                self._buffers[name] = (
                    self._state_defaults[name]
                    .clone()
                    .detach()
                    .to(self._buffers[name].device)
                )

    def enable_statefulness(self, batch_size: int):
        for m in self.children():
            if isinstance(m, Module):
                m.enable_statefulness(batch_size)
        self._init_states(batch_size)
        self._is_stateful = True

    def disable_statefulness(self):
        for m in self.children():
            if isinstance(m, Module):
                m.disable_statefulness()
        self._reset_states()
        self._is_stateful = False

    @contextmanager
    def statefulness(self, batch_size: int):
        self.enable_statefulness(batch_size)
        try:
            yield
        finally:
            self.disable_statefulness()

# https://github.com/aimagelab/meshed-memory-transformer
class BeamSearch(object):
    def __init__(self, model, max_len: int, eos_idx: int, beam_size: int):
        self.model = model
        self.max_len = max_len
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.b_s = None
        self.device = None
        self.seq_mask = None
        self.seq_logprob = None
        self.outputs = None
        self.log_probs = None
        self.selected_words = None
        self.all_log_probs = None

    def _expand_state(self, selected_beam, cur_beam_size):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(
                s.view(*([self.b_s, cur_beam_size] + shape[1:])),
                1,
                beam.expand(*([self.b_s, self.beam_size] + shape[1:])),
            )
            s = s.view(
                *(
                    [
                        -1,
                    ]
                    + shape[1:]
                )
            )
            return s

        return fn

    def _expand_visual(
        self,
        visual: TensorOrSequence,
        cur_beam_size: int,
        selected_beam: torch.Tensor,
    ):
        if isinstance(visual, torch.Tensor):
            visual_shape = visual.shape
            visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
            visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
            selected_beam_red_size = (self.b_s, self.beam_size) + tuple(
                1 for _ in range(len(visual_exp_shape) - 2)
            )
            selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[2:]
            visual_exp = visual.view(visual_exp_shape)
            selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(
                selected_beam_exp_size
            )
            visual = torch.gather(visual_exp, 1, selected_beam_exp).view(
                visual_red_shape
            )
        else:
            new_visual = []
            for im in visual:
                visual_shape = im.shape
                visual_exp_shape = (self.b_s, cur_beam_size) + visual_shape[1:]
                visual_red_shape = (self.b_s * self.beam_size,) + visual_shape[1:]
                selected_beam_red_size = (self.b_s, self.beam_size) + tuple(
                    1 for _ in range(len(visual_exp_shape) - 2)
                )
                selected_beam_exp_size = (self.b_s, self.beam_size) + visual_exp_shape[
                    2:
                ]
                visual_exp = im.view(visual_exp_shape)
                selected_beam_exp = selected_beam.view(selected_beam_red_size).expand(
                    selected_beam_exp_size
                )
                new_im = torch.gather(visual_exp, 1, selected_beam_exp).view(
                    visual_red_shape
                )
                new_visual.append(new_im)
            visual = tuple(new_visual)
        return visual

    def apply(
        self, visual: TensorOrSequence, out_size=1, return_probs=False, **kwargs
    ):
        self.b_s = get_batch_size(visual)
        self.device = get_device(visual)
        self.seq_mask = torch.ones((self.b_s, self.beam_size, 1), device=self.device)
        self.seq_logprob = torch.zeros((self.b_s, 1, 1), device=self.device)
        self.log_probs = []
        self.selected_words = None
        if return_probs:
            self.all_log_probs = []

        outputs = []
        with self.model.statefulness(self.b_s):
            for t in range(self.max_len):
                visual, outputs = self.iter(t, visual, outputs, return_probs, **kwargs)

        # Sort result
        seq_logprob, sort_idxs = torch.sort(self.seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(
            outputs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len)
        )
        log_probs = torch.cat(self.log_probs, -1)
        log_probs = torch.gather(
            log_probs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len)
        )
        if return_probs:
            all_log_probs = torch.cat(self.all_log_probs, 2)
            all_log_probs = torch.gather(
                all_log_probs,
                1,
                sort_idxs.unsqueeze(-1).expand(
                    self.b_s, self.beam_size, self.max_len, all_log_probs.shape[-1]
                ),
            )

        outputs = outputs.contiguous()[:, :out_size]
        log_probs = log_probs.contiguous()[:, :out_size]
        if out_size == 1:
            outputs = outputs.squeeze(1)
            log_probs = log_probs.squeeze(1)

        if return_probs:
            return outputs, log_probs, all_log_probs
        else:
            return outputs, log_probs

    def select(self, t, candidate_logprob, **kwargs):
        selected_logprob, selected_idx = torch.sort(
            candidate_logprob.view(self.b_s, -1), -1, descending=True
        )
        selected_logprob, selected_idx = (
            selected_logprob[:, : self.beam_size],
            selected_idx[:, : self.beam_size],
        )
        return selected_idx, selected_logprob

    def iter(
        self, t: int, visual: TensorOrSequence, outputs, return_probs, **kwargs
    ):
        cur_beam_size = 1 if t == 0 else self.beam_size

        word_logprob = self.model.step(t, self.selected_words, visual, **kwargs)
        word_logprob = word_logprob.view(self.b_s, cur_beam_size, -1)
        candidate_logprob = self.seq_logprob + word_logprob

        # Mask sequence if it reaches EOS
        if t > 0:
            mask = (
                (self.selected_words.view(self.b_s, cur_beam_size) != self.eos_idx)
                .float()
                .unsqueeze(-1)
            )
            self.seq_mask = self.seq_mask * mask
            word_logprob = word_logprob * self.seq_mask.expand_as(word_logprob)
            old_seq_logprob = self.seq_logprob.expand_as(candidate_logprob).contiguous()
            old_seq_logprob[:, :, 1:] = -999
            candidate_logprob = self.seq_mask * candidate_logprob + old_seq_logprob * (
                1 - self.seq_mask
            )

        selected_idx, selected_logprob = self.select(t, candidate_logprob, **kwargs)
        selected_beam = selected_idx // candidate_logprob.shape[-1]
        selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

        self.model.apply_to_states(self._expand_state(selected_beam, cur_beam_size))
        # visual = self._expand_visual(visual, cur_beam_size, selected_beam)

        self.seq_logprob = selected_logprob.unsqueeze(-1)
        self.seq_mask = torch.gather(self.seq_mask, 1, selected_beam.unsqueeze(-1))
        outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
        outputs.append(selected_words.unsqueeze(-1))

        if return_probs:
            if t == 0:
                self.all_log_probs.append(
                    word_logprob.expand((self.b_s, self.beam_size, -1)).unsqueeze(2)
                )
            else:
                self.all_log_probs.append(word_logprob.unsqueeze(2))

        this_word_logprob = torch.gather(
            word_logprob,
            1,
            selected_beam.unsqueeze(-1).expand(
                self.b_s, self.beam_size, word_logprob.shape[-1]
            ),
        )
        this_word_logprob = torch.gather(
            this_word_logprob, 2, selected_words.unsqueeze(-1)
        )
        self.log_probs = list(
            torch.gather(
                o, 1, selected_beam.unsqueeze(-1).expand(self.b_s, self.beam_size, 1)
            )
            for o in self.log_probs
        )
        self.log_probs.append(this_word_logprob)
        self.selected_words = selected_words.view(-1, 1)

        return visual, outputs
