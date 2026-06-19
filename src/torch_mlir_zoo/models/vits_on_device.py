"""mms-tts (VITS) on-device 재작성 — lowering 친화(결정적·static shape).

VITS는 그대로는 `export_via_iree_turbine`로 안 내려간다. 두 층의 막힘:
  (1) **stochastic duration predictor**의 rational-quadratic-spline 내 data-dependent
      `if`(GuardOnDataDependentSymNode) + duration 의존 **dynamic 출력 길이**.
  (2) WaveNet의 `fused_add_tanh_sigmoid_multiply`가 `num_channels`로 **0-dim 텐서**를
      받아 `in_act[:, :num_channels, :]` **텐서-스칼라 슬라이스** → iree.turbine **세그폴트**.

whisper의 `WhisperForwardOnly` / llama의 `LlamaOnDevice`와 같은 forward-only 재작성으로 해소:
  - duration → `attn`(정렬 행렬)을 **host에서 계산**(`host_align`) → 고정 길이 입력으로 외부화.
  - stochastic noise를 입력 `prior_noise`로 외부화(결정적; 0 = noise 없음).
  - export 시 `patch_vits_for_export`: weight_norm fold + fused activation의
    텐서-슬라이스를 **shape 기반 정적 int**로 교체.

검증(facebook/mms-tts-kor): full export → `server_side_op_hits == {}`, `dynamic == False`
(whisper와 동일 성공 지표). 전처리(uroman 로마자화 + 토크나이저)는 그래프 밖(whisper mel 추출 동급).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def patch_vits_for_export(model: nn.Module) -> nn.Module:
    """export 전 in-place 패치: weight_norm fold + fused-activation 텐서슬라이스 제거.

    - weight_norm parametrization(flow WaveNet·decoder ~64개)을 plain weight로 fold
      (수치 동일, `leave_parametrized=True`). torch.export + parametrize 훅 회피.
    - `transformers...vits.modeling_vits.fused_add_tanh_sigmoid_multiply`를 monkeypatch:
      `num_channels`(0-dim 텐서) 대신 `input_a.shape[1] // 2`(정적 int) 슬라이스.
      이 텐서-스칼라 슬라이스가 iree.turbine 세그폴트의 단일 원인.
    """
    from torch.nn.utils.parametrize import remove_parametrizations
    import transformers.models.vits.modeling_vits as V

    def _fused(input_a, input_b, num_channels):  # noqa: ARG001 (텐서 인자 무시)
        in_act = input_a + input_b
        n = input_a.shape[1] // 2
        return torch.tanh(in_act[:, :n, :]) * torch.sigmoid(in_act[:, n:, :])

    V.fused_add_tanh_sigmoid_multiply = _fused

    for _, mod in model.named_modules():
        p = getattr(mod, "parametrizations", None)
        if p is not None and "weight" in p:
            remove_parametrizations(mod, "weight", leave_parametrized=True)
    return model


@torch.no_grad()
def host_align(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """그래프 밖(host): SDP로 duration 예측 → 정렬 행렬 `attn` + output mask 계산.

    `VitsModel.forward`(transformers) 1379–1407 미러. stochastic noise_scale=0(결정적).
    반환: (attn [B, L_out, L_in], output_padding_mask [B, 1, L_out], L_out:int).
    """
    md = model.text_encoder.embed_tokens.weight.dtype
    ipm = attention_mask.unsqueeze(-1).to(md)
    enc = model.text_encoder(
        input_ids=input_ids, padding_mask=ipm, attention_mask=attention_mask, return_dict=True
    )
    hidden = enc.last_hidden_state.transpose(1, 2)
    ipm_t = ipm.transpose(1, 2)
    log_dur = model.duration_predictor(hidden, ipm_t, None, reverse=True, noise_scale=0.0)
    duration = torch.ceil(torch.exp(log_dur) * ipm_t * (1.0 / model.speaking_rate))
    pred_len = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()
    out_len = int(pred_len.max())

    idx = torch.arange(out_len, dtype=duration.dtype)
    out_pad = (idx.unsqueeze(0) < pred_len.unsqueeze(1)).unsqueeze(1).to(ipm.dtype)
    attn_mask = torch.unsqueeze(ipm_t, 2) * torch.unsqueeze(out_pad, -1)
    cum = torch.cumsum(duration, -1).view(input_ids.shape[0] * input_ids.shape[1], 1)
    valid = (idx.unsqueeze(0) < cum).to(attn_mask.dtype).view(input_ids.shape[0], input_ids.shape[1], out_len)
    padded = valid - nn.functional.pad(valid, [0, 0, 1, 0, 0, 0])[:, :-1]
    attn = (padded.unsqueeze(1).transpose(2, 3) * attn_mask).squeeze(1)
    return attn, out_pad, out_len


class VitsOnDevice(nn.Module):
    """VITS 추론의 결정적·static-shape forward (lowering 대상).

    그래프 = text_encoder → prior 확장(host `attn`) → flow → decoder → 고정길이 waveform.
    SDP/spline/dynamic length는 그래프 밖(`host_align`), noise는 입력(`prior_noise`).
    """

    def __init__(self, hf_vits: nn.Module):
        super().__init__()
        self.text_encoder = hf_vits.text_encoder
        self.flow = hf_vits.flow
        self.decoder = hf_vits.decoder
        self.noise_scale = hf_vits.noise_scale

    def forward(self, input_ids, attn, output_padding_mask, prior_noise):
        ipm = torch.ones_like(input_ids).unsqueeze(-1).to(self.text_encoder.embed_tokens.weight.dtype)
        enc = self.text_encoder(
            input_ids=input_ids, padding_mask=ipm,
            attention_mask=torch.ones_like(input_ids), return_dict=True,
        )
        prior_means = torch.matmul(attn, enc.prior_means).transpose(1, 2)
        prior_log_var = torch.matmul(attn, enc.prior_log_variances).transpose(1, 2)
        prior_latents = prior_means + prior_noise * torch.exp(prior_log_var) * self.noise_scale
        latents = self.flow(prior_latents, output_padding_mask, None, reverse=True)
        waveform = self.decoder(latents * output_padding_mask, None)
        return waveform.squeeze(1)


def build_vits_on_device(
    model_id: str = "facebook/mms-tts-kor",
    text: str = "안녕하세요. 음성 합성 테스트입니다.",
):
    """(VitsOnDevice, example_args, raw_VitsModel) — whisper `build()`와 동형.

    `example_args = (input_ids, attn, output_padding_mask, prior_noise)` — 전부 고정 shape.
    prior_noise=0(결정적). attn/길이는 `host_align`이 결정.
    """
    from transformers import VitsModel, AutoTokenizer

    raw = VitsModel.from_pretrained(model_id, torch_dtype=torch.float32).eval()
    patch_vits_for_export(raw)
    tok = AutoTokenizer.from_pretrained(model_id)

    enc = tok(text, return_tensors="pt")
    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
    attn, out_pad, out_len = host_align(raw, input_ids, attention_mask)

    # prior_noise shape = 확장된 prior_means([B, flow_size, L_out])
    prior_noise = torch.zeros(input_ids.shape[0], raw.config.flow_size, out_len)
    module = VitsOnDevice(raw).eval()
    return module, (input_ids, attn, out_pad, prior_noise), raw
