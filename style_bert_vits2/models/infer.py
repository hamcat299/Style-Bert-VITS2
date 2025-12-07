from collections.abc import Iterator
from typing import Any, Optional, Union, cast

import numpy as np
import torch
from numpy.typing import NDArray

from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons, utils
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.models import SynthesizerTrn
from style_bert_vits2.models.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)
from style_bert_vits2.nlp import (
    clean_text_with_given_phone_tone,
    cleaned_text_to_sequence,
    extract_bert_feature,
)
from style_bert_vits2.nlp.symbols import SYMBOLS


def get_net_g(
    model_path: str,
    version: str,
    device: str,
    hps: HyperParameters,
    model_dtype: Optional[torch.dtype] = None,
) -> Union[SynthesizerTrn, SynthesizerTrnJPExtra]:
    if version.endswith("JP-Extra"):
        logger.info("Using JP-Extra model")
        # use meta device to speed up instantiation
        with torch.device("meta"):
            net_g = SynthesizerTrnJPExtra(
                n_vocab=len(SYMBOLS),
                spec_channels=hps.data.filter_length // 2 + 1,
                segment_size=hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                # hps.model 以下のすべての値を引数に渡す
                use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
                use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
                use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
                use_duration_discriminator=hps.model.use_duration_discriminator,
                use_wavlm_discriminator=hps.model.use_wavlm_discriminator,
                inter_channels=hps.model.inter_channels,
                hidden_channels=hps.model.hidden_channels,
                filter_channels=hps.model.filter_channels,
                n_heads=hps.model.n_heads,
                n_layers=hps.model.n_layers,
                kernel_size=hps.model.kernel_size,
                p_dropout=hps.model.p_dropout,
                resblock=hps.model.resblock,
                resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
                resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
                upsample_rates=hps.model.upsample_rates,
                upsample_initial_channel=hps.model.upsample_initial_channel,
                upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
                n_layers_q=hps.model.n_layers_q,
                use_spectral_norm=hps.model.use_spectral_norm,
                gin_channels=hps.model.gin_channels,
                slm=hps.model.slm,
            )
    else:
        logger.info("Using normal model")
        # use meta device to speed up instantiation
        with torch.device("meta"):
            net_g = SynthesizerTrn(
                n_vocab=len(SYMBOLS),
                spec_channels=hps.data.filter_length // 2 + 1,
                segment_size=hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                # hps.model 以下のすべての値を引数に渡す
                use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
                use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
                use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
                use_duration_discriminator=hps.model.use_duration_discriminator,
                use_wavlm_discriminator=hps.model.use_wavlm_discriminator,
                inter_channels=hps.model.inter_channels,
                hidden_channels=hps.model.hidden_channels,
                filter_channels=hps.model.filter_channels,
                n_heads=hps.model.n_heads,
                n_layers=hps.model.n_layers,
                kernel_size=hps.model.kernel_size,
                p_dropout=hps.model.p_dropout,
                resblock=hps.model.resblock,
                resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
                resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
                upsample_rates=hps.model.upsample_rates,
                upsample_initial_channel=hps.model.upsample_initial_channel,
                upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
                n_layers_q=hps.model.n_layers_q,
                use_spectral_norm=hps.model.use_spectral_norm,
                gin_channels=hps.model.gin_channels,
                slm=hps.model.slm,
            )

    # net_g.state_dict() # これは不要そう
    # _ = net_g.eval() # meta device なので eval() するとエラーになるかも？ロード後にやるべき

    if model_path.endswith(".pth") or model_path.endswith(".pt"):
        # .pth の場合は assign=True は使えないので、一度 CPU に実体化してからロードして GPU に送る（従来通り）
        # ただし meta からの実体化は to_empty を使う
        net_g = net_g.to_empty(device=device)
        _ = utils.checkpoints.load_checkpoint(
            model_path, net_g, None, skip_optimizer=True, device=device
        )
    elif model_path.endswith(".safetensors"):
        # safetensors の場合は assign=True を使って直接デバイスにロード
        _ = utils.safetensors.load_safetensors(
            model_path, net_g, True, device=device, assign=True
        )
    else:
        raise ValueError(f"Unknown model format: {model_path}")

    _ = net_g.eval()

    # Dtype conversion
    if model_dtype is not None:
        net_g.to(dtype=model_dtype)
        logger.info(f"Entire model converted to {model_dtype}")

    # Generator (Decoder) の推論最適化: weight_norm を取り除く
    # 学習完了後の推論時には weight_norm は不要なオーバーヘッドとなるため除去しておく
    # 精度に影響はなく、単に計算効率が向上する
    net_g.dec.remove_weight_norm()
    logger.info(
        "Generator module weight normalization removed for inference optimization"
    )

    return net_g


def get_text(
    text: str,
    language_str: Languages,
    hps: HyperParameters,
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    use_jp_extra = hps.version.endswith("JP-Extra")
    norm_text, phone, tone, word2ph, sep_text, _, _ = clean_text_with_given_phone_tone(
        text,
        language_str,
        given_phone=given_phone,
        given_tone=given_tone,
        use_jp_extra=use_jp_extra,
        # 推論時のみ呼び出されるので、raise_yomi_error は False に設定
        raise_yomi_error=False,
    )
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert_ori = extract_bert_feature(
        norm_text,
        word2ph,
        language_str,
        device,
        assist_text,
        assist_text_weight,
        dtype=dtype,
        sep_text=sep_text,  # clean_text_with_given_phone_tone() の中間生成物を再利用して効率向上を図る
    )
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone

    if language_str == Languages.ZH:
        bert = bert_ori
        ja_bert = torch.zeros(1024, len(phone))
        en_bert = torch.zeros(1024, len(phone))
    elif language_str == Languages.JP:
        bert = torch.zeros(1024, len(phone))
        ja_bert = bert_ori
        en_bert = torch.zeros(1024, len(phone))
    elif language_str == Languages.EN:
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(1024, len(phone))
        en_bert = bert_ori
    else:
        raise ValueError("language_str should be ZH, JP or EN")

    assert bert.shape[-1] == len(phone), (
        f"Bert seq len {bert.shape[-1]} != {len(phone)}"
    )

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, en_bert, phone, tone, language


def infer(
    text: str,
    style_vec: NDArray[Any],
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
    sid: int,  # In the original Bert-VITS2, its speaker_name: str, but here it's id
    language: Languages,
    hps: HyperParameters,
    net_g: Union[SynthesizerTrn, SynthesizerTrnJPExtra],
    device: str,
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
    bert_dtype: Optional[torch.dtype] = None,
) -> NDArray[Any]:
    is_jp_extra = hps.version.endswith("JP-Extra")
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
        text,
        language,
        hps,
        device,
        assist_text=assist_text,
        assist_text_weight=assist_text_weight,
        given_phone=given_phone,
        given_tone=given_tone,
        dtype=bert_dtype,
    )
    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        bert = bert[:, 3:]
        ja_bert = ja_bert[:, 3:]
        en_bert = en_bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        bert = bert[:, :-2]
        ja_bert = ja_bert[:, :-2]
        en_bert = en_bert[:, :-2]

    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        en_bert = en_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        style_vec_tensor = torch.from_numpy(style_vec).to(device).unsqueeze(0)
        del phones
        sid_tensor = torch.LongTensor([sid]).to(device)

        if is_jp_extra:
            output = cast(SynthesizerTrnJPExtra, net_g).infer(
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
                ja_bert,
                style_vec=style_vec_tensor,
                length_scale=length_scale,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
            )
        else:
            output = cast(SynthesizerTrn, net_g).infer(
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
                bert,
                ja_bert,
                en_bert,
                style_vec=style_vec_tensor,
                length_scale=length_scale,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
            )

        audio = output[0][0, 0].data.cpu().float().numpy()

        del (
            x_tst,
            tones,
            lang_ids,
            bert,
            x_tst_lengths,
            sid_tensor,
            ja_bert,
            en_bert,
            style_vec,
        )  # , emo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio


def infer_stream(
    text: str,
    style_vec: NDArray[Any],
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
    sid: int,
    language: Languages,
    hps: HyperParameters,
    net_g: Union[SynthesizerTrn, SynthesizerTrnJPExtra],
    device: str,
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
    bert_dtype: Optional[torch.dtype] = None,
    chunk_size: int = 100,
    overlap_size: int = 16,
) -> Iterator[NDArray[np.float32]]:
    """
    ストリーミング推論を実行する関数。
    Generator 部分のみストリーミング処理を行い、音声チャンクを逐次 yield する。

    Args:
        text: 読み上げるテキスト
        style_vec: スタイルベクトル
        sdp_ratio: SDP と DP の混合比
        noise_scale: ノイズスケール
        noise_scale_w: ノイズスケール W
        length_scale: 長さスケール
        sid: 話者 ID
        language: 言語
        hps: ハイパーパラメータ
        net_g: 音声合成モデル
        device: デバイス
        skip_start: 先頭をスキップするか
        skip_end: 末尾をスキップするか
        assist_text: 補助テキスト
        assist_text_weight: 補助テキストの重み
        given_phone: 指定された音素列
        given_tone: 指定されたトーン列
        bert_dtype: BERT の dtype
        chunk_size: チャンクサイズ (フレーム数)
        overlap_size: オーバーラップサイズ (フレーム数)

    Yields:
        NDArray[np.float32]: 音声チャンク

    Reference: https://qiita.com/__dAi00/items/970f0fe66286510537dd
    """
    assert chunk_size > overlap_size, (
        f"chunk_size ({chunk_size}) must be larger than overlap_size ({overlap_size})"
    )
    assert chunk_size > 0 and overlap_size > 0, (
        "chunk_size and overlap_size must be positive"
    )
    assert overlap_size % 2 == 0, (
        "overlap_size must be even for proper margin calculation"
    )

    is_jp_extra = hps.version.endswith("JP-Extra")
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
        text,
        language,
        hps,
        device,
        assist_text=assist_text,
        assist_text_weight=assist_text_weight,
        given_phone=given_phone,
        given_tone=given_tone,
        dtype=bert_dtype,
    )
    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        bert = bert[:, 3:]
        ja_bert = ja_bert[:, 3:]
        en_bert = en_bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        bert = bert[:, :-2]
        ja_bert = ja_bert[:, :-2]
        en_bert = en_bert[:, :-2]

    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        en_bert = en_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        style_vec_tensor = torch.from_numpy(style_vec).to(device).unsqueeze(0)
        del phones
        sid_tensor = torch.LongTensor([sid]).to(device)

        # Generator への入力特徴量を生成
        if is_jp_extra:
            z, y_mask, g, attn, z_p, m_p, logs_p = cast(
                SynthesizerTrnJPExtra, net_g
            ).infer_input_feature(
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
                ja_bert,
                style_vec=style_vec_tensor,
                length_scale=length_scale,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
            )
        else:
            z, y_mask, g, attn, z_p, m_p, logs_p = cast(
                SynthesizerTrn, net_g
            ).infer_input_feature(
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
                bert,
                ja_bert,
                en_bert,
                style_vec=style_vec_tensor,
                length_scale=length_scale,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
            )

        # Generator 部分のストリーミング処理
        z_input = z * y_mask
        total_length = z_input.shape[2]

        # 全体のアップサンプリング率を計算
        total_upsample_factor = int(np.prod(hps.model.upsample_rates))
        margin_frames = overlap_size // 2

        # Decoder の dtype を取得し、g を事前に変換 (ループ外で1回だけ)
        dec_dtype = next(net_g.dec.parameters()).dtype
        g_dec = g.to(dec_dtype)
        z_input_dec = z_input.to(dec_dtype)

        for start_idx in range(0, total_length, chunk_size - overlap_size):
            end_idx = min(start_idx + chunk_size, total_length)
            chunk = z_input_dec[:, :, start_idx:end_idx]

            # Decoder を実行 (dtype変換は不要、事前に変換済み)
            chunk_output = net_g.dec(chunk, g=g_dec)

            # オーバーラップ処理
            current_output_length = chunk_output.shape[2]

            trim_left = 0
            if start_idx != 0:
                trim_left = margin_frames * total_upsample_factor

            trim_right = 0
            if end_idx != total_length:
                trim_right = margin_frames * total_upsample_factor

            start_slice = trim_left
            end_slice = current_output_length - trim_right

            if start_slice < end_slice:
                valid_audio = chunk_output[:, :, start_slice:end_slice]
                audio_chunk = valid_audio[0, 0].data.cpu().float().numpy()
                if audio_chunk.size > 0:
                    yield audio_chunk

        del (
            x_tst,
            tones,
            lang_ids,
            bert,
            x_tst_lengths,
            sid_tensor,
            ja_bert,
            en_bert,
            style_vec_tensor,
            z,
            y_mask,
            g,
            g_dec,
            z_input,
            z_input_dec,
            attn,
            z_p,
            m_p,
            logs_p,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
