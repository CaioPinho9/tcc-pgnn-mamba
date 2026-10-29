import inspect
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config.config import OPENFWI_CONFIG
from helpers.runtime import prepare_torch_extension_cache


_DEEPWAVE_MODULE = None


def _send_deepwave_wavefield(state, callback, transpose_model=False):
    wavefield = state.get_wavefield("wavefield_0", view="inner")
    if transpose_model:
        wavefield = wavefield.transpose(-2, -1)
    callback(wavefield, float(state.step) * float(state.dt))


def ricker_openfwi_torch(
    frequency_hz,
    time_spacing_s,
    time_step_count,
    device,
    dtype,
    peak_time_s=None,
):
    if peak_time_s is not None:
        time = torch.arange(time_step_count, device=device, dtype=dtype) * time_spacing_s
        alpha = torch.pi * frequency_hz * (time - float(peak_time_s))
        beta = alpha ** 2
        return (1.0 - 2.0 * beta) * torch.exp(-beta)

    wavelet_length = 2.2 / frequency_hz / time_spacing_s
    wavelet_length = int(2 * math.floor(wavelet_length / 2) + 1)
    center_index = int(math.floor(wavelet_length / 2))

    sample_indices = torch.arange(1, wavelet_length + 1, device=device, dtype=dtype)

    alpha = ((center_index - sample_indices + 1) * frequency_hz * time_spacing_s * torch.pi)
    beta = alpha ** 2
    wavelet = (1.0 - 2.0 * beta) * torch.exp(-beta)

    if time_step_count <= wavelet.numel():
        return wavelet[:time_step_count]

    padded = torch.zeros(time_step_count, device=device, dtype=dtype)
    padded[: wavelet.numel()] = wavelet

    return padded

def laplacian_2_4(pressure):
    near = (
        torch.roll(pressure, 1, dims=3) + torch.roll(pressure, -1, dims=3) +
        torch.roll(pressure, 1, dims=2) + torch.roll(pressure, -1, dims=2)
    )
    far = (
        torch.roll(pressure, 2, dims=3) + torch.roll(pressure, -2, dims=3) +
        torch.roll(pressure, 2, dims=2) + torch.roll(pressure, -2, dims=2)
    )
    return (4.0 / 3.0) * near + (-1.0 / 12.0) * far

class OpenFWITorchForward2D(nn.Module):
    def __init__(
        self,
        source_indices,
        receiver_count,
        time_step_count,
        receiver_indices=None,
        receiver_z_index=1,
        checkpoint_time_steps=None,
        config=OPENFWI_CONFIG,
    ):
        super().__init__()
        self.config = config
        self.receiver_count = int(receiver_count)
        self.receiver_z_index = int(receiver_z_index)
        self.time_step_count = int(time_step_count)
        self.checkpoint_time_steps = checkpoint_time_steps
        self.register_buffer("source_indices", torch.as_tensor(source_indices, dtype=torch.long))
        if self.source_indices.ndim != 2 or self.source_indices.shape[1] != 2:
            raise ValueError("source_indices deve ter a forma [source, 2] na ordem [x, z]")

        if receiver_indices is None:
            receiver_tensor = torch.empty((0, 2), dtype=torch.long)
        else:
            receiver_tensor = torch.as_tensor(receiver_indices, dtype=torch.long)
            if receiver_tensor.ndim != 2 or receiver_tensor.shape[1] != 2:
                raise ValueError(
                    "receiver_indices deve ter a forma [receiver, 2] na ordem [x, z]"
                )
            if receiver_tensor.shape[0] != self.receiver_count:
                raise ValueError(
                    f"receiver_count={self.receiver_count} não corresponde a "
                    f"{receiver_tensor.shape[0]} posições explícitas de receptores"
                )
        self.register_buffer("receiver_indices", receiver_tensor)

    def _make_damping(self, padded_velocity):
        nbc = self.config["absorbing_boundary_cells"]
        dx = self.config["grid_spacing_m"]

        _, padded_depth, padded_width = padded_velocity.shape
        physical_depth = padded_depth - 2 * nbc
        physical_width = padded_width - 2 * nbc
        boundary_width_m = (nbc - 1) * dx

        minimum_velocity = padded_velocity.amin(dim=(1, 2), keepdim=True).detach()
        max_damping = 3.0 * minimum_velocity * math.log(10_000_000.0) / (2.0 * boundary_width_m)

        boundary_index = torch.arange(nbc, device=padded_velocity.device, dtype=padded_velocity.dtype)
        damping_1d = max_damping * (boundary_index[None, None, :] * dx / boundary_width_m) ** 2

        damping = torch.zeros_like(padded_velocity)
        damping[:, :, :nbc] = torch.flip(damping_1d, dims=[2])
        damping[:, :, physical_width + nbc : physical_width + 2 * nbc] = damping_1d
        damping[:, :nbc, nbc : nbc + physical_width] = torch.flip(damping_1d, dims=[2]).transpose(1, 2)
        damping[:, physical_depth + nbc : physical_depth + 2 * nbc, nbc : nbc + physical_width] = damping_1d.transpose(1, 2)

        return damping

    def forward(
        self,
        velocity,
        source_selection=None,
        wavefield_callback=None,
        callback_frequency=1,
    ):
        # velocidade: [lote, 1, profundidade, largura]
        if wavefield_callback is not None:
            if torch.is_grad_enabled():
                raise ValueError("A captura de wavefield requer torch.no_grad()")
            if velocity.shape[0] != 1:
                raise ValueError("A captura de wavefield aceita uma amostra por chamada")
            if int(callback_frequency) < 1:
                raise ValueError("callback_frequency deve ser maior que zero")
        sources = self.source_indices
        if source_selection is not None:
            selection = torch.as_tensor(source_selection, dtype=torch.long, device=sources.device)
            if selection.ndim != 1 or selection.numel() == 0:
                raise ValueError("source_selection deve ser uma sequência 1D não vazia de índices de tiros")
            sources = sources.index_select(0, selection)
        velocity = velocity[:, 0].float()

        device = velocity.device
        dtype = velocity.dtype
        dx = self.config["grid_spacing_m"]
        recording_dt = float(self.config["time_spacing_s"])
        dt = float(self.config.get("simulation_time_spacing_s", recording_dt))
        if dt <= 0.0 or recording_dt <= 0.0 or dt > recording_dt:
            raise ValueError(
                "simulation_time_spacing_s deve ser positivo e não maior que time_spacing_s"
            )
        recording_stride = int(round(recording_dt / dt))
        if not math.isclose(recording_stride * dt, recording_dt, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                "time_spacing_s deve ser um múltiplo inteiro de simulation_time_spacing_s"
            )
        simulation_time_step_count = (self.time_step_count - 1) * recording_stride + 1
        nbc = self.config["absorbing_boundary_cells"]

        physical_depth, physical_width = velocity.shape[-2:]
        if torch.any(sources[:, 0] < 0) or torch.any(
            sources[:, 0] >= physical_width
        ) or torch.any(sources[:, 1] < 0) or torch.any(
            sources[:, 1] >= physical_depth
        ):
            raise ValueError("A posição de uma fonte está fora do modelo de velocidade")

        padded_velocity = F.pad(velocity[:, None], (nbc, nbc, nbc, nbc), mode="replicate")[:, 0]
        damping = self._make_damping(padded_velocity)

        batch_size, padded_depth, padded_width = padded_velocity.shape
        source_count = sources.shape[0]

        alpha = ((padded_velocity * dt / dx) ** 2)[:, None]
        kappa = (damping * dt)[:, None]
        source_scale = ((padded_velocity * dt) ** 2)[:, None]
        current_weight = 2.0 - 5.0 * alpha - kappa
        previous_weight = 1.0 - kappa

        source_wavelet = ricker_openfwi_torch(
            frequency_hz=float(self.config["frequency_hz"]),
            time_spacing_s=dt,
            time_step_count=simulation_time_step_count,
            device=device,
            dtype=dtype,
            peak_time_s=self.config.get("source_peak_time_s"),
        )

        # Máscara das fontes: [1, quantidade_de_fontes, profundidade_preenchida,
        # largura_preenchida]
        source_mask = torch.zeros(1, source_count, padded_depth, padded_width, device=device, dtype=dtype)
        source_mask[0, torch.arange(source_count, device=device), sources[:, 1] + nbc, sources[:, 0] + nbc] = 1.0

        # Vetores de índices dos receptores. A geometria explícita é usada pelo
        # Marmousi; a linha horizontal continua sendo o padrão compatível com
        # versões anteriores do OpenFWI.
        if self.receiver_indices.numel():
            if torch.any(self.receiver_indices[:, 0] < 0) or torch.any(
                self.receiver_indices[:, 0] >= physical_width
            ) or torch.any(self.receiver_indices[:, 1] < 0) or torch.any(
                self.receiver_indices[:, 1] >= physical_depth
            ):
                raise ValueError("A posição de um receptor está fora do modelo de velocidade")
            rec_x = self.receiver_indices[:, 0] + nbc
            rec_z = self.receiver_indices[:, 1] + nbc
        else:
            rec_z = torch.full(
                (self.receiver_count,),
                self.receiver_z_index + nbc,
                device=device,
                dtype=torch.long,
            )
            rec_x = torch.arange(self.receiver_count, device=device, dtype=torch.long) + nbc

        pressure_previous = torch.zeros(batch_size, source_count, padded_depth, padded_width, device=device, dtype=dtype)
        pressure_current = torch.zeros_like(pressure_previous)

        def propagate_chunk(previous, current, current_w, previous_w, alpha_value, source_value, start_t, end_t):
            recorded_chunk = []
            for t in range(start_t, end_t):
                pressure_next = (
                    current_w * current
                    - previous_w * previous
                    + alpha_value * laplacian_2_4(current)
                    + source_value * source_mask * source_wavelet[t]
                )
                if wavefield_callback is not None and t % int(callback_frequency) == 0:
                    wavefield_callback(pressure_next[0].detach(), t * dt)
                if t % recording_stride == 0:
                    recorded_chunk.append(pressure_next[:, :, rec_z, rec_x])
                previous = current
                current = pressure_next
            if recorded_chunk:
                recorded_tensor = torch.stack(recorded_chunk, dim=2)
            else:
                recorded_tensor = current.new_empty(
                    (batch_size, source_count, 0, self.receiver_count)
                )
            return previous, current, recorded_tensor

        chunk_size = self.checkpoint_time_steps
        should_checkpoint = (
                torch.is_grad_enabled()
                and velocity.requires_grad
                and chunk_size is not None
                and 0 < chunk_size < simulation_time_step_count
        )
        if should_checkpoint:
            recorded = []
            for start_t in range(0, simulation_time_step_count, chunk_size):
                end_t = min(start_t + chunk_size, simulation_time_step_count)

                def run_chunk(previous, current, current_w, previous_w, alpha_value, source_value, start_t=start_t, end_t=end_t):
                    return propagate_chunk(
                        previous, current, current_w, previous_w, alpha_value, source_value, start_t, end_t
                    )

                pressure_previous, pressure_current, recorded_chunk = checkpoint(
                    run_chunk,
                    pressure_previous, pressure_current, current_weight, previous_weight, alpha, source_scale,
                    use_reentrant=False,
                )
                recorded.append(recorded_chunk)
            result = torch.cat(recorded, dim=2)
            if result.shape[2] != self.time_step_count:
                raise RuntimeError("O solver direto produziu uma quantidade inesperada de registros")
            return result

        _, _, recorded = propagate_chunk(
            pressure_previous, pressure_current, current_weight, previous_weight, alpha, source_scale,
            0, simulation_time_step_count,
        )
        if recorded.shape[2] != self.time_step_count:
            raise RuntimeError("O solver direto produziu uma quantidade inesperada de registros")
        return recorded


def _load_deepwave():
    """Carrega o Deepwave somente quando esse mecanismo de execução é selecionado."""
    global _DEEPWAVE_MODULE
    if _DEEPWAVE_MODULE is not None:
        return _DEEPWAVE_MODULE

    prepare_torch_extension_cache(
        "/tmp/tcc-fwi-deepwave-training-extensions"
    )
    try:
        import deepwave
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "O backend direto Deepwave exige o pacote 'deepwave'. "
            "Instale-o com 'pip install deepwave'. Se o cache da extensão do PyTorch "
            "for somente leitura, defina TORCH_EXTENSIONS_DIR para um diretório gravável."
        ) from exc

    _DEEPWAVE_MODULE = deepwave
    return _DEEPWAVE_MODULE


class DeepwaveForward2D(nn.Module):
    """Operador direto acústico do Deepwave com a interface comum do projeto.

        O projeto armazena a velocidade como ``[batch, 1, depth, width]`` e a
        geometria como ``[x, z]``. O modelo 2D do Deepwave é orientado como
        ``[x, z]``, então cada modelo de velocidade é transposto antes da propagação,
        e a saída dos receptores é transposta de volta para
        ``[batch, source, time, receiver]``.
    """

    def __init__(
        self,
        source_indices,
        receiver_count,
        time_step_count,
        receiver_indices=None,
        receiver_z_index=1,
        checkpoint_time_steps=None,
        config=OPENFWI_CONFIG,
    ):
        super().__init__()
        self.config = config
        self.receiver_count = int(receiver_count)
        self.receiver_z_index = int(receiver_z_index)
        self.time_step_count = int(time_step_count)
        # Mantido por compatibilidade de interface. O Deepwave controla sua
        # própria memória interna de propagação; esta opção não é usada como
        # passo do gradiente.
        self.checkpoint_time_steps = checkpoint_time_steps
        source_tensor = torch.as_tensor(source_indices, dtype=torch.long)
        # O fluxo de trabalho PGNN-FWI usa np.flip sem um eixo, invertendo
        # também a ordem dos tiros.
        if bool(config.get("deepwave_reverse_source_order", False)):
            source_tensor = source_tensor.flip(0)
        self.register_buffer("source_indices", source_tensor)
        if self.source_indices.ndim != 2 or self.source_indices.shape[1] != 2:
            raise ValueError("source_indices deve ter a forma [source, 2] na ordem [x, z]")

        if receiver_indices is None:
            receiver_tensor = torch.empty((0, 2), dtype=torch.long)
        else:
            receiver_tensor = torch.as_tensor(receiver_indices, dtype=torch.long)
            # A mesma chamada np.flip inverte a ordem dos receptores nesses dados.
            if bool(config.get("deepwave_reverse_receiver_order", False)):
                receiver_tensor = receiver_tensor.flip(0)
            if receiver_tensor.ndim != 2 or receiver_tensor.shape[1] != 2:
                raise ValueError(
                    "receiver_indices deve ter a forma [receiver, 2] na ordem [x, z]"
                )
            if receiver_tensor.shape[0] != self.receiver_count:
                raise ValueError(
                    f"receiver_count={self.receiver_count} não corresponde a "
                    f"{receiver_tensor.shape[0]} posições explícitas de receptores"
                )
        self.register_buffer("receiver_indices", receiver_tensor)

    def _time_sampling(self):
        recording_dt = float(self.config["time_spacing_s"])
        simulation_dt = float(self.config.get("simulation_time_spacing_s", recording_dt))
        if simulation_dt <= 0.0 or recording_dt <= 0.0 or simulation_dt > recording_dt:
            raise ValueError(
                "simulation_time_spacing_s deve ser positivo e não maior que time_spacing_s"
            )
        recording_stride = int(round(recording_dt / simulation_dt))
        if not math.isclose(recording_stride * simulation_dt, recording_dt, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                "time_spacing_s deve ser um múltiplo inteiro de simulation_time_spacing_s"
            )
        simulation_time_step_count = (self.time_step_count - 1) * recording_stride + 1
        return simulation_dt, recording_stride, simulation_time_step_count

    def _locations(self, velocity, sources):
        device = velocity.device
        physical_depth, physical_width = velocity.shape

        if torch.any(sources[:, 0] < 0) or torch.any(sources[:, 0] >= physical_width) or torch.any(
            sources[:, 1] < 0
        ) or torch.any(sources[:, 1] >= physical_depth):
            raise ValueError("A posição de uma fonte está fora do modelo de velocidade")

        source_locations = sources[:, None, :].contiguous()
        if self.receiver_indices.numel():
            receivers = self.receiver_indices.to(device)
            if torch.any(receivers[:, 0] < 0) or torch.any(receivers[:, 0] >= physical_width) or torch.any(
                receivers[:, 1] < 0
            ) or torch.any(receivers[:, 1] >= physical_depth):
                raise ValueError("A posição de um receptor está fora do modelo de velocidade")
        else:
            receivers = torch.stack([
                torch.arange(self.receiver_count, device=device, dtype=torch.long),
                torch.full(
                    (self.receiver_count,),
                    self.receiver_z_index,
                    device=device,
                    dtype=torch.long,
                ),
            ], dim=1)
            if torch.any(receivers[:, 0] >= physical_width) or torch.any(
                receivers[:, 1] >= physical_depth
            ):
                raise ValueError("A posição de um receptor está fora do modelo de velocidade")
        receiver_locations = receivers.unsqueeze(0).repeat(sources.shape[0], 1, 1)
        return source_locations, receiver_locations

    def _propagate_single(
        self,
        velocity,
        sources,
        wavefield_callback=None,
        callback_frequency=1,
    ):
        deepwave = _load_deepwave()
        simulation_dt, recording_stride, simulation_time_step_count = self._time_sampling()
        source_locations, receiver_locations = self._locations(velocity, sources)

            # O fluxo Marmousi do PGNN-FWI passa o tensor armazenado [depth, width]
        # diretamente ao Deepwave, cujas coordenadas de localização usam então
        # a ordem [depth, width]. O caminho OpenFWI historicamente transpõe o
        # tensor e usa coordenadas [width, depth]. Mantenha ambas as convenções
        # explícitas para que um dataset possa reproduzir sua aquisição original.
        location_order = str(self.config.get("deepwave_location_order", "xz")).lower()
        if location_order == "zx":
            source_locations = source_locations.flip(-1)
            receiver_locations = receiver_locations.flip(-1)
        elif location_order != "xz":
            raise ValueError(
                "deepwave_location_order deve ser 'xz' ou 'zx'"
            )

        receiver_inverse = None
        if receiver_locations.shape[1]:
            unique_receivers, candidate_inverse = torch.unique(
                receiver_locations[0],
                dim=0,
                return_inverse=True,
            )
            if unique_receivers.shape[0] != receiver_locations.shape[1]:
                # A geometria é repetida para todos os tiros. O Deepwave atual
                # exige coordenadas únicas dentro de cada tiro; canais que
                # compartilham uma célula serão duplicados novamente na saída.
                receiver_inverse = candidate_inverse
                receiver_locations = unique_receivers.unsqueeze(0).expand(
                    receiver_locations.shape[0],
                    -1,
                    -1,
                ).contiguous()

        accuracy = int(self.config.get("deepwave_accuracy", 4))
        if accuracy not in (2, 4, 6, 8):
            raise ValueError("deepwave_accuracy deve ser 2, 4, 6 ou 8")
        absorbing_cells = int(self.config.get("absorbing_boundary_cells", 20))
        # O Deepwave ordena pml_width como [início_x, fim_x, início_z, fim_z].
        pml_width = [
            absorbing_cells,
            absorbing_cells,
            0 if self.config.get("is_free_surface", False) else absorbing_cells,
            absorbing_cells,
        ]

        dtype = velocity.dtype
        device = velocity.device
        source_frequency_hz = float(self.config["frequency_hz"])
        source_wavelet = deepwave.wavelets.ricker(
            freq=source_frequency_hz,
            length=simulation_time_step_count,
            dt=simulation_dt,
            peak_time=float(self.config.get("source_peak_time_s", 1.5 / float(source_frequency_hz))),
            dtype=dtype,
        ).to(device)
        source_amplitudes = source_wavelet[None, None, :].repeat(sources.shape[0], 1, 1)

        if bool(self.config.get("deepwave_transpose_model", True)):
            velocity_xy = velocity.transpose(0, 1).contiguous()
        else:
            velocity_xy = velocity.contiguous()
        deepwave_kwargs = {
            "source_amplitudes": source_amplitudes,
            "source_locations": source_locations,
            "receiver_locations": receiver_locations,
            "accuracy": accuracy,
            "pml_width": pml_width,
            "model_gradient_sampling_interval": int(
                self.config.get("deepwave_model_gradient_sampling_interval", 1)
            ),
        }
        # O padrão escalar do Deepwave é 25 Hz. Defini-lo explicitamente
        # preserva esse valor e evita o aviso da versão com callbacks.
        pml_freq = self.config.get("deepwave_pml_freq")
        deepwave_kwargs["pml_freq"] = (
            25.0 if pml_freq is None else float(pml_freq)
        )
        if self.config.get("deepwave_max_vel") is not None:
            deepwave_kwargs["max_vel"] = float(self.config["deepwave_max_vel"])
        supports_forward_callback = False
        if wavefield_callback is not None:
            try:
                scalar_parameters = inspect.signature(deepwave.scalar).parameters
                supports_forward_callback = {
                    "forward_callback",
                    "callback_frequency",
                }.issubset(scalar_parameters)
            except (TypeError, ValueError):
                supports_forward_callback = False
            if not supports_forward_callback:
                raise RuntimeError(
                    "A captura de wavefield requer deepwave==0.0.27. "
                    "Atualize o ambiente a partir de requirements.txt."
                )

        if wavefield_callback is not None and supports_forward_callback:
            deepwave_kwargs["forward_callback"] = partial(
                _send_deepwave_wavefield,
                callback=wavefield_callback,
                transpose_model=bool(self.config.get("deepwave_transpose_model", True)),
            )
            deepwave_kwargs["callback_frequency"] = int(callback_frequency)

        outputs = deepwave.scalar(
            velocity_xy,
            float(self.config["grid_spacing_m"]),
            simulation_dt,
            **deepwave_kwargs,
        )
        receiver_amplitudes = outputs[-1]
        expected_internal_steps = simulation_time_step_count
        if receiver_amplitudes.shape[-1] != expected_internal_steps:
            raise RuntimeError(
                "O Deepwave produziu uma quantidade inesperada de amostras temporais: "
                f"esperado {expected_internal_steps}, obtido {receiver_amplitudes.shape[-1]}"
            )
        recorded = receiver_amplitudes[..., ::recording_stride]
        if recorded.shape[-1] != self.time_step_count:
            raise RuntimeError("O Deepwave produziu uma quantidade inesperada de registros")
        if receiver_inverse is not None:
            recorded = recorded.index_select(1, receiver_inverse)
        return recorded.transpose(1, 2).contiguous()

    def forward(
        self,
        velocity,
        source_selection=None,
        wavefield_callback=None,
        callback_frequency=1,
    ):
        if velocity.ndim != 4 or velocity.shape[1] != 1:
            raise ValueError("velocity deve ter a forma [batch, 1, depth, width]")
        if wavefield_callback is not None:
            if torch.is_grad_enabled():
                raise ValueError("A captura de wavefield requer torch.no_grad()")
            if velocity.shape[0] != 1:
                raise ValueError("A captura de wavefield aceita uma amostra por chamada")
            if int(callback_frequency) < 1:
                raise ValueError("callback_frequency deve ser maior que zero")

        sources = self.source_indices.to(velocity.device)
        if source_selection is not None:
            selection = torch.as_tensor(source_selection, dtype=torch.long, device=velocity.device)
            if selection.ndim != 1 or selection.numel() == 0:
                raise ValueError("source_selection deve ser uma sequência 1D não vazia de índices de tiros")
            sources = sources.index_select(0, selection)

        velocity = velocity[:, 0].float()
        return torch.stack([
            self._propagate_single(
                model,
                sources,
                wavefield_callback=wavefield_callback,
                callback_frequency=callback_frequency,
            )
            for model in velocity
        ], dim=0)


def build_forward_operator(
    *,
    source_indices,
    receiver_count,
    time_step_count,
    receiver_indices=None,
    receiver_z_index=1,
    checkpoint_time_steps=None,
    config=OPENFWI_CONFIG,
):
    """Monta o operador direto diferenciável definido na configuração.

        ``openfwi`` mantém o solver de diferenças finitas existente no projeto;
        ``deepwave`` seleciona a implementação PyTorch/Deepwave para a física
        configurada no experimento.
    """
    backend = str(config.get("forward_backend", "openfwi")).strip().lower()
    operator_class = {
        "openfwi": OpenFWITorchForward2D,
        "deepwave": DeepwaveForward2D,
    }.get(backend)
    if operator_class is None:
        raise ValueError(
            f"Backend direto não suportado={backend!r}; escolha 'openfwi' ou 'deepwave'"
        )
    return operator_class(
        source_indices=source_indices,
        receiver_count=receiver_count,
        time_step_count=time_step_count,
        receiver_indices=receiver_indices,
        receiver_z_index=receiver_z_index,
        checkpoint_time_steps=checkpoint_time_steps,
        config=config,
    )
