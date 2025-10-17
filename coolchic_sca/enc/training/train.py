# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


import copy
import time
from typing import List, Tuple

import torch
from enc.component.core.quantizer import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE,
    POSSIBLE_QUANTIZER_TYPE,
)
from enc.component.frame import FrameEncoder
from enc.io.format.yuv import convert_420_to_444
from enc.training.loss import loss_function
from enc.training.presets import MODULE_TO_OPTIMIZE
from enc.training.test import test
from enc.utils.codingstructure import Frame
from enc.training.manager import FrameEncoderManager
from torch.nn.utils import clip_grad_norm_


# Custom scheduling function for the soft rounding temperature and the noise parameter
def _linear_schedule(
    initial_value: float, final_value: float, cur_itr: float, max_itr: float
) -> float:
    """Linearly schedule a function to go from initial_value at cur_itr = 0 to
    final_value when cur_itr = max_itr.

    Args:
        initial_value (float): Initial value for the scheduling
        final_value (float): Final value for the scheduling
        cur_itr (float): Current iteration index
        max_itr (float): Total number of iterations

    Returns:
        float: The linearly scheduled value @ iteration number cur_itr
    """
    assert cur_itr >= 0 and cur_itr <= max_itr, (
        f"Linear scheduling from 0 to {max_itr} iterations"
        " except to have a current iterations between those two values."
        f" Found cur_itr = {cur_itr}."
    )

    return cur_itr * (final_value - initial_value) / max_itr + initial_value

# ---- SCA: função que faz a regularização [-1, 1]
def ternary_regularizer(parameters: List[torch.Tensor], alpha_sca: float):
    reg_loss = 0.0
    for param in parameters:
        t = torch.tanh(param)
        t2 = t ** 2
        reg_loss += ((alpha_sca - t2) * t2).sum()

    return reg_loss

def calcula_loss_reg(parameters: List[torch.Tensor], alpha_sca: float):
    reg_loss = 0
    for param in parameters:
        t2 = param ** 2
        reg_loss += ((alpha_sca - t2) * t2).sum()

    return reg_loss

import torch

@torch.no_grad()
def analisar_pesos_discretizados(frame_encoder: FrameEncoder, threshold_zero: float = 0.1, threshold_saturado: float = 3.0):
    """
    Analisa os pesos dos módulos 'arm', 'synthesis' e 'upsampling' para verificar a discretização.

    Args:
        frame_encoder: O modelo treinado.
        threshold_zero: Limiar para considerar um peso como "próximo de zero".
        threshold_saturado: Limiar para considerar um peso como "saturado" (negativo ou positivo).
                           tanh(3.0) ≈ 0.995, então é um bom valor.
    """
    print(f"\n--- Análise da Discretização dos Pesos (zero < {threshold_zero}, sat > {threshold_saturado}) ---")

    for name, coolchic_encoder in frame_encoder.coolchic_enc.items():
        print(f"\nAnalisando encoder: '{name}'")
        modulos = {'arm': coolchic_encoder.arm, 'synthesis': coolchic_encoder.synthesis, 'upsampling': coolchic_encoder.upsampling}

        for mod_name, module in modulos.items():
            for param_name, param in module.named_parameters():
                if 'weight' not in param_name:
                    continue

                total_pesos = param.numel()
                pesos_zero = torch.sum(param.abs() < threshold_zero).item()                 # entre -0.1 e 0.1
                pesos_pos = torch.sum(param > threshold_saturado).item()                    # maior que 3.0
                pesos_neg = torch.sum(param < -threshold_saturado).item()                   # menor que -3.0
                pesos_intermediarios = total_pesos - pesos_zero - pesos_pos - pesos_neg     # [-3.0, -0.1] e [1.0, 3.0]

                # Calcula as porcentagens
                pct_zero = 100 * pesos_zero / total_pesos
                pct_pos = 100 * pesos_pos / total_pesos
                pct_neg = 100 * pesos_neg / total_pesos
                pct_interm = 100 * pesos_intermediarios / total_pesos

                print(f"  Módulo: {mod_name} | Parâmetro: {param_name} | Shape: {list(param.shape)}")
                print(f"    - Próximos de Zero: {pesos_zero:<7d} ({pct_zero:5.1f}%)")
                print(f"    - Saturados Pos.  : {pesos_pos:<7d} ({pct_pos:5.1f}%)")
                print(f"    - Saturados Neg.  : {pesos_neg:<7d} ({pct_neg:5.1f}%)")
                print(f"    - Intermediários  : {pesos_intermediarios:<7d} ({pct_interm:5.1f}%)")

@torch.no_grad()
def analisar_pesos_ja_arredondados(frame_encoder: FrameEncoder):
    """
    Analisa um modelo que JÁ FOI ARREDONDADO, contando exatamente
    quantos pesos são -1, 0 ou 1.
    """
    print("\n--- Análise do Modelo com Pesos Arredondados para {-1, 0, 1} ---")

    for name, coolchic_encoder in frame_encoder.coolchic_enc.items():
        print(f"\nAnalisando encoder: '{name}'")
        modulos = {'arm': coolchic_encoder.arm, 'synthesis': coolchic_encoder.synthesis}

        for mod_name, module in modulos.items():
            for param_name, param in module.named_parameters():
                if 'weight' not in param_name:
                    continue

                total_pesos = param.numel()
                # Contagem exata dos valores
                pesos_zero = torch.sum(param.data == 0.0).item()
                pesos_um_pos = torch.sum(param.data == 1.0).item()
                pesos_um_neg = torch.sum(param.data == -1.0).item()
                outros = total_pesos - pesos_zero - pesos_um_pos - pesos_um_neg

                # Calcula as porcentagens
                pct_zero = 100 * pesos_zero / total_pesos
                pct_pos = 100 * pesos_um_pos / total_pesos
                pct_neg = 100 * pesos_um_neg / total_pesos
                
                print(f"  Módulo: {mod_name} | Parâmetro: {param_name} | Shape: {list(param.shape)}")
                print(f"    - Pesos == 0 : {pesos_zero:<7d} ({pct_zero:5.1f}%)")
                print(f"    - Pesos == 1 : {pesos_um_pos:<7d} ({pct_pos:5.1f}%)")
                print(f"    - Pesos == -1: {pesos_um_neg:<7d} ({pct_neg:5.1f}%)")
                if outros > 0:
                    print(f"    - !! OUTROS VALORES: {outros}")

@torch.no_grad()
def arredondamento_suave_seletivo(
    frame_encoder: FrameEncoder,
    limiar_intermediarios_pct: float = 15.0, # Podemos ser um pouco mais permissivos aqui
    limiar_arred_zero: float = 0.2,   # Se abs(peso) < 0.2, arredonda para 0
    limiar_arred_sat: float = 2.5     # Se abs(peso) > 2.5, arredonda para +/- 1
):
    """
    Aplica um arredondamento 'suave' e seletivo.
    1. Seleciona camadas com poucos pesos intermediários (usando a análise original).
    2. Nessas camadas, arredonda APENAS os pesos que já estão muito próximos
       de zero ou dos valores de saturação. Os pesos intermediários são mantidos.
    Modifica o modelo IN-PLACE.
    """
    print(f"\n--- Iniciando Arredondamento Suave Seletivo ---")
    print(f"Critérios: Camadas com < {limiar_intermediarios_pct}% de interm.; "
          f"Arredondar pesos com |w| < {limiar_arred_zero} ou |w| > {limiar_arred_sat}")

    # Limiares da análise para selecionar a camada (thresholds originais)
    threshold_zero_analise = 0.1
    threshold_saturado_analise = 3.0

    for name, coolchic_encoder in frame_encoder.coolchic_enc.items():
        print(f"\nAnalisando encoder: '{name}'")
        modulos = {'arm': coolchic_encoder.arm, 'synthesis': coolchic_encoder.synthesis, 'upsampling': coolchic_encoder.upsampling}

        for mod_name, module in modulos.items():
            for param_name, param in module.named_parameters():
                if 'weight' not in param_name:
                    continue
                
                # --- Análise para decidir se a CAMADA será processada ---
                total_pesos = param.numel()
                pesos_zero_analise = torch.sum(param.abs() < threshold_zero_analise).item()
                pesos_pos_analise = torch.sum(param.data > threshold_saturado_analise).item()
                pesos_neg_analise = torch.sum(param.data < -threshold_saturado_analise).item()
                pesos_intermediarios = total_pesos - pesos_zero_analise - pesos_pos_analise - pesos_neg_analise
                pct_interm = 100 * pesos_intermediarios / total_pesos if total_pesos > 0 else 0

                print(f"  Módulo: {mod_name} | Parâmetro: {param_name} | Intermediários: {pct_interm:.1f}%")

                # --- Lógica Seletiva ---
                if pct_interm < limiar_intermediarios_pct:
                    print(f"    └─ CAMADA SELECIONADA. Aplicando arredondamento suave...")
                    
                    # Começamos com uma cópia exata dos pesos originais
                    novo_param = param.data.clone()

                    # Criamos máscaras para os pesos que estão "quase lá"
                    mask_quase_zero = torch.abs(param.data) < limiar_arred_zero
                    mask_quase_um_pos = param.data > limiar_arred_sat
                    mask_quase_um_neg = param.data < -limiar_arred_sat
                    
                    # Aplicamos o arredondamento APENAS nesses pesos
                    novo_param[mask_quase_zero] = 0.0
                    novo_param[mask_quase_um_pos] = 1.0
                    novo_param[mask_quase_um_neg] = -1.0
                    
                    # Calculamos quantos pesos foram realmente alterados
                    num_alterados = torch.sum(mask_quase_zero | mask_quase_um_pos | mask_quase_um_neg).item()
                    pct_alterados = 100 * num_alterados / total_pesos if total_pesos > 0 else 0
                    print(f"      - {num_alterados} / {total_pesos} pesos arredondados ({pct_alterados:.1f}%)")

                    # Atualiza os pesos no modelo
                    param.data = novo_param
                else:
                    print(f"    └─ PULANDO CAMADA.")

# FUNÇÃO DE ARREDONDAMENTO
@torch.no_grad()
def analisar_e_arredondar_seletivamente(
    frame_encoder: FrameEncoder,
    limiar_intermediarios_pct: float = 10.0,
    threshold_zero: float = 0.1,
    threshold_saturado: float = 3.0
):
    """
    Analisa e arredonda seletivamente os pesos de um modelo.
    Modifica o modelo IN-PLACE.
    """
    print(f"\n--- Tentando arredondar camadas com menos de {limiar_intermediarios_pct}% de pesos intermediários ---")

    for name, coolchic_encoder in frame_encoder.coolchic_enc.items():
        print(f"\nAnalisando encoder: '{name}'")
        modulos = {'arm': coolchic_encoder.arm, 'synthesis': coolchic_encoder.synthesis, 'upsampling': coolchic_encoder.upsampling}

        for mod_name, module in modulos.items():
            for param_name, param in module.named_parameters():
                if 'weight' not in param_name:
                    continue

                total_pesos = param.numel()
                pesos_zero = torch.sum(param.abs() < threshold_zero).item()
                pesos_pos = torch.sum(param.data > threshold_saturado).item()
                pesos_neg = torch.sum(param.data < -threshold_saturado).item()
                pesos_intermediarios = total_pesos - pesos_zero - pesos_pos - pesos_neg
                pct_interm = 100 * pesos_intermediarios / total_pesos if total_pesos > 0 else 0

                print(f"  Módulo: {mod_name} | Parâmetro: {param_name} | Intermediários: {pct_interm:.1f}%")

                if pct_interm < limiar_intermediarios_pct:
                    print(f"    └─ ARREDONDANDO!")
                    novo_param = torch.zeros_like(param.data)
                    novo_param[param.data > threshold_zero] = 1.0
                    novo_param[param.data < -threshold_zero] = -1.0
                    param.data = novo_param
                else:
                    print(f"    └─ PULANDO.")

def train(
    frame_encoder: FrameEncoder,
    frame: Frame,
    frame_encoder_manager: FrameEncoderManager,
    lmbda: float = 1e-3,
    start_lr: float = 1e-2,
    cosine_scheduling_lr: bool = True,
    max_iterations: int = 10000,
    frequency_validation: int = 100,
    patience: int = 10,
    optimized_module: List[MODULE_TO_OPTIMIZE] = ["all"],
    quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
    quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
    softround_temperature: Tuple[float, float] = (0.3, 0.2),
    noise_parameter: Tuple[float, float] = (2.0, 1.0),
) -> FrameEncoder:
    """Train a ``FrameEncoder`` and return the updated module. This function is
    supposed to be called any time we want to optimize the parameters of a
    FrameEncoder, either during the warm-up (competition of multiple possible
    initializations) or during of the stages of the actual training phase.

    The module is optimized according to the following loss function:

    .. math::

        \\mathcal{L} = ||\\mathbf{x} - \hat{\\mathbf{x}}||^2 + \\lambda
        \\mathrm{R}(\hat{\\mathbf{x}}), \\text{ with } \\begin{cases}
            \\mathbf{x} & \\text{the original image}\\\\ \\hat{\\mathbf{x}} &
            \\text{the coded image}\\\\ \\mathrm{R}(\\hat{\\mathbf{x}}) &
            \\text{A measure of the rate of } \\hat{\\mathbf{x}}
        \\end{cases}

    .. warning::

        The parameter ``frame_encoder_manager`` tracking the encoding time of
        the frame (``total_training_time_sec``) and the number of encoding
        iterations (``iterations_counter``) is modified **in place** by this
        function.

    Args:
        frame_encoder: Module to be trained.
        frame: The original image to be compressed and its references.
        frame_encoder_manager: Contains (among other things) the rate
            constraint :math:`\\lambda`. It is also used to track the total
            encoding time and encoding iterations. Modified in place.
        start_lr: Initial learning rate. Either constant for the entire
            training or schedule using a cosine scheduling, see below for more
            details. Defaults to 1e-2.
        cosine_scheduling_lr: True to schedule the learning
            rate from ``start_lr`` at iteration n°0 to 0 at iteration
            n° ``max_iterations``. Defaults to True.
        max_iterations: Do at most ``max_iterations`` iterations.
            The actual number of iterations can be made smaller through the
            patience mechanism. Defaults to 10000.
        frequency_validation: Check (and print) the performance
            each ``frequency_validation`` iterations. This drives the patience
            mechanism. Defaults to 100.
        patience: After ``patience`` iterations without any
            improvement to the results, exit the training. Patience is disabled
            by setting ``patience = max_iterations``. If patience is used alongside
            cosine_scheduling_lr, then it does not end the training. Instead,
            we simply reload the best model so far once we reach the patience,
            and the training continue. Defaults to 10.
        optimized_module: List of modules to be optimized. Most often you'd
            want to use ``optimized_module = ['all']``. Defaults to ``['all']``.
        quantizer_type: What quantizer to
            use during training. See :doc:`encoder/component/core/quantizer.py
            <../component/core/quantizer>` for more information. Defaults to
            ``"softround"``.
        quantizer_noise_type: The random noise used by the quantizer. More
            information available in
            :doc:`encoder/component/core/quantizer.py
            <../component/core/quantizer>`. Defaults to ``"kumaraswamy"``.
        softround_temperature: The softround temperature is linearly scheduled
            during the training. At iteration n° 0 it is equal to
            ``softround_temperature[0]`` while at iteration n° ``max_itr`` it is
            equal to ``softround_temperature[1]``. Note that the patience might
            interrupt the training before it reaches this last value.
            Defaults to (0.3, 0.2).
        noise_parameter: The random noise temperature is linearly scheduled
            during the training. At iteration n° 0 it is equal to
            ``noise_parameter[0]`` while at iteration n° ``max_itr`` it is equal
            to ``noise_parameter[1]``. Note that the patience might interrupt
            the training before it reaches this last value. Defaults to (2.0,
            1.0).

    Returns:
        The trained frame encoder.
    """
    start_time = time.time()

    # We train with dense reference!
    for idx_ref, ref_i in enumerate(frame.refs_data):
        if ref_i.frame_data_type == "yuv420":
            frame.refs_data[idx_ref].data = convert_420_to_444(ref_i.data)
            frame.refs_data[idx_ref].frame_data_type = "yuv444"

    raw_references_444 = [ref_i.data for ref_i in frame.refs_data]

    # ------ Keep track of the best loss and model
    # Perform a first test to get the current best logs (it includes the loss)
    initial_encoder_logs = test(frame_encoder, frame, frame_encoder_manager)
    encoder_logs_best = initial_encoder_logs
    best_model = frame_encoder.get_param()

    frame_encoder.set_to_train()

    # ------ Build the list of parameters to optimize
    # Iteratively construct the list of required parameters.

    parameters_to_optimize = []
    for cur_module_to_optimize in optimized_module:
        # No need to go further, we'll want to optimize everything!
        if cur_module_to_optimize == "all":
            parameters_to_optimize = frame_encoder.parameters()
            break

        else:
            raw_cc_name, mod_name = cur_module_to_optimize.split(".")

            if raw_cc_name == "all":
                raw_cc_name = list(frame_encoder.coolchic_enc.keys())
            else:
                raw_cc_name = [raw_cc_name]

            for cc_name in raw_cc_name:
                assert cc_name in frame_encoder.coolchic_enc, (
                    f"Trying to optimize the parameters {cur_module_to_optimize}."
                    f" However, there is no {cc_name} Cool-chic encoder. Found "
                    f"{list(frame_encoder.coolchic_enc.keys())}"
                )

                match mod_name:
                    case "all":
                        parameters_to_optimize+= [
                            *frame_encoder.coolchic_enc[cc_name].parameters()
                        ]
                    case "arm":
                        parameters_to_optimize+= [
                            *frame_encoder.coolchic_enc[cc_name].arm.parameters()
                        ]
                    case "upsampling":
                        parameters_to_optimize+= [
                            *frame_encoder.coolchic_enc[cc_name].upsampling.parameters()
                        ]
                    case "synthesis":
                        parameters_to_optimize+= [
                            *frame_encoder.coolchic_enc[cc_name].synthesis.parameters()
                        ]
                    case "latent":
                        parameters_to_optimize+= [
                            *frame_encoder.coolchic_enc[cc_name].latent_grids.parameters()
                        ]
                    case "warper":
                        if frame_encoder.frame_type != "I":
                            parameters_to_optimize+= [
                                *frame_encoder.warper.parameters()
                            ]
                        else:
                            print(
                                "Trying to optimize warper but this is an I-frame, "
                                "so it does not have a warper."
                            )



    optimizer = torch.optim.Adam(parameters_to_optimize, lr=start_lr)
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())

    if cosine_scheduling_lr:
        # TODO: I'd like to use an explicit function for this scheduler
        learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_iterations / frequency_validation,
            eta_min=0.00001,
            last_epoch=-1,
        )
    else:
        learning_rate_scheduler = None

    # Initialize soft rounding temperature and noise parameter
    cur_softround_temperature = _linear_schedule(
        softround_temperature[0],
        softround_temperature[1],
        0,
        max_iterations,
    )
    device = frame.data.data.device if frame.data.frame_data_type != "yuv420" else frame.data.data.get("y").device
    cur_softround_temperature = torch.tensor(cur_softround_temperature, device=device)

    cur_noise_parameter = _linear_schedule(
        noise_parameter[0], noise_parameter[1], 0, max_iterations
    )
    cur_noise_parameter = torch.tensor(cur_noise_parameter, device=device)

    cnt_record = 0
    show_col_name = True  # Only for a pretty display of the logs
    # Slightly faster to create the list once outside of the loop
    all_parameters = [x for x in frame_encoder.parameters()]

    # ---- SCA: separando os parâmetros das MLPs
    params_for_regularization = []
    for coolchic_encoder in frame_encoder.coolchic_enc.values():
        params_for_regularization.extend(list(coolchic_encoder.arm.parameters()))
        #params_for_regularization.extend(list(coolchic_encoder.synthesis.parameters()))
        #params_for_regularization.extend(list(coolchic_encoder.upsampling.parameters()))

    for cnt in range(max_iterations):
        # print(sum(v.abs().sum() for _, v in best_model.items()))

        # ------- Patience mechanism
        if cnt - cnt_record > patience:
            if cosine_scheduling_lr:
                # reload the best model so far
                frame_encoder.set_param(best_model)
                optimizer.load_state_dict(best_optimizer_state)

                current_lr = learning_rate_scheduler.state_dict()["_last_lr"][0]
                # actualise the best optimizer lr with current lr
                for g in optimizer.param_groups:
                    g["lr"] = current_lr

                cnt_record = cnt
            else:
                # exceeding the patience level ends the phase
                break

        # ------- Actual optimization
        # This is slightly faster than optimizer.zero_grad()
        for param in all_parameters:
            param.grad = None

        # forward / backward
        out_forward = frame_encoder.forward(
            reference_frames=raw_references_444,
            quantizer_noise_type=quantizer_noise_type,
            quantizer_type=quantizer_type,
            soft_round_temperature=cur_softround_temperature,
            noise_parameter=cur_noise_parameter,
        )

        decoded_image = out_forward.decoded_image
        target_image = frame.data.data

        loss_function_output = loss_function(
            decoded_image=decoded_image,
            rate_latent_bit=out_forward.rate,
            target_image=target_image,
            lmbda=lmbda,
            total_rate_nn_bit=0.0,
            compute_logs=False,
        )

        alpha_sca = 1e-7
        #reg_loss = ternary_regularizer(params_for_regularization, alpha_sca)
        reg_loss = calcula_loss_reg(params_for_regularization, alpha_sca=alpha_sca)

        lmbda_sca = 1e-5
        total_loss = loss_function_output.loss + lmbda_sca * reg_loss

        #loss_function_output.loss.backward()
        total_loss.backward()
        clip_grad_norm_(all_parameters, 1e-1, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()

        frame_encoder_manager.iterations_counter += 1

        # ------- Validation
        # Each freq_valid iteration or at the end of the phase, compute validation loss and log stuff
        if ((cnt + 1) % frequency_validation == 0) or (cnt + 1 == max_iterations):
            #  a. Update iterations counter and training time and test model
            frame_encoder_manager.total_training_time_sec += time.time() - start_time
            start_time = time.time()

            # b. Test the model and check whether we've beaten our record
            encoder_logs = test(frame_encoder, frame, frame_encoder_manager)

            flag_new_record = False

            if encoder_logs.loss < encoder_logs_best.loss:
                # A record must have at least -0.001 bpp or + 0.001 dB. A smaller improvement
                # does not matter.
                delta_psnr = encoder_logs.psnr_db - encoder_logs_best.psnr_db
                delta_bpp = (
                    encoder_logs.total_rate_latent_bpp
                    - encoder_logs_best.total_rate_latent_bpp
                )
                flag_new_record = delta_bpp < 0.001 or delta_psnr > 0.001

            if flag_new_record:
                # Save best model
                best_model = frame_encoder.get_param()
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())

                # ========================= reporting ========================= #
                this_phase_psnr_gain = (
                    encoder_logs.psnr_db - initial_encoder_logs.psnr_db
                )
                this_phase_bpp_gain = (
                    encoder_logs.total_rate_latent_bpp
                    - initial_encoder_logs.total_rate_latent_bpp
                )

                log_new_record = ""
                log_new_record += f"{this_phase_bpp_gain:+6.3f} bpp "
                log_new_record += f"{this_phase_psnr_gain:+6.3f} db"
                # ========================= reporting ========================= #

                # Update new record
                encoder_logs_best = encoder_logs
                cnt_record = cnt
            else:
                log_new_record = ""

            # Show column name a single time
            additional_data = {
                "lr": f"{start_lr if not cosine_scheduling_lr else learning_rate_scheduler.get_last_lr()[0]:.4f}",
                "optim": ",".join(optimized_module),
                "patience": (patience - cnt + cnt_record) // frequency_validation,
                "q_type": f"{quantizer_type:10s}",
                "sr_temp": f"{cur_softround_temperature:.3f}",
                "n_type": f"{quantizer_noise_type:12s}",
                "noise": f"{cur_noise_parameter:.2f}",
                "record": log_new_record,
            }

            print(
                encoder_logs.pretty_string(
                    show_col_name=show_col_name,
                    mode="short",
                    additional_data=additional_data,
                )
            )
            show_col_name = False

            # Update soft rounding temperature and noise_parameter
            cur_softround_temperature = _linear_schedule(
                softround_temperature[0],
                softround_temperature[1],
                cnt,
                max_iterations,
            )
            cur_softround_temperature = torch.tensor(cur_softround_temperature, device=device)

            cur_noise_parameter = _linear_schedule(
                noise_parameter[0],
                noise_parameter[1],
                cnt,
                max_iterations,
            )
            cur_noise_parameter = torch.tensor(cur_noise_parameter, device=device)

            if cosine_scheduling_lr:
                learning_rate_scheduler.step()

            frame_encoder.set_to_train()

    # At the end of the training, we load the best model
    frame_encoder.set_param(best_model)
    #analisar_pesos_discretizados(frame_encoder)
    #analisar_e_arredondar_seletivamente(frame_encoder=frame_encoder)
    #analisar_pesos_ja_arredondados(frame_encoder)
    return frame_encoder
