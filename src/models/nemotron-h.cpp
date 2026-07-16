#include "models.h"
#include <algorithm>

void llama_model_nemotron_h::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // NextN/MTP support: extra decoder block appended beyond the main stack
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    // n_layer_all is total layers (trunk + MTP)
    const uint32_t n_layer_all = hparams.n_layer_all + hparams.n_layer_nextn;

    // Set n_layer_nextn_per_head based on n_layer_nextn
    if (hparams.n_layer_nextn > 0) {
        hparams.n_layer_nextn_per_head = hparams.n_layer_nextn;
        // also store in nextn_predict_layers for backward compatibility
        hparams.nextn_predict_layers = hparams.n_layer_nextn;
    }

    // A layer is recurrent IFF the n_head_kv value is set to 0 and
    // the n_ff value is set to 0. Loop over all layers (including MTP).
    for (uint32_t i = 0; i < n_layer_all; ++i) {
        hparams.recurrent_layer_arr[i] = (hparams.n_head_kv(i) == 0 && hparams.n_ff(i) == 0);
    }

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // Load per-layer expert feed-forward lengths
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, n_layer_all, false);
    // Derive hparams.n_ff_exp_impl as max across layers
    hparams.n_ff_exp_impl = 0;
    for (uint32_t i = 0; i < n_layer_all; ++i) {
        if (hparams.n_ff_exp_arr[i] > hparams.n_ff_exp_impl) {
            hparams.n_ff_exp_impl = hparams.n_ff_exp_arr[i];
        }
    }

    // Load per-layer expert used count (top-k experts per token)
    ml.get_key_or_arr(LLM_KV_EXPERT_USED_COUNT, hparams.n_expert_used_arr, n_layer_all, false);

    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_MOE_LATENT_SIZE,                   hparams.moe_latent_size, false);

    // Distinguish homogeneous vs heterogeneous 88-layer models based on n_expert_used_arr variation
    switch (hparams.n_layer_all) {
        case 52: type = LLM_TYPE_31B_A3_5B; break; // Nemotron-H_MOE 31B
        case 56: type = LLM_TYPE_9B; break;
        case 88: {
            // Check if n_expert_used varies across layers (heterogeneous 75B.A9B)
            bool varying = false;
            uint32_t first_used = hparams.n_expert_used_arr[0];
            for (uint32_t i = 1; i < n_layer_all; ++i) {
                if (hparams.n_expert_used_arr[i] != first_used) {
                    varying = true;
                    break;
                }
            }
            type = varying ? LLM_TYPE_75B_A9B : LLM_TYPE_120B_A12B;
            break;
        }
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_nemotron_h::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    // mamba2 Mixer SSM params
    // NOTE: int64_t for tensor dimensions
    const int64_t d_conv     = hparams.ssm_d_conv;
    const int64_t d_inner    = hparams.ssm_d_inner;
    const int64_t d_state    = hparams.ssm_d_state;
    const int64_t n_ssm_head = hparams.ssm_dt_rank;
    const int64_t n_group    = hparams.ssm_n_group;
    const int64_t d_in_proj  = 2*d_inner + 2*n_group*d_state + n_ssm_head;
    const int64_t moe_n_embd = hparams.moe_latent_size > 0 ? hparams.moe_latent_size : n_embd;

    // embeddings
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    {
        output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
        // if output is NULL, init from the input tok embed, duplicated to allow offloading
        if (output == NULL) {
            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
        }
    }

    const uint32_t n_total_layers = hparams.n_layer_all + hparams.nextn_predict_layers;

    for (int i = 0; i < (int)n_total_layers; ++i) {
        auto & layer = layers[i];

        // all blocks use the attn norm
        layer.attn_norm  = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        if (hparams.is_recurrent(i)) {
            // ssm layers
            layer.ssm_in = create_tensor(tn(LLM_TENSOR_SSM_IN, "weight", i), {n_embd, d_in_proj}, 0);

            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, d_inner + 2*n_group*d_state}, 0);
            layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias", i), {d_inner + 2*n_group*d_state}, TENSOR_NOT_REQUIRED);

            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {n_ssm_head}, 0);

            // no "weight" suffix for these
            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_ssm_head}, 0);
            layer.ssm_d = create_tensor(tn(LLM_TENSOR_SSM_D, i), {1, n_ssm_head}, 0);

            layer.ssm_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {d_inner / n_group, n_group}, 0);

            // out_proj
            layer.ssm_out = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", i), {d_inner, n_embd}, 0);
        } else if (hparams.n_ff(i) == 0) {
            // attention layers (with optional bias)
            const int64_t n_head_i = hparams.n_head(i);
            const int64_t n_embd_k_gqa_i = hparams.n_embd_k_gqa(i);
            const int64_t n_embd_v_gqa_i = hparams.n_embd_v_gqa(i);
            create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head_i, n_embd_k_gqa_i, n_embd_v_gqa_i, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head_i, n_embd}, 0);
            layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
        } else {
            // MOE or MLP layers
            // Use per-layer n_ff_exp_i from hparams.n_ff_exp(i) with fallback to n_ff(i)/n_expert_used(i)
            const int64_t n_ff_exp_i = hparams.n_ff_exp_arr[i] > 0 ? hparams.n_ff_exp_arr[i] : (hparams.n_ff(i) / hparams.n_expert_used_arr[i]);
            const int64_t n_expert_used_i = hparams.n_expert_used_arr[i];
            const int64_t n_ff_shexp = hparams.n_ff_shexp;

            if (n_expert_used_i > 0) {
                layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), { n_embd, n_expert_used_i}, 0);
                layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert_used_i}, 0);

                // MoE branch
                layer.ffn_latent_down = create_tensor(tn(LLM_TENSOR_FFN_LATENT_DOWN, "weight", i), {n_embd, moe_n_embd}, TENSOR_NOT_REQUIRED);
                layer.ffn_latent_up   = create_tensor(tn(LLM_TENSOR_FFN_LATENT_UP,   "weight", i), {moe_n_embd, n_embd}, TENSOR_NOT_REQUIRED);

                layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp_i,   moe_n_embd, n_expert_used_i}, 0);
                layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {moe_n_embd, n_ff_exp_i, n_expert_used_i}, 0);

                // Shared expert branch
                if (n_ff_shexp > 0) {
                    layer.ffn_down_shexp  = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, 0);
                    layer.ffn_up_shexp    = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, 0);
                }
            } else {
                // mlp layers
                layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  hparams.n_ff(i), n_embd}, 0);
                layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   hparams.n_ff(i)}, 0);
                layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);
                layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias",   i), {hparams.n_ff(i)}, TENSOR_NOT_REQUIRED);
            }
        }

        // Create NextN/MTP specific tensors for MTP blocks starting at n_layer_all
        if (i == (int)hparams.n_layer_all) {
            // MTP blocks [n_layer_all, n_layer_all) - create nextn tensors for the first MTP block
            layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), { 2 * n_embd, n_embd }, 0);
            layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", i), { n_embd },              0);
            layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", i), { n_embd },              0);
        }

        // Create shared head norm tensor at the last block
        if (i == (int)n_total_layers - 1) {
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_nemotron_h::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_nemotron_h::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_mamba_base(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    ggml_build_forward_expand(gf, inpL);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer_all; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (hparams.is_recurrent(il)) {
            // ssm layer //
            cur = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);
        } else if (hparams.n_ff(il) == 0) {
            // attention layer //
            cur = build_attention_layer(*this, cur, inp->get_attn(), model, n_embd_head, il);
        } else {
            cur = graph::build_ffn_layer(*this, cur, model, il);
        }

        if (il == n_layer_all - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // add residual
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "nemotron_h_block_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llama_model_nemotron_h::graph::build_attention_layer(llm_graph_context & self, ggml_tensor * cur, llm_graph_input_attn_kv * inp_attn, const llama_model & model, int64_t n_embd_head, int il) {
    auto [Qcur, Kcur, Vcur] = self.build_qkv(model.layers[il], cur, n_embd_head, self.hparams.n_head(il), self.hparams.n_head_kv(il), il);

    const float kq_scale =
        self.hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : self.hparams.f_attention_scale;
    cur = self.build_attn(inp_attn,
            model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    self.cb(cur, "attn_out", il);
    return cur;
}

ggml_tensor * llama_model_nemotron_h::graph::build_ffn_layer(llm_graph_context & self, ggml_tensor * cur, const llama_model & model, int il) {
    if (model.layers[il].ffn_gate_inp == nullptr) {
        cur = self.build_ffn(cur,
                model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                NULL,                      NULL,                        NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                NULL,
                LLM_FFN_RELU_SQR, LLM_FFN_PAR, il);
        self.cb(cur, "ffn_out", il);
    } else {
        ggml_tensor * inp_emb    = cur;
        ggml_tensor * inp_latent = cur;

        if (model.layers[il].ffn_latent_down) {
            inp_latent = ggml_mul_mat(self.ctx0, model.layers[il].ffn_latent_down, cur);
        }

        ggml_tensor * router_logits = self.build_lora_mm(model.layers[il].ffn_gate_inp, cur);
        self.cb(router_logits, "ffn_moe_logits", il);

        ggml_tensor * moe_out =
            self.build_moe_ffn(inp_latent,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    nullptr, // no gate
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    self.hparams.n_expert, self.hparams.n_expert_used_impl,
                    LLM_FFN_RELU_SQR, self.hparams.expert_weights_norm,
                    self.hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
                    il,
                    router_logits, nullptr,
                    model.layers[il].ffn_up_exps_s,
                    nullptr, // no gate
                    model.layers[il].ffn_down_exps_s);
        self.cb(moe_out, "ffn_moe_out", il);

        if (model.layers[il].ffn_latent_up) {
            moe_out = ggml_mul_mat(self.ctx0, model.layers[il].ffn_latent_up, moe_out);
        }

        ggml_tensor * ffn_shexp = self.build_ffn(inp_emb,
                    model.layers[il].ffn_up_shexp,   NULL, model.layers[il].ffn_up_shexp_s,
                    NULL /* no gate */           ,   NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                    NULL,
                    LLM_FFN_RELU_SQR, LLM_FFN_PAR, il);
        self.cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(self.ctx0, moe_out, ffn_shexp);
        self.cb(cur, "ffn_out", il);
    }

    cur = self.build_cvec(cur, il);
    self.cb(cur, "l_out", il);

    return cur;
}

llama_model_nemotron_h::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {

    GGML_ASSERT(hparams.nextn_predict_layers > 0 && "Nemotron-H MTP requires nextn_predict_layers > 0");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int il = (int)hparams.n_layer_all; // first MTP block index
    const auto & layer = model.layers[il];

    // Verify required NextN tensors exist
    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm && "MTP block missing nextn.hnorm");

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // Build input embeddings
    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "mtp_inp_embd", il);

    // Create input with tokens, embeddings, and hidden state
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), ubatch.n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, ubatch.n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Input for next layer
    ggml_tensor * inpSA = inpL;

    // Get hidden state and normalize with hnorm
    cur = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_hnorm", il);

    // Get token embeddings and normalize with enorm
    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    // Concatenate and project with eh_proj
    ggml_tensor * concat = ggml_concat(ctx0, e_norm, cur, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    cur = build_lora_mm(layer.nextn.eh_proj, concat);
    cb(cur, "mtp_eh_proj", il);

    // Residual connection from input
    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_residual", il);

    // Norm
    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    // Attention
    if (hparams.n_ff(il) == 0) {
        // Attention block
        auto * inp_attn = build_attn_inp_kv();
        cur = graph::build_attention_layer(*this, cur, inp_attn, model, n_embd_head, il);
    } else {
        GGML_ASSERT(false && "MTP first block should be attention");
    }

    // Residual
    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    // Save for FFN residual
    ggml_tensor * ffn_residual = cur;

    // MoE FFN
    cur = graph::build_ffn_layer(*this, cur, model, il);
    cb(cur, "mtp_ffn_out", il);

    // Residual for FFN
    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_ffn_residual", il);

    cur = build_cvec(cur, il);
    cb(cur, "mtp_l_out", il);

    // Final norm
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "mtp_h_nextn", -1);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "mtp_result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur);
    cb(cur, "mtp_result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
