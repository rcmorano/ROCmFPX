#include "models.h"

#include <algorithm>

void llama_model_nemotron_h::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // NextN/MTP (Puzzle)
    // NOTE: hparams.nextn_predict_layers is already loaded in load_hparams() base.
    // Set nextn_predict_per_head if nextn > 0.
    if (hparams.nextn_predict_layers > 0) {
        hparams.nextn_predict_per_head = hparams.nextn_predict_layers;
    }

    // A layer is recurrent IFF the n_head_kv value is set to 0 and
    // the n_ff value is set to 0
    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        hparams.recurrent_layer_arr[i] = (hparams.n_head_kv(i) == 0 && hparams.n_ff(i) == 0);
    }

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer, false);
    hparams.n_ff_exp_impl = 0;
    for (uint32_t _il = 0; _il < hparams.n_layer; ++_il) {
        hparams.n_ff_exp_impl = std::max(hparams.n_ff_exp_impl, hparams.n_ff_exp_arr[_il]);
    }
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp,      false);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_MOE_LATENT_SIZE,                   hparams.moe_latent_size, false);

    switch (hparams.n_layer - hparams.nextn_predict_layers) {
        case 52: type = LLM_TYPE_31B_A3_5B; break;
        case 56: type = LLM_TYPE_9B; break;
        case 88:
            {
                bool heterogeneous = false;
                for (uint32_t i = 1; i < (uint32_t)(hparams.n_layer - hparams.nextn_predict_layers); ++i) {
                    heterogeneous |= hparams.n_expert_used_arr[i] != hparams.n_expert_used_arr[0];
                }
                type = heterogeneous ? LLM_TYPE_75B_A9B : LLM_TYPE_120B_A12B;
            } break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_nemotron_h::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t d_conv     = hparams.ssm_d_conv;
    const int64_t d_inner    = hparams.ssm_d_inner;
    const int64_t d_state    = hparams.ssm_d_state;
    const int64_t n_ssm_head = hparams.ssm_dt_rank;
    const int64_t n_group    = hparams.ssm_n_group;
    const int64_t d_in_proj  = 2*d_inner + 2*n_group*d_state + n_ssm_head;
    const int64_t moe_n_embd = hparams.moe_latent_size > 0 ? hparams.moe_latent_size : n_embd;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    {
        output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
        if (output == NULL) {
            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm  = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        if (hparams.is_recurrent(i)) {
            layer.ssm_in = create_tensor(tn(LLM_TENSOR_SSM_IN, "weight", i), {n_embd, d_in_proj}, 0);

            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, d_inner + 2*n_group*d_state}, 0);
            layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias", i), {d_inner + 2*n_group*d_state}, TENSOR_NOT_REQUIRED);

            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {n_ssm_head}, 0);
            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_ssm_head}, 0);
            layer.ssm_d = create_tensor(tn(LLM_TENSOR_SSM_D, i), {1, n_ssm_head}, 0);
            layer.ssm_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {d_inner / n_group, n_group}, 0);
            layer.ssm_out = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", i), {d_inner, n_embd}, 0);
        } else if (hparams.n_ff(i) == 0) {
            const int64_t n_head_i = hparams.n_head(i);
            const int64_t n_embd_k_gqa_i = hparams.n_embd_k_gqa(i);
            const int64_t n_embd_v_gqa_i = hparams.n_embd_v_gqa(i);
            create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head_i, n_embd_k_gqa_i, n_embd_v_gqa_i, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head_i, n_embd}, 0);
            layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
        } else {
            if (n_expert != 0) {
                const auto & hp = hparams;
                const int64_t n_ff_exp_i = hp.n_ff_exp(i)
                    ? (int64_t)hp.n_ff_exp(i)
                    : hp.n_ff(i) / (int64_t)hp.n_expert_used(i);
                const int64_t n_ff_shexp = hparams.n_ff_shexp;

                layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), { n_embd, n_expert}, 0);
                layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);

                layer.ffn_latent_down = create_tensor(tn(LLM_TENSOR_FFN_LATENT_DOWN, "weight", i), {n_embd, moe_n_embd}, TENSOR_NOT_REQUIRED);
                layer.ffn_latent_up   = create_tensor(tn(LLM_TENSOR_FFN_LATENT_UP,   "weight", i), {moe_n_embd, n_embd}, TENSOR_NOT_REQUIRED);

                layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp_i, moe_n_embd, n_expert}, 0);
                layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {moe_n_embd, n_ff_exp_i, n_expert}, 0);

                layer.ffn_down_shexp  = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, 0);
                layer.ffn_up_shexp    = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, 0);
            } else {
                layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {hparams.n_ff(i), n_embd}, 0);
                layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, hparams.n_ff(i)}, 0);
                layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
                layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), {hparams.n_ff(i)}, TENSOR_NOT_REQUIRED);
            }
        }

        const int n_trunk = n_layer - (int)hparams.nextn_predict_layers;
        if ((int)hparams.nextn_predict_layers > 0 && i == n_trunk) {
            layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), {2*n_embd, n_embd}, 0);
            layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,  "weight", i), {n_embd}, 0);
            layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,  "weight", i), {n_embd}, 0);
        }
        if ((int)hparams.nextn_predict_layers > 0 && i == n_layer - 1) {
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_nemotron_h::build_arch_graph(const llm_graph_params & params) const {
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

    for (int il = 0; il < (int)(n_layer - hparams.nextn_predict_layers); ++il) {
        struct ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (hparams.is_recurrent(il)) {
            cur = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);
        } else if (hparams.n_ff(il) == 0) {
            cur = build_attention_layer(*this, cur, inp->get_attn(), model, n_embd_head, il);
        } else {
            cur = build_ffn_layer(*this, cur, model, il);
        }

        if (il == (int)(n_layer - hparams.nextn_predict_layers) - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "nemotron_h_block_out", il);
        inpL = cur;
    }

    cur = inpL;
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llama_model_nemotron_h::graph::build_attention_layer(llm_graph_context & self,
                                                          ggml_tensor * cur,
                                                          llm_graph_input_attn_kv * inp_attn,
                                                          const llama_model & model,
                                                          int64_t n_embd_head,
                                                          int il) {
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

        const auto & hp = self.hparams;
        ggml_tensor * moe_out =
            self.build_moe_ffn(inp_latent,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    nullptr,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    self.n_expert, (int64_t) hp.n_expert_used(il),
                    LLM_FFN_RELU_SQR, self.hparams.expert_weights_norm,
                    self.hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
                    il,
                    router_logits, nullptr,
                    model.layers[il].ffn_up_exps_s,
                    nullptr,
                    model.layers[il].ffn_down_exps_s);
        self.cb(moe_out, "ffn_moe_out", il);

        if (model.layers[il].ffn_latent_up) {
            moe_out = ggml_mul_mat(self.ctx0, model.layers[il].ffn_latent_up, moe_out);
        }

        ggml_tensor * ffn_shexp = self.build_ffn(inp_emb,
                    model.layers[il].ffn_up_shexp,   NULL, model.layers[il].ffn_up_shexp_s,
                    NULL,                             NULL, NULL,
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
    GGML_ASSERT(hparams.nextn_predict_layers == 2 &&
            "NEMOTRON_H_MOE MTP requires exactly 2 appended sub-blocks (attention + moe)");
    GGML_ASSERT(cparams.nextn_layer_offset == 0 &&
            "NEMOTRON_H_MOE MTP is a single step made of 2 sub-blocks, not independent chained heads");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int n_trunk = n_layer - (int)hparams.nextn_predict_layers;
    const int il_attn = n_trunk;
    const int il_moe  = n_layer - 1;

    const auto & layer_attn = model.layers[il_attn];
    const auto & layer_moe  = model.layers[il_moe];

    GGML_ASSERT(layer_attn.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer_attn.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer_attn.nextn.hnorm   && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer_attn.wq && layer_attn.wk && layer_attn.wv && layer_attn.wo && "MTP attention sub-block missing tensors");
    GGML_ASSERT(layer_moe.ffn_gate_inp   && "MTP moe sub-block missing ffn_gate_inp");

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer_attn.nextn.embed_tokens ? layer_attn.nextn.embed_tokens : model.tok_embd;
        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il_attn);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * e_norm = build_norm(tok_embd, layer_attn.nextn.enorm, nullptr, LLM_NORM_RMS, il_attn);
    cb(e_norm, "mtp_enorm", il_attn);

    ggml_tensor * h_norm = build_norm(h_embd, layer_attn.nextn.hnorm, nullptr, LLM_NORM_RMS, il_attn);
    cb(h_norm, "mtp_hnorm", il_attn);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, 0);
    cb(concat, "mtp_concat", il_attn);

    ggml_tensor * cur = build_lora_mm(layer_attn.nextn.eh_proj, concat, layer_attn.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il_attn);

    ggml_tensor * inpSA = cur;
    cur = build_norm(cur, layer_attn.attn_norm, nullptr, LLM_NORM_RMS, il_attn);
    cb(cur, "mtp_attn_norm", il_attn);

    cur = graph::build_attention_layer(*this, cur, inp_attn, model, n_embd_head, il_attn);
    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il_attn);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer_moe.attn_norm, nullptr, LLM_NORM_RMS, il_moe);
    cb(cur, "mtp_ffn_norm", il_moe);

    cur = graph::build_ffn_layer(*this, cur, model, il_moe);
    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il_moe);

    ggml_tensor * head_norm_w = layer_moe.nextn.shared_head_norm ? layer_moe.nextn.shared_head_norm : model.output_norm;
    GGML_ASSERT(head_norm_w && "NEMOTRON_H_MOE MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_pre_norm = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer_moe.nextn.shared_head_head ? layer_moe.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_w && "NEMOTRON_H_MOE MTP: missing LM head");
    cur = build_lora_mm(head_w, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
