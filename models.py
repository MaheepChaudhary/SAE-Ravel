from huggingface_hub import hf_hub_download
from transformer_lens.hook_points import HookedRootModule, HookPoint

from eval_gpt2 import *
from imports import *
from ravel_data_prep import *

# torch.autograd.set_detect_anomaly(True)
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
# SAVE_DIR = Path("/workspace/1L-Sparse-Autoencoder/checkpoints")


# This would be used for the OpenAI GPT-2 model
class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["dict_size"]
        l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(cfg["act_size"], d_hidden, dtype=dtype)
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(d_hidden, cfg["act_size"], dtype=dtype)
            )
        )
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        self.to(cfg["device"])

    # inserted mask here
    def forward(self, x_base, x_source, mask, token_intervened_idx, bs):
        x_base_cent = x_base - self.b_dec
        x_source_cent = x_source - self.b_dec
        base_acts = F.relu(x_base_cent @ self.W_enc + self.b_enc)
        source_acts = F.relu(x_source_cent @ self.W_enc + self.b_enc)

        # Create a new tensor by modifying only the token_intervened_idx dimensions
        base_new_acts = (
            base_acts.clone()
        )  # Clone the original tensor to avoid modifying it in-place
        source_new_acts = source_acts.clone()

        # Ensure token_intervened_idx is a tensor
        if isinstance(token_intervened_idx, int):
            token_intervened_idx = torch.tensor(
                [token_intervened_idx], dtype=torch.long
            )

        # Reshape mask to match the new_acts dimensions
        mask = mask.view(-1, 6144)

        base_new_acts[:, token_intervened_idx, :] = (
            base_new_acts[:, token_intervened_idx, :] * mask
        )
        source_new_acts[:, token_intervened_idx, :] = (1 - mask) * source_new_acts[
            :, token_intervened_idx, :
        ]

        new_acts = base_new_acts + source_new_acts

        x_reconstruct = new_acts @ self.W_dec + self.b_dec
        return x_reconstruct
        # l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0).save()
        # l1_loss = self.l1_coeff * (acts.float().abs().sum())
        # loss = l2_loss + l1_loss
        # return loss, x_reconstruct, new_acts, l2_loss, l1_loss

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed

    def get_version(self):
        version_list = [
            int(file.name.split(".")[0])
            for file in list(SAVE_DIR.iterdir())
            if "pt" in str(file)
        ]
        if len(version_list):
            return 1 + max(version_list)
        else:
            return 0

    def save(self):
        version = self.get_version()
        torch.save(self.state_dict(), SAVE_DIR / (str(version) + ".pt"))
        with open(SAVE_DIR / (str(version) + "_cfg.json"), "w") as f:
            json.dump(cfg, f)
        print("Saved as version", version)

    @classmethod
    def load(cls, version):
        cfg = json.load(open(SAVE_DIR / (str(version) + "_cfg.json"), "r"))
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(SAVE_DIR / (str(version) + ".pt")))
        return self

    @classmethod
    def load_from_hf(cls, version, device_override=None):
        """
        Loads the saved autoencoder from HuggingFace.

        Version is expected to be an int, or "run1" or "run2"

        version 25 is the final checkpoint of the first autoencoder run,
        version 47 is the final checkpoint of the second autoencoder run.
        """
        if version == "run1":
            version = 25
        elif version == "run2":
            version = 47

        cfg = utils.download_file_from_hf(
            "NeelNanda/sparse_autoencoder", f"{version}_cfg.json"
        )
        if device_override is not None:
            cfg["device"] = device_override

        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(
            utils.download_file_from_hf(
                "NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True
            )
        )
        return self


class RotateLayer(t.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, init_orth=True):
        super().__init__()
        weight = t.empty(n, n)
        if init_orth:
            t.nn.init.orthogonal_(weight)
        self.weight = t.nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        return t.matmul(x.to(self.weight.dtype), self.weight)


class my_model(nn.Module):
    def __init__(
        self,
        model,
        DEVICE,
        method,
        expansion_factor,
        token_length_allowed,
        layer_intervened,
        intervened_token_idx,
        batch_size,
    ) -> None:
        super(my_model, self).__init__()

        self.model = model
        self.layer_intervened = t.tensor(layer_intervened, dtype=t.int32, device=DEVICE)
        self.intervened_token_idx = t.tensor(
            intervened_token_idx, dtype=t.int32, device=DEVICE
        )
        self.intervened_token_idx = intervened_token_idx
        self.expansion_factor = expansion_factor
        self.token_length_allowed = token_length_allowed
        self.method = method
        self.batch_size = batch_size

        self.DEVICE = DEVICE

        if method == "sae masking openai":
            open_sae_dim = (1, 1, 32768)
            state_dict = t.load(
                f"openai_sae/downloaded_saes/{self.layer_intervened}.pt"
            )
            self.autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict
            )
            self.l4_mask = t.nn.Parameter(
                t.zeros(open_sae_dim, device=DEVICE), requires_grad=True
            )
            for params in self.autoencoder.parameters():
                params.requires_grad = False

        elif method == "sae masking neel":
            neel_sae_dim = (1, 24576)
            self.l4_mask = t.nn.Parameter(
                t.zeros(neel_sae_dim, device=DEVICE), requires_grad=True
            )

            self.sae_neel, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.{self.layer_intervened}.hook_resid_post",  # won't always be a hook point
            )
            for params in self.sae_neel.parameters():
                params.requires_grad = False

        elif method == "sae masking apollo":
            apollo_sae_dim = (1, 46080)
            self.l4_mask = t.nn.Parameter(
                t.zeros(apollo_sae_dim, device=DEVICE), requires_grad=True
            )
            self.sae_apollo = SAETransformer.from_wandb("sparsify/gpt2/vnfh4vpi")

            for params in self.sae_apollo.parameters():
                params.requires_grad = False

        elif method == "neuron masking":
            # neuron_dim = (1,self.token_length_allowed, 768)
            neuron_dim = (1, 768)
            self.l4_mask = t.nn.Parameter(
                t.zeros(neuron_dim, device=DEVICE), requires_grad=True
            )
            self.l4_mask = self.l4_mask.to(DEVICE)

        elif method == "das masking":
            das_dim = (1, 768)
            self.l4_mask = t.nn.Parameter(
                t.zeros(das_dim, device=DEVICE), requires_grad=True
            )
            rotate_layer = RotateLayer(768)
            self.rotate_layer = t.nn.utils.parametrizations.orthogonal(rotate_layer)

        elif method == "vanilla":
            proxy_dim = (1, 1, 1)
            self.proxy = t.nn.Parameter(t.zeros(proxy_dim), requires_grad=True)

    def forward(self, source_ids, base_ids, temperature):
        l4_mask_sigmoid = t.sigmoid(self.l4_mask / temperature)
        # l4_mask_sigmoid = self.l4_mask
        if self.method == "neuron masking":
            with self.model.trace() as tracer:

                with tracer.invoke(source_ids) as runner:
                    vector_source = self.model.transformer.h[
                        self.layer_intervened
                    ].output[0][0]

                with tracer.invoke(base_ids) as runner_:
                    intermediate_output = (
                        self.model.transformer.h[self.layer_intervened]
                        .output[0]
                        .clone()
                    )
                    intermediate_output = (1 - l4_mask_sigmoid) * intermediate_output[
                        :, self.intervened_token_idx, :
                    ] + l4_mask_sigmoid * vector_source[:, self.intervened_token_idx, :]
                    assert (
                        intermediate_output.squeeze(1).shape
                        == vector_source[:, self.intervened_token_idx, :].shape
                        == torch.Size([self.batch_size, 768])
                    )
                    self.model.transformer.h[self.layer_intervened].output[0][0][
                        :, self.intervened_token_idx, :
                    ] = intermediate_output.squeeze(1)
                    # self.model.transformer.h[self.layer_intervened].output[0][0][:,self.intervened_token_idx,:] = vector_source[:,self.intervened_token_idx,:]

                    intervened_base_predicted = self.model.lm_head.output.argmax(
                        dim=-1
                    ).save()
                    intervened_base_output = self.model.lm_head.output.save()

            predicted_text = []
            for index in range(intervened_base_output.shape[0]):
                predicted_text.append(
                    self.model.tokenizer.decode(
                        intervened_base_output[index].argmax(dim=-1)
                    ).split()[-1]
                )

            return intervened_base_output, predicted_text

        elif self.method == "das masking":

            with self.model.trace() as tracer:

                with tracer.invoke(source_ids) as runner:

                    vector_source = (
                        self.model.transformer.h[self.layer_intervened].output[0].save()
                    )

                with tracer.invoke(base_ids) as runner_:

                    intermediate_output = (
                        self.model.transformer.h[self.layer_intervened]
                        .output[0][0]
                        .save()
                    )
                    # print("Intermediate shape",intermediate_output.shape)

                    # das
                    assert (
                        vector_source[0][:, self.intervened_token_idx, :].shape
                        == intermediate_output[:, self.intervened_token_idx, :].shape
                        == torch.Size([self.batch_size, 768])
                    )
                    vector_source_rotated = self.rotate_layer(
                        vector_source[0][:, self.intervened_token_idx, :]
                    )
                    cloned_intermediate_output = intermediate_output.clone()
                    intermediate_output_rotated = self.rotate_layer(
                        cloned_intermediate_output[:, self.intervened_token_idx, :]
                    )
                    assert (
                        intermediate_output_rotated.shape
                        == vector_source_rotated.shape
                        == torch.Size([self.batch_size, 768])
                    )
                    assert l4_mask_sigmoid.shape == torch.Size([1, 768])
                    masked_intermediate_output_rotated = (
                        1 - l4_mask_sigmoid
                    ) * intermediate_output_rotated
                    masked_vector_source_rotated = (
                        l4_mask_sigmoid * vector_source_rotated
                    )
                    assert (
                        masked_intermediate_output_rotated.shape
                        == masked_vector_source_rotated.shape
                        == torch.Size([self.batch_size, 768])
                    )

                    iia_vector_rotated = (
                        masked_intermediate_output_rotated
                        + masked_vector_source_rotated
                    )
                    assert iia_vector_rotated.shape == torch.Size(
                        [self.batch_size, 768]
                    )

                    # masked_intermediate_output_unrotated = torch.matmul(masked_intermediate_output_rotated,self.rotate_layer.weight.T)
                    # masked_vector_source_unrotated = torch.matmul(masked_vector_source_rotated,self.rotate_layer.weight.T)

                    iia_vector = torch.matmul(
                        iia_vector_rotated, self.rotate_layer.weight.T
                    )

                    # iia_vector = masked_intermediate_output_unrotated + masked_vector_source_unrotated

                    # intermediate_output = (1 - self.l4_mask) * intermediate_output[:,self.intervened_token_idx,:].unsqueeze(0) + self.l4_mask * vector_source[0][:,self.intervened_token_idx,:].unsqueeze(0)
                    iia_vector = iia_vector.reshape(-1, 1, 768)
                    assert (
                        (iia_vector).shape
                        == vector_source[0][:, self.intervened_token_idx, :]
                        .unsqueeze(1)
                        .shape
                        == torch.Size([self.batch_size, 1, 768])
                    )
                    # Create a new tuple with the modified intermediate_output
                    # modified_output = (intermediate_output,) + self.model.transformer.h[self.layer_intervened].output[0][1:]
                    assert (
                        self.model.transformer.h[self.layer_intervened]
                        .output[0][:, self.intervened_token_idx, :]
                        .shape
                        == iia_vector.squeeze(1).shape
                        == torch.Size([self.batch_size, 768])
                    )
                    self.model.transformer.h[self.layer_intervened].output[0][0][
                        :, self.intervened_token_idx, :
                    ] = iia_vector.squeeze(1)
                    intervened_base_predicted = self.model.lm_head.output.argmax(
                        dim=-1
                    ).save()
                    intervened_base_output = self.model.lm_head.output.save()

            predicted_text = []
            for index in range(intervened_base_output.shape[0]):
                predicted_text.append(
                    self.model.tokenizer.decode(
                        intervened_base_output[index].argmax(dim=-1)
                    ).split()[-1]
                )

            return intervened_base_output, predicted_text

        elif self.method == "sae masking neel":

            with self.model.trace() as tracer:

                with tracer.invoke(source_ids) as runner:

                    source = self.model.transformer.h[self.layer_intervened].output[0]

                with tracer.invoke(base_ids) as runner_:

                    base = (
                        self.model.transformer.h[self.layer_intervened]
                        .output[0]
                        .clone()
                    )
                    encoded_base = self.sae_neel.encode(base)
                    encoded_source = self.sae_neel.encode(source)
                    summed = (1 - l4_mask_sigmoid) * encoded_base[
                        :, self.intervened_token_idx, :
                    ] + l4_mask_sigmoid * encoded_source[
                        :, self.intervened_token_idx, :
                    ]
                    decoded_vector = self.sae_neel.decode(summed)

                    self.model.transformer.h[self.layer_intervened].output[0][0][
                        :, self.intervened_token_idx, :
                    ] = decoded_vector

                    intervened_base_predicted = self.model.lm_head.output.argmax(
                        dim=-1
                    ).save()
                    intervened_base_output = self.model.lm_head.output.save()

            predicted_text = []
            for index in range(intervened_base_output.shape[0]):
                predicted_text.append(
                    self.model.tokenizer.decode(
                        intervened_base_output[index].argmax(dim=-1)
                    ).split()[-1]
                )

            return intervened_base_output, predicted_text

        elif self.method == "sae masking apollo":

            with self.model.trace() as tracer:

                with tracer.invoke(source_ids) as runner:
                    source = (
                        self.model.transformer.h[self.layer_intervened]
                        .output[0]
                        .clone()
                    )

                with tracer.invoke(base_ids) as runner_:

                    base = (
                        self.model.transformer.h[self.layer_intervened]
                        .output[0]
                        .clone()
                    )
                    encoded_base = self.sae_apollo.saes[
                        "blocks-10-hook_resid_pre"
                    ].encoder(base)
                    encoded_source = self.sae_apollo.saes[
                        "blocks-10-hook_resid_pre"
                    ].encoder(source)

                    # Clone the tensors to avoid in-place operations
                    encoded_base_modified = encoded_base.clone()
                    encoded_source_modified = encoded_source.clone()

                    # Apply the mask in a non-inplace way
                    modified_base = encoded_base_modified[
                        :, self.intervened_token_idx, :
                    ] * (1 - l4_mask_sigmoid)
                    modified_source = (
                        encoded_source_modified[:, self.intervened_token_idx, :]
                        * l4_mask_sigmoid
                    )

                    # Assign the modified tensors to the correct indices
                    encoded_base_modified = encoded_base_modified.clone()
                    encoded_source_modified = encoded_source_modified.clone()

                    # Combine the modified tensors
                    new_acts = encoded_base_modified.clone()
                    new_acts[:, self.intervened_token_idx, :] = (
                        modified_base + modified_source
                    )

                    iia_vector = self.sae_apollo.saes[
                        "blocks-10-hook_resid_pre"
                    ].decoder(new_acts)

                    # Use a copy to avoid in-place modification
                    h_layer_output_copy = (
                        self.model.transformer.h[self.layer_intervened]
                        .output[0]
                        .clone()
                    )
                    h_layer_output_copy[:, self.intervened_token_idx, :] = iia_vector[
                        :, self.intervened_token_idx, :
                    ]

                    # Update the model's output with the modified copy
                    self.model.transformer.h[self.layer_intervened].output[0][
                        :, :, :
                    ] = h_layer_output_copy

                    intervened_base_predicted = self.model.lm_head.output.argmax(
                        dim=-1
                    ).save()
                    intervened_base_output = self.model.lm_head.output.save()

            predicted_text = []
            for index in range(intervened_base_output.shape[0]):
                predicted_text.append(
                    self.model.tokenizer.decode(
                        intervened_base_output[index].argmax(dim=-1)
                    ).split()[-1]
                )

            return intervened_base_output, predicted_text

        elif self.method == "sae masking openai":

            with self.model.trace() as tracer:

                with tracer.invoke(source_ids) as runner:
                    source = self.model.transformer.h[self.layer_intervened].output[0]

                with tracer.invoke(base_ids) as runner_:

                    base = (
                        self.model.transformer.h[self.layer_intervened]
                        .output[0]
                        .clone()
                    )
                    encoded_base, base_info = self.autoencoder.encode(base)
                    encoded_source, source_info = self.autoencoder.encode(source[0])

                    # Clone the tensors to avoid in-place operations
                    encoded_base_modified = encoded_base.clone()
                    encoded_source_modified = encoded_source.clone()

                    assert base_info == source_info

                    # Apply the mask in a non-inplace way
                    modified_base = encoded_base_modified[
                        :, self.intervened_token_idx, :
                    ] * (1 - l4_mask_sigmoid)
                    modified_source = (
                        encoded_source_modified[:, self.intervened_token_idx, :]
                        * l4_mask_sigmoid
                    )

                    # Assign the modified tensors to the correct indices
                    encoded_base_modified = encoded_base_modified.clone()
                    encoded_source_modified = encoded_source_modified.clone()

                    # Combine the modified tensors
                    new_acts = encoded_base_modified.clone()
                    new_acts[:, self.intervened_token_idx, :] = (
                        modified_base + modified_source
                    )

                    iia_vector = self.autoencoder.decode(new_acts, base_info)

                    # Use a copy to avoid in-place modification
                    h_layer_output_copy = (
                        self.model.transformer.h[self.layer_intervened]
                        .output[0]
                        .clone()
                    )
                    h_layer_output_copy[:, self.intervened_token_idx, :] = iia_vector[
                        :, self.intervened_token_idx, :
                    ]

                    # Update the model's output with the modified copy
                    self.model.transformer.h[self.layer_intervened].output[0][0][
                        :, :, :
                    ] = h_layer_output_copy

                    intervened_base_predicted = self.model.lm_head.output.argmax(
                        dim=-1
                    ).save()
                    intervened_base_output = self.model.lm_head.output.save()

            predicted_text = []
            for index in range(intervened_base_output.shape[0]):
                predicted_text.append(
                    self.model.tokenizer.decode(
                        intervened_base_output[index].argmax(dim=-1)
                    ).split()[-1]
                )

            return intervened_base_output, predicted_text

        elif self.method == "vanilla":
            intervened_token_idx = -8
            with self.model.trace() as tracer:

                with tracer.invoke(source_ids) as runner:

                    vector_source = self.model.transformer.h[
                        self.layer_intervened
                    ].output[0]

                with tracer.invoke(base_ids) as runner_:

                    self.model.transformer.h[self.layer_intervened].output[0][0][
                        :, intervened_token_idx, :
                    ] = vector_source[0][:, intervened_token_idx, :]

                    intervened_base_predicted = self.model.lm_head.output.argmax(
                        dim=-1
                    ).save()
                    intervened_base_output = self.model.lm_head.output.save()

            predicted_text = self.model.tokenizer.decode(
                intervened_base_predicted[0][-1]
            )

            return intervened_base_output, predicted_text


class eval_sae(nn.Module):
    def __init__(
        self,
        model,
        DEVICE,
        method,
        intervened_token_idx,
        batch_size,
    ) -> None:
        super(eval_sae, self).__init__()

        self.model = model
        self.intervened_token_idx = t.tensor(
            intervened_token_idx, dtype=t.int32, device=DEVICE
        )
        self.intervened_token_idx = intervened_token_idx
        self.method = method
        self.batch_size = batch_size

        self.DEVICE = DEVICE

        if method == "sae masking openai":
            state_dict0 = t.load(f"openai_sae/downloaded_saes/0.pt")
            self.sae_openai0 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict0
            )
            for params in self.sae_openai0.parameters():
                params.requires_grad = False

            state_dict1 = t.load(f"openai_sae/downloaded_saes/1.pt")
            self.sae_openai1 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict1
            )
            for params in self.sae_openai1.parameters():
                params.requires_grad = False

            state_dict2 = t.load(f"openai_sae/downloaded_saes/2.pt")
            self.sae_openai2 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict2
            )
            for params in self.sae_openai2.parameters():
                params.requires_grad = False

            state_dict3 = t.load(f"openai_sae/downloaded_saes/3.pt")
            self.sae_openai3 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict3
            )
            for params in self.sae_openai3.parameters():
                params.requires_grad = False

            state_dict4 = t.load(f"openai_sae/downloaded_saes/4.pt")
            self.sae_openai4 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict4
            )
            for params in self.sae_openai4.parameters():
                params.requires_grad = False

            state_dict5 = t.load(f"openai_sae/downloaded_saes/5.pt")
            self.sae_openai5 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict5
            )
            for params in self.sae_openai5.parameters():
                params.requires_grad = False

            state_dict6 = t.load(f"openai_sae/downloaded_saes/6.pt")
            self.sae_openai6 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict6
            )
            for params in self.sae_openai6.parameters():
                params.requires_grad = False

            state_dict7 = t.load(f"openai_sae/downloaded_saes/7.pt")
            self.sae_openai7 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict7
            )
            for params in self.sae_openai7.parameters():
                params.requires_grad = False

            state_dict8 = t.load(f"openai_sae/downloaded_saes/8.pt")
            self.sae_openai8 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict8
            )
            for params in self.sae_openai8.parameters():
                params.requires_grad = False

            state_dict9 = t.load(f"openai_sae/downloaded_saes/9.pt")
            self.sae_openai9 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict9
            )
            for params in self.sae_openai9.parameters():
                params.requires_grad = False

            state_dict10 = t.load(f"openai_sae/downloaded_saes/10.pt")
            self.sae_openai10 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict10
            )
            for params in self.sae_openai10.parameters():
                params.requires_grad = False

            state_dict11 = t.load(f"openai_sae/downloaded_saes/11.pt")
            self.sae_openai11 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict11
            )
            for params in self.sae_openai11.parameters():
                params.requires_grad = False

        elif method == "sae masking neel":

            self.sae_neel0, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.1.hook_resid_pre",  # won't always be a hook point
            )
            for params in self.sae_neel0.parameters():
                params.requires_grad = False

            self.sae_neel1, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.2.hook_resid_pre",  # won't always be a hook point
            )
            for params in self.sae_neel1.parameters():
                params.requires_grad = False

            self.sae_neel2, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.3.hook_resid_pre",  # won't always be a hook point
            )

            self.sae_neel3, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.4.hook_resid_pre",  # won't always be a hook point
            )

            self.sae_neel4, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.5.hook_resid_pre",  # won't always be a hook point
            )

            self.sae_neel5, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.6.hook_resid_pre",  # won't always be a hook point
            )
            for params in self.sae_neel2.parameters():
                params.requires_grad = False

            for params in self.sae_neel3.parameters():
                params.requires_grad = False

            for params in self.sae_neel4.parameters():
                params.requires_grad = False

            for params in self.sae_neel5.parameters():
                params.requires_grad = False

            self.sae_neel6, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.7.hook_resid_pre",  # won't always be a hook point
            )
            for params in self.sae_neel6.parameters():
                params.requires_grad = False

            self.sae_neel7, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.8.hook_resid_pre",  # won't always be a hook point
            )

            self.sae_neel8, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.9.hook_resid_pre",  # won't always be a hook point
            )

            self.sae_neel9, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.10.hook_resid_pre",  # won't always be a hook point
            )

            self.sae_neel10, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.11.hook_resid_pre",  # won't always be a hook point
            )
            for params in self.sae_neel7.parameters():
                params.requires_grad = False

            for params in self.sae_neel8.parameters():
                params.requires_grad = False

            for params in self.sae_neel9.parameters():
                params.requires_grad = False

            for params in self.sae_neel10.parameters():
                params.requires_grad = False

            self.sae_neel11, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.11.hook_resid_post",  # won't always be a hook point
            )
            for params in self.sae_neel11.parameters():
                params.requires_grad = False

        elif method == "sae masking apollo":
            self.apollo_sae_l2_e2eds = SAETransformer.from_wandb(
                "sparsify/gpt2/e26jflpq"
            )
            self.apollo_sae_l2 = SAETransformer.from_wandb("sparsify/gpt2/bst0prdd")
            self.apollo_sae_l6_e2eds = SAETransformer.from_wandb(
                "sparsify/gpt2/2lzle2f0"
            )
            self.apollo_sae_l6 = SAETransformer.from_wandb("sparsify/gpt2/tvj2owza")
            self.apollo_sae_l10_e2eds = SAETransformer.from_wandb(
                "sparsify/gpt2/u50mksr8"
            )
            self.apollo_sae_l10 = SAETransformer.from_wandb("sparsify/gpt2/vnfh4vpi")

            for params in self.apollo_sae_l2_e2eds.parameters():
                params.requires_grad = False
            for params in self.apollo_sae_l2.parameters():
                params.requires_grad = False
            for params in self.apollo_sae_l6_e2eds.parameters():
                params.requires_grad = False
            for params in self.apollo_sae_l6.parameters():
                params.requires_grad = False
            for params in self.apollo_sae_l10_e2eds.parameters():
                params.requires_grad = False
            for params in self.apollo_sae_l10.parameters():
                params.requires_grad = False

    def forward(self, x):  # where x is a tokenized sentence

        if self.method == "sae masking neel":
            with self.model.trace(x) as tracer:
                output_layer0 = self.model.transformer.h[0].output[0].clone().save()
                eout0 = self.sae_neel0.encode(output_layer0)
                dout0 = self.sae_neel0.decode(eout0).save()
                loss0 = (
                    (dout0[:, -8, :].float() - output_layer0[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer1 = self.model.transformer.h[1].output[0].save()
                eout1 = self.sae_neel1.encode(output_layer1)
                dout1 = self.sae_neel1.decode(eout1)
                loss1 = (
                    (dout1[:, -8, :].float() - output_layer1[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer2 = self.model.transformer.h[2].output[0].save()
                eout2 = self.sae_neel2.encode(output_layer2)
                dout2 = self.sae_neel2.decode(eout2)
                loss2 = (
                    (dout2[:, -8, :].float() - output_layer2[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer3 = self.model.transformer.h[3].output[0].save()
                eout3 = self.sae_neel3.encode(output_layer3)
                dout3 = self.sae_neel3.decode(eout3)
                loss3 = (
                    (dout3[:, -8, :].float() - output_layer3[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer4 = self.model.transformer.h[4].output[0].save()
                eout4 = self.sae_neel4.encode(output_layer4)
                dout4 = self.sae_neel4.decode(eout4)
                loss4 = (
                    (dout4[:, -8, :].float() - output_layer4[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer5 = self.model.transformer.h[5].output[0].save()
                eout5 = self.sae_neel5.encode(output_layer5)
                dout5 = self.sae_neel5.decode(eout5)
                loss5 = (
                    (dout5[:, -8, :].float() - output_layer5[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer6 = self.model.transformer.h[6].output[0].save()
                eout6 = self.sae_neel6.encode(output_layer6)
                dout6 = self.sae_neel6.decode(eout6)
                loss6 = (
                    (dout6[:, -8, :].float() - output_layer6[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer7 = self.model.transformer.h[7].output[0].save()
                eout7 = self.sae_neel7.encode(output_layer7)
                dout7 = self.sae_neel7.decode(eout7)
                loss7 = (
                    (dout7[:, -8, :].float() - output_layer7[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer8 = self.model.transformer.h[8].output[0].save()
                eout8 = self.sae_neel8.encode(output_layer8)
                dout8 = self.sae_neel8.decode(eout8)
                loss8 = (
                    (dout8[:, -8, :].float() - output_layer8[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer9 = self.model.transformer.h[9].output[0].save()
                eout9 = self.sae_neel9.encode(output_layer9)
                dout9 = self.sae_neel9.decode(eout9)
                loss9 = (
                    (dout9[:, -8, :].float() - output_layer9[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer10 = self.model.transformer.h[10].output[0].save()
                eout10 = self.sae_neel10.encode(output_layer10)
                dout10 = self.sae_neel10.decode(eout10)
                loss10 = (
                    (dout10[:, -8, :].float() - output_layer10[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer11 = self.model.transformer.h[11].output[0].save()
                eout11 = self.sae_neel11.encode(output_layer11)
                dout11 = self.sae_neel11.decode(eout11)
                loss11 = (
                    (dout11[:, -8, :].float() - output_layer11[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

        elif self.method == "sae masking openai":
            with self.model.trace(x) as tracer:
                output_layer0 = self.model.transformer.h[0].output[0].save()
                eout0, info = self.sae_openai0.encode(output_layer0)
                dout0 = self.sae_openai0.decode(eout0, info)
                loss0 = (
                    (dout0[:, -8, :].float() - output_layer0[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer1 = self.model.transformer.h[1].output[0].save()
                eout1, info1 = self.sae_openai1.encode(output_layer1)
                dout1 = self.sae_openai1.decode(eout1, info1)
                loss1 = (
                    (dout1[:, -8, :].float() - output_layer1[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer2 = self.model.transformer.h[2].output[0].save()
                eout2, info2 = self.sae_openai2.encode(output_layer2)
                dout2 = self.sae_openai2.decode(eout2, info2)
                loss2 = (
                    (dout2[:, -8, :].float() - output_layer2[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer3 = self.model.transformer.h[3].output[0].save()
                eout3, info3 = self.sae_openai3.encode(output_layer3)
                dout3 = self.sae_openai3.decode(eout3, info3)
                loss3 = (
                    (dout3[:, -8, :].float() - output_layer3[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer4 = self.model.transformer.h[4].output[0].save()
                eout4, info4 = self.sae_openai4.encode(output_layer4)
                dout4 = self.sae_openai4.decode(eout4, info4)
                loss4 = (
                    (dout4[:, -8, :].float() - output_layer4[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer5 = self.model.transformer.h[5].output[0].save()
                eout5, info5 = self.sae_openai5.encode(output_layer5)
                dout5 = self.sae_openai5.decode(eout5, info5)
                loss5 = (
                    (dout5[:, -8, :].float() - output_layer5[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer6 = self.model.transformer.h[6].output[0].save()
                eout6, info6 = self.sae_openai6.encode(output_layer6)
                dout6 = self.sae_openai6.decode(eout6, info6)
                loss6 = (
                    (dout6[:, -8, :].float() - output_layer6[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer7 = self.model.transformer.h[7].output[0].save()
                eout7, info7 = self.sae_openai7.encode(output_layer7)
                dout7 = self.sae_openai7.decode(eout7, info7)
                loss7 = (
                    (dout7[:, -8, :].float() - output_layer7[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer8 = self.model.transformer.h[8].output[0].save()
                eout8, info8 = self.sae_openai8.encode(output_layer8)
                dout8 = self.sae_openai8.decode(eout8, info8)
                loss8 = (
                    (dout8[:, -8, :].float() - output_layer8[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer9 = self.model.transformer.h[9].output[0].save()
                eout9, info9 = self.sae_openai9.encode(output_layer9)
                dout9 = self.sae_openai9.decode(eout9, info9)
                loss9 = (
                    (dout9[:, -8, :].float() - output_layer9[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer10 = self.model.transformer.h[10].output[0].save()
                eout10, info10 = self.sae_openai10.encode(output_layer10)
                dout10 = self.sae_openai10.decode(eout10, info10)
                loss10 = (
                    (dout10[:, -8, :].float() - output_layer10[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer11 = self.model.transformer.h[11].output[0].save()
                eout11, info11 = self.sae_openai11.encode(output_layer11)
                dout11 = self.sae_openai11.decode(eout11, info11)
                loss11 = (
                    (dout11[:, -8, :].float() - output_layer11[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

        elif self.method == "sae masking apollo":
            with self.model.trace(x) as tracer:

                output_layer1 = self.model.transformer.h[1].output[0].save()
                eout1 = self.apollo_sae_l2.saes["blocks-2-hook_resid_pre"].encoder(
                    output_layer1
                )
                dout1 = self.apollo_sae_l2.saes["blocks-2-hook_resid_pre"].decoder(
                    eout1
                )
                loss1 = (
                    (dout1[:, -8, :].float() - output_layer1[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer1_e2eds = self.model.transformer.h[1].output[0].save()
                eout1_e2eds = self.apollo_sae_l2_e2eds.saes[
                    "blocks-2-hook_resid_pre"
                ].encoder(output_layer1_e2eds)
                dout1_e2eds = self.apollo_sae_l2_e2eds.saes[
                    "blocks-2-hook_resid_pre"
                ].decoder(eout1_e2eds)
                loss1_e2eds = (
                    (
                        dout1_e2eds[:, -8, :].float()
                        - output_layer1_e2eds[:, -8, :].float()
                    )
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer5 = self.model.transformer.h[5].output[0].save()
                eout5 = self.apollo_sae_l6.saes["blocks-6-hook_resid_pre"].encoder(
                    output_layer5
                )
                dout5 = self.apollo_sae_l6.saes["blocks-6-hook_resid_pre"].decoder(
                    eout5
                )
                loss5 = (
                    (dout5[:, -8, :].float() - output_layer5[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer5_e2eds = self.model.transformer.h[5].output[0].save()
                eout5_e2eds = self.apollo_sae_l6_e2eds.saes[
                    "blocks-6-hook_resid_pre"
                ].encoder(output_layer5_e2eds)
                dout5_e2eds = self.apollo_sae_l6_e2eds.saes[
                    "blocks-6-hook_resid_pre"
                ].decoder(eout5_e2eds)
                loss5_e2eds = (
                    (
                        dout5_e2eds[:, -8, :].float()
                        - output_layer5_e2eds[:, -8, :].float()
                    )
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer9 = self.model.transformer.h[9].output[0].save()
                eout9 = self.apollo_sae_l10.saes["blocks-10-hook_resid_pre"].encoder(
                    output_layer9
                )
                dout9 = self.apollo_sae_l10.saes["blocks-10-hook_resid_pre"].decoder(
                    eout9
                )
                loss9 = (
                    (dout9[:, -8, :].float() - output_layer9[:, -8, :].float())
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )

                output_layer9_e2eds = self.model.transformer.h[9].output[0].save()
                eout9_e2eds = self.apollo_sae_l10_e2eds.saes[
                    "blocks-10-hook_resid_pre"
                ].encoder(output_layer9_e2eds)
                dout9_e2eds = self.apollo_sae_l10_e2eds.saes[
                    "blocks-10-hook_resid_pre"
                ].decoder(eout9_e2eds)
                loss9_e2eds = (
                    (
                        dout9_e2eds[:, -8, :].float()
                        - output_layer9_e2eds[:, -8, :].float()
                    )
                    .pow(2)
                    .sum(-1)
                    .mean(0)
                    .save()
                )
                zeros = torch.zeros(loss1.shape)
                loss0 = loss1_e2eds
                loss4 = loss5_e2eds
                loss8 = loss9_e2eds
                loss2 = loss3 = loss6 = loss7 = loss10 = loss11 = zeros
        return (
            loss0,
            loss1,
            loss2,
            loss3,
            loss4,
            loss5,
            loss6,
            loss7,
            loss8,
            loss9,
            loss10,
            loss11,
        )


class eval_sae_acc(nn.Module):

    def __init__(self, model, DEVICE, method, intervened_token_idx, batch_size) -> None:
        super(eval_sae_acc, self).__init__()
        self.model = model
        self.intervened_token_idx = t.tensor(
            intervened_token_idx, dtype=t.int32, device=DEVICE
        )
        self.intervened_token_idx = intervened_token_idx
        self.method = method
        self.batch_size = batch_size

        self.DEVICE = DEVICE

        if method == "acc sae masking openai":
            state_dict0 = t.load(f"openai_sae/downloaded_saes/0.pt")
            self.sae_openai0 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict0
            )
            for params in self.sae_openai0.parameters():
                params.requires_grad = False

            state_dict1 = t.load(f"openai_sae/downloaded_saes/1.pt")
            self.sae_openai1 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict1
            )
            for params in self.sae_openai1.parameters():
                params.requires_grad = False

            state_dict2 = t.load(f"openai_sae/downloaded_saes/2.pt")
            self.sae_openai2 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict2
            )
            for params in self.sae_openai2.parameters():
                params.requires_grad = False

            state_dict3 = t.load(f"openai_sae/downloaded_saes/3.pt")
            self.sae_openai3 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict3
            )
            for params in self.sae_openai3.parameters():
                params.requires_grad = False

            state_dict4 = t.load(f"openai_sae/downloaded_saes/4.pt")
            self.sae_openai4 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict4
            )
            for params in self.sae_openai4.parameters():
                params.requires_grad = False

            state_dict5 = t.load(f"openai_sae/downloaded_saes/5.pt")
            self.sae_openai5 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict5
            )
            for params in self.sae_openai5.parameters():
                params.requires_grad = False

            state_dict6 = t.load(f"openai_sae/downloaded_saes/6.pt")
            self.sae_openai6 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict6
            )
            for params in self.sae_openai6.parameters():
                params.requires_grad = False

            state_dict7 = t.load(f"openai_sae/downloaded_saes/7.pt")
            self.sae_openai7 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict7
            )
            for params in self.sae_openai7.parameters():
                params.requires_grad = False

            state_dict8 = t.load(f"openai_sae/downloaded_saes/8.pt")
            self.sae_openai8 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict8
            )
            for params in self.sae_openai8.parameters():
                params.requires_grad = False

            state_dict9 = t.load(f"openai_sae/downloaded_saes/9.pt")
            self.sae_openai9 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict9
            )
            for params in self.sae_openai9.parameters():
                params.requires_grad = False

            state_dict10 = t.load(f"openai_sae/downloaded_saes/10.pt")
            self.sae_openai10 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict10
            )
            for params in self.sae_openai10.parameters():
                params.requires_grad = False

            state_dict11 = t.load(f"openai_sae/downloaded_saes/11.pt")
            self.sae_openai11 = sparse_autoencoder.Autoencoder.from_state_dict(
                state_dict11
            )
            for params in self.sae_openai11.parameters():
                params.requires_grad = False

        elif method == "acc sae masking neel":

            self.sae_neel0, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.1.hook_resid_pre",  # won't always be a hook point
            )
            for params in self.sae_neel0.parameters():
                params.requires_grad = False

            self.sae_neel1, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.2.hook_resid_pre",  # won't always be a hook point
            )
            for params in self.sae_neel1.parameters():
                params.requires_grad = False

            self.sae_neel2, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.3.hook_resid_pre",  # won't always be a hook point
            )

            self.sae_neel3, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.4.hook_resid_pre",  # won't always be a hook point
            )

            self.sae_neel4, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.5.hook_resid_pre",  # won't always be a hook point
            )

            self.sae_neel5, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.6.hook_resid_pre",  # won't always be a hook point
            )
            for params in self.sae_neel2.parameters():
                params.requires_grad = False

            for params in self.sae_neel3.parameters():
                params.requires_grad = False

            for params in self.sae_neel4.parameters():
                params.requires_grad = False

            for params in self.sae_neel5.parameters():
                params.requires_grad = False

            self.sae_neel6, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.7.hook_resid_pre",  # won't always be a hook point
            )
            for params in self.sae_neel6.parameters():
                params.requires_grad = False

            self.sae_neel7, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.8.hook_resid_pre",  # won't always be a hook point
            )

            self.sae_neel8, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.9.hook_resid_pre",  # won't always be a hook point
            )

            self.sae_neel9, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.10.hook_resid_pre",  # won't always be a hook point
            )

            self.sae_neel10, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.11.hook_resid_pre",  # won't always be a hook point
            )
            for params in self.sae_neel7.parameters():
                params.requires_grad = False

            for params in self.sae_neel8.parameters():
                params.requires_grad = False

            for params in self.sae_neel9.parameters():
                params.requires_grad = False

            for params in self.sae_neel10.parameters():
                params.requires_grad = False

            self.sae_neel11, cfg_dict, sparsity = SAE.from_pretrained(
                release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
                sae_id=f"blocks.11.hook_resid_post",  # won't always be a hook point
            )
            for params in self.sae_neel11.parameters():
                params.requires_grad = False

        elif method == "acc sae masking apollo":
            self.apollo_sae_l2_e2eds = SAETransformer.from_wandb(
                "sparsify/gpt2/e26jflpq"
            )
            self.apollo_sae_l2 = SAETransformer.from_wandb("sparsify/gpt2/bst0prdd")
            self.apollo_sae_l6_e2eds = SAETransformer.from_wandb(
                "sparsify/gpt2/2lzle2f0"
            )
            self.apollo_sae_l6 = SAETransformer.from_wandb("sparsify/gpt2/tvj2owza")
            self.apollo_sae_l10_e2eds = SAETransformer.from_wandb(
                "sparsify/gpt2/u50mksr8"
            )
            self.apollo_sae_l10 = SAETransformer.from_wandb("sparsify/gpt2/vnfh4vpi")

            for params in self.apollo_sae_l2_e2eds.parameters():
                params.requires_grad = False
            for params in self.apollo_sae_l2.parameters():
                params.requires_grad = False
            for params in self.apollo_sae_l6_e2eds.parameters():
                params.requires_grad = False
            for params in self.apollo_sae_l6.parameters():
                params.requires_grad = False
            for params in self.apollo_sae_l10_e2eds.parameters():
                params.requires_grad = False
            for params in self.apollo_sae_l10.parameters():
                params.requires_grad = False


    def forward(self, x):

        if self.method == "acc sae masking neel":

            layers_and_encoders = [
                (0, self.sae_neel0),
                (1, self.sae_neel1),
                (2, self.sae_neel2),
                (3, self.sae_neel3),
                (4, self.sae_neel4),
                (5, self.sae_neel5),
                (6, self.sae_neel6),
                (7, self.sae_neel7),
                (8, self.sae_neel8),
                (9, self.sae_neel9),
                (10, self.sae_neel10),
                (11, self.sae_neel11),
            ]

            output_dict = {}

            for layer, sae in layers_and_encoders:
                with self.model.trace(x) as tracer:
                    output_layer = (
                        self.model.transformer.h[layer].output[0].clone().save()
                    )
                    eout = sae.encode(output_layer)
                    dout = sae.decode(eout).save()
                    self.model.transformer.h[layer].output[0][:, -8, :] = dout[:, -8, :]
                    intervened_base_output = self.model.lm_head.output.save()

                predicted_text = []
                for index in range(intervened_base_output.shape[0]):
                    predicted_text.append(
                        self.model.tokenizer.decode(
                            intervened_base_output[index].argmax(dim=-1)
                        ).split()[-1]
                    )

                output_dict[f"Predicted_L{layer}"] = [
                    intervened_base_output,
                    predicted_text,
                ]

        elif self.method == "acc sae masking openai":

            layers_and_encoders = [
                (0, self.sae_openai0),
                (1, self.sae_openai1),
                (2, self.sae_openai2),
                (3, self.sae_openai3),
                (4, self.sae_openai4),
                (5, self.sae_openai5),
                (6, self.sae_openai6),
                (7, self.sae_openai7),
                (8, self.sae_openai8),
                (9, self.sae_openai9),
                (10, self.sae_openai10),
                (11, self.sae_openai11),
            ]

            output_dict = {}

            for layer, sae in layers_and_encoders:
                with self.model.trace(x) as tracer:
                    output_layer = (
                        self.model.transformer.h[layer].output[0].clone().save()
                    )
                    eout, info = sae.encode(output_layer)
                    dout = sae.decode(eout, info).save()
                    self.model.transformer.h[layer].output[0][:, -8, :] = dout[:, -8, :]
                    intervened_base_output = self.model.lm_head.output.save()

                predicted_text = []
                for index in range(intervened_base_output.shape[0]):
                    predicted_text.append(
                        self.model.tokenizer.decode(
                            intervened_base_output[index].argmax(dim=-1)
                        ).split()[-1]
                    )

                output_dict[f"Predicted_L{layer}"] = [
                    intervened_base_output,
                    predicted_text,
                ]

        elif self.method == "acc sae masking apollo":

            layers_and_encoders = [
                (2, 0, self.apollo_sae_l2_e2eds),
                (2, 1, self.apollo_sae_l2),
                (6, 2, self.apollo_sae_l6_e2eds),
                (6, 3, self.apollo_sae_l6),
                (10, 4, self.apollo_sae_l10_e2eds),
                (10, 5, self.apollo_sae_l10),
            ]

            output_dict = {}

            for layer, layer_, sae in layers_and_encoders:
                with self.model.trace(x) as tracer:
                    output_layer = (
                        self.model.transformer.h[layer].output[0].clone().save()
                    )
                    eout = sae.saes[f"blocks-{layer}-hook_resid_pre"].encoder(output_layer)
                    dout = sae.saes[f"blocks-{layer}-hook_resid_pre"].decoder(eout)
                    self.model.transformer.h[layer].output[0][:, -8, :] = dout[:, -8, :]
                    intervened_base_output = self.model.lm_head.output.save()

                predicted_text = []
                for index in range(intervened_base_output.shape[0]):
                    predicted_text.append(
                        self.model.tokenizer.decode(
                            intervened_base_output[index].argmax(dim=-1)
                        ).split()[-1]
                    )

                output_dict[f"Predicted_L{layer_}"] = [
                    intervened_base_output,
                    predicted_text,
                ]

        return output_dict


if __name__ == "__main__":
    my_model = my_model()
