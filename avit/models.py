import torch
import torch.nn
import torch.nn.functional
import torchvision
import torchvision.ops
import transformers
from typing import *


class DeiTPatchEmbedding(torch.nn.Module):
    """
    A PyTorch model that corresponds to the patch embedding stage of a
    DeiT transformer.
    """

    def __init__(self, avit_kwargs: Dict[str, Any]) -> None:
        """
        Constructor.

        Args:
            avit_kwargs (Dict[str, Any]): The configuration arguments.
        """

        super(DeiTPatchEmbedding, self).__init__()

        # Retrieve the needed configuration arguments.
        img_dim: int = avit_kwargs["img_dim"]
        patch_dim: int = avit_kwargs["patch_dim"]
        in_channels: int = avit_kwargs["in_channels"]
        embed_dim: int = avit_kwargs["embed_dim"]

        # Compute the number of patches based on the squared image dimensions and the
        # squared patch dimensions.
        num_patches_per_dim = img_dim // patch_dim
        self.num_patches = num_patches_per_dim * num_patches_per_dim

        # Convolution projection layer to extract and embed the patches.
        self.proj = torch.nn.Conv2d(in_channels, embed_dim, patch_dim, stride=patch_dim)
        
        # Normalization layer.
        self.norm = torch.nn.LayerNorm(embed_dim, eps=1.e-6)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Feedforward for this PyTorch model.
        In particular extract and embed patches from a batch of pixels from images.

        Args:
            pixel_values (torch.Tensor): The batch of pixels to patch and embed.
            [batch_size, num_channels=3, height=224, width=224]

        Returns:
            torch.Tensor: The computed batch of patch embeddings.
            [batch_size, num_patches|seq_len=196, embed_dim=192]
        """

        # Convolve the image, then extract the patches, embed them.
        # At the end pass the patch embeddings to the normalization layer.
        return self.norm(self.proj(pixel_values).flatten(2).transpose(1, 2))


class DeiTEmbedding(torch.nn.Module):
    """
    A PyTorch model that corresponds to the embedding stage of a
    DeiT transformer.
    """
    
    def __init__(self, avit_kwargs: Dict[str, Any]) -> None:
        """
        Constructor.

        Args:
            avit_kwargs (Dict[str, Any]): The configuration arguments.
        """
        
        super(DeiTEmbedding, self).__init__()

        # Retrieve the needed configuration arguments.
        embed_dim: int = avit_kwargs["embed_dim"]

        # Patch embedding stage.
        self.patch_embed = DeiTPatchEmbedding(avit_kwargs)

        # Trainable class token embedding.
        self.cls_token = torch.nn.Parameter(data=torch.zeros((1, 1, embed_dim)))

        # Trainable position embeddings.
        self.pos_embeddings = torch.nn.Parameter(
            data=torch.zeros((1, self.patch_embed.num_patches + 1, embed_dim))
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Feedforward for this PyTorch model.
        In particular embed patches from a batch of pixels from images.

        Args:
            pixel_values (torch.Tensor): The batch of pixels to patch and embed.
            [batch_size, num_channels=3, height=224, width=224]

        Returns:
            torch.Tensor: The computed batch of patch embeddings.
            [batch_size, num_patches|seq_len=197, embed_dim=192]
        """

        # Extract and compute the patch embeddings.
        embeddings = self.patch_embed(pixel_values)
        # embeddings [batch_size, num_patches|seq_len=196, embed_dim=192]

        # Attach the class token embedding at the start of the embeddings.
        batch_size, _, _ = embeddings.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        # embeddings [batch_size, num_patches|seq_len=197, embed_dim=192]

        # Add the position embeddings to the embeddings.
        embeddings = embeddings + self.pos_embeddings
        # embeddings [batch_size, num_patches|seq_len=197, embed_dim=192]

        return embeddings


class DeiTAttention(torch.nn.Module):
    """
    A PyTorch model that corresponds to the self attention stage of a
    DeiT transformer.
    """

    def __init__(self, avit_kwargs: Dict[str, Any]) -> None:
        """
        Constructor.

        Args:
            avit_kwargs (Dict[str, Any]): The configuration arguments.
        """

        super(DeiTAttention, self).__init__()

        # Retrieve the needed configuration arguments.
        num_heads: int = avit_kwargs["num_heads"]
        embed_dim: int = avit_kwargs["embed_dim"]
        attn_bias: bool = avit_kwargs["attn_bias"]

        # Compute the number of heads, the head size and the size of all the heads
        # that should be equal to the embedding dimension.
        self.num_attn_heads = num_heads
        self.attn_head_size = embed_dim // num_heads
        self.all_heads_size = embed_dim

        # The query projection layer.
        self.query_proj = torch.nn.Linear(embed_dim, embed_dim, bias=attn_bias)

        # The key projection layer.
        self.key_proj = torch.nn.Linear(embed_dim, embed_dim, bias=attn_bias)

        # The value projection layer.
        self.value_proj = torch.nn.Linear(embed_dim, embed_dim, bias=attn_bias)

        # A final output layer.
        self.output = torch.nn.Linear(embed_dim, embed_dim)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Utility function to extract the heads from the keys, the values and
        the queries.

        Args:
            x (torch.Tensor): (batched) Keys, values or queries to extract heads from.

        Returns:
            torch.Tensor: The extracted heads from the input.
        """

        new_x_shape = x.size()[:-1] + (self.num_attn_heads, self.attn_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def transpose_for_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """
        Utility function to retrieve the hidden states from the heads.

        Args:
            x (torch.Tensor): (batched) Heads to retrieve hidden states from.

        Returns:
            torch.Tensor: The retrived hidden states from the input.
        """

        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (self.all_heads_size,)
        return x.view(new_x_shape)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Feedforward for this PyTorch model.
        In particular compute the contextualized representation of a batch of
        hidden states taking into consideration a (batched) attention mask.

        Args:
            hidden_states (torch.Tensor): The batch of hidden states to contextualize.
            [batch_size, seq_len=197, embed_dim=192]

            attn_mask (torch.Tensor): The (batched) attention mask. Unused here,
            but needed for subclassing.

        Returns:
            torch.Tensor: The computed contextualized representation of the batch
            of hidden states.
            [batch_size, seq_len=197, embed_dim=192]
        """

        # Compute the keys.
        keys = self.transpose_for_scores(self.key_proj(hidden_states))
        # keys [batch_size, num_heads=3, seq_len=197, head_dim=64]

        # Compute the values.
        values = self.transpose_for_scores(self.value_proj(hidden_states))
        # values [batch_size, num_heads=3, seq_len=197, head_dim=64]

        # Compute the queries.
        queries = self.transpose_for_scores(self.query_proj(hidden_states))
        # queries [batch_size, num_heads=3, seq_len=197, head_dim=64]

        # Compute the contextualized representation (heads).
        context = torch.nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=0.,
            is_causal=False, scale=None
        )
        # context [batch_size, num_heads=3, seq_len=197, head_dim=64]

        # Compute the contextualized representation of the hidden states.
        context = self.output(self.transpose_for_hidden_states(context))
        # context [batch_size, seq_len=197, embed_dim=192]

        return context


class AViTAttention(DeiTAttention):
    """
    A PyTorch model that corresponds to the self attention stage of an
    AViT transformer.
    """

    def __init__(self, avit_kwargs: Dict[str, Any]) -> None:
        """
        Constructor.

        Args:
            avit_kwargs (Dict[str, Any]): The configuration arguments.
        """

        super(AViTAttention, self).__init__(avit_kwargs)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Feedforward for this PyTorch model.
        In particular compute the contextualized representation of a batch of
        hidden states taking into consideration a (batched) attention mask.

        Args:
            hidden_states (torch.Tensor): The batch of hidden states to contextualize.
            [batch_size, seq_len=197, embed_dim=192]

            attn_mask (torch.Tensor): The (batched) attention mask.
            [batch_size, seq_len=197]

        Returns:
            torch.Tensor: The computed contextualized representation of the batch
            of hidden states.
            [batch_size, seq_len=197, embed_dim=192]
        """

        # Compute the keys.
        keys = self.transpose_for_scores(self.key_proj(hidden_states))
        # keys [batch_size, num_heads=3, seq_len=197, head_dim=64]

        # Compute the values.
        values = self.transpose_for_scores(self.value_proj(hidden_states))
        # values [batch_size, num_heads=3, seq_len=197, head_dim=64]

        # Compute the queries.
        queries = self.transpose_for_scores(self.query_proj(hidden_states))
        # queries [batch_size, num_heads=3, seq_len=197, head_dim=64]

        # Compute the contextualized representation (heads).
        context = torch.nn.functional.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=(
                attn_mask.view(attn_mask.shape[0], 1, 1, attn_mask.shape[1])
                * (-100_000_000.)
            ),
            dropout_p=0., is_causal=False, scale=None
        )
        # context [batch_size, num_heads=3, seq_len=197, head_dim=64]

        # Compute the contextualized representation of the hidden states.
        context = self.output(self.transpose_for_hidden_states(context))
        # context [batch_size, seq_len=197, embed_dim=192]

        return context


class DeiTLayer(torch.nn.Module):
    """
    A PyTorch model that corresponds to a layer of a DeiT transformer.
    """

    def __init__(self, avit_kwargs: Dict[str, Any]) -> None:
        """
        Constructor.

        Args:
            avit_kwargs (Dict[str, Any]): The configuration arguments.
        """

        super(DeiTLayer, self).__init__()

        # Retrieve the needed configuration arguments.
        embed_dim: int = avit_kwargs["embed_dim"]
        mlp_ratio: float = avit_kwargs["mlp_ratio"]
        p_drop_path: float = avit_kwargs["p_drop_path"]

        # The self attention stage.
        self.attn = DeiTAttention(avit_kwargs)

        # The MLP with GELU stage.
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, mlp_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(mlp_hidden_dim, embed_dim)
        )

        # A stocastic depth dropout layer along the batch for regularization.
        self.drop_path = torchvision.ops.StochasticDepth(p_drop_path, "batch")

        # The initial normalization layer.
        self.norm_before = torch.nn.LayerNorm(embed_dim, eps=1.e-6)

        # The final normalization layer.
        self.norm_after = torch.nn.LayerNorm(embed_dim, eps=1.e-6)

    def forward(
            self,
            hidden_states: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Feedforward for this PyTorch model.
        In particular compute the next hidden states from a batch of
        previous hidden states taking into consideration a (batched) mask.

        Args:
            hidden_states (torch.Tensor): The batch of previous hidden states.
            [batch_size, seq_len=197, embed_dim=192]

            mask (torch.Tensor): The (batched) mask. Unused here, but needed
            for subclassing.

        Returns:
            torch.Tensor: The computed batch of next hidden states.
            [batch_size, seq_len=197, embed_dim=192]
        """

        # Normalize the hidden states. Compute the contextualized representation
        # using the self attention + stochastic depth dropout. Perform the
        # first residual connection.
        hidden_states = (
            hidden_states
            + self.drop_path(self.attn(self.norm_before(hidden_states), None))
        )
        # hidden_states [batch_size, seq_len=197, embed_dim=192]

        # Normalize the hidden states. Compute a new representation using the
        # MLP + stochastic depth dropout. Perform the second residual connection.
        hidden_states = (
            hidden_states
            + self.drop_path(self.mlp(self.norm_after(hidden_states)))
        )
        # hidden_states [batch_size, seq_len=197, embed_dim=192]

        return hidden_states


class AViTLayer(DeiTLayer):
    """
    A PyTorch model that corresponds to a layer of an AViT transformer.
    """

    def __init__(self, avit_kwargs: Dict[str, Any]) -> None:
        """
        Constructor.

        Args:
            avit_kwargs (Dict[str, Any]): The configuration arguments.
        """

        super(AViTLayer, self).__init__(avit_kwargs)

        # The self attention stage.
        self.attn = AViTAttention(avit_kwargs)

    def forward(
            self,
            hidden_states: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Feedforward for this PyTorch model.
        In particular compute the next hidden states from a batch of
        previous hidden states taking into consideration a (batched) mask.

        Args:
            hidden_states (torch.Tensor): The batch of previous hidden states.
            [batch_size, seq_len=197, embed_dim=192]

            mask (torch.Tensor): The (batched) mask.
            [batch_size, seq_len=197]

        Returns:
            torch.Tensor: The computed batch of next hidden states.
            [batch_size, seq_len=197, embed_dim=192]
        """

        batch_size, seq_len, _ = hidden_states.shape
        new_mask = mask.view(batch_size, seq_len, 1)

        # Normalize the hidden states. Compute the contextualized representation
        # using the masked self attention + stochastic depth dropout. Perform the
        # first residual connection.
        hidden_states = (
            hidden_states
            + self.drop_path(self.attn(
                self.norm_before(hidden_states * new_mask) * new_mask,
                1. - mask
            ))
        )
        # hidden_states [batch_size, seq_len=197, embed_dim=192]

        # Normalize the hidden states. Compute a new representation using the
        # masked MLP + stochastic depth dropout. Perform the second residual
        # connection.
        hidden_states = (
            hidden_states
            + self.drop_path(self.mlp(
                self.norm_after(hidden_states * new_mask) * new_mask
            ))
        )
        # hidden_states [batch_size, seq_len=197, embed_dim=192]

        return hidden_states


class DeiTEncoder(torch.nn.Module):
    """
    A PyTorch model that corresponds to the encoder of a DeiT transformer.
    """

    def __init__(self, avit_kwargs: Dict[str, Any]) -> None:
        """
        Constructor.

        Args:
            avit_kwargs (Dict[str, Any]): The configuration arguments.
        """

        super(DeiTEncoder, self).__init__()

        # Retrieve the needed configuration arguments.
        depth: int = avit_kwargs["depth"]
        p_drop_path_max: float = avit_kwargs["p_drop_path_max"]

        # The list of DeiT layers.
        self.layers = torch.nn.ModuleList()
        p_drop_paths = torch.linspace(0., p_drop_path_max, depth)
        for i in range(depth):
            new_avit_kwargs = avit_kwargs.copy()
            new_avit_kwargs.update({"p_drop_path": p_drop_paths[i].item()})
            self.layers.append(DeiTLayer(new_avit_kwargs))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Feedforward for this PyTorch model.
        In particular compute the final hidden states from the batch of
        initial hidden states.

        Args:
            hidden_states (torch.Tensor): The batch of initial hidden states.
            [batch_size, seq_len=197, embed_dim=192]

        Returns:
            torch.Tensor: The computed batch of final hidden states.
            [batch_size, seq_len=197, embed_dim=192]
        """

        # Iterate over all the DeiT layers.
        for i, layer in enumerate(self.layers):

            # Compute the next hidden states.
            hidden_states = layer(hidden_states, None)
            # hidden_states [batch_size, seq_len=197, embed_dim=192]

        return hidden_states


class AViTEncoder(DeiTEncoder):
    """
    A PyTorch model that corresponds to the encoder of an AViT transformer.
    """

    def __init__(self, avit_kwargs: Dict[str, Any]) -> None:
        """
        Constructor.

        Args:
            avit_kwargs (Dict[str, Any]): The configuration arguments.
        """

        super(AViTEncoder, self).__init__(avit_kwargs)

        # Retrieve the needed configuration arguments.
        depth: int = avit_kwargs["depth"]
        expected_depth: int = avit_kwargs["expected_depth"]
        p_drop_path_max: float = avit_kwargs["p_drop_path_max"]
        beta: float = avit_kwargs["beta"]
        gamma: float = avit_kwargs["gamma"]
        eps: float = avit_kwargs["eps"]

        # The list of AViT layers.
        self.layers = torch.nn.ModuleList()
        p_drop_paths = torch.linspace(0., p_drop_path_max, depth)
        for i in range(depth):
            new_avit_kwargs = avit_kwargs.copy()
            new_avit_kwargs.update({"p_drop_path": p_drop_paths[i].item()})
            self.layers.append(AViTLayer(new_avit_kwargs))

        # The extra arguments for the AViT encoder. See the paper for details.
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.depth = depth
        self.expected_depth = expected_depth

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Feedforward for this PyTorch model.
        In particular compute the final hidden states and the other needed
        parameters from the batch of initial hidden states.

        Args:
            hidden_states (torch.Tensor): The batch of initial hidden states.
            [batch_size, seq_len=197, embed_dim=192]

        Returns:
            Dict[str, torch.Tensor]: The computed batches of final hidden states
            and the other parameters.
        """

        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Initialize the final hidden states as 0.
        out_hidden_states = torch.zeros_like(hidden_states)
        # out_hidden_states [batch_size, seq_len=197, embed_dim=192]

        # Initialize the mask as 1. See the paper for details.
        mask = torch.ones((batch_size, seq_len), device=device)
        # mask [batch_size, seq_len=197]

        # Initialize the cumulatives as 0. See the paper for details.
        cumulatives = torch.zeros_like(mask)
        # cumulatives [batch_size, seq_len=197]

        # Initialize the reminders as 1. See the paper for details.
        reminders = torch.ones_like(mask)
        # reminders [batch_size, seq_len=197]

        # Initialize the rhos (intermediate ponder losses at each AViT layer)
        # as 0. See the paper for details.
        rhos = torch.zeros_like(mask)
        # rhos [batch_size, seq_len=197]

        # Initialize the counter of layers for each token.
        # Needed for plotting visual results.
        counter = torch.ones_like(mask)
        # counter [batch_size, seq_len=197]

        # Range of depths.
        depths = torch.arange(self.depth, device=hidden_states.device)

        # Initialize the halting score target probability distribution as a
        # discrete standard Normal distribution centered in the value of the
        # expected depth. Then normalize it and clamp it between 0.01 and 0.99
        # to avoid numeric issues with logarithms.
        halting_score_distr_target = (
            torch.distributions.Normal(loc=self.expected_depth, scale=1)
            .log_prob(depths).exp()
        )
        halting_score_distr_target = (
            halting_score_distr_target / torch.sum(halting_score_distr_target)
        )
        halting_score_distr_target = torch.clamp(
            halting_score_distr_target, min=0.01, max=0.99
        )
        # halting_score_distr_target [depth=12]

        # Initialize the halting score probability distribution.
        halting_score_distr = list()

        # Iterate over all the DeiT layers.
        for i, layer in enumerate(self.layers):

            # Compute the next hidden states.
            hidden_states = layer(hidden_states * mask.view(batch_size, seq_len, 1), mask)
            # hidden_states [batch_size, seq_len=197, embed_dim=192]

            # Compute the halting scores. See the paper for details.
            halting_scores = (
                torch.sigmoid(hidden_states[:, :, 0] * self.gamma + self.beta)
                if i < len(self.layers) - 1
                else
                torch.ones_like(mask)
            )
            # halting_scores [batch_size, seq_len=197]

            # Populate the halting score probability distribution.
            halting_score_distr.append(torch.mean(halting_scores[0][1:]))

            # Update the cumulatives. See the paper for the details.
            cumulatives = cumulatives + halting_scores

            # Update the rhos. See the paper for the details.
            rhos = rhos + mask

            # Update the rhos or the reminders based on the reached mask.
            # Then update the hidden states. See the paper for the details.
            reached_mask = cumulatives >= 1. - self.eps
            rhos = rhos + reminders * reached_mask.float() * mask
            out_hidden_states = (
                out_hidden_states
                + hidden_states * reminders.view(batch_size, seq_len, 1)
                * reached_mask.view(batch_size, seq_len, 1).float()
            )

            not_reached_mask = torch.logical_not(reached_mask)
            reminders = reminders - (halting_scores * not_reached_mask.float())
            out_hidden_states = (
                out_hidden_states
                + hidden_states * halting_scores.view(batch_size, seq_len, 1)
                * not_reached_mask.view(batch_size, seq_len, 1).float()
            )

            # Update the mask. See the paper for details.
            mask = not_reached_mask.float()

            # Update the counter.
            counter = counter + not_reached_mask.float().detach()

        # Compute the halting score probability distribution. Then normalize it
        # and clamp it between 0.01 and 0.99 to avoid numeric issues with logarithms. 
        halting_score_distr = torch.stack(halting_score_distr)
        halting_score_distr = (
            halting_score_distr / torch.sum(halting_score_distr)
        )
        halting_score_distr = torch.clamp(
            halting_score_distr, min=0.01, max=0.99
        )

        return {
            "hidden_states": out_hidden_states,
            "rhos": rhos,
            "halting_score_distr": halting_score_distr,
            "halting_score_distr_target": halting_score_distr_target,
            "counter": counter
        }


class DeiT(torch.nn.Module):
    """
    A PyTorch model that corresponds to the actual DeiT transformer.
    """

    def __init__(self, avit_kwargs: Dict[str, Any]) -> None:
        """
        Constructor.

        Args:
            avit_kwargs (Dict[str, Any]): The configuration arguments.
        """

        super(DeiT, self).__init__()

        # Retrieve the needed configuration arguments.
        embed_dim: int = avit_kwargs["embed_dim"]

        # The embedding stage.
        self.embed = DeiTEmbedding(avit_kwargs)

        # The encoder.
        self.encoder = DeiTEncoder(avit_kwargs)

        # Normalization layer.
        self.norm = torch.nn.LayerNorm(embed_dim, eps=1.e-6)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Feedforward for this PyTorch model.
        In particular compute the final hidden states from a batch of
        pixels from images.

        Args:
            pixel_values (torch.Tensor): The batch of pixels to start with.
            [batch_size, num_channels=3, height=224, width=224]

        Returns:
            torch.Tensor: The computed batch of final hidden states.
            [batch_size, seq_len=197, embed_dim=192]
        """

        # Compute the embeddings.
        embeddings = self.embed(pixel_values)
        # embeddings [batch_size, seq_len=197, embed_dim=192]

        # Compute the final hidden states and normalize them.
        hidden_states = self.encoder(embeddings)
        hidden_states = self.norm(hidden_states)
        # hidden_states [batch_size, seq_len=197, embed_dim=192]

        return hidden_states


class AViT(DeiT):
    """
    A PyTorch model that corresponds to the actual AViT transformer.
    """

    def __init__(self, avit_kwargs: Dict[str, Any]) -> None:
        """
        Constructor.

        Args:
            avit_kwargs (Dict[str, Any]): The configuration arguments.
        """

        super(AViT, self).__init__(avit_kwargs)

        # The encoder.
        self.encoder = AViTEncoder(avit_kwargs)
    
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Feedforward for this PyTorch model.
        In particular compute the final hidden states and the other needed
        parameters from a batch of pixels from images.

        Args:
            pixel_values (torch.Tensor): The batch of pixels to start with.
            [batch_size, num_channels=3, height=224, width=224]

        Returns:
            Dict[str, torch.Tensor]: The computed batches of final hidden states
            and the other parameters.
        """

        # Compute the embeddings.
        embeddings = self.embed(pixel_values)
        # embeddings [batch_size, seq_len=197, embed_dim=192]

        # Compute the final hidden states and the other parameters.
        # Normalize the final hidden states.
        outputs = self.encoder(embeddings)
        hidden_states = self.norm(outputs["hidden_states"])
        # hidden_states [batch_size, seq_len=197, embed_dim=192]

        return {
            "hidden_states": hidden_states,
            "rhos": outputs["rhos"],
            "halting_score_distr": outputs["halting_score_distr"],
            "halting_score_distr_target": outputs["halting_score_distr_target"],
            "counter": outputs["counter"]
        }


class DeiTForImageClassification(torch.nn.Module):
    """
    A PyTorch model that corresponds to a specialization of the DeiT transformer
    for the image classification task.
    """
    
    def __init__(self, avit_kwargs: Dict[str, Any]) -> None:
        """
        Constructor.

        Args:
            avit_kwargs (Dict[str, Any]): The configuration arguments.
        """

        super(DeiTForImageClassification, self).__init__()

        # Retrieve the needed configuration arguments.
        embed_dim: int = avit_kwargs["embed_dim"]
        num_classes: int = avit_kwargs["num_classes"]

        # The DeiT transformer.
        self.model = DeiT(avit_kwargs)

        # The classification layer.
        self.classifier = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Feedforward for this PyTorch model.
        In particular compute the logits for the classification from a batch of
        pixels from images.

        Args:
            pixel_values (torch.Tensor): The batch of pixels to start with.
            [batch_size, num_channels=3, height=224, width=224]

        Returns:
            torch.Tensor: The computed batch of logits.
            [batch_size, num_classes=257]
        """

        # Compute the final hidden states.
        hidden_states = self.model(pixel_values)
        # hidden_states [batch_size, seq_len=197, embed_dim=192]

        # Pass the first embedding (class token) to the classifier.
        # Compute the logits.
        logits = self.classifier(hidden_states[:, 0, :])
        # logits [batch_size, num_classes=257]

        return logits

    def load_hf_deit_weights(
            self,
            hf_model_name: str,
            avit_kwargs: Dict[str, Any]
    ) -> None:
        """
        Manually load the weights of the chosen DeiT transformer from HuggingFace
        to avoid clashes on the names of the weights.

        Args:
            hf_model_name (str): The full HuggingFace name of the DeiT
            transformer model that will be used in the task, like
            'facebook/deit-tiny-distilled-patch16-224'.

            avit_kwargs (Dict[str, Any]): The configuration arguments.
        """

        # Retrieve and download the chosen HuggingFace DeiT transformer
        # and save it in a local directory.
        hf_model = transformers.DeiTForImageClassification.from_pretrained(
            hf_model_name, cache_dir="./hf/transformers"
        )

        # Load the weights
        self.model.embed.load_state_dict({
            "cls_token": hf_model.deit.embeddings.state_dict()["cls_token"],
            "pos_embeddings": torch.cat(
                (
                    hf_model.deit.embeddings.state_dict()["position_embeddings"][:, 0:1, :],
                    hf_model.deit.embeddings.state_dict()["position_embeddings"][:, 2:, :]
                ), dim=1
            )
        }, strict=False)
        self.model.embed.patch_embed.proj.load_state_dict(
            hf_model.deit.embeddings.patch_embeddings.projection.state_dict()
        )
        for i in range(avit_kwargs["depth"]):
            self.model.encoder.layers[i].attn.query_proj.load_state_dict(
                hf_model.deit.encoder.layer[i].attention.attention.query.state_dict()
            )
            self.model.encoder.layers[i].attn.key_proj.load_state_dict(
                hf_model.deit.encoder.layer[i].attention.attention.key.state_dict()
            )
            self.model.encoder.layers[i].attn.value_proj.load_state_dict(
                hf_model.deit.encoder.layer[i].attention.attention.value.state_dict()
            )
            self.model.encoder.layers[i].attn.output.load_state_dict(
                hf_model.deit.encoder.layer[i].attention.output.dense.state_dict()
            )
            self.model.encoder.layers[i].mlp[0].load_state_dict(
                hf_model.deit.encoder.layer[i].intermediate.dense.state_dict()
            )
            self.model.encoder.layers[i].mlp[2].load_state_dict(
                hf_model.deit.encoder.layer[i].output.dense.state_dict()
            )
            self.model.encoder.layers[i].norm_before.load_state_dict(
                hf_model.deit.encoder.layer[i].layernorm_before.state_dict()
            )
            self.model.encoder.layers[i].norm_after.load_state_dict(
                hf_model.deit.encoder.layer[i].layernorm_after.state_dict()
            )
        self.model.norm.load_state_dict(
            hf_model.deit.layernorm.state_dict()
        )


class AViTForImageClassification(DeiTForImageClassification):
    """
    A PyTorch model that corresponds to a specialization of the AViT transformer
    for the image classification task.
    """

    def __init__(self, avit_kwargs: Dict[str, Any]) -> None:
        """
        Constructor.

        Args:
            avit_kwargs (Dict[str, Any]): The configuration arguments.
        """

        super(AViTForImageClassification, self).__init__(avit_kwargs)

        # The DeiT transformer.
        self.model = AViT(avit_kwargs)
    
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Feedforward for this PyTorch model.
        In particular compute the logits for the classification and the other
        needed parameters from a batch of pixels from images.

        Args:
            pixel_values (torch.Tensor): The batch of pixels to start with.
            [batch_size, num_channels=3, height=224, width=224]

        Returns:
            torch.Tensor: The computed batch of logits.
            [batch_size, num_classes=257]
        """

        # Compute the final hidden states and the other parameters.
        outputs = self.model(pixel_values)

        # Pass the first embedding (class token) to the classifier.
        # Compute the logits.
        logits = self.classifier(outputs["hidden_states"][:, 0, :])
        # logits [batch_size, num_classes=257]

        return {
            "logits": logits,
            "rhos": outputs["rhos"],
            "halting_score_distr": outputs["halting_score_distr"],
            "halting_score_distr_target": outputs["halting_score_distr_target"],
            "counter": outputs["counter"]
        }
