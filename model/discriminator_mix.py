import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
from transformers import HubertModel, HubertConfig, Wav2Vec2Model
#from functions.attn_pooling import AttentivePooling

class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Adapter, self).__init__()
        self.adapter = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.LeakyReLU(negative_slope=0.01), #nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim))  # Ensure output matches HuBERT hidden size
        )

    def forward(self, x):
        return self.adapter(x)

class MultiHeadAttentivePooling(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        
    def forward(self, x):
        query = self.query.expand(x.size(0), -1, -1)
        attn_output, _ = self.attention(query, x, x)
        return attn_output.squeeze(1)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = spectral_norm(nn.Linear(in_features, out_features))
        self.linear2 = spectral_norm(nn.Linear(out_features, out_features))
        self.shortcut = spectral_norm(nn.Linear(in_features, out_features)) if in_features != out_features else nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        out = F.leaky_relu(self.linear1(x), 0.2)
        out = self.linear2(out)
        return F.leaky_relu(out + identity, 0.2)
class EnhancedAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super(EnhancedAdapter, self).__init__()
        
        # First projection with intermediate size
        self.down_project = spectral_norm(nn.Linear(input_dim, hidden_dim))
        
        # Add Layer Normalization for better training stability
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Intermediate processing
        self.intermediate = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim, hidden_dim * 2)),
            nn.GELU(),  # GELU typically works better than LeakyReLU for transformers
            nn.Dropout(dropout_rate),
            spectral_norm(nn.Linear(hidden_dim * 2, hidden_dim))
        )
        
        # Skip connection to help with gradient flow
        self.skip_connection = spectral_norm(nn.Linear(input_dim, hidden_dim)) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        # Skip connection path
        identity = self.skip_connection(x)
        
        # Main path
        out = self.down_project(x)
        out = self.norm1(out)
        
        # Intermediate processing
        out = self.intermediate(out)
        out = self.norm2(out)
        
        # Add skip connection
        return out + identity
class SpeakerAttentionPool(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.speaker_queries = nn.Parameter(torch.randn(1, 3, hidden_size))
        self.query_weights = nn.Parameter(torch.ones(3))
        
    def forward(self, x):
        queries = self.speaker_queries.expand(x.size(0), -1, -1)
        attn_outputs = []        
        for i in range(3):
            query = queries[:, i:i+1]
            attn_output, _ = self.attention(query, x, x)
            attn_outputs.append(attn_output * F.softmax(self.query_weights, dim=0)[i])        
        combined = torch.sum(torch.stack(attn_outputs, dim=1), dim=1)
        return combined.squeeze(1)

class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.linear1 = spectral_norm(nn.Linear(in_features, out_features))
        self.norm2 = nn.LayerNorm(out_features)
        self.linear2 = spectral_norm(nn.Linear(out_features, out_features))
        self.dropout = nn.Dropout(0.1)
        self.shortcut = spectral_norm(nn.Linear(in_features, out_features)) if in_features != out_features else nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.norm1(x)
        out = F.gelu(self.linear1(out))
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.linear2(out)
        
        return F.gelu(out + identity)


class MixupDiscriminator(nn.Module):
    """Discriminator-only ablation: a plain MLP over speaker embeddings.

    This removes the HuBERT/WavLM adapter + length-1 Transformer path while
    keeping the rest of the 3.48 DDP baseline unchanged.
    """

    def __init__(self, cache_dir="", emb_dim=192, hidden_dim=256, **kwargs):
        super(MixupDiscriminator, self).__init__()
        mid_dim = hidden_dim // 2
        self.discriminator = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.1),
            nn.Linear(mid_dim, 1),
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            "[MLPDiscriminator] architecture: "
            f"{emb_dim} -> {hidden_dim} -> {mid_dim} -> 1 "
            "with LayerNorm + Dropout "
            f"({n_params:,} trainable params)"
        )

    def forward(self, embeddings):
        return self.discriminator(embeddings)


class HubertDiscriminator(nn.Module):
    def __init__(self, hubert_model_name="facebook/hubert-large-ls960-ft",cache_dir="", proj_dim=128, emb_dim = 192):
        super(HubertDiscriminator, self).__init__()
        # Load pre-trained HuBERT
        self.hubert = HubertModel.from_pretrained(hubert_model_name, cache_dir=cache_dir)
        # for param in self.hubert.parameters():
        #     param.requires_grad = False  # Freeze HuBERT

        # Define projection layers for layers 3, 6, 9, 12
        hidden_size = self.hubert.config.hidden_size
        self.projection_3 = spectral_norm(nn.Linear(hidden_size, proj_dim))
        self.projection_6 = spectral_norm(nn.Linear(hidden_size, proj_dim))
        self.projection_9 = spectral_norm(nn.Linear(hidden_size, proj_dim))
        self.projection_12 = spectral_norm(nn.Linear(hidden_size, proj_dim))
        
        # Discriminator head
        self.discriminator = nn.Sequential(
            spectral_norm(nn.Linear(proj_dim * 4, 256)),  # Concatenate 4 layers' projections
            nn.LeakyReLU(negative_slope=0.01), #nn.ReLU(),
            spectral_norm(nn.Linear(256, 1)),  # Binary classification for real/fake
            #nn.Sigmoid()
        )
        # Adapter to align speaker encoder outputs with HuBERT
        self.adapter = Adapter(input_dim=emb_dim, hidden_dim=hidden_size)
    
    def forward(self, input_audio):
        # Extract HuBERT hidden states
        # Use adapter to align embeddings with HuBERT hidden size
        adapted_embeddings = self.adapter(input_audio)  # (batch_size, hidden_size)
         # Reshape embeddings to include a sequence length (if needed)
        if adapted_embeddings.dim() == 2:  # (batch_size, hidden_size)
            adapted_embeddings = adapted_embeddings.unsqueeze(1)  # Add sequence length: (batch_size, seq_len=1, hidden_size)
        # with torch.no_grad():  # Ensure HuBERT remains frozen
        #     outputs = self.hubert(input_audio, output_hidden_states=True)
        #     hidden_states = outputs.hidden_states
        # with torch.no_grad():  # Ensure HuBERT remains frozen
            # Feed embeddings into the HuBERT transformer (bypass feature extractor)
        encoder_outputs = self.hubert.encoder(
            hidden_states=adapted_embeddings,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = encoder_outputs.hidden_states
        # Select layers 3, 6, 9, 12
        layer_3 = hidden_states[3]
        layer_6 = hidden_states[6]
        layer_9 = hidden_states[9]
        layer_12 = hidden_states[12]
        
        # Apply projection layers
        proj_3 = self.projection_3(layer_3.mean(dim=1))  # Pool along time if necessary
        proj_6 = self.projection_6(layer_6.mean(dim=1))
        proj_9 = self.projection_9(layer_9.mean(dim=1))
        proj_12 = self.projection_12(layer_12.mean(dim=1))
        
        # Concatenate projections
        concat_proj = torch.cat([proj_3, proj_6, proj_9, proj_12], dim=-1)
        
        # Pass through discriminator head
        real_fake_logits = self.discriminator(concat_proj)
        return real_fake_logits


class Discriminator_spectral(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc1 = spectral_norm(nn.Linear(embedding_dim, 128))
        self.activation = nn.LeakyReLU(0.2)
        self.fc2 = spectral_norm(nn.Linear(128, 1))

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return self.fc2(x)
        #return torch.sigmoid(self.fc2(x))


class ProjectionDiscriminator_spectral(nn.Module):
    """Projection MLP-D conditioned on the target real/pseudo-speaker prototype."""

    requires_condition = True

    def __init__(self, embedding_dim, hidden_dim=128):
        super().__init__()
        self.embed = spectral_norm(nn.Linear(embedding_dim, hidden_dim))
        self.cond = spectral_norm(nn.Linear(embedding_dim, hidden_dim, bias=False))
        self.activation = nn.LeakyReLU(0.2)
        self.head = spectral_norm(nn.Linear(hidden_dim, 1))
        self.scale = hidden_dim ** 0.5

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            "[ProjectionMLP-D] architecture: "
            f"D(e, q) = h(e) + <phi(e), psi(q)> "
            f"({embedding_dim}->{hidden_dim}->1, {n_params:,} trainable params)"
        )

    def _embed_features(self, x):
        return self.activation(self.embed(x))

    def _condition_features(self, condition):
        return self.cond(F.normalize(condition, dim=1))

    def pairwise_compatibility(self, x, condition_bank):
        """Return every embedding-prototype compatibility score."""
        h = self._embed_features(x)
        q = self._condition_features(condition_bank)
        return torch.mm(h, q.t()) / self.scale

    def matching_logits(self, x, positive_condition, negative_condition):
        """Score assigned and mismatched prototypes with one embedding pass."""
        h = self._embed_features(x)
        q_pos = self._condition_features(positive_condition)
        q_neg = self._condition_features(negative_condition)
        pos = (h * q_pos).sum(dim=1) / self.scale
        neg = (h * q_neg).sum(dim=1) / self.scale
        return pos, neg

    def forward(self, x, condition):
        if condition is None:
            raise ValueError("ProjectionDiscriminator_spectral requires a condition tensor")
        h = self._embed_features(x)
        q = self._condition_features(condition)
        projection = (h * q).sum(dim=1, keepdim=True) / self.scale
        return self.head(h) + projection


class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(Discriminator, self).__init__()
        # self.conv1 = nn.Conv1d(1, d, 4, 2, 1)
        # self.conv2 = nn.Conv1d(d, d * 2, 4, 2, 1)
        # self.conv2_bn = nn.BatchNorm2d(d * 2)
        # self.conv3 = nn.Conv1d(d * 2, d * 4, 4, 2, 1)
        # self.conv3_bn = nn.BatchNorm2d(d * 4)
        # self.conv4 = nn.Conv1d(d * 4, d * 8, 4, 1, 1)
        # self.conv4_bn = nn.BatchNorm2d(d * 8)
        # self.conv5 = nn.Conv1d(d * 8, 1, 4, 1, 1)
        self.conv1 = spectral_norm(nn.Conv1d(1, d, 4, 2, 1))
        self.conv2 = spectral_norm(nn.Conv1d(d, d * 2, 4, 2, 1))
        self.conv2_bn = nn.BatchNorm1d(d * 2)  # Changed to BatchNorm1d for 1D Conv
        self.conv3 = spectral_norm(nn.Conv1d(d * 2, d * 4, 4, 2, 1))
        self.conv3_bn = nn.BatchNorm1d(d * 4)  # Changed to BatchNorm1d for 1D Conv
        self.conv4 = spectral_norm(nn.Conv1d(d * 4, d * 8, 4, 1, 1))
        self.conv4_bn = nn.BatchNorm1d(d * 8)  # Changed to BatchNorm1d for 1D Conv
        self.conv5 = spectral_norm(nn.Conv1d(d * 8, 1, 4, 1, 1))

    def forward(self, input):
        #x = torch.cat([input, label], 1)
        x = input.unsqueeze(1)  #
        # x = input
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        #x = F.sigmoid(self.conv5(x))
        x = x.squeeze(2)
        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
