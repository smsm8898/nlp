import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from transformer.loss import LabelSmoothingLoss
import math
from typing import Optional

# TODO: LayerNorm, Dropout, Register_buffer, masked_fill


class MultiHeadAttention(nn.Module):
    """Multi-head Attention Mechanism"""
    def __init__(
            self,
            d_model:int,
            num_heads:int
        ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model:{d_model} must be divisible by num_heads:{num_heads}"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V projection layers
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Args:
            Q: (batch_size, seq_len, d_model)
            K: (batch_size, seq_len, d_model)
            V: (batch_size, seq_len, d_model)
            mask: attention mask
        """
        # Linear Projections (batch_size, seq_len, d_model)
        Q = self.W_q(Q) 
        K = self.W_k(K)
        V = self.W_v(V)

        # Split into multiple heads (batch_size, num_heads, seq_len, d_k)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Scaled dot-product attention
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads (batch_size, seq_len, d_model)
        attn_output = self.combine_heads(attn_output)

        # Final Linear Projection
        output = self.W_o(attn_output)
        return output, attention_weights
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k)
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Args
            Q: (batch_size, num_heads, seq_len, d_k)
            K: (batch_size, num_heads, seq_len, d_k)
            V: (batch_size, num_heads, seq_len, d_k)
            mask: (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)
        """
        # 1st(MatMul)
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # 2nd(Scale)
        scores = scores / math.sqrt(self.d_k)

        # (optional) Apply Mask
        if mask is not None:
            scores = scores.masked_fill(mask ==0, -1e9)

        # 3rd(Softmax)
        attention_weights = F.softmax(scores, dim=-1)

        # 4th(Matmul)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine heads back
        Args:
            x: (batch_size, num_heads, seq_len, d_k)
        Returns:
            (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)


class PositionwiseFeedForward(nn.Module):
    def __init__(
            self, 
            d_model:int,
            d_ff:int,
            dropout:float=0.1,
        ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

class PositionalEncoding(nn.Module):
    """Positional Encoding using sinusoidal funcitons"""
    def __init__(
            self,
            d_model:int,
            max_seq_len:int=5000,
            dropout:float=0.1,
        ):
        super().__init__()
        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Positional Encoding Matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position*div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0) # (1, max_seq_len, d_model)

        # Register as buffer (not a parameter but should be saved with the model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
        

class EncoderLayer(nn.Module):
    def __init__(
            self,
            d_model:int,
            num_heads:int, 
            d_ff:int, 
            dropout:float=0.1,
        ):
        super().__init__()
        # Dropout
        self.dropout = nn.Dropout(dropout)

        # SubLayer1: Multihead self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        # SubLayer2: Feed-forward network
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        - x는 embedding + positional encoding 완료
        Args:
            x: (batch_size, seq_len, d_model)
            mask: padding mask
        """
        # SubLayer1: self-attention with residual connection and layernorm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # SubLayer2: feed-forward with residual connection and layernorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(
            self,
            d_model:int,
            num_heads:int,
            d_ff:int,
            dropout:float=0.1,
        ):
        super().__init__()
        # Dropout
        self.dropout = nn.Dropout(dropout)

        # SubLayer1: Masked multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        # SubLayer2: Encoder-decoder attention (cross-attention)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)

        # SubLayer3: Feed-forward network
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, src_mask: Optional[torch.Tensor]=None, tgt_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, tgt_seq_len, d_model)
            enc_output: (batch_size, src_seq_len, d_model)
            src_mask: source padding mask
            tgt_mask: target mask (causal + padding)
        """
        # SubLayer1: Masked self-attention
        attn_output, _ = self.self_attn(Q=x, K=x, V=x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # SubLayer2: Encoder-decoder attention
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # SubLayer3: Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            d_model:int=512,
            num_heads:int=8,
            num_encoder_layers:int=6, 
            num_decoder_layers:int=6, 
            d_ff:int=2048, 
            max_seq_len:int=5000, 
            dropout:int=0.1
        ):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder Layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder Layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Final Linear Layer
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # Initalize parameters
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            src: (batch_size, src_seq_len)
            tgt: (batch_size, tgt_seq_len)
        Returns:
            output: (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Create masks
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        # Encode
        enc_output = self.encode(src, src_mask)

        # Decode
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)

        # Final linear
        output = self.fc_out(dec_output)
        return output

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Create Mask for source padding
        - Padding Mask: 패딩(0) 토큰은 의미 없는 토큰이니 attention을 주지 않음
        - self-attention 입력 shape은 (batch_Size, heads, seq_len, seq_len)이어서 이를 맞춰서 unsqueeze
        Args:
            (batch_size, src_seq_len)
        Returns:
            (batch_size, 1, 1, src_seq_len)
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Create mask for target (causal + padding)
        - Padding Mask: 패딩(0) 토큰은 의미 없는 토큰이니 attention을 주면 안 됨
        - Causal Mask(sSubsequent Mask): 미래 토큰을 보지 마라(autoregressive)
        Args:
            tgt: (batch_size, tgt_seq_len)
        Returns:
            (batch_size, 1, tgt_seq_len, tgt_seq_len)
        """
        _, tgt_len = tgt.size()

        # Padding mask
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, tgt_len)

        # Casual mask (prevent attending to future positions)
        # torch.tril -> lower triangular part
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_sub_mask = tgt_sub_mask.unsqueeze(0).unsqueeze(0) # (1, 1, tgt_len, tgt_len)

        # Combine masks(padding or future -> False)
        tgt_mask = tgt_padding_mask & tgt_sub_mask # (batch_size, 1, tgt_len, tgt_len)
        return tgt_mask
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode source sequence
        Args:
            src: (batch_size, src_seq_len)
            src_mask: (batch_size, 1, 1, src_seq_len)
        """
        # Embedding + Position encoding
        x = self.src_embedding(src) * math.sqrt(self.d_model) # In the embedding layers, we multiply those weights by d_model
        x = self.pos_encoding(x)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x
    
    def decode(self, tgt:torch.Tensor, enc_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Decode target sequence
        Args:
            tgt: (batch_size, tgt_seq_len)
            enc_output: (batch_size, src_seq_len, d_model)
            src_mask: (batch_size, 1, 1, src_seq_len)
            tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        """
        # Embedding + Position encoding
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model) # In the embedding layers, we multiply those weights by d_model
        x = self.pos_encoding(x)

        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x


class WMT14TransformerModule(L.LightningModule):
    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            d_model: int,
            num_heads: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            d_ff: int,
            dropout: float,
            max_seq_len: int,
            warmup_steps: int,
            label_smoothing: float,
            pad_idx:int,
        ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
    
        # Loss function
        self.criterion = LabelSmoothingLoss(
            vocab_size=tgt_vocab_size,
            pad_idx=pad_idx,
            smoothing=label_smoothing,
        )

        # For logging
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def NoamLRScheduler(self, step): 
        """Noam Learning Rate Schedule"""
        if step == 0:
            step = 1
        return (self.hparams.d_model ** (-0.5)) * min(
            step ** (-0.5),
            step * (self.hparams.warmup_steps ** (-1.5))
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1.0, # Will be overridden by scheduler
            betas=(0.9, 0.98),
            eps=1e-9
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=self.NoamLRScheduler
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # Update every step
                "frequency": 1,
            }
        }

    def forward(self, src, tgt):
        return self.model(src, tgt)
    
    def training_step(self, batch, batch_idx):
        src, tgt = batch

        # Teacher forcing: using all tokens except last for input
        tgt_input = tgt[:, : -1]
        tgt_output = tgt[:, 1:]

        # Forward Pass
        output = self(src, tgt_input)

        # Loss
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        loss = self.criterion(output, tgt_output)

        # Calculate Perplexity(ppl)
        num_tokens = (tgt_output != self.hparams.pad_idx).sum()
        loss_per_token = loss / num_tokens
        perplexity = torch.exp(loss_per_token.clamp(max=20))

        # Logging
        self.log("train_loss", loss_per_token, on_step = True, on_epoch=True, prog_bar=True)
        self.log("train_ppl", perplexity, on_step = True, on_epoch=True, prog_bar=True)
        self.training_step_outputs.append({
            "loss": loss_per_token.detach(),
            "ppl": perplexity.detach(),
        })

        return loss
    
    def validation_step(self, batch, batch_idx):
        src, tgt = batch

        # Teacher forcing: using all tokens except last for input
        tgt_input = tgt[:, : -1]
        tgt_output = tgt[:, 1:]

        # Forward Pass
        output = self(src, tgt_input)

        # Loss
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        loss = self.criterion(output, tgt_output)

        # Calculate Perplexity(ppl)
        num_tokens = (tgt_output != self.hparams.pad_idx).sum()
        loss_per_token = loss / num_tokens
        perplexity = torch.exp(loss_per_token.clamp(max=20))

        # Logging
        self.log("val_loss", loss_per_token, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_ppl", perplexity, on_step=False, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append({
            "loss": loss_per_token.detach(),
            "ppl": perplexity.detach(),
        })
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        avg_ppl = torch.stack([x["ppl"] for x in self.training_step_outputs]).mean()

        self.log("train_loss_epoch", avg_loss)
        self.log("train_ppl_epoch", avg_ppl)

        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        avg_ppl = torch.stack([x["ppl"] for x in self.validation_step_outputs]).mean()

        self.log("val_loss_epoch", avg_loss)
        self.log("val_ppl_epoch", avg_ppl)

        self.validation_step_outputs.clear()

# 모델 초기화 예제
if __name__ == "__main__":
    # Hyperparameters (Base model)
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    d_ff = 2048
    max_seq_len = 5000
    dropout = 0.1
    
    # Create model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 32
    src_seq_len = 20
    tgt_seq_len = 25
    
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    output = model(src, tgt)
    print(f"Output shape: {output.shape}")  # (batch_size, tgt_seq_len, tgt_vocab_size)
    
    print("\n✅ Model implementation complete!")