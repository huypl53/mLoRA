# mLoRA

## [LLMModel](mlora/model.py#L228)

- `LLMModel` is a wrapper of `LLMForCausalLM`
- `_call_decoder_stack()`: call `LLMForCausalLM`'s `decoder_stack()` then its `norm()`

## `LLMForCausalLM`

`LLMForCausalLM` is from `transformers` and is `just the decoder stacks` of Transformer architecture.

Attributes:

- decoder*stack(): return `self.layers*`

### `LlamaForCausalLM`

- is inherited from `LLMForCausalLM`
  **Attributes**:
- `from_pretrained()`: return new self
- `self.layers_`: list of [LlamaDecoderLayer](#llamadecoderlayer)

### `LlamaDecoderLayer`

**Attrs**:

- `self.mlp_`: [FeedForward](#feedforward) that has a [LlamaMLP](#llamamlp) **self.mpl\_**

#### forward()

- `self.input_layernorm_()`: [LlamaRMSNorm]
- `self.self_attn_()`: [LlamaAttention]
- `self.post_attention_layernorm_()`: [LlamaRMSNorm]
- `self.mlp_()`: [FeedForward](#feedforward)

## [LlamaMLP](mlora/models/modeling_llama.py#L299)

## `OutputLayer`

## `LLMModelInput`

- contains multiple batchs, each has `adapter_name, batch_start_idx, batch_end_idx`

## [FeedForward](mlora/common/feed_forward.py#L13)

- where **moe** is applied
