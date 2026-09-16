# The KV cache in numbers: what it costs to remember the conversation so far.

def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_value=2):
    return 2 * layers * kv_heads * head_dim * bytes_per_value     # a key AND a value


LAYERS, HEADS, HEAD_DIM = 32, 32, 128                             # a 7B-class model
for label, kv_heads in [("every head keeps its own keys and values", HEADS),
                        ("grouped: 8 key-value heads shared", 8)]:
    per_token = kv_bytes_per_token(LAYERS, kv_heads, HEAD_DIM)
    print(label)
    print(f"  {per_token / 1024:,.0f} KB per token")
    for context in (4_096, 32_768, 128_000):
        print(f"  {context:>7,} tokens of context -> {per_token * context / 1e9:6.1f} GB "
              "for ONE sequence")
    print()

weights_gb = 7e9 * 2 / 1e9
full = kv_bytes_per_token(LAYERS, HEADS, HEAD_DIM) * 128_000 / 1e9
print(f"the weights of the same model at 16-bit: {weights_gb:.0f} GB")
print(f"one full-length conversation's cache is {full / weights_gb:.1f}x the weights")
