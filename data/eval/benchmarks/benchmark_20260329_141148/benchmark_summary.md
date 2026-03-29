# Benchmark Summary

Generated at (UTC): 2026-03-29T14:55:46.435837+00:00
Stage A runs: 12
Stage B runs: 8

## Stage A Top Families
- enhanced_v2 + off + local_minilm: score=0.6485, single_latency=2817.93ms, multi_latency=4212.62ms
- baseline_v1 + off + local_minilm: score=0.6449, single_latency=2976.05ms, multi_latency=4126.16ms
- baseline_v1 + off + openai_large: score=0.5974, single_latency=3691.80ms, multi_latency=4361.05ms
- enhanced_v2 + off + openai_large: score=0.5712, single_latency=3976.64ms, multi_latency=4598.96ms
- enhanced_v2 + llm + local_minilm: score=0.5185, single_latency=4399.99ms, multi_latency=5229.77ms

## Stage B Configs
- enhanced_v2 + off + local_minilm: single_latency=3452.74ms, multi_latency=3916.37ms
- enhanced_v2 + off + local_mpnet: single_latency=2913.41ms, multi_latency=4123.70ms
- enhanced_v2 + off + openai_small: single_latency=3477.32ms, multi_latency=4432.02ms
- enhanced_v2 + off + openai_large: single_latency=3615.45ms, multi_latency=4631.80ms
- baseline_v1 + off + local_minilm: single_latency=2730.97ms, multi_latency=3888.21ms
- baseline_v1 + off + local_mpnet: single_latency=2802.28ms, multi_latency=4032.17ms
- baseline_v1 + off + openai_small: single_latency=3559.58ms, multi_latency=4340.97ms
- baseline_v1 + off + openai_large: single_latency=3982.48ms, multi_latency=4526.26ms