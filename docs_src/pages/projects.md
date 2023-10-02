Einops tutorials cover multiple einops usages (and you'd better first follow tutorials), 
but it can also help to see einops in action.

## Selected projects

Here are some open-source projects that can teach how to leverage einops for your problems


- [@lucidrains](https://github.com/lucidrains) has a dramatic [collection of vision transformers](https://github.com/lucidrains/vit-pytorch)
    - there is a plenty of good examples how to use einops efficiently in your projects


- lambda networks (non-conventional architecture) implemented by @lucidrains
    - nice demonstration how clearer code can be with einops, even compared to description in the paper 
    - [implementation](https://github.com/lucidrains/lambda-networks) and [video](https://www.youtube.com/watch?v=3qxJ2WD8p4w)


- capsule networks (aka capsnets) [implemented in einops](https://github.com/arogozhnikov/readable_capsnet)
    - blazingly fast, concise (3-10 times less code), and memory efficient (3 times lower memory consumption) capsule networks, written with einops  


- [NuX](https://github.com/Information-Fusion-Lab-Umass/NuX) — normalizing flows in Jax
    - different rearrangement patterns in normalizing flows have nice mapping to einops


- For video recognition, look at [MotionFormer](https://github.com/facebookresearch/Motionformer) 
  and [TimeSFormer](https://github.com/lucidrains/TimeSformer-pytorch) implementations


- For protein folding, see [implementation](https://github.com/lucidrains/invariant-point-attention)
  of invariant point attention from alphafold 2

## Community introductions to einops

Tutorial in the AI summer about einops and einsum:
<https://theaisummer.com/einsum-attention/>

Introduction to einops by Kapil Sachdeva
<https://www.youtube.com/watch?v=xGy75Pjsqzo>

Implementing visual transformer in pytorch:
<https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632>

Refactoring machine learning code, one of posts in a series is devoted to einops:
<https://www.paepper.com/blog/posts/refactoring-machine-learning-code-einops/>

ML TLDR thread on einops:
<https://twitter.com/mlsummaries/status/1400505282543955970>

Book "Deep Reinforcement Learning in Action" by Brandon Brown & Alexander Zai
contains an introduction into einops in chapter 10.

[comment]: <> (MLP mixer introduction)
[comment]: <> (https://www.youtube.com/watch?v=HqytB2GUbHA)

## Other einops-based projects worth looking at:

(ordered randomly)

- <https://github.com/The-AI-Summer/self-attention-cv>
- <https://github.com/lucidrains/perceiver-pytorch>
- <https://github.com/hila-chefer/Transformer-Explainability>
- [https://github.com/microsoft/CvT](https://github.com/microsoft/CvT/blob/4cedb05b343e13ab08c0a29c5166b6e94c751112/lib/models/cls_cvt.py)
- <https://github.com/lucidrains/g-mlp-gpt>
- <https://github.com/zju3dv/LoFTR>
- <https://github.com/WangFeng18/Swin-Transformer>
- <https://github.com/kwea123/CasMVSNet_pl>
- <https://github.com/kakao/DAFT>
- <https://github.com/lucidrains/multistream-transformers>
- <https://github.com/poets-ai/elegy>
- <https://github.com/lucidrains/ponder-transformer>
- <https://github.com/isaaccorley/torchrs>
- <https://github.com/microsoft/esvit>
- <https://github.com/zyddnys/manga-image-translator>
- <https://github.com/google/jax-cfd>


## Related projects:

- [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) &mdash; grand-dad of einops, this operation is now available in all modern DL frameworks 
- einops in Rust language <https://docs.rs/einops/0.1.0/einops>
- einops in C++ for torch: <https://github.com/MaxCoo/einops-cpp>
- tensorcast in Julia language <https://juliahub.com/ui/Packages/TensorCast>
- for those chasing an extreme compactness of API, <https://github.com/cgarciae/einop> provides 'one op', as the name suggests
