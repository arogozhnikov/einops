## Tools for einops users

- [Ein Color](https://marketplace.visualstudio.com/items?itemName=MarcinJachmann.vscode-eincolor) — a VS Code extension to color axes in einops/einsum patterns
- Sonar codechecker provides a [rule](https://rules.sonarsource.com/python/RSPEC-6984/) to statically check einops patterns: 

## Selected projects implemented with einops

Einops tutorials cover many common usages (cover tutorials first!), but it is also useful to see real projects that apply einops in practice. The projects below illustrate how einops can simplify code in various domains.

- [@lucidrains](https://github.com/lucidrains) has a dramatic [collection of vision transformers](https://github.com/lucidrains/vit-pytorch)
    - there is a plenty of good examples how to use einops efficiently in your projects


- lambda networks (non-conventional architecture) implemented by @lucidrains
    - nice demonstration how clearer code can be with einops, even compared to description in the paper 
    - [implementation](https://github.com/lucidrains/lambda-networks) and [video](https://www.youtube.com/watch?v=3qxJ2WD8p4w)


- capsule networks (aka capsnets) [implemented in einops](https://github.com/arogozhnikov/readable_capsnet)
    - this implementation is blazingly fast, concise (3-10 times less code), and memory efficient


- [NuX](https://github.com/Information-Fusion-Lab-Umass/NuX) — normalizing flows in Jax
    - different rearrangement patterns in normalizing flows have nice mapping to einops


- For video recognition, look at [MotionFormer](https://github.com/facebookresearch/Motionformer) 
  and [TimeSFormer](https://github.com/lucidrains/TimeSformer-pytorch) implementations


- For protein folding, see [alphafold3-pytorch](https://github.com/lucidrains/alphafold3-pytorch) and [implementation](https://github.com/lucidrains/invariant-point-attention) of invariant point attention from AlphaFold 2


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



## Related projects

- [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) &mdash; grand-dad of einops, this operation is now available in all mainstream DL frameworks 
- einops in Rust language: <https://docs.rs/einops/0.1.0/einops>
- einops in C++ for torch: <https://github.com/dorpxam/einops-cpp>
- tensorcast in Julia language: <https://juliahub.com/ui/Packages/TensorCast>
- one-to-one einops implementation in Julia language: <https://murrellgroup.github.io/Einops.jl/stable/>
- einops in R language: <https://qile0317.github.io/einops/>
- for those chasing an extreme compactness of API, <https://github.com/cgarciae/einop> provides 'one op', as the name suggests
- <https://github.com/fferflo/einx> goes in opposite direction and creates einops-style operation for anything
