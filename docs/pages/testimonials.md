<style>
.md-typeset blockquote {
    background-color: rgba(128, 128, 128, 0.04);
    border-color: #002ee380;
    color: #333;
    margin-top: 2.5em;
    margin-bottom: -0.5em;
    margin-right: 3em;
    padding-right: 2em;
}
blockquote + p {
    text-align: right;
    padding-right: 5em;
}
</style>
Einops was created three years ago, and never hit big ML news.
However, little by little and step by step it sneaked into every major AI lab.

This all happened by word of mouth and by sharing code snippets:


> Einops simplifies and clarifies array/tensor manipulation. 👍 <br />
> You really have to try it out, you'll love it: https://github.com/arogozhnikov/einops <br />
> Plus it supports NumPy, TensorFlow, PyTorch, and more.

Aurélien Geron, <br />
author of "Hands-On Machine Learning with Scikit-Learn and TensorFlow." <br /> 
Former PM of YouTube video classification.
[(ref)](https://twitter.com/aureliengeron/status/1382829421967515648)


> This is your daily reminder to *never* trust pytorch's native reshape/view. <br /> 
> Always use einops! 
> 
> (Just spend 1 h debugging code and it turned out tensor.view was shuffling the tensor in a weird way) 


Tom Lieberum, University of Amsterdam
[(ref)](https://twitter.com/lieberum_t/status/1427282842250358787)


> TIL einops can be faster than raw PyTorch 🤯

Zach Mueller, fastai
[(ref)](https://twitter.com/TheZachMueller/status/1418003372494426113)

> einops are also a joy to work with!

Norman Casagrande, Deepmind
[(ref)](https://twitter.com/nova77t/status/1419405150805008387)


> &mdash; And btw I estimate that AI research suffers globally from a 5% loss of productivity because einops are not included in 
@PyTorch by default.
>
> &mdash; Might be true for research, but not likely to be true for engineering. I don't think a lot of people in the industry use PyTorch directly [...]
>
> &mdash; That’s why it’s 5% and not 30%
>
> &mdash; E-xa-ctly

[Discussion thread](https://twitter.com/francoisfleuret/status/1409141186326106114)


> After a while, it feels like einsum+einops is all you need ;) [...]
 
Tim Rocktäschel, Facebook AI Research 
[(ref)](https://twitter.com/_rockt/status/1390049226193788930)


> Yes, I am also using einops in that snippet! It's great! <br /> 
> I wished I knew about it from the start when it was created

Tim Dettmers, PhD Student at UoW and visiting researcher at Facebook AI
[(ref)](https://twitter.com/Tim_Dettmers/status/1390027329351520256)

> A little late to the party, but einops (https://github.com/arogozhnikov/einops) is a massive improvement to deep learning code readability. I love this!

Daniel Havir, Apple [(ref)](https://twitter.com/danielhavir/status/1389070232853966849)


Comment: some of 'late to the party' tweets are 2 years old now. You can never be late to this party.



> I recently discovered the beauty of torch.einsum and einops.rearrange <br /> 
> and at this point I'm confused why I even bothered with other tensor operations in the first place.

Robin M. Schmidt, AI/ML Resident at Apple [(ref)](https://twitter.com/robinschmidt_/status/1363709832788852736)

 
[comment]: <> (> The einops library, in particular, is magical. Best thing since baked bread and brewed beer.)

> I end up using einops for ridiculously simple things, <br />
> simply to be nice to my future self (because the code takes so little effort to read).

Nasim Rahaman, MILA [(ref)](https://twitter.com/nasim_rahaman/status/1390027557546901504)

> I love Einops for this kind of stuff, it makes code very readable, <br /> 
> even if you are just doing a simple squeeze or expand_dims. [...]

Cristian Garcia, ML Engineer @quansightai, [(ref)](https://twitter.com/cgarciae88/status/1331968395110211586)

> i might be late to the party, but einsum and the einops package are unbelievably useful

Samson Koelle, Stat PhD candidate at UoW, [(ref)](https://twitter.com/SeattleStatSam/status/1338673646898794496)

> I really recommend einops for tensor shape manipulation


Alex Mordvintsev, DeepDream creator, [(ref)](https://twitter.com/zzznah/status/1315297985585123328)

 
> The einops Python package is worth checking out: 
> it provides a powerful declarative interface 
> for manipulating and reshaping multi-dimensional arrays https://github.com/arogozhnikov/einops

Jake VanderPlas, 
Google Research, core contributor to Altair, AstroML, scikit-learn, etc.
[(ref)](https://twitter.com/jakevdp/status/1299012119761833989)



> I can't believe after this many years of programming with NumPy/PyTorch/TensorFlow, I didn't know about 𝚎𝚒𝚗𝚜𝚞𝚖. [...] <br/>
> &mdash; Arash Vahdat
> 
> They smell like regexp to me -- concise, but I know it is going to take effort to understand or modify them in the future.  <br/>
> &mdash; John Carmack
> 
> I have a much easier time to read einsum than any equivalent combinations of matmul, reshape, broadcasting... you name it.
> Regexps are ad-hoc, subtle and cryptic.
> 
> Einstein summation is uniform, succinct with simple, clear semantics. <br />
> Ein sum to rule them all ;) <br />
> &mdash; Christian Szegedy
> 
> The einops library, in particular, is magical. Best thing since baked bread and brewed beer. <br />
> &mdash; @aertherks

[Discussion thread](https://twitter.com/aertherks/status/1357054506656165889) 