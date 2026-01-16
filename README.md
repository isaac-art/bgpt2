# bgpt2

(Modded-NanoGPT Fork)

To run the current record, run the following commands.
```bash
pip install -r requirements.txt
pip install torch==2.10.0.dev20251210+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126
python data/fineweb.py 
./run.sh
```
Add torchrun to path if ./run.sh gives error `torchrun: command not found`.





## References

1. [Guilherme Penedo et al. "The fineweb datasets: Decanting the web for the finest text data at scale." arXiv preprint arXiv:2406.17557 (2024).](https://arxiv.org/abs/2406.17557)
2. Nicholas J. Higham. Functions of Matrices. Society for Industrial and Applied Mathematics (2008). Equation 5.22.
3. GÃ¼nther Schulz. Iterative Berechnung der reziproken Matrix. Z. Angew. Math. Mech., 13:57â59 (1933).
4. [Jeremy Bernstein and Laker Newhouse. "Old Optimizer, New Norm: An Anthology." arxiv preprint arXiv:2409.20325 (2024).](https://arxiv.org/abs/2409.20325)
5. [Vineet Gupta, Tomer Koren, and Yoram Singer. "Shampoo: Preconditioned stochastic tensor optimization." International Conference on Machine Learning. PMLR, 2018.](https://arxiv.org/abs/1802.09568)
6. [Rohan Anil et al. "Scalable second order optimization for deep learning." arXiv preprint arXiv:2002.09018 (2020).](https://arxiv.org/abs/2002.09018)
7. [Alexander HÃ¤gele et al. "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations." arXiv preprint arXiv:2405.18392 (2024).](https://arxiv.org/abs/2405.18392)
8. [Zhanchao Zhou et al. "Value Residual Learning For Alleviating Attention Concentration In Transformers." arXiv preprint arXiv:2410.17897 (2024).](https://arxiv.org/abs/2410.17897)
9. [Team, Gemma, et al. "Gemma 2: Improving open language models at a practical size." arXiv preprint arXiv:2408.00118 (2024).](https://arxiv.org/abs/2408.00118)
10. [Alec Radford et al. "Language models are unsupervised multitask learners." OpenAI blog 1.8 (2019).](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## Citation

```
@misc{modded_nanogpt_2024,
  author       = {Keller Jordan and Jeremy Bernstein and Brendan Rappazzo and
                  @fernbear.bsky.social and Boza Vlado and You Jiacheng and
                  Franz Cesista and Braden Koszarsky and @Grad62304977},
  title        = {modded-nanogpt: Speedrunning the NanoGPT baseline},
  year         = {2024},
  url          = {https://github.com/KellerJordan/modded-nanogpt}
}
```
