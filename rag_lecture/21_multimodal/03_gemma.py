r"""
### EmbeddingGemma 2 reads text and images

[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (Apache 2.0,
[technical report](https://arxiv.org/abs/2610.37364)) embeds text, images, video
and audio into one 768-dimensional space. Each modality encoder is loaded on
demand. The id is public and was checked against the hub at revision `f9972ac2`. We need
text and images, so we switch the audio tower off with
`config_kwargs`, which leaves about 440M parameters. In Sentence Transformers,
one input is a dict such as `{"image": page}`. See the
[multimodal guide](https://huggingface.co/blog/multimodal-sentence-transformers)
for audio, video and mixed inputs.

Two rules from the model card. Task prefixes apply to text only: questions use
the prompt `SearchQuery` and images get no prefix. And the model must not run in
float16, which the card says gives NaN or silently degraded embeddings. We load
bfloat16 on a GPU that supports it and float32 otherwise. How many parameters
did we actually load?
"""

# pin the model and revision so every run loads the same weights
GEMMA_MODEL_ID = "google/embeddinggemma-2"
GEMMA_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"

# pick a safe dtype: bf16 on capable GPUs, fp32 elsewhere (never fp16)
GEMMA_DTYPE = (
    torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
)

# load the model with the audio tower disabled, keeping only text + image
gemma_model = SentenceTransformer(
    GEMMA_MODEL_ID,
    revision=GEMMA_REVISION,
    device=DEVICE,
    model_kwargs={"dtype": GEMMA_DTYPE},
    config_kwargs={"audio_config": None},
)

# report the loaded parameter count and the text-only task prompts
print("Loaded parameters (M):", round(sum(p.numel() for p in gemma_model.parameters()) / 1e6))
print("Prompts:", {name: gemma_model.prompts[name] for name in ("SearchQuery", "Document")})
