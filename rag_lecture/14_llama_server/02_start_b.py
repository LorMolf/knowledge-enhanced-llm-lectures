r"""
### Start it, once

The first run downloads the llama.cpp build and then the 5 GB GGUF; both are cached
on disk, so a second run only starts the process. It is detached, because a server
that dies with the cell cannot be called by the next one. The wait is a poll of
`/props`, not a sleep: the weights take about a minute to load on a T4, and guessing
a duration is how notebooks break on a slower machine. On timeout the last twenty
log lines are printed instead of a stack trace.
"""

# 1. Download and unpack once; the second run skips both steps.
if not LLAMA_DIR.exists():
    archive = Path(LLAMA_ASSET)
    if not archive.exists():
        print("Downloading", LLAMA_URL)
        urllib.request.urlretrieve(LLAMA_URL, archive)
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(".", filter="data")
        except TypeError:  # Python < 3.12 has no extraction filter
            tar.extractall(".")

# 2. Start it detached, then wait for the health endpoint instead of guessing.
if not server_ready():
    # A server that died with an old kernel can hold the port; clear it first.
    subprocess.run(["pkill", "-f", str(LLAMA_DIR / "llama-server")], check=False)
    log = open("llama-server.log", "wb")
    subprocess.Popen(
        [str(LLAMA_DIR / "llama-server"), *LLAMA_FLAGS],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # survives the notebook process
    )
    for _ in range(180):
        if server_ready():
            break
        time.sleep(2)

# 3. One flag the next cells check, and the log tail when it never came up.
SERVER_READY = server_ready()
print("server ready:", SERVER_READY, "| log: llama-server.log")
if not SERVER_READY:
    tail = Path("llama-server.log").read_text(errors="replace").splitlines()[-20:]
    print("\n".join(tail))
    if "GLIBC" in "\n".join(tail):
        print(
            "\nThis image has an older glibc than the pinned llama.cpp build needs. Two ways out:\n"
            "  - build it here: cmake -B build -DGGML_CUDA=1 && cmake --build build -j --target llama-server\n"
            "  - or accept the skip: section 14 reports the failure and sections 15-17 run with\n"
            "    stub answers, so the rest of the lecture still executes end to end."
        )
