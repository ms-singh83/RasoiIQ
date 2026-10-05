# Blockers

What got in the way overnight, and what I did instead.

1. **Hugging Face was blocked from the build machine** (the cloud container's network policy returned 403). TabPFN 6.4.1 automatically falls back to Prior Labs' public mirror on Google Cloud Storage, and the 44 MB `tabpfn-v2-regressor.ckpt` downloaded from there with no login. On your Mac it will try Hugging Face first, which is also login-free for v2. **No action needed. HF_TOKEN is not required.**
2. **I couldn't test on an actual Intel Mac.** The build machine is Linux x86_64 with 4 cores. To reduce the risk:
   - The same pinned versions (torch 2.2.2, tabpfn 6.4.1, numpy 1.26.4) ran a real TabPFN prediction there.
   - `uv pip compile --python-platform x86_64-apple-darwin --python-version 3.11 --only-binary :all:` resolved every dependency to an existing Intel-Mac wheel (`requirements-macos-intel.lock.txt`).
   - On Linux, PyPI's torch 2.2.2 is the CUDA build (it ran on CPU). The Mac wheel is CPU-only anyway.
   - **Timings will differ.** Here a forecast took about 2 to 7 s and the 28-day backtest about 8 to 19 s. An older Intel iMac may be 2 to 4× slower. The page shows a spinner and the real time taken.
   - If `pip install` fails on the Mac, see "If install fails" in MORNING_SUMMARY.md.
3. **Gemma was not tested against the live API**, because no `GEMMA_API_KEY` was available. The request format, the fallback on network errors, and the rule that rejects a rewrite that changes numbers are all covered by tests with a mocked response. The default model ID `gemma-3-27b-it` is a known AI Studio Gemma model; if Google has retired it, set `GEMMA_MODEL`. Without a key the app uses the plain template, as designed.
4. **No real order data.** The results are on synthetic data only (see the caveats in the README).
5. **Proxy-only network blocking didn't simulate offline**, so the offline check was done inside a Linux network namespace with no network at all (`unshare -rn`). TabPFN forecast normally there from the cached model.
6. **Pushing to GitHub was refused at first (HTTP 403)** because the Claude GitHub App didn't have access to `ms-singh83/RasoiIQ`. A git bundle backup was sent in the meantime. **Resolved:** the push succeeded after access was fixed, and the branch is on GitHub.
