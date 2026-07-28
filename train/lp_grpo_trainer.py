"""Repository-owned terminal-stop override of TRL's experimental tool loop.

Defect (confirmed 2026-07-15, final routine eval): ``final_answer`` and
``human_escalation`` freeze the episode's ``AuditLog`` and compute reward, but
pinned TRL 1.5.1's experimental ``GRPOTrainer._tool_call_loop`` keeps executing
and regenerating for as long as the model emits tool calls. Post-terminal calls
raise ``episode over``; TRL counts a tool failure, feeds the error back, and
regenerates — all invisible to the frozen audit/reward, so a successful
completion's garbage tail inherits the same group-relative advantage as its
useful tokens (seed-2 conservative lower bound: 16.41 invisible post-terminal
generated calls per episode; seed-0 11.40).

Repair contract:

- pre-terminal calls execute exactly as pinned TRL executes them;
- the FIRST successfully executed terminal action (``final_answer`` /
  ``human_escalation``) ends that sample's tool loop immediately: any calls
  bundled later in the same assistant message are cancelled (never executed)
  and no further generation happens for that sample;
- a terminal call that RAISES (invalid args, invalid escalation code) is an
  ordinary tool failure — the episode is not over, the error is fed back, and
  the sample keeps going, exactly as before;
- message/token bookkeeping mirrors TRL's own "overlong" early-exit path: the
  final iteration's mutations are rolled back so ``completions`` stays
  consistent with ``completion_ids``/``tool_mask`` (the sample exits with its
  completion ending at the assistant message that contained the terminal
  call; the terminal action's env-side effects — frozen audit, reward,
  sandbox close — all stand);
- per-episode telemetry (generated / executed / cancelled-after-terminal /
  failed / post-terminal call counts, terminal_stopped flag) is attached to
  each environment instance as ``trainer_tool_stats`` so the rollout log can
  record it, and batch aggregates land in the trainer metrics under
  ``tools/*``.

The override is a copy of the pinned method plus clearly-marked ``LP-RL``
edits. It is guarded by BOTH the exact TRL version and a SHA-256 of the pinned
method's source: if either drifts (upgrade, patched site-packages), training
refuses to start until this copy is re-diffed against the new upstream.
Escape hatches: ``LP_RL_ALLOW_TRL_DRIFT=1`` skips the guard (after manual
review only); the launcher-side ``LP_RL_NO_TERMINAL_STOP=1`` selects the
pinned trainer class instead of this subclass (A/B diagnosis).
Never edit site-packages.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import os

import trl
from trl import GRPOTrainer
from trl.chat_template_utils import parse_response

# The two environment methods that end an episode. Must match the public
# terminal actions on least_privilege_env.env.terminal_env.TerminalEnv.
TERMINAL_TOOL_NAMES = frozenset({"final_answer", "human_escalation"})

PINNED_TRL_VERSION = "1.5.1"
# sha256 of inspect.getsource(GRPOTrainer._tool_call_loop) at the pinned
# version, computed 2026-07-15 in .venv-train. The override below is a copy of
# exactly that source; the hash guard makes silent upstream drift impossible.
PINNED_TOOL_CALL_LOOP_SHA256 = "7401e56a5e5cace78257b1e07e7b1e957e797e048146ee667c65b9e92bccae12"

_STAT_KEYS = (
    "generated_calls",
    "executed_calls",
    "cancelled_after_terminal",
    "failed_calls",
    "post_terminal_calls",
    "terminal_stopped",
)


def assert_pinned_tool_loop() -> None:
    """Refuse to run if TRL is not the exact pinned version + source."""
    if os.environ.get("LP_RL_ALLOW_TRL_DRIFT") == "1":
        return
    problems = []
    if trl.__version__ != PINNED_TRL_VERSION:
        problems.append(
            f"trl version is {trl.__version__}, override was written against {PINNED_TRL_VERSION}"
        )
    source = inspect.getsource(GRPOTrainer._tool_call_loop)
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != PINNED_TOOL_CALL_LOOP_SHA256:
        problems.append(
            "GRPOTrainer._tool_call_loop source hash "
            f"{digest[:12]}… != pinned {PINNED_TOOL_CALL_LOOP_SHA256[:12]}…"
        )
    if problems:
        raise RuntimeError(
            "LPTerminalStopGRPOTrainer overrides a copy of TRL's experimental "
            "_tool_call_loop, and the pinned upstream has changed: "
            + "; ".join(problems)
            + ". Re-diff train/lp_grpo_trainer.py against the installed TRL, "
            "update PINNED_TRL_VERSION / PINNED_TOOL_CALL_LOOP_SHA256, and "
            "re-run the terminal-stop regression tests. Set "
            "LP_RL_ALLOW_TRL_DRIFT=1 only after that manual review."
        )


def _new_stats() -> dict[str, int]:
    return {key: 0 for key in _STAT_KEYS}


class LPTerminalStopGRPOTrainer(GRPOTrainer):
    """GRPOTrainer whose tool loop stops a sample at its first terminal action."""

    def __init__(self, *args, **kwargs):
        assert_pinned_tool_loop()
        super().__init__(*args, **kwargs)

    # -- helpers ----------------------------------------------------------

    def _lp_env_for(self, idx_with_tool: int):
        """The environment instance owning sample idx's tools (or None)."""
        environments = getattr(self, "environments", None)
        if environments:
            try:
                return environments[idx_with_tool]
            except (IndexError, TypeError):
                pass
        for tool in self._sync_tool_dicts[idx_with_tool].values():
            owner = getattr(tool, "__self__", None)
            if owner is not None:
                return owner
        return None

    # -- the override ------------------------------------------------------
    # Copy of pinned TRL 1.5.1 GRPOTrainer._tool_call_loop; every deviation is
    # marked "LP-RL:". Do not restructure the copied parts — diffs against
    # upstream must stay reviewable.

    def _tool_call_loop(self, prompts, prompt_ids, completion_ids, completions, logprobs, images, multimodal_fields):
        # Tool execution loop: execute tools, then regenerate completions with tool results appended to the prompt
        tool_calls = [completion[0].get("tool_calls") for completion in completions]
        idxs_with_tool = [idx for idx, tool_call in enumerate(tool_calls) if tool_call]
        tool_calls = [tool_calls[idx] for idx in idxs_with_tool]
        tool_mask = [[1] * len(ids) for ids in completion_ids]  # 0 for tool result tokens, 1 elsewhere
        # Collect images from multimodal tool responses for the forward pass
        tool_images = [[] for _ in completion_ids]
        tool_call_count = 0
        tool_failure_count = 0
        iteration_num = 0
        stats: dict[int, dict[str, int]] = {}  # LP-RL: per-sample telemetry

        while idxs_with_tool and iteration_num < self.max_tool_calling_iterations:
            prompt_completion_tools = [prompts[i] for i in idxs_with_tool]  # select only prompts that need tool calls
            # Snapshot state so we can rollback tool results that would exceed max_completion_length
            completions_len_before = [len(completions[i]) for i in idxs_with_tool]
            tool_images_len_before = [len(tool_images[i]) for i in idxs_with_tool]
            prompts_len_before = [len(prompts[i]) for i in idxs_with_tool]
            terminated_positions: set[int] = set()  # LP-RL: samples ending at a terminal action this iteration

            # Call the tools, and build the new prompt for generation
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                tool_call_list = tool_calls[idx]
                prompt_completion_tool = prompt_completion_tools[idx]
                sync_tool_dict = self._sync_tool_dicts[idx_with_tool]
                async_tool_dict = self._async_tool_dicts[idx_with_tool]
                # Append the last assistant message (which triggered tool_calls) to the prompt
                prompt_completion_tool.append(completions[idx_with_tool][-1])
                async_coros = []
                tool_call_results = []
                # LP-RL: telemetry for this sample + terminal detection state
                st = stats.setdefault(idx_with_tool, _new_stats())
                st["generated_calls"] += len(tool_call_list)
                terminated_here = False
                for call_position, tool_call in enumerate(tool_call_list):  # LP-RL: enumerate for cancellation
                    tool_call_count += 1
                    if tool_call["type"] == "function":
                        function = tool_call["function"]
                        name = function["name"]
                        try:
                            if name in sync_tool_dict:
                                result = sync_tool_dict[name](**function["arguments"])
                                tool_call_results.append((name, result))
                                st["executed_calls"] += 1  # LP-RL
                                # LP-RL: the first SUCCESSFUL terminal action ends
                                # this sample now. Later calls bundled in the same
                                # assistant message are cancelled, never executed.
                                if name in TERMINAL_TOOL_NAMES:
                                    terminated_here = True
                                    remaining = len(tool_call_list) - call_position - 1
                                    st["cancelled_after_terminal"] += remaining
                                    st["terminal_stopped"] = 1
                                    break
                            elif name in async_tool_dict:
                                async_coros.append((name, async_tool_dict[name](**function["arguments"])))
                            else:
                                raise ValueError(f"Tool {name} not found.")
                        except Exception as e:
                            tool_failure_count += 1
                            st["failed_calls"] += 1  # LP-RL
                            if "episode over" in str(e):  # LP-RL: must stay 0 with this override
                                st["post_terminal_calls"] += 1
                            result = {"error": str(e)}
                            tool_call_results.append((name, result))
                    else:
                        tool_failure_count += 1
                        st["failed_calls"] += 1  # LP-RL
                        name = tool_call.get("name", "unknown")
                        tool_call_results.append((name, {"error": f"Unsupported tool call type: {tool_call['type']}"}))

                if async_coros:

                    async def _run_async_tools(async_coros):
                        coros = [coro for _, coro in async_coros]
                        results = await asyncio.gather(*coros, return_exceptions=True)
                        return [(name, result) for (name, _), result in zip(async_coros, results, strict=False)]

                    async_results = asyncio.run_coroutine_threadsafe(
                        _run_async_tools(async_coros), self.async_loop
                    ).result()

                    for name, result in async_results:
                        if isinstance(result, Exception):
                            tool_failure_count += 1
                            st["failed_calls"] += 1  # LP-RL
                            tool_call_results.append((name, {"error": str(result)}))
                        else:
                            st["executed_calls"] += 1  # LP-RL
                            tool_call_results.append((name, result))

                # LP-RL: a terminated sample gets NO tool-result messages — it
                # exits the loop with its completion ending at the assistant
                # message that contained the terminal call (rollback below
                # restores the prompt copy). Env-side effects already stand.
                if terminated_here:
                    terminated_positions.add(idx)
                    continue

                for name, result in tool_call_results:
                    # Support multimodal tool responses: if the tool returns a list of content blocks
                    # (e.g., [{"type": "image", "image": ...}, {"type": "text", "text": "..."}]),
                    # pass them through directly so _tokenize_prompts can extract images for VLMs.
                    content = result if isinstance(result, list) else str(result)
                    tool_message = {"role": "tool", "name": name, "content": content}
                    # Collect images from multimodal tool responses
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "image":
                                tool_images[idx_with_tool].append(part["image"])
                    prompt_completion_tool.append(tool_message)
                    completions[idx_with_tool].append(tool_message)

            # LP-RL: retire terminated samples, mirroring the pinned "overlong"
            # early-exit bookkeeping (roll back this iteration's mutations so
            # completions/tool_images stay consistent with completion_ids, and
            # drop the sample from further regeneration).
            if terminated_positions:
                for position in terminated_positions:
                    idx_with_tool = idxs_with_tool[position]
                    del completions[idx_with_tool][completions_len_before[position] :]
                    del tool_images[idx_with_tool][tool_images_len_before[position] :]
                    del prompts[idx_with_tool][prompts_len_before[position] :]
                keep = [p for p in range(len(idxs_with_tool)) if p not in terminated_positions]
                idxs_with_tool = [idxs_with_tool[p] for p in keep]
                tool_calls = [tool_calls[p] for p in keep]
                completions_len_before = [completions_len_before[p] for p in keep]
                tool_images_len_before = [tool_images_len_before[p] for p in keep]
                prompts_len_before = [prompts_len_before[p] for p in keep]
                if not idxs_with_tool:
                    break

            # Build token IDs by concatenation: prompt + completion + tool_suffix.
            prompt_completion_tool_ids = []
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                # Extract trailing tool messages from completions
                tool_messages = []
                for message in reversed(completions[idx_with_tool]):
                    if message["role"] == "tool":
                        tool_messages.insert(0, message)
                    else:
                        break
                suffix_ids = self._get_tool_suffix_ids(tool_messages)
                prompt_completion_tool_ids.append(
                    prompt_ids[idx_with_tool] + completion_ids[idx_with_tool] + suffix_ids
                )

            # Drop tool results whose addition would push the sequence past max_completion_length (the completion
            # budget) or past the backend context ceiling (vLLM and transformers will error out on inputs longer than
            # the model's max length). The sample exits the loop with its completion as-is, and the tool
            # messages/images appended this iteration are rolled back so completions and tool_images stay consistent
            # with completion_ids.
            if self.use_vllm and self.vllm_mode == "colocate":
                max_model_len = self.vllm_generation.llm.llm_engine.model_config.max_model_len
            else:
                config = self.model.config.text_config if self._is_vlm else self.model.config
                max_model_len = config.max_position_embeddings
            overlong = [
                len(pct) - len(prompt_ids[i]) > self.max_completion_length or len(pct) >= max_model_len
                for i, pct in zip(idxs_with_tool, prompt_completion_tool_ids, strict=True)
            ]
            for idx in range(len(idxs_with_tool)):
                if overlong[idx]:
                    idx_with_tool = idxs_with_tool[idx]
                    del completions[idx_with_tool][completions_len_before[idx] :]
                    del tool_images[idx_with_tool][tool_images_len_before[idx] :]
                    del prompts[idx_with_tool][prompts_len_before[idx] :]
            # Keep only non-overlong items for further processing
            idxs_with_tool = [idx for idx, o in zip(idxs_with_tool, overlong, strict=True) if not o]
            prompt_completion_tool_ids = [
                pct for pct, o in zip(prompt_completion_tool_ids, overlong, strict=True) if not o
            ]
            if not idxs_with_tool:
                break  # all overlong, exit tool loop

            # Filter images and multimodal fields to match the current subset (index into full batch).
            # Merge tool response images so the model can see visual feedback during generation.
            merged_images = images
            if any(imgs for imgs in tool_images):
                if merged_images is None:
                    merged_images = [imgs if imgs else None for imgs in tool_images]
                else:
                    merged_images = [
                        (existing or []) + new for existing, new in zip(merged_images, tool_images, strict=True)
                    ]
            loop_images = [merged_images[i] for i in idxs_with_tool] if merged_images else None
            if multimodal_fields:
                loop_multimodal_fields = {}
                for k, v in multimodal_fields.items():
                    selected = [v[i] for i in idxs_with_tool]
                    # Per-token fields (e.g. token_type_ids) need zero-padding to match extended prompt length
                    if isinstance(selected[0], list):
                        selected = [
                            s + [0] * (len(pct) - len(s))
                            for s, pct in zip(selected, prompt_completion_tool_ids, strict=True)
                        ]
                    loop_multimodal_fields[k] = selected
            else:
                loop_multimodal_fields = {}

            # Generate new completions after tool execution (using concatenated IDs, no re-tokenization)
            post_tool_ids, post_tool_logprobs = self._generate_single_turn(
                prompt_completion_tool_ids, loop_images, loop_multimodal_fields
            )

            # Truncate so that pct[len(prompt_ids[idx]) :] + post_tool does not exceed max_completion_length.
            # The pre-regen check guarantees len(completion_tool_ids) <= max_completion_length, so any
            # excess can only come from post_tool_ids. post_tool_ids is model-generated text and never
            # contains image tokens, so a plain slice is safe.
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                completion_tool_length = len(prompt_completion_tool_ids[idx]) - len(prompt_ids[idx_with_tool])
                excess_length = completion_tool_length + len(post_tool_ids[idx]) - self.max_completion_length
                if excess_length > 0:
                    new_len = len(post_tool_ids[idx]) - excess_length
                    post_tool_ids[idx] = post_tool_ids[idx][:new_len]
                    if logprobs is not None:
                        post_tool_logprobs[idx] = post_tool_logprobs[idx][:new_len]

            # Update tool_mask: the tool result should be 0 and the post-tool 1
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_completion_tool_length = len(prompt_completion_tool_ids[idx])
                prompt_length = len(prompt_ids[idx_with_tool])
                completion_length = len(completion_ids[idx_with_tool])
                post_tool_length = len(post_tool_ids[idx])
                tool_length = prompt_completion_tool_length - prompt_length - completion_length
                tool_mask[idx_with_tool] += [0] * tool_length + [1] * post_tool_length
                if logprobs is not None:
                    logprobs[idx_with_tool] += [0.0] * tool_length + post_tool_logprobs[idx]

            # Update completion_ids with the new completions (after tool execution)
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_length = len(prompt_ids[idx_with_tool])
                pct = prompt_completion_tool_ids[idx]  # = prompt-completion-tool
                completion_ids[idx_with_tool] = pct[prompt_length:] + post_tool_ids[idx]

            # Decode post-tool completions.
            post_tool_completions = [parse_response(self._tokenizer, ids) if ids else {} for ids in post_tool_ids]

            # Add post-tool completions to the existing completions
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                if post_tool_completions[idx]:  # {} if post-tool completions completely truncated
                    completions[idx_with_tool].append(post_tool_completions[idx])

            # Check for further tool calls
            tool_calls = [completion.get("tool_calls") for completion in post_tool_completions]
            idxs_with_tool = [idx for idx, tool_call in zip(idxs_with_tool, tool_calls, strict=True) if tool_call]
            tool_calls = [tool_call for tool_call in tool_calls if tool_call]
            iteration_num += 1

        # LP-RL: publish telemetry — per-episode onto the env (for the rollout
        # log, which is written at reward time, after this method returns) and
        # batch aggregates into the trainer metrics (single-process values; no
        # accelerator gather, matching this repo's single-trainer-GPU setup).
        totals = _new_stats()
        for idx_with_tool, st in stats.items():
            env = self._lp_env_for(idx_with_tool)
            if env is not None:
                try:
                    env.trainer_tool_stats = dict(st)
                except Exception:
                    pass
            for key in _STAT_KEYS:
                totals[key] += st[key]
        n_episodes = max(len(completion_ids), 1)
        mode = "train" if self.model.training else "eval"
        metrics = getattr(self, "_metrics", None)
        if metrics is not None:
            metrics[mode]["tools/generated_calls_per_episode"].append(totals["generated_calls"] / n_episodes)
            metrics[mode]["tools/executed_calls_per_episode"].append(totals["executed_calls"] / n_episodes)
            metrics[mode]["tools/cancelled_after_terminal_per_episode"].append(
                totals["cancelled_after_terminal"] / n_episodes
            )
            metrics[mode]["tools/post_terminal_calls_per_episode"].append(
                totals["post_terminal_calls"] / n_episodes
            )
            metrics[mode]["tools/terminal_stopped_frac"].append(totals["terminal_stopped"] / n_episodes)

        return tool_mask, completions, completion_ids, logprobs, tool_call_count, tool_failure_count, tool_images
