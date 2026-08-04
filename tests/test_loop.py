"""Tests for the loop agent's *Runtime* — the deterministic parts.

We never test the LLM's choices (non-deterministic). We test how the Runtime
reacts to a given action + state: validate_action, CompletionPolicy, and the
loop itself driven by a scripted (fake) policy instead of a real LLM.
"""

import unittest

from minirag.domain.models import SearchedChunk
from minirag.agents.loop.agent import (
    SearchAction,
    InspectAction,
    FinishAction,
    Step,
    RetrieverState,
    RetrieverLimits,
    CompletionPolicy,
    RetrieverAgent,
    validate_action,
)

# --- Fakes: replace the LLM policy and the real tools --------------------


class ScriptedPolicy:
    """Fake policy: yields pre-written actions instead of calling an LLM.
    This is what makes the loop deterministically testable."""

    def __init__(self, actions):
        self._actions = iter(actions)

    def decide(self, state):
        return next(self._actions)


class FakeTools:
    """Fake RetrievalTools: search returns a fixed chunk, no network/vstore."""

    def __init__(self, chunks=None):
        self._chunks = chunks if chunks is not None else [_chunk("c1")]

    def search(self, query, method, top_k):
        return self._chunks[:top_k]


def _chunk(chunk_id: str) -> SearchedChunk:
    return SearchedChunk(
        chunk_id=chunk_id,
        document=f"content of {chunk_id}",
        metadata={},
        embedding=[0.0],
        score=1.0,
    )


# --- Pure-function guards (no agent, no LLM) ------------------------------


class TestCompletionPolicy(unittest.TestCase):
    def test_rejects_finish_when_nothing_inspected(self):
        state = RetrieverState(goal="x")
        allowed, reason = CompletionPolicy().can_finish(state)
        self.assertFalse(allowed)
        self.assertEqual(reason, "no_document_inspected")

    def test_allows_finish_with_two_inspected(self):
        state = RetrieverState(goal="x")
        state.inspected_documents = {"a": "...", "b": "..."}
        allowed, _ = CompletionPolicy().can_finish(state)
        self.assertTrue(allowed)


class TestValidateAction(unittest.TestCase):
    def test_duplicate_search_rejected(self):
        state = RetrieverState(goal="x")
        action = SearchAction(type="search", query="q", top_k=5)
        state.steps.append(Step(action=action, observation={}))  # searched "q" once
        reason = validate_action(action, state, RetrieverLimits())
        self.assertEqual(reason, "duplicate_search")


# --- Loop level: scripted policy directs the story -----------------------


class TestLoop(unittest.TestCase):
    def test_premature_finish_is_rejected_then_recorded(self):
        # Model tries to finish empty-handed on step 1.
        # Given
        policy = ScriptedPolicy([FinishAction(type="finish", reason="too early")])
        agent = RetrieverAgent(
            policy=policy, tools=FakeTools(), limits=RetrieverLimits(max_steps=1)
        )

        # When
        state = agent.run("x")

        # Then
        self.assertEqual(state.steps[0].observation["status"], "rejected")
        self.assertEqual(state.steps[0].observation["reason"], "no_document_inspected")

    def test_happy_case(self):
        # Given
        policy = ScriptedPolicy(
            [
                SearchAction(type="search", query="test query"),
                InspectAction(type="inspect", chunk_id="1"),
                InspectAction(type="inspect", chunk_id="2"),
                FinishAction(type="finish", reason="good reason"),
            ]
        )
        agent = RetrieverAgent(
            policy=policy,
            tools=FakeTools([_chunk("1"), _chunk("2")]),
            limits=RetrieverLimits(max_steps=5),
        )

        # When
        state = agent.run("x")

        # Then
        self.assertEqual(state.steps[0].observation["status"], "ok")
        self.assertEqual(state.steps[1].observation["status"], "ok")
        self.assertEqual(state.steps[3].observation["status"], "finished")
        self.assertEqual(state.steps[3].observation["reason"], "good reason")

        self.assertTrue(state.finished)
        self.assertEqual(state.final_document_ids, ["1", "2"])


if __name__ == "__main__":
    unittest.main()
