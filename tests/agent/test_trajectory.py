from __future__ import annotations

from miniprophet.agent.trajectory import TrajectoryRecorder


def test_trajectory_recorder_deduplicates_by_identity() -> None:
    recorder = TrajectoryRecorder()
    msg = {"role": "system", "content": "a"}

    k1 = recorder.register(msg)[0]
    k2 = recorder.register(msg)[0]

    assert k1 == "S0"
    assert k1 == k2


def test_trajectory_recorder_records_steps() -> None:
    recorder = TrajectoryRecorder()
    recorder.record_step(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        {"role": "assistant", "content": "a"},
    )

    payload = recorder.serialize()
    assert payload["steps"][0]["input"] == ["S0", "U0"]
    assert payload["steps"][0]["output"] == "A0"
