from oprl.buffers.episodic_buffer import EpisodicReplayBuffer
from oprl.buffers.protocols import ReplayBufferProtocol


def test_replay_buffer() -> None:
    state_dim = 7
    max_episode_length = 10
    num_transitions = 100
    buffer = EpisodicReplayBuffer(
        buffer_size_transitions=num_transitions,
        state_dim=state_dim,
        action_dim=3,
        max_episode_lenth=max_episode_length,
    ).create()
    assert isinstance(buffer, ReplayBufferProtocol)

    states = buffer.states
    assert len(states.shape) == 3
    assert states.shape[0] == num_transitions // max_episode_length
    assert states.shape[1] == max_episode_length + 1
    assert states.shape[2] == state_dim
