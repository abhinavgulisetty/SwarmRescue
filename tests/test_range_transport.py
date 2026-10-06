"""Independent range receivers, typed frames and lossy delivery invariants."""

from dataclasses import replace

import numpy as np
import pytest

from ember.comms import GaussianPacket, packetize_gaussians
from ember.localization import RSSIConfig, sample_rssi
from ember.radio import RadioConfig, RadioMedium
from ember.range_comms import MAX_RANGE_UPDATES, RangePacket, packetize_ranges
from ember.range_gbp import RangeFactorGBP
from ember.range_team import RangeLocalizationTeam


def world(*, loss=0, walls=None, seed=7, beacons=2):
    poses = np.array([[0, 0], [8, 0], [0, 8], [8, 8]], float)
    cfg = RSSIConfig(noise_std_db=0.45)
    radio = RadioMedium(
        poses,
        wall_counts=walls,
        loss_seed=seed,
        config=RadioConfig(packet_error_rate=loss),
    )
    covariance = np.tile(np.eye(2) * 0.01, (4, 1, 1))
    team = RangeLocalizationTeam(poses, covariance, beacons, radio, sensor_config=cfg)
    truth = np.array([[2, 3], [6, 5]], float)[:beacons]
    rng = np.random.default_rng(19)
    readings = np.column_stack(
        [sample_rssi(poses, point, rng, config=cfg) for point in truth]
    )
    return team, readings, covariance


def packet(epoch=0):
    return RangePacket(0, 7, 2, epoch, np.array([0, 1]), np.array([-55.1, -61.7]))


def test_frame_roundtrip_quantization_forwarding_and_mtu():
    p = RangePacket(
        2,
        65535,
        1,
        0xFFFFFFFF,
        np.arange(MAX_RANGE_UPDATES),
        np.linspace(-80, -40, MAX_RANGE_UPDATES),
    )
    q = RangePacket.decode(p.encode())
    assert q.size_bytes == len(q.encode()) <= 244 and q.size_bytes == 243
    assert q.sender == 2 and q.sequence == 65535 and q.pose_epoch == 0xFFFFFFFF
    np.testing.assert_array_equal(q.rssi_dbm, p.rssi_dbm)
    np.testing.assert_array_equal(q.beacons, p.beacons)
    assert q.forwarded().ttl == 0 and q.forwarded().forwarded() is None
    assert not q.beacons.flags.writeable and not q.rssi_dbm.flags.writeable


@pytest.mark.parametrize("removed", range(1, 34))
def test_any_truncated_range_frame_rejects_both_decoders(removed):
    payload = packet(epoch=1).encode()[:-removed]
    with pytest.raises(ValueError):
        RangePacket.decode(payload)
    with pytest.raises(ValueError):
        GaussianPacket.decode(payload)


def test_gaussian_packets_never_decode_as_ranges():
    for epoch in (0, 1, 42):
        p = packetize_gaussians(
            0, 7, np.array([0, 1]), np.ones(2), np.ones(2), map_epoch=epoch
        )[0]
        with pytest.raises(ValueError):
            RangePacket.decode(p.encode())
    with pytest.raises(ValueError):
        GaussianPacket.decode(packet(epoch=42).encode())


@pytest.mark.parametrize(
    "overrides",
    [
        {"sender": True},
        {"sequence": 65536},
        {"ttl": -1},
        {"pose_epoch": 2**32},
        {"beacons": np.array([-1, 1])},
        {"beacons": np.array([0, 0])},
        {"beacons": np.array([1.0, 2.0])},
        {"rssi_dbm": np.array([float("nan"), -50])},
        {"rssi_dbm": np.array([1e100, -50])},
    ],
)
def test_invalid_range_fields(overrides):
    with pytest.raises(ValueError):
        replace(packet(), **overrides)


def test_source_batch_splits_without_sequence_wrap_or_repeated_ids():
    count = MAX_RANGE_UPDATES + 1
    packets = packetize_ranges(0, 65534, np.arange(count), np.full(count, -60.0))
    assert [p.sequence for p in packets] == [65534, 65535]
    assert all(p.size_bytes <= 244 for p in packets)
    with pytest.raises(OverflowError):
        packetize_ranges(0, 65535, np.arange(count), np.full(count, -60.0))
    with pytest.raises(ValueError):
        packetize_ranges(0, 0, np.zeros(count, int), np.full(count, -60.0))


def test_radio_requires_explicit_range_decoder_before_airtime():
    team, _, _ = world()
    with pytest.raises(ValueError):
        team.radio.flood_packet(0, packet(), np.random.default_rng(7))
    assert team.radio._next_flood_slot == 0
    result = team.radio.flood_packet(
        0, packet(), np.random.default_rng(7), decoder=RangePacket.decode
    )
    assert all(len(received) == 1 for received in result.received)
    assert result.stats.bytes_air % packet().size_bytes == 0


def test_lossless_receivers_equal_shared_gbp_on_the_same_quantized_factors():
    team, readings, covariance = world()
    outcome = team.exchange(readings, np.random.default_rng(3))
    assert (
        outcome.available
        and outcome.rounds == 1
        and outcome.factor_counts == (8, 8, 8, 8)
    )
    shared = RangeFactorGBP(team.receivers[0].anchors, covariance, 2)
    for source in range(4):
        for beacon in range(2):
            shared.observe_rssi(
                source,
                beacon,
                0,
                float(np.float32(readings[source, beacon])),
                config=team.sensor_config,
            )
    reference = shared.solve()
    for result in outcome.solutions:
        for expected, actual in zip(reference.estimates, result.estimates, strict=True):
            np.testing.assert_allclose(
                actual.position_m, expected.position_m, atol=1e-8
            )
            np.testing.assert_allclose(
                actual.covariance_m2, expected.covariance_m2, atol=1e-8
            )
    assert outcome.team_numeric_bytes > outcome.receiver_numeric_bytes
    assert not np.shares_memory(team.receivers[0].ranges, team.receivers[1].ranges)


def test_partition_does_not_inject_undelivered_table_entries():
    first, readings, _ = world(loss=1)
    second, _, _ = world(loss=1)
    changed = readings.copy()
    changed[3] += 10
    a = first.exchange(readings, np.random.default_rng(7), max_rounds=2)
    b = second.exchange(changed, np.random.default_rng(7), max_rounds=2)
    assert not a.available and not b.available
    assert a.factor_counts == (2, 2, 2, 2) and a.rounds == 2
    assert a.solutions == (None,) * 4
    for receiver in range(3):
        np.testing.assert_array_equal(
            first.receivers[receiver].ranges, second.receivers[receiver].ranges
        )
        assert np.all(first.receivers[receiver].sequences[3] == -1)
    assert a.radio_stats.randomly_dropped > 0


def test_asymmetric_delivery_produces_different_receiver_estimates():
    walls = np.zeros((4, 4), int)
    walls[3, :3] = 100
    team, readings, _ = world(walls=walls, beacons=1)
    readings[3] += 4
    result = team.exchange(readings, np.random.default_rng(7))
    assert result.available and result.factor_counts == (3, 3, 3, 4)
    a = result.solutions[0].estimates[0].position_m
    b = result.solutions[3].estimates[0].position_m
    assert np.linalg.norm(a - b) > 0.1
    assert result.radio_stats.below_sensitivity > 0


def test_retries_and_replays_never_increase_precision_for_the_same_reading():
    team, readings, _ = world()
    first = team.exchange(readings, np.random.default_rng(7))
    cov = first.solutions[1].estimates[0].covariance_m2.copy()
    p = RangePacket(0, 0, 2, 0, np.arange(2), readings[0])
    engine = team.receivers[1]
    before = tuple(
        value.copy()
        for value in (engine.ranges, engine.variances, engine.log_range_variances)
    )
    prior = engine.beacon_prior_mean.copy()
    for _ in range(100):
        assert not team.ingest(1, p)
    for old, new in zip(
        before,
        (engine.ranges, engine.variances, engine.log_range_variances),
        strict=True,
    ):
        np.testing.assert_array_equal(old, new)
    after = team.receivers[1].solve()
    np.testing.assert_allclose(
        after.estimates[0].covariance_m2, cov, atol=10 * engine.config.tolerance
    )
    np.testing.assert_array_equal(engine.beacon_prior_mean, prior)
    assert np.count_nonzero(team.receivers[1].sequences >= 0) == 8


def test_wrong_pose_epoch_rejected_before_beacon_bounds():
    team, _, _ = world()
    bad = RangePacket(0, 0, 2, 1, np.array([65535]), np.array([-60.0]))
    assert not team.ingest(0, bad) and team.rejected_epoch_packets == 1
    assert np.all(team.receivers[0].sequences == -1)


def test_bad_likelihood_batch_fails_without_publishing_sequences():
    team, readings, _ = world()
    readings[2, 1] = -1e30
    with pytest.raises(ValueError, match="representable"):
        team.exchange(readings, np.random.default_rng(7))
    assert not np.any(team.next_sequence)
    assert all(np.all(e.sequences == -1) for e in team.receivers)


def test_source_sequence_exhaustion_never_wraps():
    team, readings, _ = world()
    team.next_sequence[:] = 65536
    with pytest.raises(OverflowError):
        team.exchange(readings, np.random.default_rng(7))
    assert all(np.all(e.sequences == -1) for e in team.receivers)


def test_search_routes_to_its_assigned_receivers_estimate():
    from ember.search import localize_and_assign

    team, _, _ = world()
    survivors = np.array([[1.5, 2], [6.5, 1.5], [2, 6.5], [6, 6]], float)
    result = localize_and_assign(
        team.receivers[0].anchors,
        survivors,
        np.random.default_rng(7),
        range_gbp=True,
        range_radio=team.radio,
    )
    assert result.range_radio_enabled and result.range_gbp_converged
    assert result.range_radio_stats.bytes_air > 0
    assert result.range_radio_rounds == 1 and sorted(result.robot_to_beacon) == list(
        range(4)
    )
    for receiver, beacon in enumerate(result.robot_to_beacon):
        np.testing.assert_array_equal(
            result.estimated_positions_m[beacon],
            result.receiver_estimated_positions_m[receiver, beacon],
        )
        np.testing.assert_array_equal(
            result.covariance_m2[beacon],
            result.receiver_covariance_m2[receiver, beacon],
        )


def test_partitioned_search_reports_unavailable_instead_of_using_shared_table():
    from ember.search import localize_and_assign

    team, _, _ = world(loss=1)
    survivors = np.array([[1.5, 2], [6.5, 1.5], [2, 6.5], [6, 6]], float)
    with pytest.raises(RuntimeError, match="unavailable for receivers"):
        localize_and_assign(
            team.receivers[0].anchors,
            survivors,
            np.random.default_rng(7),
            range_gbp=True,
            range_radio=team.radio,
        )


@pytest.mark.parametrize(
    "args",
    [
        ["--range-radio", "--live-disaster"],
        ["--range-rounds", "0"],
        ["--range-rounds", "9"],
    ],
)
def test_cli_rejects_unsupported_range_delivery_configuration(args):
    from cascade.demo import main

    with pytest.raises(SystemExit) as result:
        main(args)
    assert result.value.code == 2


def test_retry_continues_when_received_anchor_count_is_sufficient_but_collinear(
    monkeypatch,
):
    poses = np.array([[0, 0], [4, 0], [8, 0], [0, 8]], float)
    radio = RadioMedium(poses, loss_seed=7, config=RadioConfig(packet_error_rate=0))
    team = RangeLocalizationTeam(poses, np.tile(np.eye(2) * 0.01, (4, 1, 1)), 1, radio)
    cfg = team.sensor_config
    readings = sample_rssi(
        poses, np.array([3, 2]), np.random.default_rng(7), config=cfg
    )[:, None]
    original = radio.transmit
    blocked = False

    def lose_fourth_source_once(frames, rng):
        nonlocal blocked
        deliveries, stats = original(frames, rng)
        if not blocked and RangePacket.decode(frames[0].payload).sender == 3:
            blocked = True
            return (), replace(stats, delivered=0, randomly_dropped=stats.delivered)
        return deliveries, stats

    monkeypatch.setattr(radio, "transmit", lose_fourth_source_once)
    result = team.exchange(readings, np.random.default_rng(7), max_rounds=3)
    assert result.available and result.rounds == 2
    assert result.factor_counts == (4, 4, 4, 4)
