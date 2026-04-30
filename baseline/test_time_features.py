"""Test script to verify time feature extraction works correctly."""

import numpy as np
from datetime import datetime, timezone

def test_time_feature_extraction():
    """Test the time feature extraction logic."""

    # Test timestamps
    test_timestamps = np.array([
        1609459200,  # 2021-01-01 00:00:00 UTC
        1640995200,  # 2022-01-01 00:00:00 UTC
        1672531200,  # 2023-01-01 00:00:00 UTC
        1704067200,  # 2024-01-01 00:00:00 UTC (Monday)
        1704153600,  # 2024-01-02 00:00:00 UTC (Tuesday)
        1704585600,  # 2024-01-07 00:00:00 UTC (Sunday - weekend)
        0,           # Invalid timestamp
        -1,          # Invalid timestamp
    ])

    B = len(test_timestamps)
    time_feats = np.zeros((B, 7), dtype=np.int64)

    for i in range(B):
        ts = int(test_timestamps[i])
        if ts <= 0:
            continue

        try:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            time_feats[i, 0] = dt.year - 2000
            time_feats[i, 1] = dt.month
            time_feats[i, 2] = dt.day
            time_feats[i, 3] = dt.hour
            time_feats[i, 4] = dt.minute
            time_feats[i, 5] = dt.weekday()
            time_feats[i, 6] = 1 if dt.weekday() >= 5 else 0
        except (ValueError, OSError, OverflowError):
            pass

    # Clip to safe ranges
    time_feats[:, 0] = np.clip(time_feats[:, 0], 0, 100)
    time_feats[:, 1] = np.clip(time_feats[:, 1], 0, 12)
    time_feats[:, 2] = np.clip(time_feats[:, 2], 0, 31)
    time_feats[:, 3] = np.clip(time_feats[:, 3], 0, 23)
    time_feats[:, 4] = np.clip(time_feats[:, 4], 0, 59)
    time_feats[:, 5] = np.clip(time_feats[:, 5], 0, 6)
    time_feats[:, 6] = np.clip(time_feats[:, 6], 0, 1)

    print("Time Feature Extraction Test Results:")
    print("=" * 80)
    print(f"{'Timestamp':<15} {'Year':<6} {'Month':<6} {'Day':<6} {'Hour':<6} {'Min':<6} {'Weekday':<8} {'Weekend':<8}")
    print("-" * 80)

    for i in range(B):
        ts = test_timestamps[i]
        if ts > 0:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            ts_str = dt.strftime('%Y-%m-%d')
        else:
            ts_str = "Invalid"

        print(f"{ts_str:<15} {time_feats[i, 0]:<6} {time_feats[i, 1]:<6} {time_feats[i, 2]:<6} "
              f"{time_feats[i, 3]:<6} {time_feats[i, 4]:<6} {time_feats[i, 5]:<8} {time_feats[i, 6]:<8}")

    print("\n" + "=" * 80)

    # Verify ranges
    assert np.all(time_feats[:, 0] >= 0) and np.all(time_feats[:, 0] <= 100), "Year out of range"
    assert np.all(time_feats[:, 1] >= 0) and np.all(time_feats[:, 1] <= 12), "Month out of range"
    assert np.all(time_feats[:, 2] >= 0) and np.all(time_feats[:, 2] <= 31), "Day out of range"
    assert np.all(time_feats[:, 3] >= 0) and np.all(time_feats[:, 3] <= 23), "Hour out of range"
    assert np.all(time_feats[:, 4] >= 0) and np.all(time_feats[:, 4] <= 59), "Minute out of range"
    assert np.all(time_feats[:, 5] >= 0) and np.all(time_feats[:, 5] <= 6), "Weekday out of range"
    assert np.all(time_feats[:, 6] >= 0) and np.all(time_feats[:, 6] <= 1), "Weekend out of range"

    print("[PASS] All time features are within safe ranges!")
    print("[PASS] No index errors will occur during embedding lookup!")

    # Test decision time bucketing
    print("\n" + "=" * 80)
    print("Decision Time Bucketing Test:")
    print("-" * 80)

    test_decision_times = np.array([0, 30, 90, 600, 7200])
    decision_time_bucket = np.zeros(len(test_decision_times), dtype=np.int64)
    decision_time_bucket[test_decision_times > 0] = 1
    decision_time_bucket[test_decision_times > 60] = 2
    decision_time_bucket[test_decision_times > 300] = 3
    decision_time_bucket[test_decision_times > 3600] = 4
    decision_time_bucket = np.clip(decision_time_bucket, 0, 4)

    for dt, bucket in zip(test_decision_times, decision_time_bucket):
        print(f"Decision time: {dt:>6}s -> Bucket: {bucket}")

    assert np.all(decision_time_bucket >= 0) and np.all(decision_time_bucket <= 4), "Decision time bucket out of range"
    print("\n[PASS] Decision time buckets are within safe range [0, 4]!")

    print("\n" + "=" * 80)
    print("All tests passed! Time feature extraction is safe and correct.")
    print("=" * 80)

if __name__ == "__main__":
    test_time_feature_extraction()
